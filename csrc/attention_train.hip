#include "attention_train.hpp"
#include "wave64.hpp"

// Training attention for gfx906: forward with log-sum-exp, backward in
// FlashAttention-2's split (DESIGN.md, "Training attention"). Head dim 128,
// fp16 in/out, fp32 accumulation, causal or full, native GQA.
//
// Every matmul is v_dot2_f32_f16 over operands that are both contiguous along
// the reduction dimension in LDS (dot_tile). Operands needed transposed are
// written transposed into LDS while staging (stage<..., TR>), as half2 pairs of
// two rows built with v_perm_b32.
//
// Thread layout (all kernels): 256 threads, tx = tid % 16 (one DPP row), ty =
// tid / 16. A thread owns output rows ty + 16 * i and columns tx + 16 * j, so
// the 16 lanes of a DPP row share their rows (row reductions by DPP; the rows
// of P / dS a wave writes are the rows that same wave reads back: no workgroup
// barrier between them).
//
// Tensors: [B, L, heads, 128] with element strides (Str) for batch, token, head;
// channels contiguous and rows 16-byte aligned (checked in binding.cpp).
// LSE / Delta: fp32 [B, H, L]. Softmax in log2 units: s = q.k * scale * log2(e).

#define TD 128
#define NT 256
#define TY 16
#define RS_N 272  // natural row: 128 halves + 16-byte pad (16 lanes, 16 rows: no bank conflicts)
#define NEG_BIG -1e30f

__device__ __forceinline__ void wave_sync_lds() {
    // rows written and read by the same wave: order the LDS accesses only
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Staging of rows row0 .. row0 + R - 1 of one head (g = its row 0, sl = token
// stride), zero past L, into LDS: natural ([R][RS_N]) and/or transposed
// ([128][TRS], dword p of channel row d = channel d of rows 2p, 2p + 1). Item i
// of R * 8 is row pair rp = i % (R / 2), 16-byte chunk c = i / (R / 2):
// consecutive lanes take consecutive row pairs of one chunk, so the transposed
// writes of a lane group hit consecutive dwords and a wave reads 64-byte row
// segments. Split in load (to registers) and store (to LDS) so a kernel can
// load the next tile while it computes on this one: at 1 wave per SIMD
// nothing else hides global latency.
template <int R> struct Pairs {
    static_assert(R * 8 % NT == 0, "items must divide over the workgroup");
    static constexpr int N = R * 8 / NT;
    uint4 a[N], b[N];  // indices are unrolled constants: registers, not scratch
};

template <int R>
__device__ __forceinline__ void load_pairs(const _Float16* g, long long sl, int row0, int L, Pairs<R>& p) {
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < Pairs<R>::N; n++) {
        const int i = threadIdx.x + n * NT, rp = i % (R / 2), c = i / (R / 2), r = row0 + 2 * rp;
        // Load from a clamped row, then select: `r < L ? load : zero` became a
        // select between the global address and a scratch copy of zero, i.e.
        // flat loads plus 32 B/lane of scratch.
        const uint4 a = ((const uint4*)(g + min(r, L - 1) * sl))[c];
        const uint4 b = ((const uint4*)(g + min(r + 1, L - 1) * sl))[c];
        p.a[n] = r < L ? a : zero;
        p.b[n] = r + 1 < L ? b : zero;
    }
}

template <int R, int TRS, bool NAT, bool TR>
__device__ __forceinline__ void store_pairs(const Pairs<R>& p, uint8_t* nat, uint8_t* T) {
#pragma unroll
    for (int n = 0; n < Pairs<R>::N; n++) {
        const int i = threadIdx.x + n * NT, rp = i % (R / 2), c = i / (R / 2);
        const uint4 a = p.a[n], b = p.b[n];
        if constexpr (NAT) {
            *(uint4*)(nat + (2 * rp) * RS_N + c * 16) = a;
            *(uint4*)(nat + (2 * rp + 1) * RS_N + c * 16) = b;
        }
        if constexpr (TR) {
            // fields written out: a local array here went to scratch (32 B/lane)
            uint8_t* col = T + c * 8 * TRS + rp * 4;
            *(uint32_t*)(col + 0 * TRS) = pair_lo(a.x, b.x);
            *(uint32_t*)(col + 1 * TRS) = pair_hi(a.x, b.x);
            *(uint32_t*)(col + 2 * TRS) = pair_lo(a.y, b.y);
            *(uint32_t*)(col + 3 * TRS) = pair_hi(a.y, b.y);
            *(uint32_t*)(col + 4 * TRS) = pair_lo(a.z, b.z);
            *(uint32_t*)(col + 5 * TRS) = pair_hi(a.z, b.z);
            *(uint32_t*)(col + 6 * TRS) = pair_lo(a.w, b.w);
            *(uint32_t*)(col + 7 * TRS) = pair_hi(a.w, b.w);
        }
    }
}

template <int R, int TRS, bool NAT, bool TR>
__device__ __forceinline__ void stage(const _Float16* g, long long sl, int row0, int L, uint8_t* nat, uint8_t* T) {
    Pairs<R> p;
    load_pairs<R>(g, sl, row0, L, p);
    store_pairs<R, TRS, NAT, TR>(p, nat, T);
}

// acc[i][j] += A[ty + 16 i] . B[tx + 16 j] over CH 16-byte chunks (8 halves each).
// A rows are shared by the 16 lanes of a DPP row (LDS broadcast).
template <int RA, int RB, int CH>
__device__ __forceinline__ void dot_tile(const uint8_t* A, int as, const uint8_t* B, int bs, int ty, int tx,
                                         float (&acc)[RA][RB]) {
#pragma unroll 2
    for (int c = 0; c < CH; c++) {
        uint4 bq[RB];
#pragma unroll
        for (int j = 0; j < RB; j++) bq[j] = *(const uint4*)(B + (tx + 16 * j) * bs + c * 16);
#pragma unroll
        for (int i = 0; i < RA; i++) {
            const uint4 aq = *(const uint4*)(A + (ty + TY * i) * as + c * 16);
#pragma unroll
            for (int j = 0; j < RB; j++) acc[i][j] = dot8h(aq, bq[j], acc[i][j]);
        }
    }
}

// ---- forward: 64 query rows per workgroup, 64-key tiles ----
// LDS: Q 17,408 + K 17,408 + V^T 18,432 + P 9,216 = 62,464 B.
__global__ void __launch_bounds__(NT) attn_fwd(const _Float16* __restrict__ Q, Str qs, const _Float16* __restrict__ K,
                                               Str ks, const _Float16* __restrict__ V, Str vs, _Float16* __restrict__ O,
                                               Str os, float* __restrict__ LSE, int H, int KVH, int L, float scale_log2,
                                               int causal) {
    constexpr int BM = 64, BN = 64, RPT = 4, KPT = 4, DPT = 8;
    constexpr int TRS = BN * 2 + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t VT[TD * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t Ps[BM * TRS];

    const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * BM, kvh = h / (H / KVH);
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const _Float16* Qg = Q + b * qs.b + h * qs.h;
    const _Float16* Kg = K + b * ks.b + kvh * ks.h;
    const _Float16* Vg = V + b * vs.b + kvh * vs.h;
    stage<BM, 0, true, false>(Qg, qs.l, q0, L, Qs, nullptr);

    float o[RPT][DPT], m[RPT], l_part[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {
        m[i] = NEG_BIG;
        l_part[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < DPT; j++) o[i][j] = 0.0f;
    }
    const int all_tiles = (L + BN - 1) / BN;
    const int n_tiles = causal ? min(all_tiles, (q0 + BM - 1) / BN + 1) : all_tiles;
    Pairs<BN> k_next, v_next;
    load_pairs<BN>(Kg, ks.l, 0, L, k_next);
    load_pairs<BN>(Vg, vs.l, 0, L, v_next);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * BN;
        __syncthreads();  // every wave is done with the previous tile's K, V^T
        store_pairs<BN, 0, true, false>(k_next, Ks, nullptr);
        store_pairs<BN, TRS, false, true>(v_next, nullptr, VT);
        __syncthreads();
        if (t + 1 < n_tiles) load_pairs<BN>(Kg, ks.l, k0 + BN, L, k_next);
        if (t + 1 < n_tiles) load_pairs<BN>(Vg, vs.l, k0 + BN, L, v_next);

        float s[RPT][KPT] = {};
        dot_tile<RPT, KPT, TD * 2 / 16>(Qs, RS_N, Ks, RS_N, ty, tx, s);
#pragma unroll
        for (int i = 0; i < RPT; i++) {
            const int row = q0 + ty + TY * i;
            float tmax = NEG_BIG;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                const int key = k0 + tx + 16 * j;
                s[i][j] = (key >= L || (causal && key > row)) ? NEG_BIG : s[i][j] * scale_log2;
                tmax = fmaxf(tmax, s[i][j]);
            }
            tmax = row_max16(tmax);
            // key 0 is in tile 0 and visible to every row: m is finite from the first tile on
            const float m_new = fmaxf(m[i], tmax);
            const float alpha = exp2f(m[i] - m_new);
            m[i] = m_new;
            l_part[i] *= alpha;
#pragma unroll
            for (int j = 0; j < DPT; j++) o[i][j] *= alpha;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                const float p = exp2f(s[i][j] - m_new);
                l_part[i] += p;
                *(_Float16*)(Ps + (ty + TY * i) * TRS + (tx + 16 * j) * 2) = (_Float16)p;
            }
        }
        wave_sync_lds();
        dot_tile<RPT, DPT, BN * 2 / 16>(Ps, TRS, VT, TRS, ty, tx, o);
    }

#pragma unroll
    for (int i = 0; i < RPT; i++) {
        const float l = row_sum16(l_part[i]);
        const int row = q0 + ty + TY * i;
        if (row >= L) continue;
        _Float16* dst = O + b * os.b + row * os.l + h * os.h;
#pragma unroll
        for (int j = 0; j < DPT; j++) dst[tx + 16 * j] = (_Float16)(o[i][j] / l);
        if (tx == 0) LSE[((long long)b * H + h) * L + row] = m[i] + log2f(l);
    }
}

// ---- backward preprocess: Delta = rowsum(dO * O), one DPP row per query row ----
__global__ void __launch_bounds__(NT) attn_bwd_delta(const _Float16* __restrict__ O, Str os,
                                                     const _Float16* __restrict__ dO, Str dos, float* __restrict__ Delta,
                                                     int H, int L, long long rows) {
    const long long r = (long long)blockIdx.x * TY + threadIdx.x / 16;
    const int tx = threadIdx.x % 16;
    // whole DPP rows leave together: the reduction below stays row-local
    if (r >= rows) return;
    const int l = r % L, h = (r / L) % H, b = r / ((long long)L * H);
    const uint4 ov = ((const uint4*)(O + b * os.b + l * os.l + h * os.h))[tx];
    const uint4 dv = ((const uint4*)(dO + b * dos.b + l * dos.l + h * dos.h))[tx];
    const float sum = row_sum16(dot8h(ov, dv, 0.0f));
    if (tx == 0) Delta[r] = sum;
}

// ---- backward dK, dV: 32 keys per workgroup, 32-query tiles, all G query heads of the KV head ----
// LDS: K, V, Q, dO natural 4 x 8,704; Q^T, dO^T 2 x 10,240; P^T, dS^T 2 x 2,560 = 60,416 B (+ 256).
__global__ void __launch_bounds__(NT) attn_bwd_dkdv(const _Float16* __restrict__ Q, Str qs,
                                                    const _Float16* __restrict__ K, Str ks,
                                                    const _Float16* __restrict__ V, Str vs,
                                                    const _Float16* __restrict__ dO, Str dos,
                                                    const float* __restrict__ LSE, const float* __restrict__ Delta,
                                                    _Float16* __restrict__ dK, Str dks, _Float16* __restrict__ dV, Str dvs,
                                                    int H, int KVH, int L, float scale_log2, float scale, int causal) {
    constexpr int BN = 32, BM = 32, RA = BN / TY, RB = BM / 16, DPT = 8;
    constexpr int TRS = BM * 2 + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Vs[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t dOs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t QT[TD * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t dOT[TD * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t PT[BN * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t dST[BN * TRS];
    __shared__ float lse_s[BM], delta_s[BM];

    const int b = blockIdx.z, kvh = blockIdx.y, k0 = blockIdx.x * BN, G = H / KVH;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    stage<BN, 0, true, false>(K + b * ks.b + kvh * ks.h, ks.l, k0, L, Ks, nullptr);
    stage<BN, 0, true, false>(V + b * vs.b + kvh * vs.h, vs.l, k0, L, Vs, nullptr);

    float dk[RA][DPT] = {}, dv[RA][DPT] = {};
    // causal: query tiles before the block's first key see none of its keys
    const int i_begin = causal ? (k0 / BM) * BM : 0;
    const int n_qt = (L - i_begin + BM - 1) / BM;
    // one step = one query tile of one query head of the group, flattened so the
    // prefetch crosses head boundaries too
    Pairs<BM> q_next, do_next;
    float lse_next = 0.0f, delta_next = 0.0f;
    auto load_step = [&](int step) {
        const int h = kvh * G + step / n_qt, i0 = i_begin + (step % n_qt) * BM;
        load_pairs<BM>(Q + b * qs.b + h * qs.h, qs.l, i0, L, q_next);
        load_pairs<BM>(dO + b * dos.b + h * dos.h, dos.l, i0, L, do_next);
        const long long row = ((long long)b * H + h) * L + i0 + tid;
        if (tid < BM) lse_next = i0 + tid < L ? LSE[row] : 0.0f;
        if (tid < BM) delta_next = i0 + tid < L ? Delta[row] : 0.0f;
    };
    load_step(0);
    for (int step = 0; step < G * n_qt; step++) {
        const int i0 = i_begin + (step % n_qt) * BM;
        __syncthreads();  // every wave is done with the previous tile
        store_pairs<BM, TRS, true, true>(q_next, Qs, QT);
        store_pairs<BM, TRS, true, true>(do_next, dOs, dOT);
        if (tid < BM) {
            lse_s[tid] = lse_next;
            delta_s[tid] = delta_next;
        }
        __syncthreads();
        if (step + 1 < G * n_qt) load_step(step + 1);

        float st[RA][RB] = {}, dpt[RA][RB] = {};
        dot_tile<RA, RB, TD * 2 / 16>(Ks, RS_N, Qs, RS_N, ty, tx, st);
        dot_tile<RA, RB, TD * 2 / 16>(Vs, RS_N, dOs, RS_N, ty, tx, dpt);
#pragma unroll
        for (int a = 0; a < RA; a++) {
            const int key = k0 + ty + TY * a;
#pragma unroll
            for (int c = 0; c < RB; c++) {
                const int qi = tx + 16 * c, q = i0 + qi;
                const bool masked = key >= L || q >= L || (causal && q < key);
                const float p = masked ? 0.0f : exp2f(st[a][c] * scale_log2 - lse_s[qi]);
                const float ds = p * (dpt[a][c] - delta_s[qi]);
                *(_Float16*)(PT + (ty + TY * a) * TRS + qi * 2) = (_Float16)p;
                *(_Float16*)(dST + (ty + TY * a) * TRS + qi * 2) = (_Float16)ds;
            }
        }
        wave_sync_lds();
        dot_tile<RA, DPT, BM * 2 / 16>(PT, TRS, dOT, TRS, ty, tx, dv);
        dot_tile<RA, DPT, BM * 2 / 16>(dST, TRS, QT, TRS, ty, tx, dk);
    }
#pragma unroll
    for (int a = 0; a < RA; a++) {
        const int key = k0 + ty + TY * a;
        if (key >= L) continue;
        _Float16* dk_dst = dK + b * dks.b + key * dks.l + kvh * dks.h;
        _Float16* dv_dst = dV + b * dvs.b + key * dvs.l + kvh * dvs.h;
#pragma unroll
        for (int j = 0; j < DPT; j++) {
            dk_dst[tx + 16 * j] = (_Float16)(dk[a][j] * scale);
            dv_dst[tx + 16 * j] = (_Float16)dv[a][j];
        }
    }
}

// ---- backward dQ: 32 query rows per workgroup, 32-key tiles ----
// LDS: Q, dO, K, V natural 4 x 8,704; K^T 10,240; dS 2,560 = 47,616 B.
__global__ void __launch_bounds__(NT) attn_bwd_dq(const _Float16* __restrict__ Q, Str qs,
                                                  const _Float16* __restrict__ K, Str ks,
                                                  const _Float16* __restrict__ V, Str vs,
                                                  const _Float16* __restrict__ dO, Str dos,
                                                  const float* __restrict__ LSE, const float* __restrict__ Delta,
                                                  _Float16* __restrict__ dQ, Str dqs, int H, int KVH, int L,
                                                  float scale_log2, float scale, int causal) {
    constexpr int BM = 32, BN = 32, RA = BM / TY, RB = BN / 16, DPT = 8;
    constexpr int TRS = BN * 2 + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t dOs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Vs[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t KT[TD * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t dSs[BM * TRS];

    const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * BM, kvh = h / (H / KVH);
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    stage<BM, 0, true, false>(Q + b * qs.b + h * qs.h, qs.l, q0, L, Qs, nullptr);
    stage<BM, 0, true, false>(dO + b * dos.b + h * dos.h, dos.l, q0, L, dOs, nullptr);
    const _Float16* Kg = K + b * ks.b + kvh * ks.h;
    const _Float16* Vg = V + b * vs.b + kvh * vs.h;

    float lse_r[RA], delta_r[RA];
#pragma unroll
    for (int a = 0; a < RA; a++) {
        const int row = q0 + ty + TY * a;
        lse_r[a] = row < L ? LSE[((long long)b * H + h) * L + row] : 0.0f;
        delta_r[a] = row < L ? Delta[((long long)b * H + h) * L + row] : 0.0f;
    }
    float dq[RA][DPT] = {};
    const int all_tiles = (L + BN - 1) / BN;
    const int n_tiles = causal ? min(all_tiles, (q0 + BM - 1) / BN + 1) : all_tiles;
    Pairs<BN> k_next, v_next;
    load_pairs<BN>(Kg, ks.l, 0, L, k_next);
    load_pairs<BN>(Vg, vs.l, 0, L, v_next);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * BN;
        __syncthreads();  // every wave is done with the previous tile
        store_pairs<BN, TRS, true, true>(k_next, Ks, KT);
        store_pairs<BN, 0, true, false>(v_next, Vs, nullptr);
        __syncthreads();
        if (t + 1 < n_tiles) load_pairs<BN>(Kg, ks.l, k0 + BN, L, k_next);
        if (t + 1 < n_tiles) load_pairs<BN>(Vg, vs.l, k0 + BN, L, v_next);

        float s[RA][RB] = {}, dp[RA][RB] = {};
        dot_tile<RA, RB, TD * 2 / 16>(Qs, RS_N, Ks, RS_N, ty, tx, s);
        dot_tile<RA, RB, TD * 2 / 16>(dOs, RS_N, Vs, RS_N, ty, tx, dp);
#pragma unroll
        for (int a = 0; a < RA; a++) {
            const int row = q0 + ty + TY * a;
#pragma unroll
            for (int c = 0; c < RB; c++) {
                const int kj = tx + 16 * c, key = k0 + kj;
                const bool masked = key >= L || row >= L || (causal && key > row);
                const float p = masked ? 0.0f : exp2f(s[a][c] * scale_log2 - lse_r[a]);
                *(_Float16*)(dSs + (ty + TY * a) * TRS + kj * 2) = (_Float16)(p * (dp[a][c] - delta_r[a]));
            }
        }
        wave_sync_lds();
        dot_tile<RA, DPT, BN * 2 / 16>(dSs, TRS, KT, TRS, ty, tx, dq);
    }
#pragma unroll
    for (int a = 0; a < RA; a++) {
        const int row = q0 + ty + TY * a;
        if (row >= L) continue;
        _Float16* dst = dQ + b * dqs.b + row * dqs.l + h * dqs.h;
#pragma unroll
        for (int j = 0; j < DPT; j++) dst[tx + 16 * j] = (_Float16)(dq[a][j] * scale);
    }
}

// ---- host dispatch (binding.cpp validates shapes, strides, alignment) ----
static constexpr float LOG2E = 1.4426950408889634f;

void attn_train_fwd_dispatch(const void* q, Str qs, const void* k, Str ks, const void* v, Str vs, void* o, Str os,
                             float* lse, int B, int H, int KVH, int L, float scale, int causal, hipStream_t stream) {
    dim3 grid((L + 63) / 64, H, B);
    attn_fwd<<<grid, NT, 0, stream>>>((const _Float16*)q, qs, (const _Float16*)k, ks, (const _Float16*)v, vs,
                                      (_Float16*)o, os, lse, H, KVH, L, scale * LOG2E, causal);
}

void attn_train_bwd_dispatch(const void* q, Str qs, const void* k, Str ks, const void* v, Str vs, const void* o, Str os,
                             const void* dout, Str dos, const float* lse, float* delta, void* dq, Str dqs, void* dk,
                             Str dks, void* dv, Str dvs, int B, int H, int KVH, int L, float scale, int causal,
                             int stages, hipStream_t stream) {
    const long long rows = (long long)B * H * L;
    if (stages & BWD_DELTA)
        attn_bwd_delta<<<(rows + TY - 1) / TY, NT, 0, stream>>>((const _Float16*)o, os, (const _Float16*)dout, dos,
                                                                delta, H, L, rows);
    if (stages & BWD_DKDV)
        attn_bwd_dkdv<<<dim3((L + 31) / 32, KVH, B), NT, 0, stream>>>(
        (const _Float16*)q, qs, (const _Float16*)k, ks, (const _Float16*)v, vs, (const _Float16*)dout, dos, lse, delta,
        (_Float16*)dk, dks, (_Float16*)dv, dvs, H, KVH, L, scale * LOG2E, scale, causal);
    if (stages & BWD_DQ)
        attn_bwd_dq<<<dim3((L + 31) / 32, H, B), NT, 0, stream>>>(
        (const _Float16*)q, qs, (const _Float16*)k, ks, (const _Float16*)v, vs, (const _Float16*)dout, dos, lse, delta,
        (_Float16*)dq, dqs, H, KVH, L, scale * LOG2E, scale, causal);
}
