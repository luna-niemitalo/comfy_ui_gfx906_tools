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
// nothing else hides global latency. THREADS = the workgroup size; with fewer
// items than threads the rest idle.
//
// Swizzled transposed layout (SWZ, t_row): channel row d starts at d * TRS +
// (d >> 3) * 16. Lanes that read rows 8 apart (dkdv's 8 lanes of a key) then hit
// disjoint banks: 8 * TRS is a multiple of 128 B for any 16-byte TRS, and the
// extra 16 B per 8-row group moves each group 4 banks on. Monotone on purpose:
// a wrapping offset ((d >> 3) & 7) let a group's last row overlap the next
// group's first row (dK/dV 12% off).
template <int TRS> constexpr int t_bytes_swz() { return TD * TRS + (TD / 8 - 1) * 16; }
template <int TRS, bool SWZ> __device__ __forceinline__ int t_row(int d) {
    return d * TRS + (SWZ ? (d >> 3) * 16 : 0);
}

template <int R, int THREADS = NT> struct Pairs {
    static constexpr int ITEMS = R * 8;
    static constexpr int N = (ITEMS + THREADS - 1) / THREADS;
    uint4 a[N], b[N];  // indices are unrolled constants: registers, not scratch
};

template <int R, int THREADS>
__device__ __forceinline__ void load_pairs(const _Float16* g, long long sl, int row0, int L, Pairs<R, THREADS>& p) {
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < Pairs<R, THREADS>::N; n++) {
        const int i = threadIdx.x + n * THREADS, rp = i % (R / 2), c = i / (R / 2), r = row0 + 2 * rp;
        if (Pairs<R, THREADS>::ITEMS % THREADS != 0 && i >= Pairs<R, THREADS>::ITEMS) break;
        // Load from a clamped row, then select: `r < L ? load : zero` became a
        // select between the global address and a scratch copy of zero, i.e.
        // flat loads plus 32 B/lane of scratch.
        const uint4 a = ((const uint4*)(g + min(r, L - 1) * sl))[c];
        const uint4 b = ((const uint4*)(g + min(r + 1, L - 1) * sl))[c];
        p.a[n] = r < L ? a : zero;
        p.b[n] = r + 1 < L ? b : zero;
    }
}

template <int R, int TRS, bool NAT, bool TR, bool SWZ = false, int THREADS = NT>
__device__ __forceinline__ void store_pairs(const Pairs<R, THREADS>& p, uint8_t* nat, uint8_t* T) {
#pragma unroll
    for (int n = 0; n < Pairs<R, THREADS>::N; n++) {
        const int i = threadIdx.x + n * THREADS, rp = i % (R / 2), c = i / (R / 2);
        if (Pairs<R, THREADS>::ITEMS % THREADS != 0 && i >= Pairs<R, THREADS>::ITEMS) break;
        const uint4 a = p.a[n], b = p.b[n];
        if constexpr (NAT) {
            *(uint4*)(nat + (2 * rp) * RS_N + c * 16) = a;
            *(uint4*)(nat + (2 * rp + 1) * RS_N + c * 16) = b;
        }
        if constexpr (TR) {
            // fields written out: a local array here went to scratch (32 B/lane)
            uint8_t* col = T + t_row<TRS, SWZ>(c * 8) + rp * 4;  // rows c*8 .. c*8+7 share one swizzle
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

// ---- backward dK, dV: 64 keys per workgroup, 32-query tiles, all G query heads of the KV head ----
// Each key's K and V rows stay in registers, split over the 8 lanes of an aligned
// lane group: sub = lane % 8 holds 16-byte chunks sub and sub + 8 of the row, so
// the 8 lanes' reads of one Q / dO row hit disjoint LDS banks. S^T and dP^T are
// per-lane partial dots over those 16 channels finished by reduce_scatter8 (DPP):
// each lane then owns the full scores of a different query, and exp / dS run
// once per (key, query). dV and dK: each lane accumulates its own 16 channels,
// dot2 over query pairs from the swizzled Q^T / dO^T.
// 512 threads: LDS allows one workgroup per CU, so the workgroup size sets the
// waves per SIMD (v1: 256 threads, 1 wave/SIMD, 4.3 TFLOPS).
// LDS: Q, dO natural 2 x 8,704; Q^T, dO^T 2 x 10,480; P^T, dS^T 2 x 5,120 = 48,608 B (+ 256).
constexpr int KV_NT = 512;
__global__ void __launch_bounds__(KV_NT) attn_bwd_dkdv(const _Float16* __restrict__ Q, Str qs,
                                                    const _Float16* __restrict__ K, Str ks,
                                                    const _Float16* __restrict__ V, Str vs,
                                                    const _Float16* __restrict__ dO, Str dos,
                                                    const float* __restrict__ LSE, const float* __restrict__ Delta,
                                                    _Float16* __restrict__ dK, Str dks, _Float16* __restrict__ dV, Str dvs,
                                                    int H, int KVH, int L, float scale_log2, float scale, int causal) {
    constexpr int BN = KV_NT / 8, BM = 32;
    constexpr int TRS = BM * 2 + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t dOs[BM * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t QT[t_bytes_swz<TRS>()];
    __shared__ __attribute__((aligned(16))) uint8_t dOT[t_bytes_swz<TRS>()];
    __shared__ __attribute__((aligned(16))) uint8_t PT[BN * TRS];
    __shared__ __attribute__((aligned(16))) uint8_t dST[BN * TRS];
    __shared__ float lse_s[BM], delta_s[BM];

    const int b = blockIdx.z, kvh = blockIdx.y, k0 = blockIdx.x * BN, G = H / KVH;
    const int tid = threadIdx.x, sub = tid % 8, kl = tid / 8, key = k0 + kl;
    // rows past L load row L - 1; their scores are masked below
    const uint4* krow = (const uint4*)(K + b * ks.b + (long long)min(key, L - 1) * ks.l + kvh * ks.h);
    const uint4* vrow = (const uint4*)(V + b * vs.b + (long long)min(key, L - 1) * vs.l + kvh * vs.h);
    const uint4 k_lo = krow[sub], k_hi = krow[sub + 8], v_lo = vrow[sub], v_hi = vrow[sub + 8];

    // [hb][r]: channel 64 * hb + 8 * sub + r
    float dk[2][8] = {}, dv[2][8] = {};
    // causal: query tiles before the block's first key see none of its keys
    const int i_begin = causal ? (k0 / BM) * BM : 0;
    const int n_qt = (L - i_begin + BM - 1) / BM;
    // one step = one query tile of one query head of the group, flattened so the
    // prefetch crosses head boundaries too
    Pairs<BM, KV_NT> q_next, do_next;
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
        store_pairs<BM, TRS, true, true, true>(q_next, Qs, QT);
        store_pairs<BM, TRS, true, true, true>(do_next, dOs, dOT);
        if (tid < BM) {
            lse_s[tid] = lse_next;
            delta_s[tid] = delta_next;
        }
        __syncthreads();
        if (step + 1 < G * n_qt) load_step(step + 1);

        // outer chunk loops stay rolled: unrolled, the hoisted LDS loads spilled
        // 152 B/lane at the 128-VGPR cap of 2 waves/SIMD
#pragma unroll 1
        for (int c = 0; c < BM / 8; c++) {
            float s[8], dp[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint8_t* q_row = Qs + (8 * c + j) * RS_N;
                const uint8_t* do_row = dOs + (8 * c + j) * RS_N;
                s[j] = dot8h(k_hi, *(const uint4*)(q_row + (sub + 8) * 16),
                             dot8h(k_lo, *(const uint4*)(q_row + sub * 16), 0.0f));
                dp[j] = dot8h(v_hi, *(const uint4*)(do_row + (sub + 8) * 16),
                              dot8h(v_lo, *(const uint4*)(do_row + sub * 16), 0.0f));
            }
            const float s_own = reduce_scatter8(s, sub), dp_own = reduce_scatter8(dp, sub);
            const int qi = 8 * c + sub, q = i0 + qi;
            const bool masked = key >= L || q >= L || (causal && q < key);
            const float p = masked ? 0.0f : exp2f(s_own * scale_log2 - lse_s[qi]);
            *(_Float16*)(PT + kl * TRS + qi * 2) = (_Float16)p;
            *(_Float16*)(dST + kl * TRS + qi * 2) = (_Float16)(p * (dp_own - delta_s[qi]));
        }
        wave_sync_lds();  // a key's P^T / dS^T row is written and read by its own 8 lanes
#pragma unroll 1
        for (int c = 0; c < BM * 2 / 16; c++) {
            const uint4 pv = *(const uint4*)(PT + kl * TRS + c * 16);
            const uint4 dsv = *(const uint4*)(dST + kl * TRS + c * 16);
#pragma unroll
            for (int hb = 0; hb < 2; hb++)
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const int off = t_row<TRS, true>(64 * hb + 8 * sub + r) + c * 16;
                    dv[hb][r] = dot8h(pv, *(const uint4*)(dOT + off), dv[hb][r]);
                    dk[hb][r] = dot8h(dsv, *(const uint4*)(QT + off), dk[hb][r]);
                }
        }
    }
    if (key >= L) return;
    uint4* dk_row = (uint4*)(dK + b * dks.b + (long long)key * dks.l + kvh * dks.h);
    uint4* dv_row = (uint4*)(dV + b * dvs.b + (long long)key * dvs.l + kvh * dvs.h);
#pragma unroll
    for (int hb = 0; hb < 2; hb++) {
#pragma unroll
        for (int r = 0; r < 8; r++) dk[hb][r] *= scale;
        dk_row[sub + 8 * hb] = pack8h(dk[hb]);
        dv_row[sub + 8 * hb] = pack8h(dv[hb]);
    }
}

// ---- backward dQ: 64 query rows per workgroup (512 threads), 32-key tiles ----
// dkdv's layout with the roles swapped: each query's Q and dO rows in registers
// over 8 lanes (chunks sub, sub + 8), scores by reduce_scatter8, dQ over key pairs
// from the swizzled K^T. A lane's query row is fixed: lse / delta are scalars.
// LDS: K, V natural 2 x 8,704; K^T 10,480; dS 5,120 = 33,008 B.
__global__ void __launch_bounds__(KV_NT) attn_bwd_dq(const _Float16* __restrict__ Q, Str qs,
                                                  const _Float16* __restrict__ K, Str ks,
                                                  const _Float16* __restrict__ V, Str vs,
                                                  const _Float16* __restrict__ dO, Str dos,
                                                  const float* __restrict__ LSE, const float* __restrict__ Delta,
                                                  _Float16* __restrict__ dQ, Str dqs, int H, int KVH, int L,
                                                  float scale_log2, float scale, int causal) {
    constexpr int BM = KV_NT / 8, BN = 32;
    constexpr int TRS = BN * 2 + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t Vs[BN * RS_N];
    __shared__ __attribute__((aligned(16))) uint8_t KT[t_bytes_swz<TRS>()];
    __shared__ __attribute__((aligned(16))) uint8_t dSs[BM * TRS];

    const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * BM, kvh = h / (H / KVH);
    const int tid = threadIdx.x, sub = tid % 8, ql = tid / 8, row = q0 + ql;
    // rows past L read row L - 1; their scores are masked below
    const long long r_in = min(row, L - 1);
    const uint4* q_row = (const uint4*)(Q + b * qs.b + r_in * qs.l + h * qs.h);
    const uint4* do_row = (const uint4*)(dO + b * dos.b + r_in * dos.l + h * dos.h);
    const uint4 q_lo = q_row[sub], q_hi = q_row[sub + 8], do_lo = do_row[sub], do_hi = do_row[sub + 8];
    const float lse = LSE[((long long)b * H + h) * L + r_in], delta = Delta[((long long)b * H + h) * L + r_in];
    const _Float16* Kg = K + b * ks.b + kvh * ks.h;
    const _Float16* Vg = V + b * vs.b + kvh * vs.h;

    // [hb][r]: channel 64 * hb + 8 * sub + r
    float dq[2][8] = {};
    const int all_tiles = (L + BN - 1) / BN;
    const int n_tiles = causal ? min(all_tiles, (q0 + BM - 1) / BN + 1) : all_tiles;
    Pairs<BN, KV_NT> k_next, v_next;
    load_pairs<BN>(Kg, ks.l, 0, L, k_next);
    load_pairs<BN>(Vg, vs.l, 0, L, v_next);
    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * BN;
        __syncthreads();  // every wave is done with the previous tile
        store_pairs<BN, TRS, true, true, true>(k_next, Ks, KT);
        store_pairs<BN, 0, true, false>(v_next, Vs, nullptr);
        __syncthreads();
        if (t + 1 < n_tiles) load_pairs<BN>(Kg, ks.l, k0 + BN, L, k_next);
        if (t + 1 < n_tiles) load_pairs<BN>(Vg, vs.l, k0 + BN, L, v_next);

#pragma unroll 1
        for (int c = 0; c < BN / 8; c++) {
            float s[8], dp[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint8_t* k_row = Ks + (8 * c + j) * RS_N;
                const uint8_t* v_row = Vs + (8 * c + j) * RS_N;
                s[j] = dot8h(q_hi, *(const uint4*)(k_row + (sub + 8) * 16),
                             dot8h(q_lo, *(const uint4*)(k_row + sub * 16), 0.0f));
                dp[j] = dot8h(do_hi, *(const uint4*)(v_row + (sub + 8) * 16),
                              dot8h(do_lo, *(const uint4*)(v_row + sub * 16), 0.0f));
            }
            const float s_own = reduce_scatter8(s, sub), dp_own = reduce_scatter8(dp, sub);
            const int kj = 8 * c + sub, key = k0 + kj;
            const bool masked = key >= L || row >= L || (causal && key > row);
            const float p = masked ? 0.0f : exp2f(s_own * scale_log2 - lse);
            *(_Float16*)(dSs + ql * TRS + kj * 2) = (_Float16)(p * (dp_own - delta));
        }
        wave_sync_lds();  // a query's dS row is written and read by its own 8 lanes
#pragma unroll 1
        for (int c = 0; c < BN * 2 / 16; c++) {
            const uint4 dsv = *(const uint4*)(dSs + ql * TRS + c * 16);
#pragma unroll
            for (int hb = 0; hb < 2; hb++)
#pragma unroll
                for (int r = 0; r < 8; r++)
                    dq[hb][r] = dot8h(dsv, *(const uint4*)(KT + t_row<TRS, true>(64 * hb + 8 * sub + r) + c * 16),
                                      dq[hb][r]);
        }
    }
    if (row >= L) return;
    uint4* dq_row = (uint4*)(dQ + b * dqs.b + (long long)row * dqs.l + h * dqs.h);
#pragma unroll
    for (int hb = 0; hb < 2; hb++) {
#pragma unroll
        for (int r = 0; r < 8; r++) dq[hb][r] *= scale;
        dq_row[sub + 8 * hb] = pack8h(dq[hb]);
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
        attn_bwd_dkdv<<<dim3((L + KV_NT / 8 - 1) / (KV_NT / 8), KVH, B), KV_NT, 0, stream>>>(
        (const _Float16*)q, qs, (const _Float16*)k, ks, (const _Float16*)v, vs, (const _Float16*)dout, dos, lse, delta,
        (_Float16*)dk, dks, (_Float16*)dv, dvs, H, KVH, L, scale * LOG2E, scale, causal);
    if (stages & BWD_DQ)
        attn_bwd_dq<<<dim3((L + KV_NT / 8 - 1) / (KV_NT / 8), H, B), KV_NT, 0, stream>>>(
        (const _Float16*)q, qs, (const _Float16*)k, ks, (const _Float16*)v, vs, (const _Float16*)dout, dos, lse, delta,
        (_Float16*)dq, dqs, H, KVH, L, scale * LOG2E, scale, causal);
}
