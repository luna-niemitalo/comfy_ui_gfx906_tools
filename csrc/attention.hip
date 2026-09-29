#include <hip/hip_runtime.h>
#include <cstdint>

// Fused (flash-style) attention for gfx906, head dim 128, native GQA.
// out[b, l, h*128 + d] = softmax(q . k^T) v, one workgroup per BQ query rows of
// one head, looping over 64-key tiles of that head's KV head.
//
// Q.K precision (compile time): fp16 (v_dot2_f32_f16), int8 per-token
// (v_dot4_i32_i8), int4 per-token (v_dot8_i32_i4). P.V: fp16, or u8 with a
// per-(row, key tile) P scale and per-(key tile, channel) V scale
// (v_dot4_u32_u8; V stored +128 so it is unsigned, corrected per tile).
// Accuracy study behind these choices: CLAUDE.md "Attention exploration".
//
// Inputs are prepared by attention.py (scaling, smoothing, rotation, packing):
// scores come out in log2 units (softmax scale * log2(e) folded into Q / the
// Q scale), so softmax uses exp2.
//
// Thread layout, both phases: tx = tid % 16, ty = tid / 16 (TY = NT / 16 of
// them); a thread owns query rows ty + TY*i (i < 4). S phase: keys tx + 16*j
// (j < 4). P.V phase: channels tx + 16*j (j < 8). Strided so the 16 tx lanes
// of a wave read 16 distinct LDS rows (conflict-free with the 16-byte pad)
// while lanes sharing ty read the same Q/P row (broadcast).
//
// NT: int modes use 512 threads / 128 rows -- one 8-wave workgroup per CU (2
// waves/SIMD) within 64 KB LDS, and each K/V tile in LDS serves 128 rows. fp16
// Q.K rows are twice as wide and only fit at 256 threads / 64 rows.

#define ATT_D 128
#define BKV 64
#define RPT 4
#define KPT 4
#define DPT 8
#define NO_SCORE -1e30f

enum { QK_F16 = 0, QK_I8 = 1, QK_I4 = 2 };
enum { PV_F16 = 0, PV_U8 = 1 };

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

template <int QK> struct QKRow;
template <> struct QKRow<QK_F16> { static constexpr int BYTES = ATT_D * 2; static constexpr int THREADS = 256; };
template <> struct QKRow<QK_I8> { static constexpr int BYTES = ATT_D; static constexpr int THREADS = 512; };
template <> struct QKRow<QK_I4> { static constexpr int BYTES = ATT_D / 2; static constexpr int THREADS = 512; };

__device__ __forceinline__ half2_t as_h2(uint32_t v) { return __builtin_bit_cast(half2_t, v); }

// 16 bytes of a against 16 bytes of b, accumulated
template <int QK> __device__ __forceinline__ void qk_dot(uint4 a, uint4 b, float& facc, int& iacc);
template <> __device__ __forceinline__ void qk_dot<QK_F16>(uint4 a, uint4 b, float& facc, int&) {
    facc = __builtin_amdgcn_fdot2(as_h2(a.x), as_h2(b.x), facc, false);
    facc = __builtin_amdgcn_fdot2(as_h2(a.y), as_h2(b.y), facc, false);
    facc = __builtin_amdgcn_fdot2(as_h2(a.z), as_h2(b.z), facc, false);
    facc = __builtin_amdgcn_fdot2(as_h2(a.w), as_h2(b.w), facc, false);
}
template <> __device__ __forceinline__ void qk_dot<QK_I8>(uint4 a, uint4 b, float&, int& iacc) {
    iacc = __builtin_amdgcn_sdot4((int)a.x, (int)b.x, iacc, false);
    iacc = __builtin_amdgcn_sdot4((int)a.y, (int)b.y, iacc, false);
    iacc = __builtin_amdgcn_sdot4((int)a.z, (int)b.z, iacc, false);
    iacc = __builtin_amdgcn_sdot4((int)a.w, (int)b.w, iacc, false);
}
template <> __device__ __forceinline__ void qk_dot<QK_I4>(uint4 a, uint4 b, float&, int& iacc) {
    iacc = __builtin_amdgcn_sdot8((int)a.x, (int)b.x, iacc, false);
    iacc = __builtin_amdgcn_sdot8((int)a.y, (int)b.y, iacc, false);
    iacc = __builtin_amdgcn_sdot8((int)a.z, (int)b.z, iacc, false);
    iacc = __builtin_amdgcn_sdot8((int)a.w, (int)b.w, iacc, false);
}

// Reductions over the 16 lanes of a DPP row (= the 16 tx lanes of one ty).
// DPP moves are plain VALU; __shfl_xor compiles to ds_bpermute, an LDS round
// trip per step -- 4 chained per reduction, ~40 per thread per tile.
template <int CTRL> __device__ __forceinline__ float dpp(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, false));
}
#define DPP_QUAD_XOR1 0xB1  // quad_perm [1, 0, 3, 2]
#define DPP_QUAD_XOR2 0x4E  // quad_perm [2, 3, 0, 1]
#define DPP_ROW_ROR4 0x124
#define DPP_ROW_ROR8 0x128
__device__ __forceinline__ float row_max16(float v) {
    v = fmaxf(v, dpp<DPP_QUAD_XOR1>(v));
    v = fmaxf(v, dpp<DPP_QUAD_XOR2>(v));
    v = fmaxf(v, dpp<DPP_ROW_ROR4>(v));
    return fmaxf(v, dpp<DPP_ROW_ROR8>(v));
}
__device__ __forceinline__ float row_sum16(float v) {
    v += dpp<DPP_QUAD_XOR1>(v);
    v += dpp<DPP_QUAD_XOR2>(v);
    v += dpp<DPP_ROW_ROR4>(v);
    return v + dpp<DPP_ROW_ROR8>(v);
}

// Q [B, H, L, QROW] bytes, q_scale [B, H, L] (int modes), K [B, KVH, L, QROW],
// k_scale [B, KVH, L], Vt [B, KVH, n_tiles, 128, 64] (fp16 or u8+128),
// v_scale [B, KVH, n_tiles, 128] (u8 mode). out [B, L, H, 128] fp16.
// A key tile whose largest P (relative to each row's running max) is below
// skip_below for every row of the workgroup contributes to the row sums but
// its P.V is skipped (0 disables).
template <int QK, int PV>
__global__ void __launch_bounds__(QKRow<QK>::THREADS) attention_kernel(
    const uint8_t* __restrict__ Q, const float* __restrict__ q_scale,
    const uint8_t* __restrict__ K, const float* __restrict__ k_scale,
    const uint8_t* __restrict__ Vt, const float* __restrict__ v_scale,
    _Float16* __restrict__ out, int H, int KVH, int L, int n_tiles, float skip_below) {
    constexpr int NT = QKRow<QK>::THREADS;
    constexpr int TY = NT / 16;
    constexpr int BQ = RPT * TY;
    constexpr int WAVES = NT / 64;
    constexpr int RB = QKRow<QK>::BYTES;
    constexpr int RS = RB + 16;
    constexpr int PV_BYTES = PV == PV_F16 ? 2 : 1;
    constexpr int VROW = BKV * PV_BYTES;
    constexpr int VRS = VROW + 16;
    constexpr int PRS = BKV * PV_BYTES + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BQ * RS];
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BKV * RS];
    __shared__ __attribute__((aligned(16))) uint8_t Vs[ATT_D * VRS];
    __shared__ __attribute__((aligned(16))) uint8_t Ps[BQ * PRS];
    __shared__ float Ks_scale[BKV];
    __shared__ float Vs_scale[ATT_D];
    __shared__ float wave_rel[WAVES];

    const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * BQ;
    const int kvh = h / (H / KVH);
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;

    const uint8_t* Qg = Q + (size_t)(b * H + h) * L * RB;
    const uint8_t* Kg = K + (size_t)(b * KVH + kvh) * L * RB;
    const uint8_t* Vg = Vt + (size_t)(b * KVH + kvh) * n_tiles * ATT_D * VROW;
    const float* ksg = k_scale + (size_t)(b * KVH + kvh) * L;
    const float* vsg = v_scale + (size_t)(b * KVH + kvh) * n_tiles * ATT_D;

    for (int i = tid; i < BQ * RB / 16; i += NT) {
        int r = i / (RB / 16), c = i % (RB / 16);
        uint4 val = q0 + r < L ? ((const uint4*)(Qg + (size_t)(q0 + r) * RB))[c] : make_uint4(0, 0, 0, 0);
        *(uint4*)(Qs + r * RS + c * 16) = val;
    }
    float qs[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {
        int r = q0 + ty + TY * i;
        qs[i] = (QK != QK_F16 && r < L) ? q_scale[(size_t)(b * H + h) * L + r] : 1.0f;
    }

    float o[RPT][DPT], m[RPT], l_part[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {
        m[i] = NO_SCORE;
        l_part[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < DPT; j++) o[i][j] = 0.0f;
    }

    // The next tile's K and V are loaded into registers during this tile's P.V
    // (K issued right after the softmax, V at the start of P.V) and written to
    // LDS after it: global latency hides behind P.V instead of stalling all
    // waves at the tile's first barrier (~34 ms of 130 at L=14540 before). The
    // S-phase operand registers are dead during P.V, so peak VGPRs don't grow.
    constexpr int K_CHUNKS = BKV * RB / 16;
    constexpr int V_CHUNKS = ATT_D * VROW / 16;
    constexpr int K_PER = (K_CHUNKS + NT - 1) / NT;
    constexpr int V_PER = (V_CHUNKS + NT - 1) / NT;
    uint4 k_pre[K_PER], v_pre[V_PER];
#pragma unroll
    for (int n = 0; n < K_PER; n++) k_pre[n] = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < V_PER; n++) v_pre[n] = make_uint4(0, 0, 0, 0);
    float ks_pre = 0.0f, vs_pre = 0.0f;
    auto load_k = [&](int t) {
        const int k0 = t * BKV;
#pragma unroll
        for (int n = 0; n < K_PER; n++) {
            const int i = min(tid + n * NT, K_CHUNKS - 1), r = i / (RB / 16), c = i % (RB / 16);
            const uint4 val = ((const uint4*)(Kg + (size_t)min(k0 + r, L - 1) * RB))[c];
            k_pre[n] = k0 + r < L ? val : make_uint4(0, 0, 0, 0);
        }
        if (QK != QK_F16 && tid < BKV) ks_pre = k0 + tid < L ? ksg[k0 + tid] : 0.0f;
    };
    auto load_v = [&](int t) {
        const uint8_t* Vtile = Vg + (size_t)t * ATT_D * VROW;
#pragma unroll
        for (int n = 0; n < V_PER; n++) {
            const int i = min(tid + n * NT, V_CHUNKS - 1), d = i / (VROW / 16), c = i % (VROW / 16);
            v_pre[n] = ((const uint4*)(Vtile + (size_t)d * VROW))[c];
        }
        if (PV == PV_U8 && tid < ATT_D) vs_pre = vsg[(size_t)t * ATT_D + tid];
    };
    auto store_kv = [&]() {
#pragma unroll
        for (int n = 0; n < K_PER; n++) {
            const int i = tid + n * NT, r = i / (RB / 16), c = i % (RB / 16);
            if (i < K_CHUNKS) *(uint4*)(Ks + r * RS + c * 16) = k_pre[n];
        }
#pragma unroll
        for (int n = 0; n < V_PER; n++) {
            const int i = tid + n * NT, d = i / (VROW / 16), c = i % (VROW / 16);
            if (i < V_CHUNKS) *(uint4*)(Vs + d * VRS + c * 16) = v_pre[n];
        }
        if (QK != QK_F16 && tid < BKV) Ks_scale[tid] = ks_pre;
        if (PV == PV_U8 && tid < ATT_D) Vs_scale[tid] = vs_pre;
    };
    load_k(0);
    load_v(0);
    store_kv();

    for (int t = 0; t < n_tiles; t++) {
        const int k0 = t * BKV;
        const bool has_next = t + 1 < n_tiles;
        __syncthreads();  // this tile's K/V (and Q on t = 0) are in LDS

        float facc[RPT][KPT];
        int iacc[RPT][KPT];
#pragma unroll
        for (int i = 0; i < RPT; i++)
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                facc[i][j] = 0.0f;
                iacc[i][j] = 0;
            }
#pragma unroll 1
        for (int c = 0; c < RB / 16; c++) {
            uint4 kq[KPT];
#pragma unroll
            for (int j = 0; j < KPT; j++) kq[j] = *(const uint4*)(Ks + (tx + 16 * j) * RS + c * 16);
#pragma unroll
            for (int i = 0; i < RPT; i++) {
                uint4 qq = *(const uint4*)(Qs + (ty + TY * i) * RS + c * 16);
#pragma unroll
                for (int j = 0; j < KPT; j++) qk_dot<QK>(qq, kq[j], facc[i][j], iacc[i][j]);
            }
        }

        // online softmax; P to LDS
        float rel_max = 0.0f;
        float p_scale[RPT], pq_sum[RPT];
#pragma unroll
        for (int i = 0; i < RPT; i++) {
            float s[KPT];
            float tmax = NO_SCORE;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                int key = tx + 16 * j;
                s[j] = QK == QK_F16 ? facc[i][j] : (float)iacc[i][j] * qs[i] * Ks_scale[key];
                if (k0 + key >= L) s[j] = NO_SCORE;
                tmax = fmaxf(tmax, s[j]);
            }
            tmax = row_max16(tmax);
            const float m_new = fmaxf(m[i], tmax);
            const float alpha = exp2f(m[i] - m_new);
            m[i] = m_new;
            l_part[i] *= alpha;
#pragma unroll
            for (int j = 0; j < DPT; j++) o[i][j] *= alpha;
            const float rel = exp2f(tmax - m_new);  // this row's largest P in the tile
            rel_max = fmaxf(rel_max, rel);
            // u8: P / (rel / 255) in [0, 255]. A denormal rel (block 0's outlier
            // heads) would make 255 / rel inf and 0 * inf NaN.
            p_scale[i] = rel * (1.0f / 255.0f);
            const float inv_scale = rel > 1e-30f ? 255.0f / rel : 0.0f;
            float qsum = 0.0f;
            const int r = ty + TY * i;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                const float p = exp2f(s[j] - m_new);
                l_part[i] += p;
                const int key = tx + 16 * j;
                if (PV == PV_F16) {
                    *(_Float16*)(Ps + r * PRS + key * 2) = (_Float16)p;
                } else {
                    const float pq = rintf(p * inv_scale);
                    qsum += pq;
                    Ps[r * PRS + key] = (uint8_t)pq;
                }
            }
            pq_sum[i] = PV == PV_U8 ? row_sum16(qsum) : 0.0f;
        }
        if (has_next) load_k(t + 1);
        rel_max = fmaxf(rel_max, __shfl_xor(rel_max, 16));
        rel_max = fmaxf(rel_max, __shfl_xor(rel_max, 32));
        rel_max = row_max16(rel_max);
        if ((tid & 63) == 0) wave_rel[tid / 64] = rel_max;
        __syncthreads();
        float tile_rel = 0.0f;
#pragma unroll
        for (int w = 0; w < WAVES; w++) tile_rel = fmaxf(tile_rel, wave_rel[w]);
        if (has_next) load_v(t + 1);

        if (tile_rel < skip_below) {
            // skipped: this tile's P.V contribution is negligible
        } else if (PV == PV_F16) {
#pragma unroll 1
            for (int c = 0; c < VROW / 16; c++) {
                uint4 pp[RPT];
#pragma unroll
                for (int i = 0; i < RPT; i++) pp[i] = *(const uint4*)(Ps + (ty + TY * i) * PRS + c * 16);
#pragma unroll
                for (int j = 0; j < DPT; j++) {
                    const uint4 vv = *(const uint4*)(Vs + (tx + 16 * j) * VRS + c * 16);
#pragma unroll
                    for (int i = 0; i < RPT; i++) {
                        int unused = 0;
                        qk_dot<QK_F16>(pp[i], vv, o[i][j], unused);
                    }
                }
            }
        } else {
            uint32_t acc[RPT][DPT];
#pragma unroll
            for (int i = 0; i < RPT; i++)
#pragma unroll
                for (int j = 0; j < DPT; j++) acc[i][j] = 0;
#pragma unroll 1
            for (int c = 0; c < VROW / 16; c++) {
                uint4 pp[RPT];
#pragma unroll
                for (int i = 0; i < RPT; i++) pp[i] = *(const uint4*)(Ps + (ty + TY * i) * PRS + c * 16);
#pragma unroll
                for (int j = 0; j < DPT; j++) {
                    const uint4 vv = *(const uint4*)(Vs + (tx + 16 * j) * VRS + c * 16);
#pragma unroll
                    for (int i = 0; i < RPT; i++) {
                        acc[i][j] = __builtin_amdgcn_udot4(pp[i].x, vv.x, acc[i][j], false);
                        acc[i][j] = __builtin_amdgcn_udot4(pp[i].y, vv.y, acc[i][j], false);
                        acc[i][j] = __builtin_amdgcn_udot4(pp[i].z, vv.z, acc[i][j], false);
                        acc[i][j] = __builtin_amdgcn_udot4(pp[i].w, vv.w, acc[i][j], false);
                    }
                }
            }
            // V was stored +128: subtract 128 * sum(Pq) per row
#pragma unroll
            for (int i = 0; i < RPT; i++)
#pragma unroll
                for (int j = 0; j < DPT; j++)
                    o[i][j] = fmaf(p_scale[i] * Vs_scale[tx + 16 * j], (float)acc[i][j] - 128.0f * pq_sum[i], o[i][j]);
        }
        if (has_next) {
            __syncthreads();  // every wave is done reading this tile's Ks/Vs/Ps
            store_kv();
        }
    }

#pragma unroll
    for (int i = 0; i < RPT; i++) {
        const float l = row_sum16(l_part[i]);
        const int r = q0 + ty + TY * i;
        if (r >= L) continue;
        _Float16* dst = out + ((size_t)b * L + r) * H * ATT_D + (size_t)h * ATT_D;
#pragma unroll
        for (int j = 0; j < DPT; j++) dst[tx + 16 * j] = (_Float16)(o[i][j] / l);
    }
}

template <int QK, int PV>
static void attention_launch(const uint8_t* Q, const float* qs, const uint8_t* K, const float* ks, const uint8_t* Vt,
                             const float* vs, void* out, int B, int H, int KVH, int L, float skip_below, hipStream_t stream) {
    constexpr int NT = QKRow<QK>::THREADS;
    constexpr int BQ = RPT * NT / 16;
    const int n_tiles = (L + BKV - 1) / BKV;
    dim3 grid((L + BQ - 1) / BQ, H, B);
    attention_kernel<QK, PV><<<grid, NT, 0, stream>>>(Q, qs, K, ks, Vt, vs, (_Float16*)out, H, KVH, L, n_tiles, skip_below);
}

void attention_dispatch(int qk_mode, int pv_mode, const uint8_t* Q, const float* qs, const uint8_t* K, const float* ks,
                        const uint8_t* Vt, const float* vs, void* out, int B, int H, int KVH, int L, float skip_below,
                        hipStream_t stream) {
#define ATT_CASE(QKM, PVM) \
    if (qk_mode == QKM && pv_mode == PVM) return attention_launch<QKM, PVM>(Q, qs, K, ks, Vt, vs, out, B, H, KVH, L, skip_below, stream);
    ATT_CASE(QK_F16, PV_F16) ATT_CASE(QK_F16, PV_U8)
    ATT_CASE(QK_I8, PV_F16) ATT_CASE(QK_I8, PV_U8)
    ATT_CASE(QK_I4, PV_F16) ATT_CASE(QK_I4, PV_U8)
#undef ATT_CASE
}
