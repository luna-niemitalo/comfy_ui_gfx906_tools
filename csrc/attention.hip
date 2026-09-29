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
// PV_ADAPT: per wave and key tile, u8 as PV_U8, or 4-bit P against a 4-bit V
// copy (v_dot8_u32_u4, twice the MACs per instruction) when the tile's
// largest P is below int4_below of the rows' running max -- such tiles carry
// little of the output, so their coarser error is diluted.
enum { PV_F16 = 0, PV_U8 = 1, PV_ADAPT = 2 };
enum { TIER_SKIP, TIER_P4, TIER_P8 };

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

__device__ __forceinline__ float wave_max_of_rows(float row_uniform) {
    const int v = __builtin_bit_cast(int, row_uniform);
    return fmaxf(fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 0)),
                       __builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 16))),
                 fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 32)),
                       __builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 48))));
}

// Q [B, H, Lq, QROW] bytes, q_scale [B, H, Lq] (int modes), K [B, KVH, Lk, QROW],
// k_scale [B, KVH, Lk], Vt [B, KVH, n_tiles, 128, 64] (fp16 or u8+128; tiles of
// Lk), v_scale [B, KVH, n_tiles, 128] (u8 modes), V4t [B, KVH, n_tiles, 128, 32]
// (PV_ADAPT: u4+8, key 2i in the low nibble, scale v_scale * 127/7).
// out [B, Lq, H, 128] fp16. Lq != Lk when queries are a subset (the two-GPU row split).
// A key tile whose largest P (relative to each row's running max) is below
// skip_below for every row of a wave contributes to that wave's row sums but
// its P.V is skipped (0 disables). Decisions are per wave: wave-uniform
// branches cost nothing, and each wave reads only its own P rows.
template <int QK, int PV>
__global__ void __launch_bounds__(QKRow<QK>::THREADS) attention_kernel(
    const uint8_t* __restrict__ Q, const float* __restrict__ q_scale,
    const uint8_t* __restrict__ K, const float* __restrict__ k_scale,
    const uint8_t* __restrict__ Vt, const float* __restrict__ v_scale, const uint8_t* __restrict__ V4t,
    _Float16* __restrict__ out, int H, int KVH, int Lq, int Lk, int n_tiles, float skip_below, float int4_below) {
    constexpr int NT = QKRow<QK>::THREADS;
    constexpr int TY = NT / 16;
    constexpr int BQ = RPT * TY;
    constexpr int RB = QKRow<QK>::BYTES;
    constexpr int RS = RB + 16;
    constexpr int PV_BYTES = PV == PV_F16 ? 2 : 1;
    constexpr bool P_U8 = PV != PV_F16;
    constexpr int VROW = BKV * PV_BYTES;
    constexpr int VRS = VROW + 16;
    constexpr int PRS = BKV * PV_BYTES + 16;
    constexpr int V4ROW = BKV / 2;
    constexpr int V4RS = V4ROW + 16;
    __shared__ __attribute__((aligned(16))) uint8_t Qs[BQ * RS];
    __shared__ __attribute__((aligned(16))) uint8_t Ks[BKV * RS];
    __shared__ __attribute__((aligned(16))) uint8_t Vs[ATT_D * VRS];
    __shared__ __attribute__((aligned(16))) uint8_t V4s[PV == PV_ADAPT ? ATT_D * V4RS : 16];
    __shared__ __attribute__((aligned(16))) uint8_t Ps[BQ * PRS];
    // P4 rows use the first half of their P8 row: waves pick tiers
    // independently, so a tier-specific stride would overlap other waves' rows.
    __shared__ float Ks_scale[BKV];
    __shared__ float Vs_scale[ATT_D];

    const int b = blockIdx.z, h = blockIdx.y, q0 = blockIdx.x * BQ;
    const int kvh = h / (H / KVH);
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;

    const uint8_t* Qg = Q + (size_t)(b * H + h) * Lq * RB;
    const uint8_t* Kg = K + (size_t)(b * KVH + kvh) * Lk * RB;
    const uint8_t* Vg = Vt + (size_t)(b * KVH + kvh) * n_tiles * ATT_D * VROW;
    const float* ksg = k_scale + (size_t)(b * KVH + kvh) * Lk;
    const float* vsg = v_scale + (size_t)(b * KVH + kvh) * n_tiles * ATT_D;
    const uint8_t* V4g = V4t + (size_t)(b * KVH + kvh) * n_tiles * ATT_D * (BKV / 2);

    for (int i = tid; i < BQ * RB / 16; i += NT) {
        int r = i / (RB / 16), c = i % (RB / 16);
        uint4 val = q0 + r < Lq ? ((const uint4*)(Qg + (size_t)(q0 + r) * RB))[c] : make_uint4(0, 0, 0, 0);
        *(uint4*)(Qs + r * RS + c * 16) = val;
    }
    float qs[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {
        int r = q0 + ty + TY * i;
        qs[i] = (QK != QK_F16 && r < Lq) ? q_scale[(size_t)(b * H + h) * Lq + r] : 1.0f;
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
    constexpr int V4_CHUNKS = ATT_D * V4ROW / 16;
    static_assert(V4_CHUNKS <= NT, "one V4 chunk per thread");
    // int4 K fills only half the threads' K chunk registers: the other half
    // (whole waves, so the branch is uniform) carry the V4 chunk in them.
    // A separate V4 register spilled (+80 B scratch, adaptive 13% slower).
    constexpr bool V4_IN_K = PV == PV_ADAPT && K_PER == 1 && K_CHUNKS + V4_CHUNKS <= NT;
    const bool carries_v4 = V4_IN_K && tid >= K_CHUNKS;
    uint4 k_pre[K_PER], v_pre[V_PER], v4_pre = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < K_PER; n++) k_pre[n] = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < V_PER; n++) v_pre[n] = make_uint4(0, 0, 0, 0);
    float ks_pre = 0.0f, vs_pre = 0.0f;
    auto load_k = [&](int t) {
        const int k0 = t * BKV;
#pragma unroll
        for (int n = 0; n < K_PER; n++) {
            if (carries_v4) break;
            const int i = min(tid + n * NT, K_CHUNKS - 1), r = i / (RB / 16), c = i % (RB / 16);
            const uint4 val = ((const uint4*)(Kg + (size_t)min(k0 + r, Lk - 1) * RB))[c];
            k_pre[n] = k0 + r < Lk ? val : make_uint4(0, 0, 0, 0);
        }
        if (QK != QK_F16 && tid < BKV) ks_pre = k0 + tid < Lk ? ksg[k0 + tid] : 0.0f;
    };
    auto load_v = [&](int t) {
        const uint8_t* Vtile = Vg + (size_t)t * ATT_D * VROW;
#pragma unroll
        for (int n = 0; n < V_PER; n++) {
            const int i = min(tid + n * NT, V_CHUNKS - 1), d = i / (VROW / 16), c = i % (VROW / 16);
            v_pre[n] = ((const uint4*)(Vtile + (size_t)d * VROW))[c];
        }
        if (P_U8 && tid < ATT_D) vs_pre = vsg[(size_t)t * ATT_D + tid];
        if (carries_v4) {
            const int i = tid - K_CHUNKS, d = i / (V4ROW / 16), c = i % (V4ROW / 16);
            k_pre[0] = ((const uint4*)(V4g + ((size_t)t * ATT_D + d) * V4ROW))[c];
        } else if (PV == PV_ADAPT && !V4_IN_K) {
            const int i = min(tid, V4_CHUNKS - 1), d = i / (V4ROW / 16), c = i % (V4ROW / 16);
            v4_pre = ((const uint4*)(V4g + ((size_t)t * ATT_D + d) * V4ROW))[c];
        }
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
        if (P_U8 && tid < ATT_D) Vs_scale[tid] = vs_pre;
        if (carries_v4) {
            const int i = tid - K_CHUNKS, d = i / (V4ROW / 16), c = i % (V4ROW / 16);
            *(uint4*)(V4s + d * V4RS + c * 16) = k_pre[0];
        } else if (PV == PV_ADAPT && !V4_IN_K && tid < V4_CHUNKS) {
            const int d = tid / (V4ROW / 16), c = tid % (V4ROW / 16);
            *(uint4*)(V4s + d * V4RS + c * 16) = v4_pre;
        }
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

        // online softmax: row maxima first, so the wave's tier is known before P is written
        float s[RPT][KPT], rel[RPT];
        float rows_rel = 0.0f;
#pragma unroll
        for (int i = 0; i < RPT; i++) {
            float tmax = NO_SCORE;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                int key = tx + 16 * j;
                s[i][j] = QK == QK_F16 ? facc[i][j] : (float)iacc[i][j] * qs[i] * Ks_scale[key];
                if (k0 + key >= Lk) s[i][j] = NO_SCORE;
                tmax = fmaxf(tmax, s[i][j]);
            }
            tmax = row_max16(tmax);
            const float m_new = fmaxf(m[i], tmax);
            const float alpha = exp2f(m[i] - m_new);
            m[i] = m_new;
            l_part[i] *= alpha;
#pragma unroll
            for (int j = 0; j < DPT; j++) o[i][j] *= alpha;
            rel[i] = exp2f(tmax - m_new);  // this row's largest P in the tile
            rows_rel = fmaxf(rows_rel, rel[i]);
        }
        const float wave_rel = wave_max_of_rows(rows_rel);
        const int tier = wave_rel < skip_below                        ? TIER_SKIP
                         : PV == PV_ADAPT && wave_rel < int4_below ? TIER_P4
                                                                     : TIER_P8;
        const float levels = tier == TIER_P4 ? 15.0f : 255.0f;

        // P to LDS: fp16, or P / (rel / levels) in [0, levels]
        float p_scale[RPT], pq_sum[RPT];
#pragma unroll
        for (int i = 0; i < RPT; i++) {
            // A denormal rel (block 0's outlier heads) would make levels / rel
            // inf and 0 * inf NaN.
            p_scale[i] = rel[i] / levels;
            const float inv_scale = rel[i] > 1e-30f ? levels / rel[i] : 0.0f;
            float qsum = 0.0f;
            const int r = ty + TY * i;
#pragma unroll
            for (int j = 0; j < KPT; j++) {
                const float p = exp2f(s[i][j] - m[i]);
                l_part[i] += p;
                const int key = tx + 16 * j;
                if (tier == TIER_SKIP) continue;
                if (PV == PV_F16) {
                    *(_Float16*)(Ps + r * PRS + key * 2) = (_Float16)p;
                    continue;
                }
                const float pq = rintf(p * inv_scale);
                qsum += pq;
                if (tier == TIER_P8) {
                    Ps[r * PRS + key] = (uint8_t)pq;
                    continue;
                }
                // keys tx, tx ^ 1 share a byte; the even lane writes it
                const int q4 = (int)pq;
                const int odd = __builtin_amdgcn_update_dpp(0, q4, DPP_QUAD_XOR1, 0xF, 0xF, false);
                if (!(tx & 1)) Ps[r * PRS + key / 2] = (uint8_t)(q4 | (odd << 4));
            }
            pq_sum[i] = P_U8 && tier != TIER_SKIP ? row_sum16(qsum) : 0.0f;
        }
        if (has_next) load_k(t + 1);
        if (has_next) load_v(t + 1);
        // P rows are written and read only by their own wave: no workgroup
        // barrier, but keep the compiler from moving the reads above the writes.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        if (tier == TIER_SKIP) {
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
            if (PV == PV_ADAPT && tier == TIER_P4) {
#pragma unroll 1
                for (int c = 0; c < V4ROW / 16; c++) {
                    uint4 pp[RPT];
#pragma unroll
                    for (int i = 0; i < RPT; i++) pp[i] = *(const uint4*)(Ps + (ty + TY * i) * PRS + c * 16);
#pragma unroll
                    for (int j = 0; j < DPT; j++) {
                        const uint4 vv = *(const uint4*)(V4s + (tx + 16 * j) * V4RS + c * 16);
#pragma unroll
                        for (int i = 0; i < RPT; i++) {
                            acc[i][j] = __builtin_amdgcn_udot8(pp[i].x, vv.x, acc[i][j], false);
                            acc[i][j] = __builtin_amdgcn_udot8(pp[i].y, vv.y, acc[i][j], false);
                            acc[i][j] = __builtin_amdgcn_udot8(pp[i].z, vv.z, acc[i][j], false);
                            acc[i][j] = __builtin_amdgcn_udot8(pp[i].w, vv.w, acc[i][j], false);
                        }
                    }
                }
            } else {
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
            }
            // V was stored +128 (u4: +8, scale 127/7 of the u8 one): subtract offset * sum(Pq) per row
            const float v_factor = tier == TIER_P4 ? 127.0f / 7.0f : 1.0f;
            const float v_offset = tier == TIER_P4 ? 8.0f : 128.0f;
#pragma unroll
            for (int i = 0; i < RPT; i++)
#pragma unroll
                for (int j = 0; j < DPT; j++)
                    o[i][j] = fmaf(p_scale[i] * v_factor * Vs_scale[tx + 16 * j],
                                   (float)acc[i][j] - v_offset * pq_sum[i], o[i][j]);
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
        if (r >= Lq) continue;
        _Float16* dst = out + ((size_t)b * Lq + r) * H * ATT_D + (size_t)h * ATT_D;
#pragma unroll
        for (int j = 0; j < DPT; j++) dst[tx + 16 * j] = (_Float16)(o[i][j] / l);
    }
}

template <int QK, int PV>
static void attention_launch(const uint8_t* Q, const float* qs, const uint8_t* K, const float* ks, const uint8_t* Vt,
                             const float* vs, const uint8_t* V4t, void* out, int B, int H, int KVH, int Lq, int Lk,
                             float skip_below, float int4_below, hipStream_t stream) {
    constexpr int NT = QKRow<QK>::THREADS;
    constexpr int BQ = RPT * NT / 16;
    const int n_tiles = (Lk + BKV - 1) / BKV;
    dim3 grid((Lq + BQ - 1) / BQ, H, B);
    attention_kernel<QK, PV><<<grid, NT, 0, stream>>>(Q, qs, K, ks, Vt, vs, V4t, (_Float16*)out, H, KVH, Lq, Lk,
                                                      n_tiles, skip_below, int4_below);
}

void attention_dispatch(int qk_mode, int pv_mode, const uint8_t* Q, const float* qs, const uint8_t* K, const float* ks,
                        const uint8_t* Vt, const float* vs, const uint8_t* V4t, void* out, int B, int H, int KVH,
                        int Lq, int Lk, float skip_below, float int4_below, hipStream_t stream) {
#define ATT_CASE(QKM, PVM)                                                                                     \
    if (qk_mode == QKM && pv_mode == PVM)                                                                       \
        return attention_launch<QKM, PVM>(Q, qs, K, ks, Vt, vs, V4t, out, B, H, KVH, Lq, Lk, skip_below, int4_below, \
                                          stream);
    ATT_CASE(QK_F16, PV_F16) ATT_CASE(QK_F16, PV_U8) ATT_CASE(QK_F16, PV_ADAPT)
    ATT_CASE(QK_I8, PV_F16) ATT_CASE(QK_I8, PV_U8) ATT_CASE(QK_I8, PV_ADAPT)
    ATT_CASE(QK_I4, PV_F16) ATT_CASE(QK_I4, PV_U8) ATT_CASE(QK_I4, PV_ADAPT)
#undef ATT_CASE
}
