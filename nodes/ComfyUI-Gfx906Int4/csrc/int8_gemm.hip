#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// Block-scaled int8 GEMM for gfx906: C[M,N] = A[M,K] @ B[N,K]^T (+ bias), the int8
// counterpart of int4_gemm_scaled.hip (same tiling, tile order and LDS padding; see its
// header for the why). A (activation) and B (weight) are int8, 32 elements (= 32 bytes =
// two uint4 loads) per float32 scale; each block is 8 v_dot4_i32_i8 into an int32 partial,
// rescaled into an fp32 accumulator.
// Why int8 and not the int4 kernel: for the Qwen-Image VAE's convs, W4A4 costs 37-40 dB
// against fp32 (14% latent error on encode); W8A8 58 dB (0.6%)
// (develop/experiments/2026-09-29/vae_int_study.py).
#define QBLOCK 32
#define BK_BLOCKS 4
#define HALVES (2 * BK_BLOCKS)  // uint4 per row per stage
#define GROUP_M 8
#define THREADS 256
#define ROW (HALVES + 1)  // +1 uint4 padding against LDS bank conflicts

namespace {

// Implicit-GEMM conv: A is the channels-last int8 input [B*H*W, C] (+ scales [B*H*W, C/32])
// and row m / K block kb of the virtual im2col matrix are resolved in the load stage:
// m -> output pixel (b, oy, ox), kb -> (tap, 32-channel block), tap -> (dy, dx). Out-of-image
// taps load zero values and zero scales (= zero padding). Skips writing the 9x-expanded
// matrix: the gather cost as much as the int8 GEMM saved (2026-09-29, vae_profile.py int8).
struct ConvGeom {
    int H, W, c_blocks, Ho, Wo, kw, sh, sw, ph, pw;
};

struct PixelRef {
    int base, iy0, ix0;  // base = b*H*W; base < 0 marks a row past M
};

__device__ __forceinline__ PixelRef pixel_ref(int m, int M, const ConvGeom& g) {
    if (m >= M) return {-1, 0, 0};
    int hw = g.Ho * g.Wo, b = m / hw, rem = m - b * hw, oy = rem / g.Wo, ox = rem - oy * g.Wo;
    return {b * g.H * g.W, oy * g.sh - g.ph, ox * g.sw - g.pw};
}

// input pixel index of K block kb for this output pixel, or -1 outside the image
__device__ __forceinline__ int tap_pixel(const PixelRef& p, int kb, int& cb, const ConvGeom& g) {
    int tap = kb / g.c_blocks;
    cb = kb - tap * g.c_blocks;
    int dy = tap / g.kw, dx = tap - dy * g.kw;
    int iy = p.iy0 + dy, ix = p.ix0 + dx;
    if (p.base < 0 || iy < 0 || iy >= g.H || ix < 0 || ix >= g.W) return -1;
    return p.base + iy * g.W + ix;
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void from_float(float v, float* out) { *out = v; }
__device__ __forceinline__ void from_float(float v, __half* out) { *out = __float2half(v); }
__device__ __forceinline__ void from_float(float v, __hip_bfloat16* out) { *out = __float2bfloat16(v); }

__device__ __forceinline__ int dot16(uint4 a, uint4 b, int acc) {
    acc = __builtin_amdgcn_sdot4((int)a.x, (int)b.x, acc, false);
    acc = __builtin_amdgcn_sdot4((int)a.y, (int)b.y, acc, false);
    acc = __builtin_amdgcn_sdot4((int)a.z, (int)b.z, acc, false);
    return __builtin_amdgcn_sdot4((int)a.w, (int)b.w, acc, false);
}

template <int BM, int BN, int TM, int TN, bool CONV, typename T>
__global__ void __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(2))) int8_gemm_kernel(
    const uint4* __restrict__ A, const float* __restrict__ A_scale,
    const uint4* __restrict__ B, const float* __restrict__ B_scale,
    const T* __restrict__ bias, T* __restrict__ C, int M, int N, int n_blocks, ConvGeom geom) {
    constexpr int THREADS_X = BN / TN;
    constexpr int THREADS_Y = BM / TM;
    static_assert(THREADS_X * THREADS_Y == THREADS, "256 threads per workgroup");
    static_assert(BM >= 64 && BN >= 64, "one scale load per thread per operand at least");
    constexpr int A_LOADS = BM * HALVES / THREADS, B_LOADS = BN * HALVES / THREADS;
    constexpr int A_SCALES = BM * BK_BLOCKS / THREADS, B_SCALES = BN * BK_BLOCKS / THREADS;
    __shared__ uint4 As[BM * ROW];
    __shared__ uint4 Bs[BN * ROW];
    __shared__ float As_scale[BK_BLOCKS][BM];
    __shared__ float Bs_scale[BK_BLOCKS][BN];

    const int tiles_m = (M + BM - 1) / BM;
    const int tiles_n = (N + BN - 1) / BN;
    const int pid = blockIdx.x;
    const int group = pid / (GROUP_M * tiles_n);
    const int first_m = group * GROUP_M;
    const int group_rows = min(tiles_m - first_m, GROUP_M);
    const int tile_m = first_m + (pid % (GROUP_M * tiles_n)) % group_rows;
    const int tile_n = (pid % (GROUP_M * tiles_n)) / group_rows;
    const int row0 = tile_m * BM;
    const int col0 = tile_n * BN;

    const int tid = threadIdx.x;
    const int tx = tid % THREADS_X;
    const int ty = tid / THREADS_X;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    // Per stage: rows x 8 uint4 per operand (consecutive threads walk one row's 128
    // contiguous bytes), plus rows x 4 scales.
    const int n_halves = 2 * n_blocks;
    uint4 a_pref[A_LOADS], b_pref[B_LOADS];
    float as_pref[A_SCALES], bs_pref[B_SCALES];
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    PixelRef a_pix[CONV ? A_LOADS : 1], s_pix[CONV ? A_SCALES : 1];
    if constexpr (CONV) {
#pragma unroll
        for (int l = 0; l < A_LOADS; l++) a_pix[l] = pixel_ref(row0 + (tid + l * THREADS) / HALVES, M, geom);
#pragma unroll
        for (int l = 0; l < A_SCALES; l++) s_pix[l] = pixel_ref(row0 + (tid + l * THREADS) / BK_BLOCKS, M, geom);
    }
    auto load_stage = [&](int kt) {
#pragma unroll
        for (int l = 0; l < A_LOADS; l++) {
            int idx = tid + l * THREADS, r = idx / HALVES, kh = kt * HALVES + idx % HALVES;
            if constexpr (CONV) {
                int cb, pix = kh < n_halves ? tap_pixel(a_pix[l], kh >> 1, cb, geom) : -1;
                a_pref[l] = pix >= 0 ? A[(size_t)pix * (2 * geom.c_blocks) + 2 * cb + (kh & 1)] : zero4;
            } else {
                a_pref[l] = row0 + r < M && kh < n_halves ? A[(size_t)(row0 + r) * n_halves + kh] : zero4;
            }
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; l++) {
            int idx = tid + l * THREADS, r = idx / HALVES, kh = kt * HALVES + idx % HALVES;
            b_pref[l] = col0 + r < N && kh < n_halves ? B[(size_t)(col0 + r) * n_halves + kh] : zero4;
        }
#pragma unroll
        for (int l = 0; l < A_SCALES; l++) {
            int idx = tid + l * THREADS, r = idx / BK_BLOCKS, kb = kt * BK_BLOCKS + idx % BK_BLOCKS;
            if constexpr (CONV) {
                int cb, pix = kb < n_blocks ? tap_pixel(s_pix[l], kb, cb, geom) : -1;
                as_pref[l] = pix >= 0 ? A_scale[(size_t)pix * geom.c_blocks + cb] : 0.0f;
            } else {
                as_pref[l] = row0 + r < M && kb < n_blocks ? A_scale[(size_t)(row0 + r) * n_blocks + kb] : 0.0f;
            }
        }
#pragma unroll
        for (int l = 0; l < B_SCALES; l++) {
            int idx = tid + l * THREADS, r = idx / BK_BLOCKS, kb = kt * BK_BLOCKS + idx % BK_BLOCKS;
            bs_pref[l] = col0 + r < N && kb < n_blocks ? B_scale[(size_t)(col0 + r) * n_blocks + kb] : 0.0f;
        }
    };

    const int k_tiles = (n_blocks + BK_BLOCKS - 1) / BK_BLOCKS;
    load_stage(0);
    for (int kt = 0; kt < k_tiles; kt++) {
#pragma unroll
        for (int l = 0; l < A_LOADS; l++) {
            int idx = tid + l * THREADS;
            As[(idx / HALVES) * ROW + idx % HALVES] = a_pref[l];
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; l++) {
            int idx = tid + l * THREADS;
            Bs[(idx / HALVES) * ROW + idx % HALVES] = b_pref[l];
        }
#pragma unroll
        for (int l = 0; l < A_SCALES; l++) {
            int idx = tid + l * THREADS;
            As_scale[idx % BK_BLOCKS][idx / BK_BLOCKS] = as_pref[l];
        }
#pragma unroll
        for (int l = 0; l < B_SCALES; l++) {
            int idx = tid + l * THREADS;
            Bs_scale[idx % BK_BLOCKS][idx / BK_BLOCKS] = bs_pref[l];
        }
        __syncthreads();
        if (kt + 1 < k_tiles) load_stage(kt + 1);

#pragma unroll 1
        for (int c = 0; c < BK_BLOCKS; c++) {
            uint4 bq[TN][2];
            float bsc[TN];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                bq[j][0] = Bs[(tx + j * THREADS_X) * ROW + 2 * c];
                bq[j][1] = Bs[(tx + j * THREADS_X) * ROW + 2 * c + 1];
                bsc[j] = Bs_scale[c][tx + j * THREADS_X];
            }
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint4 a0 = As[(ty + i * THREADS_Y) * ROW + 2 * c];
                uint4 a1 = As[(ty + i * THREADS_Y) * ROW + 2 * c + 1];
                float asc = As_scale[c][ty + i * THREADS_Y];
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    int ip = dot16(a1, bq[j][1], dot16(a0, bq[j][0], 0));
                    acc[i][j] = fmaf((float)ip, asc * bsc[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = row0 + ty + i * THREADS_Y;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = col0 + tx + j * THREADS_X;
            if (n >= N) continue;
            float v = acc[i][j];
            if (bias) v += to_float(bias[n]);
            from_float(v, &C[(size_t)m * N + n]);
        }
    }
}

// One thread per 32-element block: amax, scale = amax / 127, round half-even, store.
// The same kernel quantizes weights and activations, so both share one convention.
template <typename T>
__global__ void quantize_int8_kernel(const T* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scale,
                                     long total_blocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_blocks) return;
    const T* src = x + b * QBLOCK;
    float v[QBLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < QBLOCK; e++) {
        v[e] = to_float(src[e]);
        amax = fmaxf(amax, fabsf(v[e]));
    }
    float s = fmaxf(amax, 1e-8f) * (1.0f / 127.0f);
    uint32_t words[8];
#pragma unroll
    for (int w = 0; w < 8; w++) {
        uint32_t word = 0;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int qv = (int)fminf(fmaxf(rintf(v[4 * w + e] / s), -127.0f), 127.0f);
            word |= (uint32_t)(qv & 0xFF) << (8 * e);
        }
        words[w] = word;
    }
    uint4* dst = (uint4*)(q + b * QBLOCK);
    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
    scale[b] = s;
}

template <typename T>
void quantize_launch(const void* x, void* q, float* scale, long total_blocks, hipStream_t stream) {
    long grid = (total_blocks + THREADS - 1) / THREADS;
    quantize_int8_kernel<T><<<grid, THREADS, 0, stream>>>((const T*)x, (int8_t*)q, scale, total_blocks);
}

template <int BM, int BN, int TM, int TN, typename T>
void gemm_launch(const void* A, const float* As, const void* B, const float* Bs, const void* bias, void* C, int M,
                 int N, int n_blocks, const ConvGeom* geom, hipStream_t stream) {
    int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    if (geom)
        int8_gemm_kernel<BM, BN, TM, TN, true, T><<<tiles, THREADS, 0, stream>>>(
            (const uint4*)A, As, (const uint4*)B, Bs, (const T*)bias, (T*)C, M, N, n_blocks, *geom);
    else
        int8_gemm_kernel<BM, BN, TM, TN, false, T><<<tiles, THREADS, 0, stream>>>(
            (const uint4*)A, As, (const uint4*)B, Bs, (const T*)bias, (T*)C, M, N, n_blocks, ConvGeom{});
}

template <typename T>
void gemm_config(int config, const void* A, const float* As, const void* B, const float* Bs, const void* bias, void* C,
                 int M, int N, int n_blocks, const ConvGeom* geom, hipStream_t stream) {
    switch (config) {
        case 1: gemm_launch<128, 64, 8, 4, T>(A, As, B, Bs, bias, C, M, N, n_blocks, geom, stream); break;
        case 2: gemm_launch<64, 128, 4, 8, T>(A, As, B, Bs, bias, C, M, N, n_blocks, geom, stream); break;
        case 3: gemm_launch<128, 128, 8, 8, T>(A, As, B, Bs, bias, C, M, N, n_blocks, geom, stream); break;
        default: gemm_launch<64, 64, 4, 4, T>(A, As, B, Bs, bias, C, M, N, n_blocks, geom, stream); break;
    }
}

}  // namespace

// dtype: 0 = float32, 1 = float16, 2 = bfloat16
void quantize_int8_dispatch(int dtype, const void* x, void* q, float* scale, long total_blocks, hipStream_t stream) {
    if (dtype == 0) quantize_launch<float>(x, q, scale, total_blocks, stream);
    else if (dtype == 1) quantize_launch<__half>(x, q, scale, total_blocks, stream);
    else quantize_launch<__hip_bfloat16>(x, q, scale, total_blocks, stream);
}

// geom (optional): implicit-GEMM conv, A/As = channels-last input (see ConvGeom)
void gemm_int8_dispatch(int dtype, int config, const void* A, const float* As, const void* B, const float* Bs,
                        const void* bias, void* C, int M, int N, int n_blocks, const int* geom, hipStream_t stream) {
    ConvGeom g{};
    if (geom) g = ConvGeom{geom[0], geom[1], geom[2], geom[3], geom[4], geom[5], geom[6], geom[7], geom[8], geom[9]};
    const ConvGeom* gp = geom ? &g : nullptr;
    if (dtype == 0) gemm_config<float>(config, A, As, B, Bs, bias, C, M, N, n_blocks, gp, stream);
    else if (dtype == 1) gemm_config<__half>(config, A, As, B, Bs, bias, C, M, N, n_blocks, gp, stream);
    else gemm_config<__hip_bfloat16>(config, A, As, B, Bs, bias, C, M, N, n_blocks, gp, stream);
}
