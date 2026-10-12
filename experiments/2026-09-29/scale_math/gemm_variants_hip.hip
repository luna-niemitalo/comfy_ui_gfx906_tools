#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// SCALE-MATH STUDY COPY of custom_nodes/ComfyUI-Gfx906Int4/csrc/int4_gemm_scaled.hip
// (2026-09-29). SCALE_MODE picks the per-block rescale; modes 1-4 are SPEED BOUNDS ONLY
// (numerically wrong on today's format, each stands for a format change):
//   0 production: acc += float(ip) * (as * bs)                         per block, per pair
//   1 64-value blocks: 8 dots into one ip, one rescale per 2 blocks  (scale of block 0)
//   2 one-sided scale (per-token activation or per-channel weight): acc += float(ip) * bs
//   3 two-level scales (block scale = int sub-scale x super scale): int32 accumulate
//     ip * (a_sub * b_sub) with v_mad_i24, one float rescale per stage (4 blocks)
//   4 no rescale at all: acc_int += ip (ceiling)
#ifndef SCALE_MODE
#define SCALE_MODE 0
#endif
// Block-scaled int4 GEMM for gfx906: C[M,N] = A[M,K] @ B[N,K]^T (+ bias).
// A (activation) and B (weight) are nibble-packed two's-complement int4,
// 32 elements (= 16 bytes = one uint4 load) per float32 scale. Each 32-elem
// block is 4 v_dot8_i32_i4 into an int32 partial, then rescaled into an
// fp32 accumulator (the scale is not constant across K).
//
// gfx906 is GCN: wave64 only, 64 KB LDS/CU, 256 VGPRs/lane. Tuning notes
// (develop/harness/bench_linear.py, 2026-09-28, ~40 TOPS at M>=2304 vs ~13
// for the previous 64x64 byte-load kernel):
// - one 16-byte global load == one quant block; BK = 4 blocks per LDS stage,
//   the next stage prefetched into registers while the current one computes
// - LDS rows padded by 16 bytes: unpadded 64-byte rows put every ty-row of a
//   wave in the same bank for ds_read_b128
// - thread owns rows ty + i*16 and cols tx + j*16 (strided, not contiguous)
//   so the 16 tx lanes of a wave read 16 distinct B rows without conflicts
// - 128x128/8x8 and 128x64/8x4 tiles were slower: 234/100+ VGPRs -> one
//   wave per SIMD. Forcing 3-4 waves/EU spills. The block loop is kept
//   rolled (unrolling raises VGPRs for no gain).
// - grouped tile order: GROUP_M M-tiles walk N together so co-resident WGs
//   share A and B slices in L2
#define QBLOCK 32
#define BK_BLOCKS 4
#define GROUP_M 8
#define BM 64
#define BN 64
#define TM 4
#define TN 4
#define THREADS 256
#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)
#define ROW (BK_BLOCKS + 1)

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void from_float(float v, float* out) { *out = v; }
__device__ __forceinline__ void from_float(float v, __half* out) { *out = __float2half(v); }
__device__ __forceinline__ void from_float(float v, __hip_bfloat16* out) { *out = __float2bfloat16(v); }

// Split-K: blockIdx.y picks a slice of `tiles_per_split` K stages. With one
// split C is the output; otherwise C is an fp32 workspace [splits, M, N] with
// no bias, summed by splitk_reduce_kernel (gfx906 has no global float atomics).
template <typename T>
__global__ void __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(2))) int4_gemm_kernel(
    const uint4* __restrict__ A, const float* __restrict__ A_scale,
    const uint4* __restrict__ B, const float* __restrict__ B_scale,
    const T* __restrict__ bias, T* __restrict__ C, int M, int N, int n_blocks, int tiles_per_split) {
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

    // BM * BK_BLOCKS == THREADS: exactly one A and one B block per thread per
    // stage. Consecutive threads take consecutive blocks of one row, so each
    // row's 64-byte K slice is one contiguous global segment.
    const int load_r = tid / BK_BLOCKS;
    const int load_c = tid % BK_BLOCKS;
    const int load_m = row0 + load_r;
    const int load_n = col0 + load_r;
    uint4 a_pref, b_pref;
    float as_pref, bs_pref;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    auto load_stage = [&](int kt) {
        int kb = kt * BK_BLOCKS + load_c;
        bool a_ok = load_m < M && kb < n_blocks;
        bool b_ok = load_n < N && kb < n_blocks;
        a_pref = a_ok ? A[(size_t)load_m * n_blocks + kb] : zero4;
        as_pref = a_ok ? A_scale[(size_t)load_m * n_blocks + kb] : 0.0f;
        b_pref = b_ok ? B[(size_t)load_n * n_blocks + kb] : zero4;
        bs_pref = b_ok ? B_scale[(size_t)load_n * n_blocks + kb] : 0.0f;
    };

    const int k_first = blockIdx.y * tiles_per_split;
    const int k_end = min((n_blocks + BK_BLOCKS - 1) / BK_BLOCKS, k_first + tiles_per_split);
    C += (size_t)blockIdx.y * M * N;
    load_stage(k_first);
    for (int kt = k_first; kt < k_end; kt++) {
        As[load_r * ROW + load_c] = a_pref;
        As_scale[load_c][load_r] = as_pref;
        Bs[load_r * ROW + load_c] = b_pref;
        Bs_scale[load_c][load_r] = bs_pref;
        __syncthreads();
        if (kt + 1 < k_end) load_stage(kt + 1);

#if SCALE_MODE == 3 || SCALE_MODE == 4
        int iacc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) iacc[i][j] = 0;
#endif
#pragma unroll 1
        for (int c = 0; c < BK_BLOCKS; c += (SCALE_MODE == 1 ? 2 : 1)) {
            uint4 bq[TN];
            float bsc[TN];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                bq[j] = Bs[(tx + j * THREADS_X) * ROW + c];
                bsc[j] = Bs_scale[c][tx + j * THREADS_X];
            }
#if SCALE_MODE == 1
            uint4 bq2[TN];
#pragma unroll
            for (int j = 0; j < TN; j++) bq2[j] = Bs[(tx + j * THREADS_X) * ROW + c + 1];
#endif
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint4 aq = As[(ty + i * THREADS_Y) * ROW + c];
                float asc = As_scale[c][ty + i * THREADS_Y];
#if SCALE_MODE == 1
                uint4 aq2 = As[(ty + i * THREADS_Y) * ROW + c + 1];
#endif
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    int ip = __builtin_amdgcn_sdot8((int)aq.x, (int)bq[j].x, 0, false);
                    ip = __builtin_amdgcn_sdot8((int)aq.y, (int)bq[j].y, ip, false);
                    ip = __builtin_amdgcn_sdot8((int)aq.z, (int)bq[j].z, ip, false);
                    ip = __builtin_amdgcn_sdot8((int)aq.w, (int)bq[j].w, ip, false);
#if SCALE_MODE == 0
                    acc[i][j] = fmaf((float)ip, asc * bsc[j], acc[i][j]);
#elif SCALE_MODE == 1
                    ip = __builtin_amdgcn_sdot8((int)aq2.x, (int)bq2[j].x, ip, false);
                    ip = __builtin_amdgcn_sdot8((int)aq2.y, (int)bq2[j].y, ip, false);
                    ip = __builtin_amdgcn_sdot8((int)aq2.z, (int)bq2[j].z, ip, false);
                    ip = __builtin_amdgcn_sdot8((int)aq2.w, (int)bq2[j].w, ip, false);
                    acc[i][j] = fmaf((float)ip, asc * bsc[j], acc[i][j]);
#elif SCALE_MODE == 2
                    acc[i][j] = fmaf((float)ip, bsc[j], acc[i][j]);
#elif SCALE_MODE == 3
                    // sub-scales stand in as the scales' low bits (the kernel would store int8 sub-scales)
                    int sub = __mul24(__float_as_int(asc) & 0xF, __float_as_int(bsc[j]) & 0xF);
                    iacc[i][j] = __mul24(ip, sub) + iacc[i][j];  // -> v_mad_i32_i24
#else
                    iacc[i][j] += ip;
#endif
                }
            }
        }
#if SCALE_MODE == 3 || SCALE_MODE == 4
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] = fmaf((float)iacc[i][j], As_scale[0][ty + i * THREADS_Y] * Bs_scale[0][tx + j * THREADS_X], acc[i][j]);
#endif
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

template <typename T>
__global__ void splitk_reduce_kernel(const float* __restrict__ partial, const T* __restrict__ bias, T* __restrict__ C,
                                     int splits, long MN, int N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    float v = 0.0f;
    for (int s = 0; s < splits; s++) v += partial[s * MN + idx];
    if (bias) v += to_float(bias[idx % N]);
    from_float(v, &C[idx]);
}

// One thread per 32-element quant block: read, amax, scale, round, pack.
// Must match ops.quantize_pack_int4 bit-for-bit (same fp32 divide, rint =
// round-half-even like torch.round), so weights quantized by the Python
// path and activations quantized here share one convention.
template <typename T>
__global__ void quantize_int4_kernel(const T* __restrict__ x, uint4* __restrict__ packed,
                                     float* __restrict__ scale, long total_blocks) {
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
    // torch evaluates `tensor / 7.0` (scalar divisor) as a reciprocal multiply
    float s = fmaxf(amax, 1e-8f) * (1.0f / 7.0f);
    uint32_t words[4] = {0, 0, 0, 0};
#pragma unroll
    for (int e = 0; e < QBLOCK; e++) {
        int q = (int)fminf(fmaxf(rintf(v[e] / s), -8.0f), 7.0f);
        words[e / 8] |= (uint32_t)(q & 0xF) << (4 * (e % 8));
    }
    packed[b] = make_uint4(words[0], words[1], words[2], words[3]);
    scale[b] = s;
}

// dtype: 0 = float32, 1 = float16, 2 = bfloat16
template <typename T>
static void quantize_launch(const void* x, void* packed, float* scale, long total_blocks, hipStream_t stream) {
    long grid = (total_blocks + THREADS - 1) / THREADS;
   hipLaunchKernelGGL(( quantize_int4_kernel<T>), dim3(grid), dim3(THREADS), 0, stream, (const T*)x, (uint4*)packed, scale, total_blocks);
}

// splits > 1 needs `workspace`: fp32 [splits, M, N]
template <typename T>
static void gemm_launch(const void* A, const float* As, const void* B, const float* Bs, const void* bias, void* C,
                        int M, int N, int n_blocks, int splits, float* workspace, hipStream_t stream) {
    int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    int k_tiles = (n_blocks + BK_BLOCKS - 1) / BK_BLOCKS;
    int tiles_per_split = (k_tiles + splits - 1) / splits;
    if (splits == 1) {
       hipLaunchKernelGGL(( int4_gemm_kernel<T>), dim3(dim3(tiles, 1)), dim3(THREADS), 0, stream, 
            (const uint4*)A, As, (const uint4*)B, Bs, (const T*)bias, (T*)C, M, N, n_blocks, tiles_per_split);
        return;
    }
   hipLaunchKernelGGL(( int4_gemm_kernel<float>), dim3(dim3(tiles, splits)), dim3(THREADS), 0, stream, 
        (const uint4*)A, As, (const uint4*)B, Bs, nullptr, workspace, M, N, n_blocks, tiles_per_split);
    long MN = (long)M * N;
   hipLaunchKernelGGL(( splitk_reduce_kernel<T>), dim3((MN + THREADS - 1) / THREADS), dim3(THREADS), 0, stream, 
        workspace, (const T*)bias, (T*)C, splits, MN, N);
}

void quantize_int4_dispatch(int dtype, const void* x, void* packed, float* scale, long total_blocks, hipStream_t stream) {
    if (dtype == 0) quantize_launch<float>(x, packed, scale, total_blocks, stream);
    else if (dtype == 1) quantize_launch<__half>(x, packed, scale, total_blocks, stream);
    else quantize_launch<__hip_bfloat16>(x, packed, scale, total_blocks, stream);
}

void gemm_dispatch(int dtype, const void* A, const float* As, const void* B, const float* Bs, const void* bias,
                   void* C, int M, int N, int n_blocks, int splits, float* workspace, hipStream_t stream) {
    if (dtype == 0) gemm_launch<float>(A, As, B, Bs, bias, C, M, N, n_blocks, splits, workspace, stream);
    else if (dtype == 1) gemm_launch<__half>(A, As, B, Bs, bias, C, M, N, n_blocks, splits, workspace, stream);
    else gemm_launch<__hip_bfloat16>(A, As, B, Bs, bias, C, M, N, n_blocks, splits, workspace, stream);
}

int gemm_tiles(int M, int N) { return ((M + BM - 1) / BM) * ((N + BN - 1) / BN); }
int gemm_k_tiles(int n_blocks) { return (n_blocks + BK_BLOCKS - 1) / BK_BLOCKS; }
