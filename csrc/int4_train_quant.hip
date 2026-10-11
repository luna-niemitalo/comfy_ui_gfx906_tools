#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

// Quantizers for the int4 block-16 training GEMMs (int4_gemm_scaled.hip, qblock
// 16): an int4 backward with fp16 forward (develop/experiments/2026-10-11/
// RESULTS.md 5: int4 b16 ~ NVFP4 error; recipe from arXiv 2509.25149 minus 2D
// weight tiles). Output layout = the GEMM's: packed int8 [rows, K / 2] with
// element e of each 32-element chunk in word e / 8, nibble e % 8 (so the two
// 16-element blocks of a chunk are words 0-1 and 2-3), fp32 scales [rows, K / 16].
// Scale = amax / 7, codes clamped to [-8, 7] (the inference quantizer's
// convention); round to nearest even, or stochastic (floor(v + u)).
//
// Emulation counterpart, checked against in develop/experiments/2026-10-11/
// int4_train_test.py: quant_study.q_int / rht16.

// lowbias32 (Chris Wellons): a counter-based hash, so SR needs no RNG state
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// uniform [0, 1) per (seed, element); 24 bits are all a float in [0, 1) holds
__device__ __forceinline__ float uniform(uint32_t seed, unsigned long long element) {
    const uint32_t h = hash32(seed ^ hash32((uint32_t)element ^ hash32((uint32_t)(element >> 32))));
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

// 16 values -> two packed words, returns the scale
template <bool SR>
__device__ __forceinline__ float quant16(const float (&v)[16], uint32_t seed, unsigned long long first, uint32_t& w0,
                                         uint32_t& w1) {
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; e++) amax = fmaxf(amax, fabsf(v[e]));
    const float s = fmaxf(amax, 1e-8f) * (1.0f / 7.0f);
    w0 = 0;
    w1 = 0;
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const float t = v[e] / s;
        const float r = SR ? floorf(t + uniform(seed, first + e)) : rintf(t);
        const uint32_t nib = (uint32_t)((int)fminf(fmaxf(r, -8.0f), 7.0f) & 0xF);
        if (e < 8) w0 |= nib << (4 * e);
        else w1 |= nib << (4 * (e - 8));
    }
    return s;
}

// Blocks along each row: x [R, C] (rows `row_stride` apart, C % 32 == 0) ->
// packed [R, C / 2], scale [R, C / 16]. One thread per 32-element chunk.
template <bool SR>
__global__ void __launch_bounds__(256) quant_rows_b16(const __half* __restrict__ x, long long row_stride, int R, int C,
                                                      uint32_t seed, uint4* __restrict__ packed,
                                                      float* __restrict__ scale) {
    const long long chunk = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int chunks = C / 32;
    if (chunk >= (long long)R * chunks) return;
    const int r = chunk / chunks, c = chunk % chunks;
    const uint4* src = (const uint4*)(x + r * row_stride + c * 32);
    uint32_t words[4];
    float s[2];
#pragma unroll
    for (int hb = 0; hb < 2; hb++) {
        const uint4 lo = src[2 * hb], hi = src[2 * hb + 1];
        const uint32_t raw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float v[16];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const __half2 h2 = __builtin_bit_cast(__half2, raw[e]);
            v[2 * e] = __low2float(h2);
            v[2 * e + 1] = __high2float(h2);
        }
        s[hb] = quant16<SR>(v, seed, (unsigned long long)r * C + c * 32 + hb * 16, words[2 * hb], words[2 * hb + 1]);
    }
    packed[chunk] = make_uint4(words[0], words[1], words[2], words[3]);
    *(float2*)(scale + (long long)r * (C / 16) + 2 * c) = make_float2(s[0], s[1]);
}

// Blocks along each column, i.e. transposed: x [R, C] (R % 32 == 0) -> packed
// [C, R / 2], scale [C, R / 16]. One thread per (column, 16-row group): the
// group is one quantization block and, with RHT, one Hadamard transform: v <-
// (H16 / 4) diag(signs) v, an in-register fast Walsh-Hadamard (H16 is the
// Sylvester matrix, entry (-1)^popcount(i & j), as quant_study.rht16 builds it).
// Consecutive lanes take consecutive columns: each of the 16 loads is coalesced.
template <bool RHT, bool SR>
__global__ void __launch_bounds__(256) quant_cols_b16(const __half* __restrict__ x, long long row_stride, int R, int C,
                                                      const float* __restrict__ signs, uint32_t seed,
                                                      uint2* __restrict__ packed, float* __restrict__ scale) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = R / 16;
    if (id >= (long long)C * groups) return;
    const int c = id % C, g = id / C;
    float v[16];
#pragma unroll
    for (int t = 0; t < 16; t++) v[t] = __half2float(x[(long long)(g * 16 + t) * row_stride + c]);
    if constexpr (RHT) {
#pragma unroll
        for (int t = 0; t < 16; t++) v[t] *= signs[t];
#pragma unroll
        for (int h = 1; h < 16; h <<= 1)
#pragma unroll
            for (int i = 0; i < 16; i++)
                if ((i & h) == 0) {
                    const float a = v[i], b = v[i + h];
                    v[i] = a + b;
                    v[i + h] = a - b;
                }
#pragma unroll
        for (int t = 0; t < 16; t++) v[t] *= 0.25f;
    }
    uint32_t w0, w1;
    const float s = quant16<SR>(v, seed, (unsigned long long)c * R + g * 16, w0, w1);
    packed[(long long)c * groups + g] = make_uint2(w0, w1);
    scale[(long long)c * groups + g] = s;
}

void quant_rows_b16_dispatch(const void* x, long long row_stride, int R, int C, bool sr, uint32_t seed, void* packed,
                             float* scale, hipStream_t stream) {
    const long long threads = (long long)R * (C / 32);
    const dim3 grid((threads + 255) / 256);
    // one launch per braced block: hipify glued `else kernel<<<...>>>` into `elsehipLaunchKernelGGL`
    if (sr) {
        quant_rows_b16<true><<<grid, 256, 0, stream>>>((const __half*)x, row_stride, R, C, seed, (uint4*)packed, scale);
    } else {
        quant_rows_b16<false><<<grid, 256, 0, stream>>>((const __half*)x, row_stride, R, C, seed, (uint4*)packed, scale);
    }
}

void quant_cols_b16_dispatch(const void* x, long long row_stride, int R, int C, const float* signs, bool sr,
                             uint32_t seed, void* packed, float* scale, hipStream_t stream) {
    const long long threads = (long long)C * (R / 16);
    const dim3 grid((threads + 255) / 256);
    const __half* xh = (const __half*)x;
#define COLS_CASE(RHT, SRM) \
    { quant_cols_b16<RHT, SRM><<<grid, 256, 0, stream>>>(xh, row_stride, R, C, signs, seed, (uint2*)packed, scale); }
    if (signs && sr) {
        COLS_CASE(true, true)
    } else if (signs) {
        COLS_CASE(true, false)
    } else if (sr) {
        COLS_CASE(false, true)
    } else {
        COLS_CASE(false, false)
    }
#undef COLS_CASE
}
