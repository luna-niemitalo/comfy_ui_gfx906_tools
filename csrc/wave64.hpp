#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// gfx906 wave64 helpers shared by attention.hip and attention_train.hip.
// Lane layout they assume: 16 consecutive lanes form one DPP row; kernels map
// tx = tid % 16 onto them.

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ half2_t as_h2(uint32_t v) { return __builtin_bit_cast(half2_t, v); }

// 8 fp16 pairs of a against b, accumulated in fp32 (v_dot2_f32_f16)
__device__ __forceinline__ float dot8h(uint4 a, uint4 b, float acc) {
    acc = __builtin_amdgcn_fdot2(as_h2(a.x), as_h2(b.x), acc, false);
    acc = __builtin_amdgcn_fdot2(as_h2(a.y), as_h2(b.y), acc, false);
    acc = __builtin_amdgcn_fdot2(as_h2(a.z), as_h2(b.z), acc, false);
    return __builtin_amdgcn_fdot2(as_h2(a.w), as_h2(b.w), acc, false);
}

// v_perm_b32 selectors: {lo16(lo), lo16(hi)} and {hi16(lo), hi16(hi)} -- the
// half2 pair of one channel across two rows, for transposed LDS writes.
__device__ __forceinline__ uint32_t pair_lo(uint32_t lo, uint32_t hi) { return __builtin_amdgcn_perm(hi, lo, 0x05040100); }
__device__ __forceinline__ uint32_t pair_hi(uint32_t lo, uint32_t hi) { return __builtin_amdgcn_perm(hi, lo, 0x07060302); }

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

// readlane must run with every lane active (mantlecore isa_cross_lane_probe:
// after a divergent branch it returns the caller's own value).
__device__ __forceinline__ float wave_max_of_rows(float row_uniform) {
    const int v = __builtin_bit_cast(int, row_uniform);
    return fmaxf(fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 0)),
                       __builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 16))),
                 fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 32)),
                       __builtin_bit_cast(float, __builtin_amdgcn_readlane(v, 48))));
}
