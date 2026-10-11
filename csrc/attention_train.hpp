#pragma once
#include <hip/hip_runtime.h>

// Interface of attention_train.hip, shared with binding.cpp.

// Element strides of a [B, L, heads, 128] tensor (channels contiguous).
struct Str {
    long long b, l, h;
};

void attn_train_fwd_dispatch(const void* q, Str qs, const void* k, Str ks, const void* v, Str vs, void* o, Str os,
                             float* lse, int B, int H, int KVH, int L, float scale, int causal, hipStream_t stream);

// Backward kernels a call launches (profiling: time one at a time). dkdv and dq
// read the delta of an earlier call when BWD_DELTA is left out.
enum { BWD_DELTA = 1, BWD_DKDV = 2, BWD_DQ = 4, BWD_ALL = 7 };

// delta: fp32 [B, H, L] scratch.
void attn_train_bwd_dispatch(const void* q, Str qs, const void* k, Str ks, const void* v, Str vs, const void* o, Str os,
                             const void* dout, Str dos, const float* lse, float* delta, void* dq, Str dqs, void* dk,
                             Str dks, void* dv, Str dvs, int B, int H, int KVH, int L, float scale, int causal,
                             int stages, hipStream_t stream);
