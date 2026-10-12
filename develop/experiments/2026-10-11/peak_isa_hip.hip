#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

// Pure-register throughput of each gfx906 multiply-add class an LLM training
// step could lean on. Same scheme as develop/harness/peak_dot8.hip (its
// counterpart): 16 independent accumulator chains per lane hide latency, the
// sum is stored so nothing is optimized away.

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

constexpr int kChains = 16;

// ops per lane per instruction, counting a multiply-add as 2
enum Op { FMA_F32 = 0, PK_FMA_F16 = 1, DOT2_F32_F16 = 2, DOT4_I8 = 3, DOT8_I4 = 4 };
constexpr int kOpsPerInst[] = {2, 4, 4, 8, 16};

template <int OP>
__global__ void __launch_bounds__(256) peak_kernel(float* out, int seed, int iters) {
    const int ia = seed ^ threadIdx.x, ib = seed * 7 + blockIdx.x;
    const float fa = 1.0f + 1e-7f * threadIdx.x, fb = 0.999999f;
    const half2_t ha = {(_Float16)fa, (_Float16)0.5f}, hb = {(_Float16)fb, (_Float16)0.25f};
    float facc[kChains];
    int iacc[kChains];
    half2_t hacc[kChains];
#pragma unroll
    for (int k = 0; k < kChains; k++) {
        facc[k] = k;
        iacc[k] = k;
        hacc[k] = half2_t{(_Float16)k, (_Float16)k};
    }
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int k = 0; k < kChains; k++) {
            if constexpr (OP == FMA_F32) facc[k] = __builtin_fmaf(fa, fb, facc[k]);
            if constexpr (OP == PK_FMA_F16) hacc[k] = __builtin_elementwise_fma(ha, hb, hacc[k]);
            if constexpr (OP == DOT2_F32_F16) facc[k] = __builtin_amdgcn_fdot2(ha, hb, facc[k], false);
            if constexpr (OP == DOT4_I8) iacc[k] = __builtin_amdgcn_sdot4(ia + k, ib, iacc[k], false);
            if constexpr (OP == DOT8_I4) iacc[k] = __builtin_amdgcn_sdot8(ia + k, ib, iacc[k], false);
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < kChains; k++) s += facc[k] + (float)iacc[k] + (float)hacc[k].x + (float)hacc[k].y;
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Launches once; returns the ops that launch performs.
double run(torch::Tensor out, int64_t op, int64_t blocks, int64_t iters) {
    TORCH_CHECK(op >= 0 && op <= DOT8_I4, "op in [0, ", (int)DOT8_I4, "], got ", op);
    TORCH_CHECK(out.numel() >= blocks * 256, "out needs ", blocks * 256, " elements, got ", out.numel());
    const c10::cuda::CUDAGuard guard(out.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    float* p = out.data_ptr<float>();
    switch (op) {
        case FMA_F32:hipLaunchKernelGGL(( peak_kernel<FMA_F32>), dim3(blocks), dim3(256), 0, stream, p, 3, (int)iters); break;
        case PK_FMA_F16:hipLaunchKernelGGL(( peak_kernel<PK_FMA_F16>), dim3(blocks), dim3(256), 0, stream, p, 3, (int)iters); break;
        case DOT2_F32_F16:hipLaunchKernelGGL(( peak_kernel<DOT2_F32_F16>), dim3(blocks), dim3(256), 0, stream, p, 3, (int)iters); break;
        case DOT4_I8:hipLaunchKernelGGL(( peak_kernel<DOT4_I8>), dim3(blocks), dim3(256), 0, stream, p, 3, (int)iters); break;
        case DOT8_I4:hipLaunchKernelGGL(( peak_kernel<DOT8_I4>), dim3(blocks), dim3(256), 0, stream, p, 3, (int)iters); break;
    }
    return (double)blocks * 256 * iters * kChains * kOpsPerInst[op];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
