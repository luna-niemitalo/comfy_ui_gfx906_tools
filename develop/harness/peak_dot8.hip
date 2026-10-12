#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Pure-register v_dot8_i32_i4 throughput: the ceiling any int4 GEMM on this
// GPU can reach. 16 independent accumulator chains per lane hide the dot
// latency; the result is stored so nothing is optimized away.
__global__ void __launch_bounds__(256) peak_kernel(int* out, int seed, int iters) {
    int a = seed ^ threadIdx.x, b = seed * 7 + blockIdx.x;
    int acc[16];
#pragma unroll
    for (int k = 0; k < 16; k++) acc[k] = k;
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int k = 0; k < 16; k++) acc[k] = __builtin_amdgcn_sdot8(a + k, b, acc[k], false);
    }
    int s = 0;
#pragma unroll
    for (int k = 0; k < 16; k++) s += acc[k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Launches once; returns the int4 ops (2 per multiply-add) that launch performs.
double run(torch::Tensor out, int64_t blocks, int64_t iters) {
    TORCH_CHECK(out.numel() >= blocks * 256, "out needs ", blocks * 256, " elements, got ", out.numel());
    const c10::cuda::CUDAGuard guard(out.device());
    peak_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(out.data_ptr<int>(), 3, (int)iters);
    return (double)blocks * 256 * iters * 16 * 8 * 2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
