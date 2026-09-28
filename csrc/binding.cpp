#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

void int4_gemm_scaled_launch(const int8_t* A, const float* As, const int8_t* B, const float* Bs,
                              float* C, int M, int N, int K, hipStream_t stream);

// A_packed: [M, K/2] int8 (nibble-packed, two's complement, block=32)
// A_scale:  [M, K/32] float32
// B_packed: [N, K/2] int8
// B_scale:  [N, K/32] float32
// returns:  [M, N] float32
torch::Tensor int4_gemm_scaled(torch::Tensor A_packed, torch::Tensor A_scale,
                                torch::Tensor B_packed, torch::Tensor B_scale, int64_t K) {
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda(), "inputs must be on device");
    TORCH_CHECK(A_packed.device() == B_packed.device(), "A and B must be on the same device");
    TORCH_CHECK(A_packed.dtype() == torch::kInt8 && B_packed.dtype() == torch::kInt8, "packed tensors must be int8");
    TORCH_CHECK(A_scale.dtype() == torch::kFloat32 && B_scale.dtype() == torch::kFloat32, "scales must be float32");

    // Multi-GPU correctness: without this, kernel launches and stream
    // selection below silently use whatever device happens to be "current"
    // rather than the device these tensors actually live on -- causes a
    // GPU memory access fault on any GPU other than device 0.
    const c10::cuda::CUDAGuard device_guard(A_packed.device());

    auto A_c = A_packed.contiguous();
    auto B_c = B_packed.contiguous();
    auto As_c = A_scale.contiguous();
    auto Bs_c = B_scale.contiguous();

    int64_t M = A_c.size(0);
    int64_t N = B_c.size(0);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A_c.device()));

    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int4_gemm_scaled_launch(
        A_c.data_ptr<int8_t>(), As_c.data_ptr<float>(),
        B_c.data_ptr<int8_t>(), Bs_c.data_ptr<float>(),
        C.data_ptr<float>(), (int)M, (int)N, (int)K, stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_gemm_scaled", &int4_gemm_scaled, "gfx906 block-scaled int4 GEMM");
}
