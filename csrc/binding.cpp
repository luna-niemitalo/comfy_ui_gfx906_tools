#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

void quantize_int4_dispatch(int dtype, const void* x, void* packed, float* scale, long total_blocks, hipStream_t stream);
void gemm_dispatch(int dtype, const void* A, const float* As, const void* B, const float* Bs,
                   const void* bias, void* C, int M, int N, int n_blocks, hipStream_t stream);

static int dtype_code(torch::ScalarType t) {
    if (t == torch::kFloat32) return 0;
    if (t == torch::kFloat16) return 1;
    TORCH_CHECK(t == torch::kBFloat16, "unsupported dtype ", t, ", expected float32/float16/bfloat16");
    return 2;
}

// x: [M, K] float32/16/bf16 -> (packed int8 [M, K/2], scale float32 [M, K/32])
std::vector<torch::Tensor> quantize_int4(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be on device");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0, "x must be [M, K] with K % 32 == 0, got ", x.sizes());
    // Multi-GPU correctness: without this, kernel launches and stream
    // selection silently use whatever device is "current" rather than the
    // one the tensors live on -- a memory access fault on any GPU but 0.
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xc = x.contiguous();
    int64_t M = xc.size(0), K = xc.size(1);
    auto packed = torch::empty({M, K / 2}, xc.options().dtype(torch::kInt8));
    auto scale = torch::empty({M, K / 32}, xc.options().dtype(torch::kFloat32));
    if (M == 0) return {packed, scale};
    quantize_int4_dispatch(dtype_code(xc.scalar_type()), xc.data_ptr(), packed.data_ptr(), scale.data_ptr<float>(),
                           M * (K / 32), at::cuda::getCurrentCUDAStream().stream());
    return {packed, scale};
}

// A_packed/A_scale: from quantize_int4. B_packed [N, K/2], B_scale [N, K/32].
// Returns [M, N] in out_dtype; bias (optional, out_dtype) is fused into the store.
torch::Tensor gemm_int4(torch::Tensor A_packed, torch::Tensor A_scale, torch::Tensor B_packed, torch::Tensor B_scale,
                        c10::optional<torch::Tensor> bias, torch::ScalarType out_dtype) {
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda(), "inputs must be on device");
    TORCH_CHECK(A_packed.device() == B_packed.device(), "A and B must be on the same device");
    TORCH_CHECK(A_packed.dtype() == torch::kInt8 && B_packed.dtype() == torch::kInt8, "packed tensors must be int8");
    TORCH_CHECK(A_packed.size(1) == B_packed.size(1), "K mismatch: A ", A_packed.sizes(), " B ", B_packed.sizes());
    TORCH_CHECK(A_scale.dtype() == torch::kFloat32 && B_scale.dtype() == torch::kFloat32, "scales must be float32");
    const c10::cuda::CUDAGuard device_guard(A_packed.device());
    auto A = A_packed.contiguous(), B = B_packed.contiguous();
    auto As = A_scale.contiguous(), Bs = B_scale.contiguous();
    int64_t M = A.size(0), N = B.size(0), n_blocks = A.size(1) / 16;
    auto C = torch::empty({M, N}, A.options().dtype(out_dtype));
    if (M == 0) return C;
    const void* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == out_dtype, "bias dtype ", bias->scalar_type(), " != out dtype ", out_dtype);
        bias_c = bias->contiguous();
        bias_ptr = bias_c.data_ptr();
    }
    gemm_dispatch(dtype_code(out_dtype), A.data_ptr(), As.data_ptr<float>(), B.data_ptr(), Bs.data_ptr<float>(),
                  bias_ptr, C.data_ptr(), (int)M, (int)N, (int)n_blocks, at::cuda::getCurrentCUDAStream().stream());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4, "fused block-32 int4 activation quantize");
    m.def("gemm_int4", &gemm_int4, "gfx906 block-scaled int4 GEMM");
}
