#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

void quantize_int4_dispatch(int dtype, const void* x, void* packed, float* scale, long total_blocks, hipStream_t stream);
void gemm_dispatch(int dtype, const void* A, const float* As, const void* B, const float* Bs, const void* bias,
                   void* C, int M, int N, int n_blocks, int splits, float* workspace, hipStream_t stream);
int gemm_tiles(int M, int N);
int gemm_k_tiles(int n_blocks);

// Workgroups resident at once per CU: __launch_bounds__(256) with
// amdgpu_waves_per_eu(2) -> 2 waves/SIMD x 4 SIMDs / 4 waves per WG.
constexpr int WGS_PER_CU = 2;
constexpr int MAX_SPLITS = 8;
// A split shorter than this many K stages (x 4 quant blocks x 32) spends
// more on prologue/epilogue and the fp32 partial round trip than it gains.
constexpr int MIN_STAGES_PER_SPLIT = 8;

// Split-K only when the output tiles can't cover the resident workgroup
// slots; then aim for ~2.4 waves (12/5) of workgroups. Swept 2026-09-29
// (bench_linear --splits, M 64..1152 x the four Krea2 shapes): best counts
// were 4-5 for M=192 N=1536 (72 tiles, 1.5x) and 3 for 96 tiles (1.1-1.2x);
// any split of a GEMM that already fills the GPU lost 2-40%.
static int auto_splits(int M, int N, int n_blocks, int cus) {
    const int tiles = gemm_tiles(M, N);
    const int slots = cus * WGS_PER_CU;
    if (tiles >= slots) return 1;
    const int max_splits = std::min(MAX_SPLITS, std::max(1, gemm_k_tiles(n_blocks) / MIN_STAGES_PER_SPLIT));
    const int wanted = (12 * slots + 5 * tiles - 1) / (5 * tiles);
    return std::min(wanted, max_splits);
}

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
// splits: split-K count, 0 = auto_splits.
torch::Tensor gemm_int4(torch::Tensor A_packed, torch::Tensor A_scale, torch::Tensor B_packed, torch::Tensor B_scale,
                        c10::optional<torch::Tensor> bias, torch::ScalarType out_dtype, int64_t splits) {
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
    TORCH_CHECK(splits >= 0 && splits <= MAX_SPLITS, "splits must be in [0, ", MAX_SPLITS, "], got ", splits);
    if (splits == 0) {
        int cus = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        splits = auto_splits((int)M, (int)N, (int)n_blocks, cus);
    }
    torch::Tensor workspace;
    if (splits > 1) workspace = torch::empty({splits, M, N}, A.options().dtype(torch::kFloat32));
    gemm_dispatch(dtype_code(out_dtype), A.data_ptr(), As.data_ptr<float>(), B.data_ptr(), Bs.data_ptr<float>(),
                  bias_ptr, C.data_ptr(), (int)M, (int)N, (int)n_blocks, (int)splits,
                  splits > 1 ? workspace.data_ptr<float>() : nullptr, at::cuda::getCurrentCUDAStream().stream());
    return C;
}

int64_t gemm_int4_auto_splits(int64_t M, int64_t N, int64_t K) {
    int cus = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return auto_splits((int)M, (int)N, (int)(K / 32), cus);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4, "fused block-32 int4 activation quantize");
    m.def("gemm_int4", &gemm_int4, "gfx906 block-scaled int4 GEMM", py::arg("A_packed"), py::arg("A_scale"),
          py::arg("B_packed"), py::arg("B_scale"), py::arg("bias"), py::arg("out_dtype"), py::arg("splits") = 0);
    m.def("gemm_int4_auto_splits", &gemm_int4_auto_splits, "split-K count gemm_int4 picks for [M,K] x [N,K]^T");
}
