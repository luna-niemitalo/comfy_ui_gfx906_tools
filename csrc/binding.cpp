#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "train_kernels.hpp"

void quantize_int4_dispatch(int dtype, const void* x, void* packed, float* scale, long total_blocks, hipStream_t stream);
void gemm_dispatch(int dtype, int qblock, const void* A, const float* As, const void* B, const float* Bs,
                   const void* bias, void* C, int M, int N, int n_blocks, int splits, float* workspace,
                   hipStream_t stream);
int gemm_tiles(int M, int N);
int gemm_k_tiles(int n_blocks);
void attention_dispatch(int qk_mode, int pv_mode, const uint8_t* Q, const float* qs, const uint8_t* K, const float* ks,
                        const uint8_t* Vt, const float* vs, const uint8_t* keep, void* out, int B, int H, int KVH,
                        int Lq, int Lk, float skip_below, hipStream_t stream);
int attention_block_rows(int qk_mode);
void quantize_int8_dispatch(int dtype, const void* x, void* q, float* scale, long total_blocks, hipStream_t stream);
void gemm_int8_dispatch(int dtype, int config, const void* A, const float* As, const void* B, const float* Bs,
                        const void* bias, void* C, int M, int N, int n_blocks, const int* geom, hipStream_t stream);
int attention_max_tiles();

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
                        c10::optional<torch::Tensor> bias, torch::ScalarType out_dtype, int64_t splits,
                        int64_t qblock) {
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda(), "inputs must be on device");
    TORCH_CHECK(A_packed.device() == B_packed.device(), "A and B must be on the same device");
    TORCH_CHECK(A_packed.dtype() == torch::kInt8 && B_packed.dtype() == torch::kInt8, "packed tensors must be int8");
    TORCH_CHECK(A_packed.dim() == 2 && B_packed.dim() == 2 && A_packed.size(1) == B_packed.size(1) &&
                    A_packed.size(1) % 16 == 0,
                "expected A [M, K/2], B [N, K/2], K % 32 == 0; got A ", A_packed.sizes(), " B ", B_packed.sizes());
    TORCH_CHECK(A_scale.dtype() == torch::kFloat32 && B_scale.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(qblock == 32 || qblock == 16, "qblock must be 32 or 16, got ", qblock);
    const int64_t K = A_packed.size(1) * 2;
    TORCH_CHECK(A_scale.numel() == A_packed.size(0) * K / qblock && B_scale.numel() == B_packed.size(0) * K / qblock,
                "scales must be [M, K/", qblock, "] / [N, K/", qblock, "] = ", A_packed.size(0) * K / qblock, " / ",
                B_packed.size(0) * K / qblock, " elements, got ", A_scale.numel(), " / ", B_scale.numel());
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
    gemm_dispatch(dtype_code(out_dtype), (int)qblock, A.data_ptr(), As.data_ptr<float>(), B.data_ptr(), Bs.data_ptr<float>(),
                  bias_ptr, C.data_ptr(), (int)M, (int)N, (int)n_blocks, (int)splits,
                  splits > 1 ? workspace.data_ptr<float>() : nullptr, at::cuda::getCurrentCUDAStream().stream());
    return C;
}

// Fused attention (attention.hip). Tensors come from attention.py's prepare step:
// Q [B, H, Lq, qrow] uint8, K [B, KVH, Lk, qrow] uint8 (qrow = 256 fp16 / 128 int8 /
// 64 int4), q_scale [B, H, Lq] / k_scale [B, KVH, Lk] fp32 (int modes, else empty),
// Vt [B, KVH, ceil(Lk / 64), 128, 64 * vbytes] uint8, v_scale [B, KVH, ceil(Lk / 64),
// 128] fp32 (u8 mode, else empty). keep (optional) uint8 [B, H, ceil(Lq / block_rows),
// ceil(Lk / 64)], block_rows = attention_block_rows(qk_mode): nonzero = visit that key
// tile; every query block must keep at least one. Returns [B, Lq, H * 128] fp16.
torch::Tensor attention(torch::Tensor Q, torch::Tensor q_scale, torch::Tensor K, torch::Tensor k_scale,
                        torch::Tensor Vt, torch::Tensor v_scale, int64_t qk_mode, int64_t pv_mode, double skip_below,
                        c10::optional<torch::Tensor> keep) {
    TORCH_CHECK(qk_mode >= 0 && qk_mode <= 2, "qk_mode must be 0 fp16 / 1 int8 / 2 int4, got ", qk_mode);
    TORCH_CHECK(pv_mode >= 0 && pv_mode <= 1, "pv_mode must be 0 fp16 / 1 u8, got ", pv_mode);
    const int64_t qrow = qk_mode == 0 ? 256 : qk_mode == 1 ? 128 : 64;
    const int64_t vbytes = pv_mode == 0 ? 2 : 1;
    TORCH_CHECK(Q.dim() == 4 && Q.scalar_type() == torch::kUInt8 && Q.size(3) == qrow,
                "Q must be uint8 [B, H, L, ", qrow, "], got ", Q.scalar_type(), " ", Q.sizes());
    const int64_t B = Q.size(0), H = Q.size(1), Lq = Q.size(2);
    TORCH_CHECK(K.dim() == 4 && K.scalar_type() == torch::kUInt8 && K.size(0) == B && K.size(3) == qrow,
                "K must be uint8 [", B, ", KVH, Lk, ", qrow, "], got ", K.scalar_type(), " ", K.sizes());
    const int64_t KVH = K.size(1), Lk = K.size(2);
    TORCH_CHECK(KVH > 0 && H % KVH == 0, "H (", H, ") must be a multiple of KVH (", KVH, ")");
    const int64_t n_tiles = (Lk + 63) / 64;
    TORCH_CHECK(Vt.dim() == 5 && Vt.scalar_type() == torch::kUInt8 && Vt.size(0) == B && Vt.size(1) == KVH &&
                    Vt.size(2) == n_tiles && Vt.size(3) == 128 && Vt.size(4) == 64 * vbytes,
                "Vt must be uint8 [", B, ", ", KVH, ", ", n_tiles, ", 128, ", 64 * vbytes, "], got ", Vt.scalar_type(), " ",
                Vt.sizes());
    if (qk_mode != 0) {
        TORCH_CHECK(q_scale.numel() == B * H * Lq && k_scale.numel() == B * KVH * Lk,
                    "q_scale/k_scale must have ", B * H * Lq, "/", B * KVH * Lk, " elements, got ", q_scale.numel(), "/",
                    k_scale.numel());
    }
    if (pv_mode == 1) {
        TORCH_CHECK(v_scale.numel() == B * KVH * n_tiles * 128, "v_scale must have ", B * KVH * n_tiles * 128,
                    " elements, got ", v_scale.numel());
    }
    torch::Tensor keep_c;
    if (keep.has_value()) {
        const int64_t rows = attention_block_rows((int)qk_mode), n_qb = (Lq + rows - 1) / rows;
        TORCH_CHECK(keep->scalar_type() == torch::kUInt8 && keep->dim() == 4 && keep->size(0) == B &&
                        keep->size(1) == H && keep->size(2) == n_qb && keep->size(3) == n_tiles,
                    "keep must be uint8 [", B, ", ", H, ", ", n_qb, ", ", n_tiles, "], got ", keep->scalar_type(), " ",
                    keep->sizes());
        TORCH_CHECK(n_tiles <= attention_max_tiles(), "keep supports at most ", attention_max_tiles(),
                    " key tiles, got ", n_tiles);
        keep_c = keep->contiguous();
    }
    const c10::cuda::CUDAGuard device_guard(Q.device());
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = Vt.contiguous();
    auto qs = q_scale.to(torch::kFloat32).contiguous(), ks = k_scale.to(torch::kFloat32).contiguous();
    auto vs = v_scale.to(torch::kFloat32).contiguous();
    auto out = torch::empty({B, Lq, H * 128}, Q.options().dtype(torch::kFloat16));
    attention_dispatch((int)qk_mode, (int)pv_mode, Qc.data_ptr<uint8_t>(), qs.numel() ? qs.data_ptr<float>() : nullptr,
                       Kc.data_ptr<uint8_t>(), ks.numel() ? ks.data_ptr<float>() : nullptr, Vc.data_ptr<uint8_t>(),
                       vs.numel() ? vs.data_ptr<float>() : nullptr,
                       keep_c.defined() ? keep_c.data_ptr<uint8_t>() : nullptr, out.data_ptr(), (int)B, (int)H,
                       (int)KVH, (int)Lq, (int)Lk, (float)skip_below, at::cuda::getCurrentCUDAStream().stream());
    return out;
}

// x: [M, K] float32/16/bf16 -> (int8 [M, K], scale float32 [M, K/32]); scale = block amax / 127
std::vector<torch::Tensor> quantize_int8(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be on device");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0, "x must be [M, K] with K % 32 == 0, got ", x.sizes());
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xc = x.contiguous();
    int64_t M = xc.size(0), K = xc.size(1);
    auto q = torch::empty({M, K}, xc.options().dtype(torch::kInt8));
    auto scale = torch::empty({M, K / 32}, xc.options().dtype(torch::kFloat32));
    if (M == 0) return {q, scale};
    quantize_int8_dispatch(dtype_code(xc.scalar_type()), xc.data_ptr(), q.data_ptr(), scale.data_ptr<float>(),
                           M * (K / 32), at::cuda::getCurrentCUDAStream().stream());
    return {q, scale};
}

// A/A_scale: from quantize_int8, [M, K] / [M, K/32]. B [N, K], B_scale [N, K/32].
// Returns [M, N] in out_dtype; bias (optional, out_dtype) is fused into the store.
torch::Tensor gemm_int8(torch::Tensor A, torch::Tensor A_scale, torch::Tensor B, torch::Tensor B_scale,
                        c10::optional<torch::Tensor> bias, torch::ScalarType out_dtype, int64_t config) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be on device");
    TORCH_CHECK(config >= 0 && config <= 3, "config must be in [0, 3], got ", config);
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");
    TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8, "A and B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(1) && A.size(1) % 32 == 0,
                "expected A [M, K], B [N, K], K % 32 == 0; got A ", A.sizes(), " B ", B.sizes());
    const int64_t M = A.size(0), N = B.size(0), n_blocks = A.size(1) / 32;
    TORCH_CHECK(A_scale.dtype() == torch::kFloat32 && B_scale.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(A_scale.numel() == M * n_blocks && B_scale.numel() == N * n_blocks, "scales must be [M, K/32] / [N, K/32], got ",
                A_scale.sizes(), " / ", B_scale.sizes());
    const c10::cuda::CUDAGuard device_guard(A.device());
    auto Ac = A.contiguous(), Bc = B.contiguous(), As = A_scale.contiguous(), Bs = B_scale.contiguous();
    auto C = torch::empty({M, N}, Ac.options().dtype(out_dtype));
    if (M == 0) return C;
    const void* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == out_dtype && bias->numel() == N, "bias must be ", out_dtype, " [", N, "], got ",
                    bias->scalar_type(), " ", bias->sizes());
        bias_c = bias->contiguous();
        bias_ptr = bias_c.data_ptr();
    }
    gemm_int8_dispatch(dtype_code(out_dtype), (int)config, Ac.data_ptr(), As.data_ptr<float>(), Bc.data_ptr(), Bs.data_ptr<float>(),
                       bias_ptr, C.data_ptr(), (int)M, (int)N, (int)n_blocks, nullptr, at::cuda::getCurrentCUDAStream().stream());
    return C;
}

// Implicit-GEMM int8 conv. x/x_scale: quantize_int8 of the channels-last input, [B*H*W, C] /
// [B*H*W, C/32]. W/W_scale: quantize_int8 of the weight as [Cout, kh*kw*C] (tap-major).
// Returns [B*Ho*Wo, Cout] (channels-last output) in out_dtype.
torch::Tensor conv_int8(torch::Tensor x, torch::Tensor x_scale, torch::Tensor W, torch::Tensor W_scale,
                        c10::optional<torch::Tensor> bias, torch::ScalarType out_dtype, int64_t B, int64_t H, int64_t Wd,
                        int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw, int64_t config) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && x.device() == W.device(), "x and W must be on the same device");
    TORCH_CHECK(x.dtype() == torch::kInt8 && W.dtype() == torch::kInt8, "x and W must be int8");
    TORCH_CHECK(config >= 0 && config <= 3, "config must be in [0, 3], got ", config);
    const int64_t C = x.size(1), N = W.size(0);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == B * H * Wd && C % 32 == 0, "x must be [", B * H * Wd, ", C % 32 == 0], got ",
                x.sizes());
    TORCH_CHECK(W.dim() == 2 && W.size(1) == kh * kw * C, "W must be [Cout, ", kh * kw * C, "], got ", W.sizes());
    TORCH_CHECK(x_scale.dtype() == torch::kFloat32 && W_scale.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(x_scale.numel() == x.size(0) * (C / 32) && W_scale.numel() == N * kh * kw * (C / 32),
                "scale sizes: expected ", x.size(0) * (C / 32), " / ", N * kh * kw * (C / 32), ", got ", x_scale.numel(), " / ",
                W_scale.numel());
    const int64_t Ho = (H + 2 * ph - kh) / sh + 1, Wo = (Wd + 2 * pw - kw) / sw + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "empty output: ", H, "x", Wd, " kernel ", kh, "x", kw);
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xc = x.contiguous(), Wc = W.contiguous(), xs = x_scale.contiguous(), Ws = W_scale.contiguous();
    const int64_t M = B * Ho * Wo;
    auto out = torch::empty({M, N}, xc.options().dtype(out_dtype));
    const void* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias.has_value()) {
        TORCH_CHECK(bias->scalar_type() == out_dtype && bias->numel() == N, "bias must be ", out_dtype, " [", N, "], got ",
                    bias->scalar_type(), " ", bias->sizes());
        bias_c = bias->contiguous();
        bias_ptr = bias_c.data_ptr();
    }
    const int geom[10] = {(int)H, (int)Wd, (int)(C / 32), (int)Ho, (int)Wo, (int)kw, (int)sh, (int)sw, (int)ph, (int)pw};
    gemm_int8_dispatch(dtype_code(out_dtype), (int)config, xc.data_ptr(), xs.data_ptr<float>(), Wc.data_ptr(),
                       Ws.data_ptr<float>(), bias_ptr, out.data_ptr(), (int)M, (int)N, (int)(kh * kw * C / 32), geom,
                       at::cuda::getCurrentCUDAStream().stream());
    return out;
}

// Training attention (attention_train.hip). IN: q [B, L, H, 128], k/v [B, L, KVH, 128],
// fp16, any batch/token/head strides; the kernels read 16-byte chunks, so channels
// must be contiguous and every stride and the base 16-byte aligned (critical: a
// misaligned row is a wrong read, not a crash).
static Str head_strides(const char* name, const torch::Tensor& t, int64_t B, int64_t L, int64_t heads) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat16 && t.dim() == 4,
                name, " must be a device fp16 [B, L, heads, 128] tensor, got ", t.scalar_type(), " ", t.sizes());
    TORCH_CHECK(t.size(0) == B && t.size(1) == L && (heads < 0 || t.size(2) == heads) && t.size(3) == 128,
                name, " must be [", B, ", ", L, ", ", heads < 0 ? "KVH" : std::to_string(heads), ", 128], got ", t.sizes());
    TORCH_CHECK(t.stride(3) == 1 && t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0,
                name, " needs contiguous channels and strides that are multiples of 8, got strides ", t.strides());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " data must be 16-byte aligned");
    return Str{t.stride(0), t.stride(1), t.stride(2)};
}

std::vector<torch::Tensor> attention_train_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                                               double scale) {
    TORCH_CHECK(q.dim() == 4, "q must be [B, L, H, 128], got ", q.sizes());
    const int64_t B = q.size(0), L = q.size(1), H = q.size(2);
    const Str qs = head_strides("q", q, B, L, H), ks = head_strides("k", k, B, L, -1);
    const int64_t KVH = k.size(2);
    const Str vs = head_strides("v", v, B, L, KVH);
    TORCH_CHECK(KVH > 0 && H % KVH == 0, "H (", H, ") must be a multiple of KVH (", KVH, ")");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on one device");
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto o = torch::empty({B, L, H, 128}, q.options());
    auto lse = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
    attn_train_fwd_dispatch(q.data_ptr(), qs, k.data_ptr(), ks, v.data_ptr(), vs, o.data_ptr(),
                            Str{o.stride(0), o.stride(1), o.stride(2)}, lse.data_ptr<float>(), (int)B, (int)H, (int)KVH,
                            (int)L, (float)scale, causal, at::cuda::getCurrentCUDAStream().stream());
    return {o, lse};
}

std::vector<torch::Tensor> attention_train_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                                               torch::Tensor lse, torch::Tensor dout, bool causal, double scale,
                                               int64_t stages) {
    TORCH_CHECK(stages >= 1 && stages <= BWD_ALL, "stages must be a mask in [1, ", (int)BWD_ALL, "], got ", stages);
    TORCH_CHECK(q.dim() == 4, "q must be [B, L, H, 128], got ", q.sizes());
    const int64_t B = q.size(0), L = q.size(1), H = q.size(2);
    const Str qs = head_strides("q", q, B, L, H), ks = head_strides("k", k, B, L, -1);
    const int64_t KVH = k.size(2);
    const Str vs = head_strides("v", v, B, L, KVH), os = head_strides("o", o, B, L, H);
    const Str dos = head_strides("dout", dout, B, L, H);
    TORCH_CHECK(KVH > 0 && H % KVH == 0, "H (", H, ") must be a multiple of KVH (", KVH, ")");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous() && lse.dim() == 3 && lse.size(0) == B &&
                    lse.size(1) == H && lse.size(2) == L,
                "lse must be contiguous fp32 [", B, ", ", H, ", ", L, "], got ", lse.scalar_type(), " ", lse.sizes());
    const c10::cuda::CUDAGuard device_guard(q.device());
    auto dq = torch::empty({B, L, H, 128}, q.options());
    auto dk = torch::empty({B, L, KVH, 128}, q.options());
    auto dv = torch::empty({B, L, KVH, 128}, q.options());
    auto delta = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
    const auto str = [](const torch::Tensor& t) { return Str{t.stride(0), t.stride(1), t.stride(2)}; };
    attn_train_bwd_dispatch(q.data_ptr(), qs, k.data_ptr(), ks, v.data_ptr(), vs, o.data_ptr(), os, dout.data_ptr(), dos,
                            lse.data_ptr<float>(), delta.data_ptr<float>(), dq.data_ptr(), str(dq), dk.data_ptr(),
                            str(dk), dv.data_ptr(), str(dv), (int)B, (int)H, (int)KVH, (int)L, (float)scale, causal,
                            (int)stages, at::cuda::getCurrentCUDAStream().stream());
    return {dq, dk, dv};
}

// int4 block-16 training quantizers. IN: x fp16 [R, C], channels contiguous (rows
// may be strided). OUT: (packed int8, scale fp32) in gemm_int4(qblock=16)'s layout.
static void check_train_quant_input(const torch::Tensor& x, int64_t mult_r, int64_t mult_c) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.dim() == 2 && x.stride(1) == 1,
                "x must be a device fp16 [R, C] tensor with contiguous rows, got ", x.scalar_type(), " ", x.sizes(),
                " strides ", x.strides());
    TORCH_CHECK(x.size(0) % mult_r == 0 && x.size(1) % mult_c == 0, "x must be [R % ", mult_r, " == 0, C % ", mult_c,
                " == 0], got ", x.sizes());
    TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x rows must be 16-byte aligned, got row stride ", x.stride(0));
}

std::vector<torch::Tensor> quantize_int4_rows(torch::Tensor x, bool sr, int64_t seed) {
    check_train_quant_input(x, 1, 32);
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t R = x.size(0), C = x.size(1);
    auto packed = torch::empty({R, C / 2}, x.options().dtype(torch::kInt8));
    auto scale = torch::empty({R, C / 16}, x.options().dtype(torch::kFloat32));
    quant_rows_b16_dispatch(x.data_ptr(), x.stride(0), (int)R, (int)C, sr, (uint32_t)seed, packed.data_ptr(),
                            scale.data_ptr<float>(), at::cuda::getCurrentCUDAStream().stream());
    return {packed, scale};
}

std::vector<torch::Tensor> quantize_int4_cols(torch::Tensor x, c10::optional<torch::Tensor> signs, bool sr,
                                              int64_t seed) {
    check_train_quant_input(x, 32, 1);
    if (signs.has_value()) {
        TORCH_CHECK(signs->is_cuda() && signs->scalar_type() == torch::kFloat32 && signs->numel() == 16 &&
                        signs->is_contiguous(),
                    "signs must be a contiguous device fp32 tensor of 16, got ", signs->scalar_type(), " ",
                    signs->sizes());
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t R = x.size(0), C = x.size(1);
    auto packed = torch::empty({C, R / 2}, x.options().dtype(torch::kInt8));
    auto scale = torch::empty({C, R / 16}, x.options().dtype(torch::kFloat32));
    quant_cols_b16_dispatch(x.data_ptr(), x.stride(0), (int)R, (int)C,
                            signs.has_value() ? signs->data_ptr<float>() : nullptr, sr, (uint32_t)seed,
                            packed.data_ptr(), scale.data_ptr<float>(), at::cuda::getCurrentCUDAStream().stream());
    return {packed, scale};
}

int64_t gemm_int4_auto_splits(int64_t M, int64_t N, int64_t K) {
    int cus = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return auto_splits((int)M, (int)N, (int)(K / 32), cus);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4, "fused block-32 int4 activation quantize");
    m.def("gemm_int4", &gemm_int4, "gfx906 block-scaled int4 GEMM", py::arg("A_packed"), py::arg("A_scale"),
          py::arg("B_packed"), py::arg("B_scale"), py::arg("bias"), py::arg("out_dtype"), py::arg("splits") = 0,
          py::arg("qblock") = 32);
    m.def("gemm_int4_auto_splits", &gemm_int4_auto_splits, "split-K count gemm_int4 picks for [M,K] x [N,K]^T");
    m.def("quantize_int8", &quantize_int8, "block-32 int8 quantize (weights and activations)");
    m.def("gemm_int8", &gemm_int8, "gfx906 block-scaled int8 GEMM", py::arg("A"), py::arg("A_scale"), py::arg("B"),
          py::arg("B_scale"), py::arg("bias"), py::arg("out_dtype"), py::arg("config") = 0);
    m.def("conv_int8", &conv_int8, "gfx906 implicit-GEMM int8 conv, channels-last", py::arg("x"), py::arg("x_scale"),
          py::arg("W"), py::arg("W_scale"), py::arg("bias"), py::arg("out_dtype"), py::arg("B"), py::arg("H"),
          py::arg("W_in"), py::arg("kh"), py::arg("kw"), py::arg("sh"), py::arg("sw"), py::arg("ph"), py::arg("pw"),
          py::arg("config") = 1);
    m.def("attention", &attention, "gfx906 fused attention, head dim 128 (inputs from attention.py)", py::arg("Q"),
          py::arg("q_scale"), py::arg("K"), py::arg("k_scale"), py::arg("Vt"), py::arg("v_scale"), py::arg("qk_mode"),
          py::arg("pv_mode"), py::arg("skip_below"), py::arg("keep") = py::none());
    m.def("attention_block_rows", &attention_block_rows, "query rows per attention workgroup (= keep-mask block)");
    m.def("attention_max_tiles", &attention_max_tiles, "most key tiles a keep mask supports");
    m.def("quantize_int4_rows", &quantize_int4_rows, "int4 block-16 quantize along rows (training)", py::arg("x"),
          py::arg("sr"), py::arg("seed"));
    m.def("quantize_int4_cols", &quantize_int4_cols,
          "int4 block-16 quantize along columns, transposed, optional random Hadamard (training)", py::arg("x"),
          py::arg("signs"), py::arg("sr"), py::arg("seed"));
    m.def("attention_train_fwd", &attention_train_fwd, "gfx906 training attention forward -> (o, lse)", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("scale"));
    m.def("attention_train_bwd", &attention_train_bwd, "gfx906 training attention backward -> (dq, dk, dv)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"), py::arg("lse"), py::arg("dout"), py::arg("causal"),
          py::arg("scale"), py::arg("stages") = (int64_t)BWD_ALL);
}
