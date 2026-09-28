"""
Torch binding for the gfx906 native int4 (v_dot8_i32_i4) GEMM kernel.

Packing convention: native two's-complement nibbles (NOT the +8 offset
binary that ggml/quantize_model_int4.py use for the dequant-then-fp16-matmul
path) -- __builtin_amdgcn_sdot8 is a signed dot instruction and reinterprets
each nibble as signed 4-bit directly. See int4_gemm_scaled.hip's
correctness harness in the scratchpad for why this distinction matters.
"""
import os
import torch
from torch.utils.cpp_extension import load

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="gfx906_int4_ext",
            sources=[
                os.path.join(this_dir, "csrc", "binding.cpp"),
                os.path.join(this_dir, "csrc", "int4_gemm_scaled.hip"),
            ],
            extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
            verbose=False,
        )
    return _ext


_HADAMARD_CACHE = {}
_ROTATION_CACHE = {}
ROTATION_SEED = 0  # fixed so the weight-fold (transcode time) and the
                   # activation rotation (forward time) always agree


def _hadamard(n: int, device, dtype=torch.float32):
    key = (n, device, dtype)
    if key not in _HADAMARD_CACHE:
        assert n & (n - 1) == 0, f"Hadamard order {n} must be a power of two"
        h = torch.ones(1, 1, device=device, dtype=dtype)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        _HADAMARD_CACHE[key] = h / (n ** 0.5)
    return _HADAMARD_CACHE[key]


def choose_rotation_group(in_features: int, max_group: int = 128, min_group: int = 32) -> int | None:
    """Largest power-of-two group size, capped at max_group, that divides
    in_features -- the rotation is block-diagonal over these groups, so it
    never needs the token/resolution axis to divide evenly, only the
    per-layer channel count. Returns None if no group >= min_group fits,
    in which case that layer just skips rotation."""
    g = max_group
    while g >= min_group:
        if in_features % g == 0:
            return g
        g //= 2
    return None


def get_group_rotation(in_features: int, group: int, device) -> torch.Tensor:
    """Randomized Hadamard rotation R = H @ diag(+-1), block-diagonal over
    `group`-sized chunks of the channel axis. H is orthonormal and
    symmetric (Sylvester construction), and the random signs are what
    make the rotation spread outlier energy regardless of which channels
    happen to be large for a given input -- no calibration data needed,
    unlike the SmoothQuant channel scales tried earlier. Deterministic
    (fixed ROTATION_SEED) so weight-fold time and forward time agree
    without storing the matrix per-weight."""
    key = (in_features, group, device)
    if key not in _ROTATION_CACHE:
        g = torch.Generator(device="cpu").manual_seed(ROTATION_SEED ^ group)
        signs = (torch.randint(0, 2, (group,), generator=g).float() * 2 - 1).to(device)
        _ROTATION_CACHE[key] = _hadamard(group, device) * signs
    return _ROTATION_CACHE[key]


def apply_group_rotation(t: torch.Tensor, R: torch.Tensor, group: int) -> torch.Tensor:
    """t: [..., K], K % group == 0. Rotates every group-sized chunk of the
    last dim by R. The same call folds a weight (offline, once) or rotates
    an activation (forward, every call): for a row vector v, v @ R^T ==
    R @ v elementwise, so W_g @ R^T (weight fold) and R @ x_g (activation
    rotate) are both this one expression.

    R is shared across every group, so this is one plain matmul over all
    rows*groups flattened together, not a true batched matmul -- avoids a
    real segfault this ROCm PyTorch build's CPU BLAS has with 3D matmul
    (and, it turns out, with CPU matmul at all). A weight can legitimately
    still be CPU-resident here (lowvram/offload, not yet moved for
    inference), so route through CUDA rather than assume GPU residency."""
    orig_shape = t.shape
    K = orig_shape[-1]
    flat = t.reshape(-1, group)
    if flat.device.type == "cpu" and torch.cuda.is_available():
        rotated = (flat.cuda() @ R.t().cuda()).to("cpu")
    else:
        rotated = flat @ R.t()
    return rotated.reshape(orig_shape)


def quantize_pack_int4(t: torch.Tensor, block: int = 32):
    """t: [..., K] float. Returns (packed int8 [..., K/2], scale float32 [..., K/block])."""
    orig_shape = t.shape
    K = orig_shape[-1]
    assert K % block == 0, f"last dim {K} not divisible by block {block}"

    tb = t.reshape(-1, K // block, block).float()
    amax = tb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 7.0
    codes = torch.round(tb / scale).clamp(-8, 7).to(torch.int32).reshape(-1, K)

    lo = codes[:, 0::2] & 0xF
    hi = codes[:, 1::2] & 0xF
    packed = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)

    rows = orig_shape[:-1]
    packed = packed.reshape(*rows, K // 2).contiguous()
    scale = scale.reshape(*rows, K // block).contiguous()
    return packed, scale


def repack_q4_0_native(raw_bytes: torch.Tensor, out_features: int, in_features: int, block: int = 32):
    """Lossless repack of genuine ggml Q4_0 bytes into our native
    two's-complement, interleaved-nibble format. Q4_0 and our scheme share
    the same value range (-8..7, symmetric, no zero point) and block size
    (32) -- this is a pure bit-shuffle (nibble re-encoding + reordering),
    not a dequant-then-requantize round trip, so there's no additional
    quantization error versus the source file. Verified bit-exact against
    ggml's own dequantizer in test_q4_0_repack.py.

    raw_bytes: flat/reshapable uint8-viewable tensor, out_features *
    (in_features//block) * 18 bytes, row-major ggml Q4_0 blocks (2 bytes
    fp16 scale + 16 bytes packed nibbles, split-half order per block)."""
    TYPE_SIZE = 18
    n_blocks_per_row = in_features // block
    total_blocks = out_features * n_blocks_per_row
    blocks = raw_bytes.reshape(-1).view(torch.uint8).reshape(total_blocks, TYPE_SIZE)

    scale_f32 = blocks[:, :2].contiguous().view(torch.float16).reshape(total_blocks).float()

    qs = blocks[:, 2:].to(torch.int32)
    lo = qs & 0xF
    hi = (qs >> 4) & 0xF
    lo_native = (lo - 8) & 0xF  # exact 2's-complement re-encoding, no rounding
    hi_native = (hi - 8) & 0xF

    elems = torch.empty(total_blocks, block, dtype=torch.int32, device=raw_bytes.device)
    elems[:, 0 : block // 2] = lo_native   # ggml split-half: byte i low nibble = elem[i]
    elems[:, block // 2 :] = hi_native     # byte i high nibble = elem[i + block/2]

    e_lo = elems[:, 0::2]  # our interleaved order: byte j low = elem[2j]
    e_hi = elems[:, 1::2]  # byte j high = elem[2j+1]
    packed = (e_lo | (e_hi << 4)).to(torch.uint8).view(torch.int8)

    packed = packed.reshape(out_features, in_features // 2).contiguous()
    scale = scale_f32.reshape(out_features, n_blocks_per_row).contiguous()
    return packed, scale


def int4_linear(x: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, block: int = 32, rotation_group: int = 0) -> torch.Tensor:
    """x: [..., K] float. w_packed/w_scale: from quantize_pack_int4(weight), already
    folded through the same rotation via rotation_group (0 = none). Returns [..., N] float32."""
    ext = _get_ext()
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K)

    if rotation_group:
        R = get_group_rotation(K, rotation_group, x2d.device)
        x2d = apply_group_rotation(x2d, R, rotation_group)

    a_packed, a_scale = quantize_pack_int4(x2d, block)
    out = ext.int4_gemm_scaled(a_packed, a_scale, w_packed, w_scale, K)
    return out.reshape(*orig_shape[:-1], w_packed.shape[0])


# ---------------------------------------------------------------------------
# custom_operations integration: replaces F.linear with the gfx906 int4
# kernel from model-construction time, so weights load directly into packed
# int4 buffers and never get materialized as full-precision tensors -- the
# real VRAM win the earlier Gfx906Int4Patch MODEL-patch node couldn't give.
# Only Linear is overridden; every other op (Conv2d/RMSNorm/Embedding/...)
# falls through to comfy's normal disable_weight_init behavior unchanged.
# ---------------------------------------------------------------------------
import comfy.ops

BLOCK = 32


class Gfx906Int4Ops(comfy.ops.disable_weight_init):
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, rotation_group=0):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rotation_group = rotation_group  # 0 = none; else the group size the weight was folded with
            # Some layers in a real model (e.g. Krea2's txtfusion.projector,
            # in_features=12) are far too narrow for a 32-wide quant block.
            # Those fall back to a plain float Linear instead of asserting --
            # the same shape falls through unquantized in _transcode_linears'
            # skip check; custom_operations construction has no such escape
            # hatch, since comfy calls operations.Linear() uniformly.
            self.quantized = in_features % BLOCK == 0 and in_features >= BLOCK
            if self.quantized:
                self.register_buffer("weight_q4", torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device))
                self.register_buffer("weight_scale", torch.empty(out_features, in_features // BLOCK, dtype=torch.float32, device=device))
            else:
                self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)

        def forward(self, x):
            if not self.quantized:
                bias = self.bias.to(x.device, x.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(x, self.weight.to(x.device, x.dtype), bias)
            # Comfy's lowvram path can leave weights CPU-resident and cast them
            # on demand per forward call -- this class bypasses comfy.ops'
            # cast_bias_weight machinery (it isn't a CastWeightBiasOp), so it
            # has to do that same per-call cast itself, or the kernel gets
            # handed CPU tensors and RuntimeErrors on "inputs must be on device".
            weight_q4 = self.weight_q4.to(x.device)
            weight_scale = self.weight_scale.to(x.device)
            out = int4_linear(x, weight_q4, weight_scale, BLOCK, self.rotation_group)
            out = out.to(x.dtype)
            if self.bias is not None:
                out = out + self.bias.to(x.device, x.dtype)
            return out
