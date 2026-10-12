"""Linear layer with an int4 backward for training on gfx906.

Forward: fp16 (exact; NVIDIA's NVFP4 study found the forward quantization is
where most of the 4-bit loss gap comes from); int4_fprop=True quantizes it too,
for the A/B in develop/experiments/2026-10-11/train_gate.py. Backward, both GEMMs on int4
block-16 operands through the v_dot8 kernel (int4_gemm_scaled.hip, qblock 16):
  dgrad  dX = dY W:    dY blocked along N with stochastic rounding, W^T along N
  wgrad  dW = dY^T X:  both blocked along tokens after a 16-point random Hadamard
                       (one fixed sign vector), dY with stochastic rounding
Numerics = develop/experiments/2026-10-11/train_gate.py's int4_b16_sr arm (emulated);
int4_train_test.py checks this against it.
"""
import torch
import torch.nn.functional as F

from . import ops

_SIGNS = {}
_seed = [0]


def _signs(device):
    if device not in _SIGNS:
        g = torch.Generator().manual_seed(1234)
        _SIGNS[device] = torch.where(torch.rand(16, generator=g) < 0.5, -1.0, 1.0).to(device)
    return _SIGNS[device]


def _next_seed():
    _seed[0] = (_seed[0] + 1) & 0x7FFFFFFF
    return _seed[0]


def _fprop_packed(x, wq, wsc):
    """y = x W^T, x quantized here and W packed: block-16 round to nearest along K"""
    ext = ops._get_ext()
    xq, xsc = ext.quantize_int4_rows(x.reshape(-1, x.shape[-1]).contiguous(), False, 0)
    return ext.gemm_int4(xq, xsc, wq, wsc, None, x.dtype, 0, 16).view(*x.shape[:-1], -1)


def _int4_fprop(x, w):
    return _fprop_packed(x, *ops._get_ext().quantize_int4_rows(w.contiguous(), False, 0))


class _Int4BwdLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, int4_fprop):
        ctx.save_for_backward(x, w)
        return _int4_fprop(x, w) if int4_fprop else F.linear(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        ext = ops._get_ext()
        g = dy.reshape(-1, dy.shape[-1]).contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            gq, gs = ext.quantize_int4_rows(g, True, _next_seed())
            wq, ws = ext.quantize_int4_cols(w, None, False, 0)  # W [N, K] -> W^T blocked along N
            dx = ext.gemm_int4(gq, gs, wq, ws, None, dy.dtype, 0, 16).view(*dy.shape[:-1], -1)
        if ctx.needs_input_grad[1]:
            signs = _signs(dy.device)
            xs = x.reshape(-1, x.shape[-1])
            gtq, gts = ext.quantize_int4_cols(g, signs, True, _next_seed())
            xtq, xts = ext.quantize_int4_cols(xs if xs.stride(-1) == 1 else xs.contiguous(), signs, False, 0)
            dw = ext.gemm_int4(gtq, gts, xtq, xts, None, torch.float32, 0, 16).to(w.dtype)
        return dx, dw, None


def linear(x, w, int4_fprop=False):
    """F.linear(x, w) with the int4 backward; int4_fprop also quantizes the forward
    (x and W block-16 along K, round to nearest). Tokens (x's leading dims) must be a
    multiple of 32."""
    return _Int4BwdLinear.apply(x, w, int4_fprop)


class _FrozenInt4(torch.autograd.Function):
    """Saves nothing per call: no wgrad, so x isn't needed; dgrad uses the stored W^T."""

    @staticmethod
    def forward(ctx, x, wq, wsc, wtq, wtsc):
        ctx.wt = (wtq, wtsc)
        return _fprop_packed(x, wq, wsc)

    @staticmethod
    def backward(ctx, dy):
        ext = ops._get_ext()
        gq, gs = ext.quantize_int4_rows(dy.reshape(-1, dy.shape[-1]).contiguous(), True, _next_seed())
        return ext.gemm_int4(gq, gs, *ctx.wt, None, dy.dtype, 0, 16).view(*dy.shape[:-1], -1), None, None, None, None


class FrozenInt4Linear(torch.nn.Module):
    """A frozen bias-free linear stored only as int4: W blocked along K (forward) and
    W^T blocked along N (dgrad), both quantized once from fp16 -- 1.5 B/param (0.5 codes
    + 0.25 fp32 scales, per copy) instead of 2, exact vs requantizing per call (bit-identical to linear(x, w, int4_fprop=True)
    with w frozen; develop/experiments/2026-10-11/frozen_int4_test.py)."""

    def __init__(self, linear):
        super().__init__()
        # IN, critical: the packed layouts need fp16, contiguous, K and N multiples of 32
        w = linear.weight
        if linear.bias is not None or w.dtype != torch.float16 or w.shape[0] % 32 or w.shape[1] % 32:
            raise ValueError(f"FrozenInt4Linear: expected a bias-free fp16 weight with both dims % 32 == 0, got "
                             f"bias={linear.bias is not None} dtype={w.dtype} shape={tuple(w.shape)}")
        ext = ops._get_ext()
        w = w.detach().contiguous()
        for name, t in zip(("wq", "wsc", "wtq", "wtsc"),
                           (*ext.quantize_int4_rows(w, False, 0), *ext.quantize_int4_cols(w, None, False, 0))):
            self.register_buffer(name, t)
        self.in_features, self.out_features = w.shape[1], w.shape[0]

    def forward(self, x):
        return _FrozenInt4.apply(x, self.wq, self.wsc, self.wtq, self.wtsc)
