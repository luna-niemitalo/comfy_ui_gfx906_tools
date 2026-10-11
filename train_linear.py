"""Linear layer with an int4 backward for training on gfx906.

Forward: fp16 (exact; NVIDIA's NVFP4 study found the forward quantization is
where most of the 4-bit loss gap comes from). Backward, both GEMMs on int4
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


class _Int4BwdLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return F.linear(x, w)

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
        return dx, dw


def linear(x, w):
    """F.linear(x, w) with the int4 backward. Tokens (x's leading dims) must be a multiple of 32."""
    return _Int4BwdLinear.apply(x, w)
