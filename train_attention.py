"""Trainable attention on gfx906 (csrc/attention_train.hip; design: DESIGN.md).

attention(q, k, v, causal=True, scale=None) with q [B, L, H, 128] and k, v
[B, L, KVH, 128] fp16 -- token-major, so the views of a fused qkv projection go
in without copies. Returns [B, L, H, 128]. Gradients flow to q, k, v.
"""
import torch

from . import ops


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        o, lse = ops._get_ext().attention_train_fwd(q, k, v, causal, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx, dout):
        q, k, v, o, lse = ctx.saved_tensors
        # the kernels need 16-byte rows; an expanded or transposed grad is not
        dq, dk, dv = ops._get_ext().attention_train_bwd(q, k, v, o, lse, dout.contiguous(), ctx.causal, ctx.scale)
        return dq, dk, dv, None, None


def attention(q, k, v, causal=True, scale=None):
    return _Attention.apply(q, k, v, causal, q.shape[-1] ** -0.5 if scale is None else scale)
