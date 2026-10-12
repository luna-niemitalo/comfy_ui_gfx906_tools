"""int8 convolutions for image VAEs on gfx906 (csrc/int8_gemm.hip).

A conv as a GEMM: M = output pixels, K = (tap, in channel), N = out channels. x is
quantized once per (pixel, 32 channels), channels-last; the kernel reads each tap's pixel
straight from it (implicit GEMM, conv_int8), so every 32-element K block is one pixel's 32
channels -- exactly the kernel's per-block scheme. Weights: per (out channel, tap, 32 in
channels), quantized once and cached. Taps outside the image read as zero.

Quality (Qwen-Image VAE, W8A8, 2560x1440 render): decode 58.3 dB, encode 59.0 dB vs fp32,
latent error 0.6% (develop/experiments/2026-09-29/vae_int_study.py). W4A4 was 37-40 dB,
14% latent error -- not used.

Outputs stay channels-last (an NCHW view of [pixels, channels] storage): norms, SiLU and
adds keep that layout, and the next conv quantizes it without a transpose.
"""
import copy
import logging

import torch

from . import ops

BLOCK = 32
# ComfyUI frees room for vae.memory_used_* before each call. Measured int8 / fp16 working memory
# (peak allocated above the loaded weights, Qwen-Image VAE, 2560x1440, vae_mem_bench.py):
# tiled decode 0.54, plain decode 0.51, tiled encode 0.39, plain encode 0.81. Rounded up.
DECODE_MEMORY_RATIO = 0.6
ENCODE_MEMORY_RATIO = 0.85
# 128x64 tiles, 8x4 per thread: best of the four configs at every VAE conv shape
# (develop/harness/bench_int8.py, 2026-09-29).
GEMM_CONFIG = 1


def eligible(weight, groups):
    """Convs the int8 path takes: 2-D kernels (or 3-D ones used on one frame), in channels
    a multiple of 32, at least 32 out channels (the 96->3 image conv stays fp32)."""
    return groups == 1 and weight.shape[1] % BLOCK == 0 and weight.shape[0] >= BLOCK


def quantize_weight(weight):
    """[Cout, Cin, kh, kw] -> (int8 [Cout, kh*kw*Cin], scale [Cout, kh*kw*Cin/32]), tap-major K."""
    w = weight.float().permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()
    return ops._get_ext().quantize_int8(w)


def conv2d(x, wq, ws, kernel, bias, stride, padding):
    """x [B, C, H, W] (any memory format) -> [B, Cout, Ho, Wo] channels-last, x.dtype."""
    B, C, H, W = x.shape
    ext = ops._get_ext()
    q, s = ext.quantize_int8(x.permute(0, 2, 3, 1).reshape(-1, C))
    out = ext.conv_int8(q, s, wq, ws, None if bias is None else bias.to(x.dtype), x.dtype, B, H, W, *kernel, *stride,
                        *padding, GEMM_CONFIG)
    Ho, Wo = (H + 2 * padding[0] - kernel[0]) // stride[0] + 1, (W + 2 * padding[1] - kernel[1]) // stride[1] + 1
    return out.view(B, Ho, Wo, -1).permute(0, 3, 1, 2)


def _int8_forward(module):
    """Replacement for module._conv_forward: int8 when the conv is 2-D in effect, else the
    module's own forward. Weights are quantized on first use per (device, tensor) -- the
    patcher may move or recast them."""
    cache = {}
    fallback = type(module)._conv_forward

    def forward(input, weight, bias, *args, **kwargs):
        if weight.dim() == 5:
            autopad = args[0] if args else kwargs.get("autopad")
            # one frame through a causal kernel (Wan / Qwen-Image image VAE) = its last slice
            single = input.shape[2] == 1 and module.padding[0] == 0 and module.stride[0] == 1 and (
                autopad == "causal_zero" or weight.shape[2] == 1)
            if not single:
                return fallback(module, input, weight, bias, *args, **kwargs)
            x, w = input[:, :, 0], weight[:, :, -1]
            stride, padding = module.stride[1:], module.padding[1:]
        else:
            x, w = input, weight
            stride, padding = module.stride, module.padding
        # (inference tensors have no version counter; VAE weights are never edited in place)
        key = (w.device, w.data_ptr(), w.dtype)
        if cache.get("key") != key:
            cache["key"], cache["w"] = key, quantize_weight(w)
        out = conv2d(x, *cache["w"], tuple(w.shape[2:]), bias, stride, padding)
        return out.unsqueeze(2) if weight.dim() == 5 else out
    return forward


def patch_vae(vae):
    """VAE -> a VAE whose eligible convs run int8 (object patches on a patcher clone: the
    input VAE is untouched). Returns (vae, number of convs patched)."""
    patched = copy.copy(vae)
    patched.patcher = vae.patcher.clone()
    patched.memory_used_decode = lambda shape, dtype: vae.memory_used_decode(shape, dtype) * DECODE_MEMORY_RATIO
    patched.memory_used_encode = lambda shape, dtype: vae.memory_used_encode(shape, dtype) * ENCODE_MEMORY_RATIO
    count = 0
    for name, m in vae.first_stage_model.named_modules():
        if not isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)) or not eligible(m.weight, m.groups):
            continue
        if any(d != 1 for d in m.dilation) or m.padding_mode != "zeros":
            continue
        patched.patcher.add_object_patch(f"{name}._conv_forward", _int8_forward(m))
        count += 1
    return patched, count


class Gfx906Int8VAE:
    """VAE -> VAE with int8 convolutions (W8A8, 32-channel block scales) on gfx906: a VRAM
    option. Qwen-Image VAE (Krea2) on fp16, 2560x1440 tiled 512: reserved 1.19 vs 1.82 GB
    decode, 0.71 vs 0.92 GB encode; ~same speed as fp16 (2.47 vs 2.29 s decode); 58 dB vs
    fp32 (fp16 alone: 66 dB). CLAUDE.md "VAE int path"."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Int8 VAE"

    def patch(self, vae):
        patched, count = patch_vae(vae)
        if count == 0:
            raise ValueError("Gfx906 Int8 VAE: expected convs with in channels a multiple of 32 and >= 32 out "
                             f"channels, found none in {type(vae.first_stage_model).__name__}")
        logging.info(f"[Gfx906 Int8 VAE] {count} convs int8 in {type(vae.first_stage_model).__name__}")
        return (patched,)
