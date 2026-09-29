"""Fused gfx906 attention (csrc/attention.hip): input preparation + entry point.

Preparation is O(L * D) torch work around the O(L^2) kernel:
- softmax scale * log2(e) folded into Q (fp16) or the Q scale (int), so the
  kernel's softmax runs on exp2
- int modes: K mean over tokens subtracted per KV head (exact -- shifts every
  score of a query row by one constant), then per-token symmetric quantization
- int4: head dim Hadamard-rotated first (q.k unchanged, outliers spread out)
- V split into 64-key tiles and transposed to [tile, channel, key] so the P.V
  loop reads 16 consecutive keys of one channel per load; u8 mode quantizes
  per (tile, channel) and stores +128 (the kernel corrects for the offset)
Accuracy of each mode: CLAUDE.md "Attention exploration".
"""
import math

import torch

from . import ops

HEAD_DIM = 128
TILE = 64
QK_MODES = {"fp16": 0, "int8": 1, "int4": 2}
PV_MODES = {"fp16": 0, "u8": 1}
_LOG2E = 1.4426950408889634


def _quant_tokens(x, bits):
    """Per-token (last dim) symmetric quantization -> (int8 values, fp32 scale)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1).clamp_min(1e-8) / qmax
    return (x / scale[..., None]).round().clamp(-qmax - 1, qmax).to(torch.int8), scale


def _pack_int4(x):
    """int8 values in [-8, 7], last dim even -> uint8 with element 2i in the low nibble."""
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).to(torch.int16)
    return ((pairs[..., 0] & 0xF) | ((pairs[..., 1] & 0xF) << 4)).to(torch.uint8)


def _prepare_qk(q, k, mode):
    scale = HEAD_DIM ** -0.5 * _LOG2E
    empty = torch.empty(0, device=q.device)
    if mode == "fp16":
        return (q.float() * scale).half().view(torch.uint8), empty, k.half().contiguous().view(torch.uint8), empty
    qf, kf = q.float(), k.float()
    kf = kf - kf.mean(dim=2, keepdim=True)
    if mode == "int4":
        rot = ops._hadamard(HEAD_DIM, q.device)
        qf, kf = qf @ rot, kf @ rot
    bits = 8 if mode == "int8" else 4
    qi, qs = _quant_tokens(qf, bits)
    ki, ks = _quant_tokens(kf, bits)
    if mode == "int4":
        return _pack_int4(qi), qs * scale, _pack_int4(ki), ks
    return qi.view(torch.uint8), qs * scale, ki.view(torch.uint8), ks


def _prepare_v(v, mode):
    B, KVH, L, D = v.shape
    n_tiles = -(-L // TILE)
    tiles = torch.nn.functional.pad(v, (0, 0, 0, n_tiles * TILE - L)).reshape(B, KVH, n_tiles, TILE, D)
    tiles = tiles.transpose(-1, -2)  # [B, KVH, n_tiles, D, TILE]
    if mode == "fp16":
        return tiles.half().contiguous().view(torch.uint8), torch.empty(0, device=v.device)
    tf = tiles.float()
    scale = tf.abs().amax(-1).clamp_min(1e-8) / 127
    vq = (tf / scale[..., None]).round().clamp(-127, 127) + 128
    return vq.to(torch.uint8).contiguous(), scale.contiguous()


def fused_attention(q, k, v, qk="int8", pv="u8", skip_below=0.0):
    """q [B, H, L, 128] post-RoPE, k/v [B, KVH, L, 128] (not GQA-expanded).
    Returns [B, L, H * 128] fp16."""
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {q.shape[-1]}")
    if qk not in QK_MODES or pv not in PV_MODES:
        raise ValueError(f"qk in {list(QK_MODES)}, pv in {list(PV_MODES)}; got qk={qk!r} pv={pv!r}")
    qb, qs, kb, ks = _prepare_qk(q, k, qk)
    vt, vs = _prepare_v(v, pv)
    return ops._get_ext().attention(qb, qs, kb, ks, vt, vs, QK_MODES[qk], PV_MODES[pv], float(skip_below))
