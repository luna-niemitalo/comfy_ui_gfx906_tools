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
import torch

from . import block_sparse, ops

HEAD_DIM = 128
TILE = 64
QK_MODES = {"fp16": 0, "int8": 1, "int4": 2}
PV_MODES = {"fp16": 0, "u8": 1}
_LOG2E = 1.4426950408889634
# Below this many tokens the per-call prep costs more than the kernel saves
# (2026-09-29, 600 text tokens: 128² = 664 tokens 565 vs 542 ms/step with
# ComfyUI's quad attention; 256² = 856 tokens already 742 vs 764).
MIN_FUSED_TOKENS = 768


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


def fused_attention(q, k, v, qk="int8", pv="u8", skip_below=0.0, keep=None):
    """q [B, H, Lq, 128] post-RoPE, k/v [B, KVH, Lk, 128] (not GQA-expanded;
    Lq != Lk for the two-GPU row split). keep: optional block-sparse mask,
    uint8 [B, H, ceil(Lq / block_rows(qk)), ceil(Lk / 64)]. Returns [B, Lq, H * 128] fp16."""
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {q.shape[-1]}")
    if qk not in QK_MODES or pv not in PV_MODES:
        raise ValueError(f"qk in {list(QK_MODES)}, pv in {list(PV_MODES)}; got qk={qk!r} pv={pv!r}")
    qb, qs, kb, ks = _prepare_qk(q, k, qk)
    vt, vs = _prepare_v(v, pv)
    return ops._get_ext().attention(qb, qs, kb, ks, vt, vs, QK_MODES[qk], PV_MODES[pv], float(skip_below), keep)


def block_rows(qk):
    """Query rows per kernel workgroup = rows per keep-mask block."""
    return ops._get_ext().attention_block_rows(QK_MODES[qk])


def _gqa_repeat(diffusion_model):
    """Query heads per KV head. Krea2 repeat_interleaves K/V up to the query
    head count before calling attention; the override undoes that with a
    stride so the kernel reads each KV head once."""
    attn = next((m for m in diffusion_model.modules() if hasattr(m, "kvheads") and hasattr(m, "heads")), None)
    return attn.heads // attn.kvheads if attn is not None else 1


def _make_override(qk, pv, skip_below, gqa_repeat, sparse_tau, max_missed):
    sparsity = block_sparse.SparsityState(sparse_tau, max_missed, block_rows(qk)) if sparse_tau > 0 else None
    max_tiles = ops._get_ext().attention_max_tiles()

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        # The kernel's contract: [B, H, L, 128] inputs, no mask, no custom
        # scale, [B, L, H*128] output. Anything else (e.g. masked text-encoder
        # attention) and short sequences go to ComfyUI's own backend.
        handled = (mask is None and skip_reshape and not skip_output_reshape and "scale" not in kwargs
                   and q.dim() == 4 and q.shape[1] == heads and q.shape[-1] == HEAD_DIM
                   and k.shape[2] >= MIN_FUSED_TOKENS)
        if not handled:
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)
        if gqa_repeat > 1 and k.shape[1] == q.shape[1]:
            k, v = k[:, ::gqa_repeat], v[:, ::gqa_repeat]
        keep = None
        if sparsity is not None and -(-k.shape[2] // TILE) <= max_tiles:
            keep = sparsity.keep(q, k, kwargs.get("transformer_options") or {})
        return fused_attention(q, k, v, qk, pv, skip_below, keep).to(q.dtype)
    return override


class Gfx906FusedAttention:
    """MODEL -> MODEL. Routes the model's attention through the fused gfx906
    kernel (head dim 128, unmasked); other attention calls fall through to
    ComfyUI's backend. Accuracy per mode: CLAUDE.md "Attention exploration"
    (renders indistinguishable at int4/u8 in Luna's check)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "qk": (list(QK_MODES), {"default": "int4", "tooltip": "Q.K precision. int4 (Hadamard-rotated) is fastest; fp16 is the exact reference."}),
                "pv": (list(PV_MODES), {"default": "u8", "tooltip": "softmax.V precision. u8 uses a per-64-key-tile scale."}),
                "skip_below": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 0.1, "step": 1e-5, "tooltip": "Skip a key tile's P.V when its largest attention weight is below this fraction of every row's peak (0 = never)."}),
                "sparse_tau": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.999, "step": 0.001, "tooltip": "Block-sparse attention: skip whole (query block, key tile) pairs outside the predicted top tau of attention mass (0 = off; 0.995 tested). The first step of a run is dense and decides per layer and head where the prediction is safe."}),
                "max_missed": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "Block-sparse calibration: a head runs sparse only if the prediction drops at most this fraction of its exact attention mass on the first step."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Fused Attention"

    def patch(self, model, qk, pv, skip_below, sparse_tau, max_missed):
        gqa_repeat = _gqa_repeat(model.model.diffusion_model)
        model = model.clone()
        model.model_options["transformer_options"]["optimized_attention_override"] = _make_override(
            qk, pv, skip_below, gqa_repeat, sparse_tau, max_missed)
        return (model,)
