#!/usr/bin/env python3
"""Which 4-bit (or lower) training format can gfx906 use? Emulated GEMM error on
real Qwen3-4B tensors, per training GEMM (fprop Y = X W^T, dgrad dX = dY W,
wgrad dW = dY^T X), per format. NVFP4 with NVIDIA's recipe (arXiv 2509.25149:
1x16 E2M1 blocks with E4M3 scales, 16x16 weight blocks, stochastic rounding on
gradients, 16-point random Hadamard on wgrad inputs) is the anchor: it trained a
12B model within ~1-1.5% loss of FP8. gfx906 has no FP4; it has int4 dot8
(v_dot8_i32_i4, ~96 TOPS) and int8 dot4 (~48).

Tensors: one fp16 forward + backward of Qwen3-4B (/media/luna/models/text_encoders/
qwen_3_4b.safetensors, read-only) on 1024 tokens of ComfyUI's README, capturing X,
W, dY of every linear in layers 0, 1, 17, 34, 35. Errors are rel. Frobenius vs
fp32 GEMMs of the same fp16 tensors. Stochastic rounding errors are one sample
(its value is unbiasedness, which a single error figure does not show).

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 1200 -- \
        direnv exec . python develop/experiments/2026-10-11/quant_study.py
"""
import json
import math
import os
import struct

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CKPT = "/media/luna/models/text_encoders/qwen_3_4b.safetensors"
TOKENIZER = os.path.join(REPO, "comfy", "text_encoders", "qwen25_tokenizer")
TEXT = os.path.join(REPO, "README.md")
LAYERS = (0, 1, 17, 34, 35)
TOKENS = 1024
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---- model + capture ----

def load_model(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from safetensors import safe_open
    config = Qwen3Config(vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                         num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=1000000.0,
                         rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=40960)
    # built on the GPU in fp16: on meta, non-checkpoint buffers (rotary frequencies)
    # stay empty; on the host, fp32 init would take 16 GB of RAM
    torch.set_default_dtype(torch.float16)
    with torch.device(device):
        model = Qwen3ForCausalLM(config)
    torch.set_default_dtype(torch.float32)
    params = model.state_dict()
    # streamed one tensor at a time: a whole-file load plus its fp16 copy peaked at
    # ~16 GB of host RAM (kswapd Oops on record, CLAUDE.md Hazards)
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        # IN, critical: a key set that doesn't match means a different checkpoint layout
        expected = {k for k in params if k != "lm_head.weight"}
        if keys != expected:
            raise SystemExit(f"quant_study: tried to load {CKPT} -- expected keys like {sorted(expected)[:3]} "
                             f"({len(expected)}) -- actual: missing {sorted(expected - keys)[:5]}, "
                             f"extra {sorted(keys - expected)[:5]}")
        with torch.no_grad():
            for k in keys:
                params[k].copy_(f.get_tensor(k))
    model.tie_weights()
    # 8 GB of weights + 36 layers of eager-attention activations did not fit 16 GB.
    # Checkpointing needs train mode in HF; Qwen3 has no dropout, so outputs are
    # unchanged (capture hooks just fire again on recompute, with the same values).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


def capture(model, ids):
    records = {}
    hooks = []
    for li in LAYERS:
        layer = model.model.layers[li]
        for name, mod in [*layer.self_attn.named_children(), *layer.mlp.named_children()]:
            if not isinstance(mod, torch.nn.Linear):
                continue
            key = f"L{li}.{name}"
            records[key] = {"W": mod.weight.detach()}
            hooks.append(mod.register_forward_hook(lambda m, i, o, key=key: records[key].__setitem__("X", i[0].detach()[0])))
            hooks.append(mod.register_full_backward_hook(
                lambda m, gi, go, key=key: records[key].__setitem__("dY", go[0].detach()[0])))
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.model.embed_tokens(ids).detach().requires_grad_()
    logits = model(inputs_embeds=emb).logits[0, :-1].float()
    # summed, not mean: per-token gradient magnitudes stay clear of fp16 underflow
    loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum")
    loss.backward()
    for h in hooks:
        h.remove()
    return records, (loss / (ids.shape[1] - 1)).item()


# ---- quantizers (fp32 emulation; return dequantized values) ----

def blocks_last(x, block):
    """x [..., n] -> [..., n / block, block]"""
    return x.reshape(*x.shape[:-1], x.shape[-1] // block, block)


def round_(v, sr):
    return torch.floor(v + torch.rand_like(v)) if sr else torch.round(v)


def q_int(x, bits, block, sr=False):
    """symmetric per-block int (ours: amax / qmax scale, codes clamped to [-qmax - 1, qmax])"""
    qmax = 2 ** (bits - 1) - 1
    b = blocks_last(x, block)
    scale = b.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (round_(b / scale, sr).clamp(-qmax - 1, qmax) * scale).reshape(x.shape)


def to_e2m1(v, sr):
    """values already scaled into [-6, 6] -> nearest (or stochastic) E2M1 grid point"""
    grid = E2M1.to(v.device)
    a = v.abs().clamp(max=6.0)
    hi_i = torch.searchsorted(grid, a.contiguous()).clamp(1, len(grid) - 1)
    lo, hi = grid[hi_i - 1], grid[hi_i]
    frac = (a - lo) / (hi - lo)
    up = (torch.rand_like(a) < frac) if sr else (frac >= 0.5)
    return torch.sign(v) * torch.where(up, hi, lo)


def q_nvfp4(x, sr=False, block=16):
    """1x16 E2M1 blocks, E4M3 block scales, fp32 tensor scale (NVFP4)"""
    t_scale = x.abs().amax().clamp_min(1e-12) / (6.0 * 448.0)
    b = blocks_last(x, block)
    bs = (b.abs().amax(-1, keepdim=True) / 6.0 / t_scale).to(torch.float8_e4m3fn).float().clamp_min(2**-9)
    scale = bs * t_scale
    return (to_e2m1(b / scale, sr) * scale).reshape(x.shape)


def q_mxfp4(x, sr=False):
    """1x32 E2M1 blocks, power-of-two (E8M0) scales: 2^(floor(log2 amax) - 2), OCP MX"""
    b = blocks_last(x, 32)
    scale = torch.exp2(torch.floor(torch.log2(b.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - 2)
    return (to_e2m1(b / scale, sr) * scale).reshape(x.shape)


def two_d(q_tile, w, block=16):
    """2D blocks: one scale per 16x16 weight tile, so W and W^T quantize to the same
    values. q_tile quantizes rows of 256 (a flattened tile) with one scale each."""
    n, k = w.shape
    tiles = w.reshape(n // block, block, k // block, block).permute(0, 2, 1, 3).reshape(-1, block * block)
    out = q_tile(tiles)
    return out.reshape(n // block, k // block, block, block).permute(0, 2, 1, 3).reshape(n, k)


def rht16(x, signs):
    """random Hadamard along dim 0 in blocks of 16 (orthonormal: preserves dY^T X)"""
    h = torch.tensor([[1.0]], device=x.device)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / 4.0 * signs[None, :]
    m = x.shape[0]
    return torch.einsum("ij,bjk->bik", h, x.reshape(m // 16, 16, -1)).reshape(m, -1)


def ternary(w):
    """BitNet b1.58 absmean ternary weights"""
    s = w.abs().mean().clamp_min(1e-12)
    return (w / s).round().clamp(-1, 1) * s


# ---- formats: quantizer per role; the GEMM always contracts the last dim ----

def fmt_int(bits, block):
    q = lambda t, sr=False: q_int(t, bits, block, sr)
    return {"x": q, "w": q, "g": q, "wgrad_rht": False, "grad_sr": False, "w2d": None}


NVFP4 = {"x": q_nvfp4, "w": q_nvfp4, "g": q_nvfp4, "wgrad_rht": False, "grad_sr": False, "w2d": None}
RECIPE = {"wgrad_rht": True, "grad_sr": True}
FORMATS = {
    "int8_b32 (ours)": fmt_int(8, 32),
    "int4_b32 (ours)": fmt_int(4, 32),
    "int4_b16": fmt_int(4, 16),
    "int4_b16 + 2D W": {**fmt_int(4, 16), "w2d": lambda t: q_int(t, 4, 256)},
    "int4_b16 + recipe": {**fmt_int(4, 16), **RECIPE, "w2d": lambda t: q_int(t, 4, 256)},
    "mxfp4": {"x": q_mxfp4, "w": q_mxfp4, "g": q_mxfp4, "wgrad_rht": False, "grad_sr": False, "w2d": None},
    "nvfp4": NVFP4,
    "nvfp4 + 2D W": {**NVFP4, "w2d": lambda t: q_nvfp4(t, block=256)},
    "nvfp4 + recipe": {**NVFP4, **RECIPE, "w2d": lambda t: q_nvfp4(t, block=256)},
}


def gemm_errors(f, X, W, dY, signs):
    X, W, dY = X.float(), W.float(), dY.float()
    exact = {"fprop": X @ W.t(), "dgrad": dY @ W, "wgrad": dY.t() @ X}
    if f["w2d"]:
        wq = two_d(f["w2d"], W)
        w_fprop, w_dgrad = wq, wq  # one quantized W serves both directions
    else:
        w_fprop = f["w"](W)  # blocks along K
        w_dgrad = f["w"](W.t().contiguous()).t()  # blocks along N
    sr = f["grad_sr"]
    got = {
        "fprop": f["x"](X) @ w_fprop.t(),
        "dgrad": f["g"](dY, sr=sr) @ w_dgrad,
    }
    xw, gw = (rht16(X, signs), rht16(dY, signs)) if f["wgrad_rht"] else (X, dY)
    xq_w = f["x"](xw.t().contiguous())
    got["wgrad"] = f["g"](gw.t().contiguous(), sr=sr) @ xq_w.t()
    err = {k: ((got[k] - exact[k]).norm() / exact[k].norm()).item() for k in exact}
    if sr:
        # what accumulation sees: the mean of 8 rounding samples (RTN's bias would stay)
        mean8 = sum(f["g"](gw.t().contiguous(), sr=True) @ xq_w.t() for _ in range(8)) / 8
        err["wgrad_mean8"] = ((mean8 - exact["wgrad"]).norm() / exact["wgrad"].norm()).item()
    return err


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ids = tok(open(TEXT).read(), return_tensors="pt").input_ids[:, :TOKENS].to(device)
    if ids.shape[1] < TOKENS:
        raise SystemExit(f"quant_study: tried to tokenize {TEXT} -- expected >= {TOKENS} tokens -- actual {ids.shape[1]}")
    model = load_model(device)
    records, loss = capture(model, ids)
    finite = all(torch.isfinite(r[t]).all() for r in records.values() for t in ("X", "W", "dY"))
    print(json.dumps({"loss_per_token": round(loss, 4), "linears": len(records), "all_finite": finite}), flush=True)
    if not finite:
        raise SystemExit("quant_study: fp16 forward/backward produced non-finite tensors -- results would be meaningless")
    del model
    torch.cuda.empty_cache()

    signs = torch.where(torch.rand(16, device=device) < 0.5, -1.0, 1.0)  # one fixed sign vector, as in the paper
    totals = {}
    for key, r in records.items():
        row = {}
        for name, f in FORMATS.items():
            row[name] = {k: round(v, 4) for k, v in gemm_errors(f, r["X"], r["W"], r["dY"], signs).items()}
            for k, v in row[name].items():
                totals.setdefault(name, {}).setdefault(k, []).append(v)
        X, W = r["X"].float(), r["W"].float()
        xq = q_int(X, 8, X.shape[-1])  # BitNet: per-token int8 activations
        exact = X @ W.t()
        row["ternary W, int8 A (fprop)"] = round(((xq @ ternary(W).t() - exact).norm() / exact.norm()).item(), 4)
        totals.setdefault("ternary W, int8 A (fprop)", {}).setdefault("fprop", []).append(row["ternary W, int8 A (fprop)"])
        print(json.dumps({"linear": key, "shape": list(W.shape), **row}), flush=True)
    summary = {name: {k: round(sum(v) / len(v), 4) for k, v in d.items()} for name, d in totals.items()}
    print(json.dumps({"mean_rel_err": summary}), flush=True)


if __name__ == "__main__":
    main()
