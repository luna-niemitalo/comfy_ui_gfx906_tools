#!/usr/bin/env python3
"""Does an int4 backward hurt fine-tuning? Short emulated A/B on real Qwen3-4B
before any int4 training kernel is written (RESULTS.md 5, proposed step 1).

Arms (same seed, data, order):
  fp16         -- plain
  int4_b16_sr  -- fprop fp16; dgrad and wgrad from int4 block-16 operands, stochastic
                  rounding on dY, 16-point random Hadamard on the wgrad inputs
                  (NVIDIA's NVFP4 recipe minus the 2D weight tiles: quant_study showed
                  they cost int4 more than they buy, and fprop stays fp16 here)
  int4_b32_rtn -- dgrad and wgrad from int4 block-32 round-to-nearest operands (the
                  numerics of our current int4 kernel)
Quantization is emulated (fp32 math on dequantized values), so speed means nothing.

Setup fits one 16 GB card: layers 16-19 trained (fp32 master weights and AdamW on the
host -- a small version of the planned host-offloaded optimizer); layers 20-31 frozen
but backpropagated through (their dgrad is quantized too); layers 32-35 fp16 (the
recipe keeps the last blocks in high precision); layers 0-15 run without autograd.
Data: ComfyUI's comfy/ Python source in 2048-token sequences; comfy/ldm/ held out
for validation.

Usage (repo root, ~1 h for 3 arms):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 5400 -- \
        direnv exec . python develop/experiments/2026-10-11/train_gate.py
"""
import argparse
import glob
import importlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_study import TOKENIZER, load_model, q_int, rht16  # noqa: E402
from train_bench import load_node_package  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEQ = 2048
TRAINED = range(16, 20)
QUANT_BWD = range(16, 32)
LOSS_SCALE = 1024.0  # fp16 grads of a mean loss underflow without it; quantization is scale-invariant
# arm -> (block, stochastic rounding on dY, random Hadamard on wgrad inputs); None = fp16
ARMS = {
    "fp16": None,
    "int4_b16_kernel": "kernel",  # the recipe through the real kernels (train_linear.py), not emulated
    "int4_b16_sr": (16, True, True),  # the recipe (name kept from the first run's log)
    "int4_b32_rtn": (32, False, False),  # our current kernel's numerics
    "int4_b16_rtn": (16, False, False),  # ablation: neither
    "int4_b16_sronly": (16, True, False),  # ablation: SR without the Hadamard
}


class QuantBwdLinear(torch.autograd.Function):
    """y = x W^T in fp16; the backward GEMMs see quantized operands (cfg)."""

    @staticmethod
    def forward(ctx, x, w, cfg):
        ctx.save_for_backward(x, w)
        ctx.cfg = cfg
        return F.linear(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        cfg = ctx.cfg
        shape = dy.shape
        g, xf, wf = dy.reshape(-1, shape[-1]).float(), x.reshape(-1, x.shape[-1]).float(), w.float()
        q = lambda t, sr=False: q_int(t, 4, cfg["block"], sr)
        # dgrad: reduction over N -- dY blocks along N, W^T blocks along N
        dx = (q(g, cfg["sr"]) @ q(wf.t().contiguous()).t()).to(dy.dtype).reshape(*shape[:-1], -1)
        dw = None
        if w.requires_grad:
            gw, xw = (rht16(g, cfg["signs"]), rht16(xf, cfg["signs"])) if cfg["rht"] else (g, xf)
            # wgrad: reduction over tokens -- both operands blocked along tokens
            dw = (q(gw.t().contiguous(), cfg["sr"]) @ q(xw.t().contiguous()).t()).to(w.dtype)
        return dx, dw, None


def patch_linears(model, cfg):
    kernel_linear = None
    if cfg == "kernel":
        load_node_package()
        kernel_linear = importlib.import_module("gfx906int4.train_linear").linear
    for li in QUANT_BWD:
        layer = model.model.layers[li]
        for mod in [*layer.self_attn.children(), *layer.mlp.children()]:
            if not isinstance(mod, torch.nn.Linear):
                continue
            if kernel_linear:
                mod.forward = lambda x, mod=mod: kernel_linear(x, mod.weight)
            else:
                mod.forward = lambda x, mod=mod: QuantBwdLinear.apply(x, mod.weight, cfg)


def chunked_loss(model, hidden, ids, rows=512):
    """mean next-token CE; logits of one chunk at a time (head peak: RESULTS.md 4)"""
    emb = model.lm_head.weight
    h, target = hidden[0, :-1], ids[0, 1:]
    part = lambda hc, tc: F.cross_entropy(F.linear(hc, emb).float(), tc, reduction="sum")
    total = sum(checkpoint(part, h[i:i + rows], target[i:i + rows], use_reentrant=False)
                for i in range(0, h.shape[0], rows))
    return total / target.numel()


def corpus(tok):
    files = sorted(glob.glob(os.path.join(REPO, "comfy", "**", "*.py"), recursive=True))
    val_files = [f for f in files if "/comfy/ldm/" in f]
    train_files = [f for f in files if "/comfy/ldm/" not in f]
    cut = lambda fs: (lambda ids: ids[: len(ids) // SEQ * SEQ].view(-1, SEQ))(
        torch.tensor(tok("\n\n".join(open(f).read() for f in fs)).input_ids))
    return cut(train_files), cut(val_files)


def run_arm(arm, args, train, val, device):
    torch.manual_seed(0)
    model = load_model(device)
    model.config.use_cache = False
    if ARMS[arm] == "kernel":
        patch_linears(model, "kernel")
    elif ARMS[arm]:
        block, sr, rht = ARMS[arm]
        cfg = {"block": block, "sr": sr, "rht": rht,
               "signs": torch.where(torch.rand(16, device=device) < 0.5, -1.0, 1.0)}
        patch_linears(model, cfg)
    params = [p for li in TRAINED for p in model.model.layers[li].parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    master = [p.detach().float().cpu().requires_grad_() for p in params]
    opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    order = torch.randperm(train.shape[0], generator=torch.Generator().manual_seed(1))

    def evaluate():
        with torch.no_grad():
            losses = [chunked_loss(model, model.model(input_ids=v[None].to(device)).last_hidden_state,
                                   v[None].to(device)).item() for v in val[: args.val_seqs]]
        return sum(losses) / len(losses)

    log = {"arm": arm, "step": 0, "val": round(evaluate(), 4)}
    print(json.dumps(log), flush=True)
    t0, skipped = time.perf_counter(), 0
    for step in range(1, args.steps + 1):
        ids = train[order[step % len(order)]][None].to(device)
        loss = chunked_loss(model, model.model(input_ids=ids).last_hidden_state, ids)
        (loss * LOSS_SCALE).backward()
        grads = [p.grad for p in params]
        if not all(torch.isfinite(g).all() for g in grads):
            skipped += 1  # overflow at this scale: skip, as a dynamic scaler would
        else:
            warm = min(1.0, step / args.warmup)
            for g_group in opt.param_groups:
                g_group["lr"] = args.lr * warm
            for m, g in zip(master, grads):
                m.grad = g.float().cpu() / LOSS_SCALE
            opt.step()
            with torch.no_grad():
                for p, m in zip(params, master):
                    p.copy_(m.to(device, torch.float16))
        for p in params:
            p.grad = None
        if step % args.log_every == 0:
            print(json.dumps({"arm": arm, "step": step, "train": round(loss.item(), 4),
                              "s_per_step": round((time.perf_counter() - t0) / step, 2), "skipped": skipped}),
                  flush=True)
        if step % args.eval_every == 0:
            print(json.dumps({"arm": arm, "step": step, "val": round(evaluate(), 4)}), flush=True)
    del model, opt, master
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default="fp16,int4_b16_kernel,int4_b16_sr,int4_b32_rtn,int4_b16_rtn,int4_b16_sronly")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--val-seqs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    from transformers import AutoTokenizer
    train, val = corpus(AutoTokenizer.from_pretrained(TOKENIZER))
    print(json.dumps({"train_seqs": train.shape[0], "val_seqs": val.shape[0], "seq": SEQ}), flush=True)
    if train.shape[0] < args.steps or val.shape[0] < args.val_seqs:
        raise SystemExit(f"train_gate: tried to build the corpus -- expected >= {args.steps} train / "
                         f"{args.val_seqs} val sequences -- actual {train.shape[0]} / {val.shape[0]}")
    for arm in args.arms.split(","):
        run_arm(arm, args, train, val, device)


if __name__ == "__main__":
    main()
