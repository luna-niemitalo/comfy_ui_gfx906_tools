#!/usr/bin/env python3
"""Memory of LLM-training pieces on one gfx906 (card0), with the training attention
kernel: per transformer block (4B / 8B shapes, 4096 tokens) the weights, the
activations a fwd+bwd holds, and the same with activation checkpointing (time
included: recompute costs a forward); the LM head plain vs chunked cross-entropy.

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 900 -- \
        direnv exec . python develop/experiments/2026-10-11/mem_bench.py
"""
import importlib
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402
from attn_train_test import OursBlock  # noqa: E402
import dual_bench  # noqa: E402

GIB = 2**30


def emit(**row):
    print(json.dumps(row), flush=True)


def peak_of(fn):
    """GiB allocated at peak above what was allocated before fn."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / GIB


def blocks(attention, ops, device):
    for model, c in dual_bench.CONFIGS.items():
        for mode in dual_bench.MODES:
            before = torch.cuda.memory_allocated()
            block = OursBlock(c, mode, ops, device)
            block.attention = attention
            weights = (torch.cuda.memory_allocated() - before) / GIB
            x = torch.randn(1, dual_bench.TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
            grad = torch.randn_like(x)

            def plain():
                block(x).backward(grad)

            def ckpt():
                checkpoint(block, x, use_reentrant=False).backward(grad)

            plain()  # gradients allocated once: the peaks below are activations + workspace
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            out = block(x)
            saved = (torch.cuda.memory_allocated() - base) / GIB  # kept for backward, incl. the output
            out.backward(grad)
            del out
            row = {"weights_gib": round(weights, 3), "saved_per_layer_gib": round(saved, 3),
                   "grads_gib": round(sum(p.grad.numel() * 2 for p in block.parameters() if p.grad is not None) / GIB, 3)}
            for name, fn in (("plain", plain), ("ckpt", ckpt)):
                row[f"{name}_peak_gib"] = round(peak_of(fn), 3)
                row[f"{name}_ms"] = round(time_s(fn, min_s=1.0) * 1e3, 1)
            # what checkpointing keeps per layer between forward and backward: the block input
            row["ckpt_kept_per_layer_gib"] = round(x.numel() * 2 / GIB, 4)
            emit(section="block", model=model, mode=mode, layers=c["layers"], **row)
            del block, x, grad
            torch.cuda.empty_cache()


def head(device):
    for model, c in dual_bench.CONFIGS.items():
        emb = torch.nn.Parameter(torch.randn(c["vocab"], c["d"], device=device, dtype=torch.float16) * 0.02)
        h = torch.randn(dual_bench.TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, c["vocab"], (dual_bench.TOKENS,), device=device)

        def plain():
            F.cross_entropy(F.linear(h, emb).float(), target).backward()

        def chunk_loss(hc, tc):
            return F.cross_entropy(F.linear(hc, emb).float(), tc, reduction="sum")

        def chunked(rows=1024):
            # logits of one chunk live at a time; backward recomputes them
            loss = sum(checkpoint(chunk_loss, h[i:i + rows], target[i:i + rows], use_reentrant=False)
                       for i in range(0, dual_bench.TOKENS, rows))
            (loss / dual_bench.TOKENS).backward()

        plain()
        row = {"emb_gib": round(emb.numel() * 2 / GIB, 3)}
        for name, fn in (("plain", plain), ("chunked1024", chunked)):
            row[f"{name}_peak_gib"] = round(peak_of(fn), 3)
            row[f"{name}_ms"] = round(time_s(fn, min_s=1.0) * 1e3, 1)
        emit(section="head", model=model, vocab=c["vocab"], **row)
        del emb, h, target
        torch.cuda.empty_cache()


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ops, _ = load_node_package()
    attention = importlib.import_module("gfx906int4.train_attention").attention
    blocks(attention, ops, device)
    head(device)


if __name__ == "__main__":
    main()
