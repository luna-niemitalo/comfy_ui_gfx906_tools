#!/usr/bin/env python3
"""fp16 weight-gradient GEMM dW[N, K] = dY^T X (dY [M, N], X [M, K], M = tokens) on
one gfx906 in every operand order torch can be asked for. train_bench measured the
autograd order (dy.t() @ x) at 2.9-18 TFLOPS depending on the shape; this asks
whether a layout choice alone recovers the ~17 of the fwd / dgrad passes.

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 600 -- \
        direnv exec . python develop/experiments/2026-10-11/wgrad_layouts.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import time_s  # noqa: E402

TOKENS = 4096
SHAPES = {  # (name, K in, N out); dual_bench's 4B / 8B blocks
    "4B": [("qkv", 2560, 6144), ("o", 4096, 2560), ("gate_up", 2560, 19456), ("down", 9728, 2560)],
    "8B": [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096)],
}


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    for model, shapes in SHAPES.items():
        for name, k, n in shapes:
            x = torch.randn(TOKENS, k, device=device, dtype=torch.float16)
            dy = torch.randn(TOKENS, n, device=device, dtype=torch.float16)
            ref = (dy.float().t() @ x.float())
            flop = 2 * TOKENS * n * k
            variants = {
                "dyT@x (autograd)": lambda: dy.t() @ x,
                "(xT@dy)^T": lambda: (x.t() @ dy).t(),
                "dyT.contig@x": lambda: dy.t().contiguous() @ x,
                "dyT@xT.contig^T": lambda: dy.t() @ x.t().contiguous().t(),
                "mm out=": lambda: torch.mm(dy.t(), x, out=torch.empty(n, k, device=device, dtype=torch.float16)),
                "fp32 accumulate view": lambda: torch.mm(dy.t(), x).float(),
            }
            row = {}
            for vname, fn in variants.items():
                err = ((fn().float() - ref).norm() / ref.norm()).item()
                row[vname] = {"tflops": round(flop / time_s(fn) / 1e12, 2), "rel_err": round(err, 5)}
            print(json.dumps({"model": model, "shape": name, "k": k, "n": n, **row}), flush=True)
            del x, dy, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
