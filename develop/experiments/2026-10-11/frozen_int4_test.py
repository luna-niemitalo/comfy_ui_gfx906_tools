#!/usr/bin/env python3
"""Gate + bench for FrozenInt4Linear (custom_nodes/ComfyUI-Gfx906Int4/train_linear.py):
a frozen linear stored as two packed int4 copies (W along K, W^T along N).

Per 4B shape, x requires grad, W frozen:
1. y and dx bit-identical to linear(x, w, int4_fprop=True) (same SR seed)
2. resident weight bytes and bytes held between forward and backward, vs fp16 and
   the requantize-per-call path
3. forward + dgrad time

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 600 -- \
        direnv exec . python develop/experiments/2026-10-11/frozen_int4_test.py
"""
import importlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402
from int4_train_test import M_TOKENS, SHAPES_4B  # noqa: E402


def emit(**row):
    print(json.dumps(row), flush=True)


def held_bytes(fn, x):
    """bytes the autograd graph keeps alive between forward and backward (the input is a
    fresh clone, so a saved x counts, as it would inside a model)"""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    y = fn(x.clone())
    held = torch.cuda.memory_allocated() - before - y.numel() * y.element_size()
    del y
    return held


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    load_node_package()
    tl = importlib.import_module("gfx906int4.train_linear")
    ok = True
    for name, k, n in SHAPES_4B:
        lin = torch.nn.Linear(k, n, bias=False, device=device, dtype=torch.float16).requires_grad_(False)
        lin.weight.normal_(0, k**-0.5)
        before = torch.cuda.memory_allocated()
        frozen = tl.FrozenInt4Linear(lin)
        packed_bytes = torch.cuda.memory_allocated() - before
        x = torch.randn(M_TOKENS, k, device=device, dtype=torch.float16).requires_grad_()
        dy = torch.randn(M_TOKENS, n, device=device, dtype=torch.float16)
        paths = {"fp16": lambda x: torch.nn.functional.linear(x, lin.weight),
                 "requant": lambda x: tl.linear(x, lin.weight, int4_fprop=True), "frozen": frozen}

        got = {}
        for p in ("requant", "frozen"):
            tl._seed[0] = 0
            y = paths[p](x)
            (dx,) = torch.autograd.grad(y, x, dy)
            got[p] = (y, dx)
        same = torch.equal(got["requant"][0], got["frozen"][0]) and torch.equal(got["requant"][1], got["frozen"][1])
        ok &= same
        ms = {p: round(time_s(lambda f=f: torch.autograd.grad(f(x), x, dy)) * 1e3, 2) for p, f in paths.items()}
        emit(shape=name, bit_identical=same, weight_mb={"fp16": round(n * k * 2 / 2**20, 1),
                                                        "frozen": round(packed_bytes / 2**20, 1)},
             held_mb={p: round(held_bytes(f, x) / 2**20, 1) for p, f in paths.items()},
             fwd_dgrad_ms=ms)
    if not ok:
        raise SystemExit("frozen_int4_test: expected FrozenInt4Linear bit-identical to linear(int4_fprop=True) -- "
                         "a shape differed (see bit_identical above)")


if __name__ == "__main__":
    main()
