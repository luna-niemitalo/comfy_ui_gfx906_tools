#!/usr/bin/env python3
"""Microbenchmark of one int4 Linear at real Krea2 shapes, split into its
phases (activation quantize, int4 GEMM) against an fp16 rocBLAS matmul of
the same shape. Answers "where does a linear's time go" without loading a
model. One JSON line per (M, shape) on stdout.

Usage:
    python develop/harness/bench_linear.py --device cuda:0 --m 64,2304,14400
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
OPS_PATH = os.path.join(REPO_ROOT, "custom_nodes", "ComfyUI-Gfx906Int4", "ops.py")

# (name, K, N) for one Krea2 SingleStreamBlock; see comfy/ldm/krea2/model.py
SHAPES = [
    ("wq/gate/wo", 6144, 6144),
    ("wk/wv", 6144, 1536),
    ("mlp.gate/up", 6144, 16384),
    ("mlp.down", 16384, 6144),
]


def load_ops():
    spec = importlib.util.spec_from_file_location("gfx906_int4_ops", OPS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_ms(fn, iters):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", default="64,2304,14400", help="comma-separated token counts")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--splits", default="0", help="comma-separated split-K counts to time (0 = the kernel's auto pick)")
    args = parser.parse_args()

    ops = load_ops()
    ext = ops._get_ext()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    for m in (int(v) for v in args.m.split(",")):
        for name, k, n in SHAPES:
            x = torch.randn(m, k, device=device, dtype=torch.float16)
            w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.02
            w_packed, w_scale = ops.quantize_pack_int4(w)
            a_packed, a_scale = ext.quantize_int4(x)

            quant_ms = time_ms(lambda: ext.quantize_int4(x), args.iters)
            fp16_ms = time_ms(lambda: x @ w.t(), args.iters)
            flop = 2 * m * n * k
            unsplit = ext.gemm_int4(a_packed, a_scale, w_packed, w_scale, None, torch.float32, 1)
            for splits in (int(v) for v in args.splits.split(",")):
                gemm_ms = time_ms(lambda: ext.gemm_int4(a_packed, a_scale, w_packed, w_scale, None, x.dtype, splits), args.iters)
                split_out = ext.gemm_int4(a_packed, a_scale, w_packed, w_scale, None, torch.float32, splits)
                print(json.dumps({
                    "m": m, "shape": name, "k": k, "n": n, "splits": splits or f"auto={ext.gemm_int4_auto_splits(m, n, k)}",
                    "quant_ms": round(quant_ms, 3), "gemm_ms": round(gemm_ms, 3), "fp16_ms": round(fp16_ms, 3),
                    "int4_tops": round(flop / gemm_ms / 1e9, 2), "fp16_tflops": round(flop / fp16_ms / 1e9, 2),
                    "vs_unsplit_rel": float(f"{((split_out - unsplit).abs().max() / unsplit.abs().max()).item():.1e}"),
                }), flush=True)
            del x, w, w_packed, w_scale, a_packed, a_scale


if __name__ == "__main__":
    main()
