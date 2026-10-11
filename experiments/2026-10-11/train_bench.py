#!/usr/bin/env python3
"""Throughput of LLM-training-shaped work on ONE gfx906 (card0): stock torch
paths next to this repo's custom ISA paths (dot2/dot4/dot8 peaks, block-scaled
int8/int4 GEMMs, the fused attention kernel). One JSON line per measurement.

Cross-card rows (gradient exchange) are not measured here: card1 drives the
display, and peer copies into it hung the machine (~/.claude/rules/gpu-lock.md).

Timing is wall clock between host syncs: HIP events overlap on this build
(CLAUDE.md, Hazards).

Usage (repo root, through mantlecore's guard so HIP sees card0 only):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 1200 -- \
        direnv exec . python develop/experiments/2026-10-11/train_bench.py
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NODE_DIR = os.path.join(REPO_ROOT, "custom_nodes", "ComfyUI-Gfx906Int4")
sys.path.insert(0, REPO_ROOT)

TOKENS = 4096  # tokens per micro-batch (M of every linear)

# Llama-style blocks: (name, K = in features, N = out features)
MODELS = {
    "1B": [("qkv", 2048, 6144), ("o", 2048, 2048), ("gate_up", 2048, 16384), ("down", 8192, 2048)],
    "7B": [("qkv", 4096, 12288), ("o", 4096, 4096), ("gate_up", 4096, 22016), ("down", 11008, 4096)],
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
PEAK_OPS = ["fma_f32", "pk_fma_f16", "dot2_f32_f16", "dot4_i8", "dot8_i4"]  # counterpart: peak_isa.hip enum Op


def emit(**row):
    print(json.dumps(row), flush=True)


def time_s(fn, min_s=0.3):
    """Seconds per call: best of 3 sync-bracketed batches, each >= min_s / 3."""
    fn()
    torch.cuda.synchronize()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        if time.perf_counter() - t0 >= min_s / 3:
            break
        n *= 2
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def load_node_package():
    """The node's modules without running its __init__ (that registers ComfyUI nodes)."""
    spec = importlib.util.spec_from_file_location("gfx906int4", os.path.join(NODE_DIR, "__init__.py"),
                                                  submodule_search_locations=[NODE_DIR])
    sys.modules["gfx906int4"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("gfx906int4.ops"), importlib.import_module("gfx906int4.attention")


def bench_peaks(device):
    ext = load(name="peak_isa", sources=[os.path.join(HERE, "peak_isa.hip")],
               extra_cuda_cflags=["--offload-arch=gfx906", "-O3"], verbose=False)
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    blocks = cus * 4
    out = torch.empty(blocks * 256, dtype=torch.float32, device=device)
    for op, name in enumerate(PEAK_OPS):
        iters = 4000
        ops = ext.run(out, op, blocks, iters)
        seconds = time_s(lambda: ext.run(out, op, blocks, iters), min_s=1.5)
        emit(section="peak", op=name, tops=round(ops / seconds / 1e12, 2))


def bench_gemm_stock(model, shapes, device):
    for name, k, n in shapes:
        for dname, dtype in DTYPES.items():
            x = torch.randn(TOKENS, k, device=device, dtype=dtype)
            w = torch.randn(n, k, device=device, dtype=dtype) * 0.02
            dy = torch.randn(TOKENS, n, device=device, dtype=dtype)
            flop = 2 * TOKENS * n * k
            passes = {"fwd": lambda: x @ w.t(), "dgrad": lambda: dy @ w, "wgrad": lambda: dy.t() @ x}
            for pname, fn in passes.items():
                emit(section="gemm", model=model, shape=name, k=k, n=n, path=dname, pass_=pname,
                     tflops=round(flop / time_s(fn) / 1e12, 2))
            del x, w, dy


def bench_gemm_custom(model, shapes, ops, device):
    """int8 / int4 block-32 GEMMs, activation-side quantization included in the
    time (weight quantization excluded: once per optimizer step, amortized).
    Error vs an fp32 matmul of the same fp16 inputs."""
    ext = ops._get_ext()
    for name, k, n in shapes:
        x = torch.randn(TOKENS, k, device=device, dtype=torch.float16)
        w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.02
        dy = torch.randn(TOKENS, n, device=device, dtype=torch.float16)
        flop = 2 * TOKENS * n * k
        ref = {"fwd": x.float() @ w.float().t(), "dgrad": dy.float() @ w.float(), "wgrad": dy.float().t() @ x.float()}

        w8, w8s = ext.quantize_int8(w)
        wt8, wt8s = ext.quantize_int8(w.t().contiguous())
        int8 = {
            "fwd": lambda c: ext.gemm_int8(*ext.quantize_int8(x), w8, w8s, None, torch.float16, c),
            "dgrad": lambda c: ext.gemm_int8(*ext.quantize_int8(dy), wt8, wt8s, None, torch.float16, c),
            "wgrad": lambda c: ext.gemm_int8(*ext.quantize_int8(dy.t().contiguous()),
                                             *ext.quantize_int8(x.t().contiguous()), None, torch.float16, c),
        }
        for pname, fn in int8.items():
            timed = {c: time_s(lambda: fn(c)) for c in range(4)}
            config = min(timed, key=timed.get)
            emit(section="gemm", model=model, shape=name, k=k, n=n, path="int8_b32", pass_=pname, config=config,
                 tflops=round(flop / timed[config] / 1e12, 2), rel_err=round(rel_err(fn(config), ref[pname]), 4))

        # int4: frozen-weight passes only (QLoRA-style base); an int4 wgrad would quantize dY^T to 4 bits
        w4, w4s = ops.quantize_pack_int4(w)
        wt4, wt4s = ops.quantize_pack_int4(w.t().contiguous())
        int4 = {
            "fwd": lambda: ext.gemm_int4(*ext.quantize_int4(x), w4, w4s, None, torch.float16),
            "dgrad": lambda: ext.gemm_int4(*ext.quantize_int4(dy), wt4, wt4s, None, torch.float16),
        }
        for pname, fn in int4.items():
            emit(section="gemm", model=model, shape=name, k=k, n=n, path="int4_b32", pass_=pname,
                 tflops=round(flop / time_s(fn) / 1e12, 2), rel_err=round(rel_err(fn(), ref[pname]), 4))
        del x, w, dy, ref


def causal_keep(attention, qk, heads, length, device):
    """Block-causal keep mask: a key tile is kept when it starts at or before the
    block's last query row. Diagonal tiles stay unmasked inside, so this is a
    throughput proxy for causal attention, not causal attention."""
    rows = attention.block_rows(qk)
    q_blocks, k_tiles = -(-length // rows), -(-length // attention.TILE)
    last_row = (torch.arange(q_blocks, device=device) + 1) * rows - 1
    tile_start = torch.arange(k_tiles, device=device) * attention.TILE
    keep = (tile_start[None, :] <= last_row[:, None]).to(torch.uint8)
    return keep.expand(1, heads, q_blocks, k_tiles).contiguous(), keep.sum().item() * rows * attention.TILE


def bench_attention(attention, device):
    heads, dim = 16, 128
    for length in (2048, 4096, 8192):
        q, k, v = (torch.randn(1, heads, length, dim, device=device, dtype=torch.float16) for _ in range(3))
        flop_full = 4 * heads * length * length * dim
        flop_causal = flop_full / 2
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        emit(section="attention", L=length, path="torch_sdpa_fp16", pass_="fwd causal",
             tflops=round(flop_causal / time_s(sdpa) / 1e12, 2))

        qg, kg, vg = (t.clone().requires_grad_() for t in (q, k, v))
        grad = torch.randn_like(q)

        def fwd_bwd():
            torch.nn.functional.scaled_dot_product_attention(qg, kg, vg, is_causal=True).backward(grad)

        # backward = 2.5x forward's matmul work (5 matmuls against 2)
        emit(section="attention", L=length, path="torch_sdpa_fp16", pass_="fwd+bwd causal",
             tflops=round(3.5 * flop_causal / time_s(fwd_bwd) / 1e12, 2))
        del qg, kg, vg, grad

        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        ref = ref.transpose(1, 2).reshape(1, length, heads * dim)
        for qk, pv in (("fp16", "fp16"), ("int8", "u8"), ("int4", "u8")):
            full = lambda: attention.fused_attention(q, k, v, qk=qk, pv=pv)
            emit(section="attention", L=length, path=f"fused_{qk}_{pv}", pass_="fwd full",
                 tflops=round(flop_full / time_s(full) / 1e12, 2), rel_err=round(rel_err(full(), ref), 4))
            keep, kept_pairs = causal_keep(attention, qk, heads, length, device)
            causal = lambda: attention.fused_attention(q, k, v, qk=qk, pv=pv, keep=keep)
            emit(section="attention", L=length, path=f"fused_{qk}_{pv}", pass_="fwd block-causal",
                 tflops=round(flop_causal / time_s(causal) / 1e12, 2),
                 computed_vs_causal=round(kept_pairs / (length * length / 2), 3))
        del q, k, v, ref


def bench_memory(device):
    numel = 256 * 2**20  # 1 GiB of fp32
    a = torch.empty(numel, device=device)
    seconds = time_s(lambda: a.clone())
    emit(section="memory", op="device copy", gbps=round(2 * a.nbytes / seconds / 1e9, 1))
    del a

    params = [torch.nn.Parameter(torch.randn(16 * 2**20, device=device)) for _ in range(16)]  # 256M params
    for p in params:
        p.grad = torch.randn_like(p)
    count = sum(p.numel() for p in params)
    for impl in ("foreach", "fused"):
        try:
            opt = torch.optim.AdamW(params, lr=1e-4, **{impl: True})
            opt.step()
        except RuntimeError as err:
            emit(section="memory", op=f"AdamW fp32 ({impl})", error=str(err).splitlines()[0])
            continue
        seconds = time_s(opt.step)
        # reads param, grad, exp_avg, exp_avg_sq; writes param, exp_avg, exp_avg_sq
        emit(section="memory", op=f"AdamW fp32 ({impl})", ms_per_1b_params=round(seconds * 1e9 / count * 1e3, 1),
             gbps=round(28 * count / seconds / 1e9, 1))
        del opt
    del params

    host = torch.empty(64 * 2**20, dtype=torch.float32, pin_memory=True)  # 256 MiB
    dev = torch.empty_like(host, device=device)
    emit(section="pcie", op="H2D pinned 256 MiB", gbps=round(host.nbytes / time_s(lambda: dev.copy_(host)) / 1e9, 2))
    emit(section="pcie", op="D2H pinned 256 MiB", gbps=round(host.nbytes / time_s(lambda: host.copy_(dev)) / 1e9, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", default="peak,gemm,attention,memory")
    args = parser.parse_args()
    sections = args.sections.split(",")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    emit(section="device", name=props.name, arch=props.gcnArchName, cus=props.multi_processor_count,
         visible=torch.cuda.device_count(), torch=torch.__version__)
    ops, attention = load_node_package()
    if "peak" in sections:
        bench_peaks(device)
    if "gemm" in sections:
        for model, shapes in MODELS.items():
            bench_gemm_stock(model, shapes, device)
            bench_gemm_custom(model, shapes, ops, device)
    if "attention" in sections:
        bench_attention(attention, device)
    if "memory" in sections:
        bench_memory(device)


if __name__ == "__main__":
    main()
