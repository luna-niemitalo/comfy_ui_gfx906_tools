#!/usr/bin/env python3
"""Gate and benchmark for the gfx906 training attention (custom_nodes/
ComfyUI-Gfx906Int4/csrc/attention_train.hip).

1. correctness vs an fp32 reference on the same fp16 inputs: odd lengths, GQA,
   causal and full, q/k/v as strided views of one fused qkv tensor
2. speed vs torch SDPA (math: the only gfx906 path) at L 2048..8192, 32/8 heads
3. dual_bench's 4B / 8B training blocks with the kernel in place of SDPA

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 900 -- \
        direnv exec . python develop/experiments/2026-10-11/attn_train_test.py
"""
import argparse
import importlib
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402
import dual_bench  # noqa: E402

TOLERANCE = 2e-2  # rel. L2 error vs fp32; fp16 P / dS in LDS and fp16 outputs, FA2 on fp16 is ~1e-3


def emit(**row):
    print(json.dumps(row), flush=True)


def fused_qkv(B, L, H, KVH, device, requires_grad=True):
    qkv = torch.randn(B, L, (H + 2 * KVH) * 128, device=device, dtype=torch.float16)
    qkv.requires_grad_(requires_grad)
    q, k, v = qkv.split([H * 128, KVH * 128, KVH * 128], dim=-1)
    return qkv, q.unflatten(-1, (H, 128)), k.unflatten(-1, (KVH, 128)), v.unflatten(-1, (KVH, 128))


def reference(q, k, v, causal):
    """fp32 SDPA on [B, L, heads, 128] inputs, GQA expanded."""
    rep = q.shape[2] // k.shape[2]
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    kf, vf = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal).transpose(1, 2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(attention, device):
    ok = True
    for L in (100, 257, 1000):
        for causal in (True, False):
            torch.manual_seed(L)
            qkv, q, k, v = fused_qkv(2, L, 8, 2, device)
            grad = torch.randn(2, L, 8, 128, device=device, dtype=torch.float16)
            out = attention(q, k, v, causal=causal)
            out.backward(grad)
            got = {"out": out, "dq": qkv.grad[..., :1024], "dk": qkv.grad[..., 1024:1280], "dv": qkv.grad[..., 1280:]}

            ref_qkv = qkv.detach().float().requires_grad_()
            rq, rk, rv = ref_qkv.split([1024, 256, 256], dim=-1)
            ref_out = reference(rq.unflatten(-1, (8, 128)), rk.unflatten(-1, (2, 128)), rv.unflatten(-1, (2, 128)), causal)
            ref_out.backward(grad.float())
            want = {"out": ref_out, "dq": ref_qkv.grad[..., :1024], "dk": ref_qkv.grad[..., 1024:1280],
                    "dv": ref_qkv.grad[..., 1280:]}
            errors = {name: round(rel(got[name], want[name]), 5) for name in got}
            passed = all(e < TOLERANCE and e == e for e in errors.values())
            ok &= passed
            emit(test="correctness", L=L, causal=causal, rel_err=errors, expected=f"< {TOLERANCE}", passed=passed)
    return ok


def speed(attention, device):
    H, KVH = 32, 8
    for L in (2048, 4096, 8192):
        flop = 3.5 * 2 * L * L * H * 128  # causal fwd + 2.5x bwd
        qkv, q, k, v = fused_qkv(1, L, H, KVH, device)
        grad = torch.randn(1, L, H, 128, device=device, dtype=torch.float16)
        ours_fwd = time_s(lambda: attention(q, k, v), min_s=1.0)
        ours = time_s(lambda: attention(q, k, v).backward(grad), min_s=1.0)
        if L > 4096:  # SDPA's math path allocates the full L x L scores: 4 GiB at 8192, OOM next to ours
            emit(test="speed", L=L, heads=f"{H}/{KVH}", ours_fwd_ms=round(ours_fwd * 1e3, 1),
                 ours_fwd_bwd_ms=round(ours * 1e3, 1), ours_tflops=round(flop / ours / 1e12, 2), sdpa="skipped")
            continue
        qt, kt, vt = (t.detach().transpose(1, 2).contiguous().requires_grad_() for t in (q, k, v))
        gt = grad.transpose(1, 2).contiguous()
        sdpa_fwd = time_s(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True), min_s=1.0)
        sdpa = time_s(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True).backward(gt),
                      min_s=1.0)
        emit(test="speed", L=L, heads=f"{H}/{KVH}",
             ours_fwd_ms=round(ours_fwd * 1e3, 1), ours_fwd_tflops=round(flop / 3.5 / ours_fwd / 1e12, 2),
             ours_fwd_bwd_ms=round(ours * 1e3, 1), ours_tflops=round(flop / ours / 1e12, 2),
             sdpa_fwd_ms=round(sdpa_fwd * 1e3, 1), sdpa_fwd_bwd_ms=round(sdpa * 1e3, 1),
             sdpa_tflops=round(flop / sdpa / 1e12, 2), speedup=round(sdpa / ours, 2))
        del qkv, q, k, v, qt, kt, vt
        torch.cuda.empty_cache()


def stages(ops, device):
    """Each kernel alone at L 4096, 32/8 heads (BWD_* masks: attention_train.hpp)."""
    ext = ops._get_ext()
    L, H, KVH = 4096, 32, 8
    _, q, k, v = fused_qkv(1, L, H, KVH, device, requires_grad=False)
    dout = torch.randn(1, L, H, 128, device=device, dtype=torch.float16)
    o, lse = ext.attention_train_fwd(q, k, v, True, 128**-0.5)
    flop_fwd = 2 * L * L * H * 128  # causal
    row = {"fwd_ms": time_s(lambda: ext.attention_train_fwd(q, k, v, True, 128**-0.5), min_s=1.0) * 1e3}
    for name, mask, flop in (("delta", 1, 0), ("dkdv", 2, 4 * flop_fwd / 2), ("dq", 4, 3 * flop_fwd / 2)):
        ms = time_s(lambda: ext.attention_train_bwd(q, k, v, o, lse, dout, True, 128**-0.5, mask), min_s=1.0) * 1e3
        row[f"{name}_ms"] = ms
        if flop:
            row[f"{name}_tflops"] = flop / ms / 1e9
    emit(test="stages", L=L, **{key: round(val, 2) for key, val in row.items()})


class OursBlock(dual_bench.Block):
    """dual_bench's block with the training kernel: q/k/v stay token-major views."""

    def forward(self, x):
        c, length = self.c, x.shape[1]
        q, k, v = self.qkv(dual_bench.rms_norm(x, self.n1)).split([c["q"], c["kv"], c["kv"]], dim=-1)
        a = self.attention(q.unflatten(-1, (-1, 128)), k.unflatten(-1, (-1, 128)), v.unflatten(-1, (-1, 128)))
        x = x + self.o(a.flatten(-2))
        gate, up = self.gate_up(dual_bench.rms_norm(x, self.n2)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


def blocks(attention, ops, device):
    for model, c in dual_bench.CONFIGS.items():
        for mode in dual_bench.MODES:
            block = OursBlock(c, mode, ops, device)
            block.attention = attention
            x = torch.randn(1, dual_bench.TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
            grad = torch.randn_like(x)
            seconds = time_s(lambda: block(x).backward(grad), min_s=1.5)
            emit(test="block", model=model, mode=mode, ms_per_block=round(seconds * 1e3, 1),
                 model_tokens_per_s=round(dual_bench.TOKENS / (seconds * c["layers"]), 1))
            del block, x, grad
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", default="check,speed,blocks")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ops, _ = load_node_package()
    attention = importlib.import_module("gfx906int4.train_attention").attention
    sections = args.sections.split(",")
    if "check" in sections and not check(attention, device):
        raise SystemExit("attn_train_test: correctness gate failed -- speed numbers would be meaningless")
    if "stages" in sections:
        stages(ops, device)
    if "speed" in sections:
        speed(attention, device)
    if "blocks" in sections:
        blocks(attention, ops, device)


if __name__ == "__main__":
    main()
