#!/usr/bin/env python3
"""Gate + bench for the int4 block-16 training backward (custom_nodes/
ComfyUI-Gfx906Int4: int4_gemm_scaled.hip qblock 16, int4_train_quant.hip,
train_linear.py).

1. inference regression: gemm_int4 (qblock 32) bit-identical to a build of the
   committed sources (--ref-src: `git archive HEAD csrc` of the node repo)
2. quantizers vs quant_study's emulation (round-to-nearest; rows, cols, cols + RHT)
3. stochastic rounding is unbiased: error of the mean over 64 seeds vs one sample
4. block-16 GEMM vs fp32 matmul of the dequantized operands
5. train_linear's backward vs exact fp32 grads, next to the emulated recipe's error
6. speed: dgrad + wgrad per 4B shape, int4 path (quantize + GEMM) vs torch fp16

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --timeout 1800 -- \
        direnv exec . python develop/experiments/2026-10-11/int4_train_test.py --ref-src <dir with csrc/>
"""
import argparse
import importlib
import json
import os
import sys

import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402
from quant_study import q_int, rht16  # noqa: E402
from train_gate import QuantBwdLinear  # noqa: E402

M_TOKENS = 4096
SHAPES_4B = [("qkv", 2560, 6144), ("o", 4096, 2560), ("gate_up", 2560, 19456), ("down", 9728, 2560)]


def emit(**row):
    print(json.dumps(row), flush=True)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def dequant(packed, scale, block):
    """packed int8 [R, K/2] (nibble e%8 of word e/8 per 32-chunk), scale [R, K/block] -> fp32 [R, K]"""
    words = packed.view(torch.int32)
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    nib = (words[..., None] >> shifts) & 0xF
    codes = torch.where(nib > 7, nib - 16, nib).reshape(packed.shape[0], -1).float()
    return codes * scale.repeat_interleave(block, dim=1)


def ref_ext(ref_src):
    csrc = os.path.join(ref_src, "csrc")
    sources = [os.path.join(csrc, f) for f in ("binding.cpp", "int4_gemm_scaled.hip", "attention.hip", "int8_gemm.hip",
                                               "attention_train.hip")]
    return load(name="gfx906_int4_ref", sources=sources, extra_cuda_cflags=["--offload-arch=gfx906", "-O3"],
                verbose=False)


def check_regression(ext, ref, device):
    ok = True
    for m, k, n in ((4096, 9728, 2560), (64, 2560, 6144), (300, 4096, 1000)):
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.02
        aq, as_ = ext.quantize_int4(a)
        wq, ws = ext.quantize_int4(w)
        for splits in (0, 1, 3):
            same = torch.equal(ext.gemm_int4(aq, as_, wq, ws, None, torch.float16, splits),
                               ref.gemm_int4(aq, as_, wq, ws, None, torch.float16, splits))
            ok &= same
            emit(test="b32 regression", m=m, k=k, n=n, splits=splits, bit_identical=same)
    return ok


def check_quantizers(ext, signs, device):
    ok = True
    x = torch.randn(512, 2560, device=device, dtype=torch.float16) * torch.rand(1, 2560, device=device, dtype=torch.float16) * 4
    cases = {
        "rows": (ext.quantize_int4_rows(x, False, 0), q_int(x.float(), 4, 16)),
        "cols": (ext.quantize_int4_cols(x, None, False, 0), q_int(x.float().t().contiguous(), 4, 16)),
        "cols+rht": (ext.quantize_int4_cols(x, signs, False, 0), q_int(rht16(x.float(), signs).t().contiguous(), 4, 16)),
    }
    for name, ((pk, sc), emu) in cases.items():
        got = dequant(pk, sc, 16)
        mismatch = (got != emu).float().mean().item()  # scale = amax * (1/7) vs amax / 7: rare ties may flip
        passed = mismatch < 1e-3 and rel(got, emu) < 1e-3
        ok &= passed
        emit(test="quantizer vs emulation", case=name, mismatch_fraction=round(mismatch, 6),
             rel_diff=round(rel(got, emu), 7), expected="< 1e-3 both", passed=passed)
    one = dequant(*ext.quantize_int4_rows(x, True, 1), 16)
    mean = sum(dequant(*ext.quantize_int4_rows(x, True, s), 16) for s in range(1, 65)) / 64
    err1, err64 = rel(one, x), rel(mean, x)
    passed = err64 < err1 / 5  # unbiased: ~1/8 of one sample's error
    ok &= passed
    emit(test="SR unbiased", one_sample=round(err1, 4), mean_of_64=round(err64, 4), expected="mean < one / 5",
         passed=passed)
    return ok


def check_gemm_b16(ext, device):
    ok = True
    for m, k, n in ((4096, 2560, 6144), (2560, 4096, 9728), (96, 4096, 160)):
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(n, k, device=device, dtype=torch.float16)
        aq, as_ = ext.quantize_int4_rows(a, False, 0)
        bq, bs = ext.quantize_int4_rows(b, False, 0)
        want = dequant(aq, as_, 16) @ dequant(bq, bs, 16).t()
        for splits in (0, 1):
            got = ext.gemm_int4(aq, as_, bq, bs, None, torch.float32, splits, 16)
            d = rel(got, want)
            ok &= d < 1e-5
            emit(test="b16 gemm vs fp32 of dequantized", m=m, k=k, n=n, splits=splits, rel_diff=d, expected="< 1e-5",
                 passed=d < 1e-5)
    return ok


def check_linear(linear, signs, device):
    for name, k, n in SHAPES_4B:
        x = (torch.randn(M_TOKENS, k, device=device) * 0.5).half().requires_grad_()
        w = (torch.randn(n, k, device=device) * k**-0.5).half().requires_grad_()
        dy = torch.randn(M_TOKENS, n, device=device).half() * 1e-2
        exact_dx, exact_dw = dy.float() @ w.float(), dy.float().t() @ x.float()
        linear(x, w).backward(dy)
        got = {"dgrad": rel(x.grad, exact_dx), "wgrad": rel(w.grad, exact_dw)}
        x.grad = w.grad = None
        cfg = {"block": 16, "sr": True, "rht": True, "signs": signs}
        QuantBwdLinear.apply(x, w, cfg).backward(dy)
        emu = {"dgrad": rel(x.grad, exact_dx), "wgrad": rel(w.grad, exact_dw)}
        emit(test="int4 backward vs exact", shape=name, kernel={k_: round(v, 4) for k_, v in got.items()},
             emulated_recipe={k_: round(v, 4) for k_, v in emu.items()})


def speed(linear, device):
    for name, k, n in SHAPES_4B:
        x = torch.randn(M_TOKENS, k, device=device, dtype=torch.float16).requires_grad_()
        w = (torch.randn(n, k, device=device) * 0.02).half().requires_grad_()
        dy = torch.randn(M_TOKENS, n, device=device, dtype=torch.float16)
        out_fp16, out_int4 = torch.nn.functional.linear(x, w), linear(x, w)
        flop = 2 * 2 * M_TOKENS * n * k  # dgrad + wgrad
        t16 = time_s(lambda: torch.autograd.grad(out_fp16, (x, w), dy, retain_graph=True))
        t4 = time_s(lambda: torch.autograd.grad(out_int4, (x, w), dy, retain_graph=True))
        emit(test="speed dgrad+wgrad", shape=name, fp16_ms=round(t16 * 1e3, 2), int4_ms=round(t4 * 1e3, 2),
             fp16_tflops=round(flop / t16 / 1e12, 1), int4_tflops=round(flop / t4 / 1e12, 1),
             speedup=round(t16 / t4, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-src", required=True)
    parser.add_argument("--sections", default="regression,quant,gemm,linear,speed")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    ops, _ = load_node_package()
    ext = ops._get_ext()
    linear = importlib.import_module("gfx906int4.train_linear").linear
    signs = importlib.import_module("gfx906int4.train_linear")._signs(device)
    sections = args.sections.split(",")
    ok = True
    if "regression" in sections:
        ok &= check_regression(ext, ref_ext(args.ref_src), device)
    if "quant" in sections:
        ok &= check_quantizers(ext, signs, device)
    if "gemm" in sections:
        ok &= check_gemm_b16(ext, device)
    if not ok:
        raise SystemExit("int4_train_test: a gate failed -- not timing")
    if "linear" in sections:
        check_linear(linear, signs, device)
    if "speed" in sections:
        speed(linear, device)


if __name__ == "__main__":
    main()
