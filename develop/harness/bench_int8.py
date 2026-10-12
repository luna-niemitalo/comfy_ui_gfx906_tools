"""Needs the int4 extension at d21d717 (the int8 path was removed in 5f98c0d: fp16 won).
gemm_int8 (csrc/int8_gemm.hip): correctness vs a dequantized fp32 matmul of the same
int8 operands, and throughput vs fp32/fp16 torch matmul at the Qwen-Image VAE's 3x3 conv
GEMM shapes (M = pixels of a 512-px tile incl. overlap, K = 9 * Cin, N = Cout).

Usage: bench_int8.py [device]"""
import sys
import time

import torch

from bench_linear import load_ops

ext = load_ops()._get_ext()
device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
torch.cuda.set_device(device)


def bench(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


CONFIGS = {0: "64x64/4x4", 1: "128x64/8x4", 2: "64x128/4x8", 3: "128x128/8x8"}


def dequant(q, s):
    return (q.float().reshape(q.shape[0], -1, 32) * s[..., None]).reshape(q.shape)


# correctness, including ragged M / N (tile edges) and bias
for M, K, N in ((1000, 864, 96), (77, 96, 3), (4096, 3456, 384)):
    a = torch.randn(M, K, device=device)
    w = torch.randn(N, K, device=device)
    bias = torch.randn(N, device=device)
    aq, as_ = ext.quantize_int8(a)
    wq, ws = ext.quantize_int8(w)
    ref = dequant(aq, as_) @ dequant(wq, ws).T + bias
    err = max(((ext.gemm_int8(aq, as_, wq, ws, bias, torch.float32, c) - ref).abs().max() / ref.abs().max()).item()
              for c in CONFIGS)
    quant_err = ((dequant(aq, as_) - a).abs().max() / a.abs().max()).item()
    print(f"check M{M} K{K} N{N}: max rel err {err:.2e} (vs dequant ref), quant err {quant_err:.2e}")
    assert err < 1e-5, err

for M, K, N in ((576 * 576, 864, 96), (576 * 576, 1728, 96), (288 * 288, 1728, 192), (288 * 288, 3456, 192),
                (144 * 144, 3456, 384), (72 * 72, 3456, 384)):
    a = torch.randn(M, K, device=device)
    w = torch.randn(N, K, device=device)
    aq, as_ = ext.quantize_int8(a)
    wq, ws = ext.quantize_int8(w)
    flop = 2 * M * K * N
    t32 = bench(lambda: a @ w.T)
    ah, wh = a.half(), w.half()
    t16 = bench(lambda: ah @ wh.T)
    configs = {c: bench(lambda c=c: ext.gemm_int8(aq, as_, wq, ws, None, torch.float32, c)) for c in CONFIGS}
    best = min(configs, key=configs.get)
    print(f"M{M:>6} K{K:>5} N{N:>4}: fp32 {flop / t32 / 1e12:5.1f} TF  fp16 {flop / t16 / 1e12:5.1f} TF  int8 TOPS "
          + " ".join(f"{CONFIGS[c]} {flop / t / 1e12:5.1f}" for c, t in configs.items())
          + f"  best {t32 / configs[best]:.2f}x fp32")

# implicit-GEMM conv (conv_int8) vs F.conv2d on the same dequantized int8 operands
F = torch.nn.functional
for B, C, H, W, N, k, s, p in ((1, 96, 37, 53, 96, 3, 1, 1), (2, 64, 20, 18, 64, 3, 2, 0), (1, 384, 72, 72, 384, 3, 1, 1),
                               (1, 192, 30, 30, 384, 1, 1, 0)):
    x = torch.randn(B, C, H, W, device=device)
    w = torch.randn(N, C, k, k, device=device)
    bias = torch.randn(N, device=device)
    xq, xs = ext.quantize_int8(x.permute(0, 2, 3, 1).reshape(-1, C))
    wq, ws = ext.quantize_int8(w.permute(0, 2, 3, 1).reshape(N, -1).contiguous())
    xd = dequant(xq, xs).view(B, H, W, C).permute(0, 3, 1, 2)
    wd = dequant(wq, ws).view(N, k, k, C).permute(0, 3, 1, 2)
    ref = F.conv2d(xd, wd, bias, s, p)
    out = ext.conv_int8(xq, xs, wq, ws, bias, torch.float32, B, H, W, k, k, s, s, p, p)
    out = out.view(B, ref.shape[2], ref.shape[3], N).permute(0, 3, 1, 2)
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f"conv check B{B} C{C} {H}x{W} N{N} k{k} s{s} p{p}: max rel err {err:.2e}")
    assert err < 1e-5, err

for C, HW, N in ((96, 576, 96), (192, 288, 192), (384, 144, 384), (384, 72, 384)):
    x = torch.randn(1, C, HW, HW, device=device)
    w = torch.randn(N, C, 3, 3, device=device)
    xq, xs = ext.quantize_int8(x.permute(0, 2, 3, 1).reshape(-1, C))
    wq, ws = ext.quantize_int8(w.permute(0, 2, 3, 1).reshape(N, -1).contiguous())
    torch.backends.cudnn.enabled = False
    t32 = bench(lambda: F.conv2d(x, w, None, 1, 1))
    xh, wh = x.half(), w.half()
    t16 = bench(lambda: F.conv2d(xh, wh, None, 1, 1))
    tc = bench(lambda: ext.conv_int8(xq, xs, wq, ws, None, torch.float32, 1, HW, HW, 3, 3, 1, 1, 1, 1))
    tq = bench(lambda: ext.quantize_int8(x.permute(0, 2, 3, 1).reshape(-1, C)))
    print(f"conv 3x3 {C}->{N} at {HW}x{HW}: fp32 {t32 * 1e3:.2f} ms  fp16 {t16 * 1e3:.2f} ms  "
          f"int8 {tc * 1e3:.2f} ms + quantize {tq * 1e3:.2f} ms (from NCHW)")
