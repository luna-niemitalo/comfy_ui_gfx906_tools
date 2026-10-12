"""Speed bounds of int4 GEMM scale-math reductions (gemm_variants.hip, SCALE_MODE 0-4) at
Krea2's linear shapes, M = 14400 tokens (2560x1440) and 4096 + text (a growth tile).
Mode 0 is checked against the production kernel's output (must be identical).

Usage: bench_scale_math.py [device]"""
import json, os, sys, time
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {0: "production", 1: "64-value blocks", 2: "one-sided scale", 3: "two-level i24", 4: "no rescale (ceiling)"}
SHAPES = [("wq/gate/wo", 6144, 6144), ("wk/wv", 6144, 1536), ("mlp.gate/up", 6144, 16384), ("mlp.down", 16384, 6144)]
device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
torch.cuda.set_device(device)
exts = {m: load(name=f"scale_math_mode{m}", sources=[os.path.join(HERE, "binding.cpp"), os.path.join(HERE, "gemm_variants.hip")],
                extra_cuda_cflags=["--offload-arch=gfx906", "-O3", f"-DSCALE_MODE={m}"], verbose=False) for m in MODES}


def bench(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/develop/harness")
from bench_linear import load_ops  # noqa: E402
prod = load_ops()._get_ext()
for M in (14400, 4224):
    for name, K, N in SHAPES:
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        w = torch.randn(N, K, device=device, dtype=torch.float16)
        a, a_s = exts[0].quantize(x)
        b, b_s = exts[0].quantize(w)
        ref = prod.gemm_int4(a, a_s, b, b_s, None, torch.float16, 1)
        assert torch.equal(exts[0].gemm(a, a_s, b, b_s), ref), "mode 0 must match production"
        flop = 2 * M * K * N
        row = {"M": M, "shape": name}
        for m, label in MODES.items():
            row[label] = round(flop / bench(lambda: exts[m].gemm(a, a_s, b, b_s)) / 1e12, 1)
        print(json.dumps(row), flush=True)
