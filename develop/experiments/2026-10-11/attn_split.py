#!/usr/bin/env python3
"""Where a dual_bench block's time goes: causal attention fwd+bwd at the blocks'
shape (32 q heads, 8 kv heads, L 4096) per SDPA path, against the block without
attention. Answers whether attention backward is the training bottleneck."""
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402
import dual_bench  # noqa: E402

L, HQ, HKV, D = 4096, 32, 8, 128


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    q = torch.randn(1, HQ, L, D, device=device, dtype=torch.float16, requires_grad=True)
    k, v = (torch.randn(1, HKV, L, D, device=device, dtype=torch.float16, requires_grad=True) for _ in range(2))
    grad = torch.randn(1, HQ, L, D, device=device, dtype=torch.float16)
    variants = {
        "gqa_default": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
        "repeat_kv_default": lambda: F.scaled_dot_product_attention(
            q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True),
    }
    for backend in (SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION):
        def run(backend=backend):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True)
        variants[f"repeat_kv_{backend.name.lower()}"] = run
    flop = 3.5 * 2 * L * L * HQ * D  # causal fwd (half of 4 L^2 D per head) + bwd 2.5x
    for name, fn in variants.items():
        try:
            seconds = time_s(lambda: fn().backward(grad), min_s=1.0)
        except RuntimeError as err:
            print(json.dumps({"attention": name, "error": str(err).splitlines()[0][:160]}), flush=True)
            continue
        print(json.dumps({"attention": name, "fwd_bwd_ms": round(seconds * 1e3, 1),
                          "tflops": round(flop / seconds / 1e12, 2)}), flush=True)

    ops, _ = load_node_package()
    dual_bench.F = F
    for model in ("4B", "8B"):
        for mode in dual_bench.MODES:
            block = dual_bench.Block(dual_bench.CONFIGS[model], mode, ops, device)
            # attention replaced by its input's q slice: same shapes, no attention cost
            dual_bench.F = type("NoAttn", (), {"scaled_dot_product_attention": staticmethod(lambda q, k, v, **kw: q),
                                               "__getattr__": lambda self, n: getattr(F, n)})()
            x = torch.randn(1, L, dual_bench.CONFIGS[model]["d"], device=device, dtype=torch.float16, requires_grad=True)
            g = torch.randn_like(x)
            seconds = time_s(lambda: block(x).backward(g), min_s=1.0)
            dual_bench.F = F
            print(json.dumps({"model": model, "mode": mode, "block_without_attention_ms": round(seconds * 1e3, 1)}), flush=True)
            del block, x, g
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
