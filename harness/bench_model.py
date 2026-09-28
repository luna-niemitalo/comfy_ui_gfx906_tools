#!/usr/bin/env python3
"""Time one Krea2 diffusion-model forward (= one sampling step at cfg 1) per
resolution, with the real int4 UNet loaded once. Skips text encode, sampler
and VAE: this is the per-step cost the sampler multiplies by `steps`.
Optionally splits each forward's device time into categories.

Attention backend is chosen here, not inherited: bootstrap.py doesn't match
run.nu (quad attention + noflash_attention's SDPA patch), and comfy picks
optimized_attention once at import time, so it must be set before bootstrap.

Usage:
    python develop/harness/bench_model.py --attention quad --res 128x128,1024x576,2560x1440
    python develop/harness/bench_model.py --attention pytorch --noflash --breakdown --res 1024x576
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--unet", default="krea2/museByStableYogi_v35Int4Extended.safetensors")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--attention", choices=["quad", "split", "pytorch", "ck"], default="quad")
parser.add_argument("--noflash", action="store_true", help="import noflash_attention (patches SDPA), as main.py does")
parser.add_argument("--res", default="128x128,1024x576,2560x1440", help="comma-separated WxH")
parser.add_argument("--txt-tokens", type=int, default=128)
parser.add_argument("--iters", type=int, default=2)
parser.add_argument("--breakdown", action="store_true", help="split one forward into int4 linear / attention / other (single GPU only)")
parser.add_argument("--split-device", default=None, help="apply Gfx906RowSplitKrea2 with this second GPU")
parser.add_argument("--check", action="store_true", help="max relative error of the (possibly split) forward vs a plain one")
cli = parser.parse_args()

import comfy.cli_args  # noqa: E402  (must precede bootstrap: attention is picked at comfy import)
comfy.cli_args.args.use_quad_cross_attention = cli.attention == "quad"
comfy.cli_args.args.use_split_cross_attention = cli.attention == "split"
comfy.cli_args.args.use_pytorch_cross_attention = cli.attention == "pytorch"
comfy.cli_args.args.use_ck_attention = cli.attention == "ck"
comfy.cli_args.args.force_fp16 = True
if cli.noflash:
    import noflash_attention  # noqa: F401,E402

import torch  # noqa: E402

import bootstrap  # noqa: E402
import pipeline  # noqa: E402
import comfy.model_management  # noqa: E402


def parse_res(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def make_inputs(model, w, h, txt_tokens, device, dtype):
    dm = model.model.diffusion_model
    x = torch.randn(1, dm.channels, h // 8, w // 8, device=device, dtype=dtype)
    t = torch.full((1,), 0.5, device=device, dtype=dtype)
    context = torch.randn(1, txt_tokens, dm.txtlayers * dm.txtdim, device=device, dtype=dtype)
    return x, t, context


def forward_ms(forward, iters):
    forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        forward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


class SyncTimer:
    """Wraps a function so each call is bracketed by device syncs and timed
    on the host. torch.profiler records no device activity on this ROCm
    build, and HIP event pairs overlapped (categories summed past the
    total), so this pays a sync per call instead -- noise at small sizes,
    accurate where the breakdown matters."""

    def __init__(self):
        self.totals = {}

    def wrap(self, name, fn):
        self.totals.setdefault(name, 0.0)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            self.totals[name] += (time.perf_counter() - t0) * 1000
            return out
        return timed

    def totals_ms(self):
        return self.totals


def breakdown(dm, forward):
    """Per-category device time of one forward: int4 linears, attention, rest."""
    import comfy.ldm.krea2.model as krea2
    int4_linear_cls = next(type(m) for m in dm.modules() if hasattr(m, "weight_q4"))
    int4_ops = sys.modules[int4_linear_cls.__module__]
    timer = SyncTimer()
    orig_attn, orig_linear = krea2.optimized_attention_masked, int4_ops.int4_linear
    krea2.optimized_attention_masked = timer.wrap("attention", orig_attn)
    int4_ops.int4_linear = timer.wrap("int4_linear", orig_linear)
    try:
        total_ms = forward_ms(forward, 1)
    finally:
        krea2.optimized_attention_masked, int4_ops.int4_linear = orig_attn, orig_linear
    parts = {k: round(v, 1) for k, v in timer.totals_ms().items()}
    parts["other"] = round(total_ms - sum(parts.values()), 1)
    return {"total_ms": round(total_ms, 1), **parts}


def main():
    bootstrap.ensure_custom_nodes()
    device = torch.device(cli.device)
    with torch.inference_mode():
        t0 = time.perf_counter()
        (model,) = pipeline.call_node("Gfx906Int4UnetLoader", unet_name=cli.unet, device=cli.device)
        comfy.model_management.load_models_gpu([model])
        load_s = time.perf_counter() - t0
        dm = model.model.diffusion_model
        dtype = model.model.get_dtype()
        print(json.dumps({"load_s": round(load_s, 1), "dtype": str(dtype), "attention": cli.attention,
                          "noflash": cli.noflash, "vram_gb": round(torch.cuda.memory_allocated(device) / 2**30, 2)}), flush=True)

        # The sampler hands the patcher's wrappers to the model via
        # transformer_options; do the same so wrapper-based nodes take effect.
        if cli.split_device:
            (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device=cli.split_device, min_image_tokens=0)
        options = {"wrappers": model.wrappers}

        for res in cli.res.split(","):
            w, h = parse_res(res)
            inputs = make_inputs(model, w, h, cli.txt_tokens, device, dtype)

            def forward():
                return dm(*inputs, transformer_options=options)

            torch.cuda.reset_peak_memory_stats(device)
            ms = forward_ms(forward, cli.iters)
            record = {"res": res, "img_tokens": (w // 16) * (h // 16), "step_ms": round(ms, 1),
                      "peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
            if cli.breakdown:
                record["breakdown_ms"] = breakdown(dm, forward)
            if cli.check:
                ref = dm(*inputs)
                out = forward()
                record["check_rel_err"] = float(f"{((out - ref).abs().max() / ref.abs().max()).item():.2e}")
            print(json.dumps(record), flush=True)
            del inputs


if __name__ == "__main__":
    main()
