"""How far does resolution growth hold up, and how do growth schedules compare? (Luna:
consistent 720p -> 2160p; find where it falls apart, report between every step.)
Each run grows the same 1280x720 composition through the ResolutionGrowth node's own grow().
After every step: decode, save <label>/step_<k>_<w>x<h>.png, print one JSON line:
  drift_db     PSNR of the step image shrunk to 1280x720 vs the composition (layout / colour hold)
  change_db    PSNR of the step image shrunk to the previous size vs the previous image
  noise        Immerkaer single-image noise sigma (8-bit units; also counts fine texture)
  noise_720    the same on the image shrunk to 1280x720 (Luna's stop criterion, default < 1)
  detail       mean |Laplacian| per pixel (falls with resolution by itself)
  detail_720   the same on the image shrunk to 1280x720 (content structure at a fixed scale)
Ends with metrics_<labels>.png: every run's curves against output megapixels, overlaid.

Usage: scale_test.py [mode:WxH:n ...]   mode geometric (n = steps) or linear (n = step px)
       default: geometric:7680x4320:10 (the 8K study)"""
import asyncio, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402
import tiled_test as tt  # noqa: E402

import nodes  # noqa: E402
NODE_DIR = os.path.join(os.path.dirname(nodes.__file__), "custom_nodes", "ComfyUI-ResolutionGrowth")
asyncio.run(nodes.load_custom_node(NODE_DIR))  # progressive.py's whitelist excludes it

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

growth = sys.modules[next(name for name in sys.modules if name.endswith("ComfyUI-ResolutionGrowth.growth"))]
OUT = os.path.join(prog.OUT, "scale")


def parse_run(spec):
    mode, size, n = spec.split(":")
    if mode not in ("geometric", "linear"):
        sys.exit(f"run spec {spec!r}: mode must be geometric|linear, got {mode!r}")
    w, h = (int(v) for v in size.split("x"))
    return {"label": spec.replace(":", "_"), "mode": mode, "target": (w, h), "n": int(n)}


RUNS = [parse_run(s) for s in (sys.argv[1:] or ["geometric:7680x4320:10"])]


def shrink(arr, size):
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 10 * math.log10(255 ** 2 / max(mse, 1e-9))


def noise_sigma(arr):
    """Immerkaer (1996): sigma from the |response| of a Laplacian-difference mask on luma."""
    y = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r = (y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:] - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1]
         - 2 * y[1:-1, 2:] + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:])
    return float(math.sqrt(math.pi / 2) * np.abs(r).mean() / 6)


def decode(vae, samples):
    with growth.no_miopen():  # every step is a new size
        (images,) = pipeline.call_node("VAEDecodeTiled", samples={"samples": samples}, vae=vae, tile_size=512,
                                       overlap=64, temporal_size=64, temporal_overlap=8)
    return (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                             "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                  sparse_tau=0.0, max_missed=0.02)
    (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device="auto", min_image_tokens=2048)
    (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
    (detail_positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=tt.DETAIL_PROMPT)
    (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
    (composition,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                        steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                        positive=positive, negative=positive, latent_image=latent,
                                        start_at_step=0, end_at_step=8, return_with_leftover_noise="disable")
    comp = decode(vae, composition["samples"])

    for run in RUNS:
        run_dir = os.path.join(OUT, run["label"])
        os.makedirs(run_dir, exist_ok=True)
        run["rows"] = []
        state = {"prev": comp, "t": time.perf_counter(), "total": 0.0}

        def report(k, size, sigma, x0, run=run, run_dir=run_dir, state=state):
            step_s = time.perf_counter() - state["t"]
            state["total"] += step_s
            img = decode(vae, x0)
            Image.fromarray(img).save(os.path.join(run_dir, f"step_{k + 1}_{size[0]}x{size[1]}.png"))
            small = shrink(img, prog.NATIVE)
            row = {"run": run["label"], "step": k + 1, "size": f"{size[0]}x{size[1]}",
                   "mp": round(size[0] * size[1] / 1e6, 2), "sigma": round(sigma, 3), "step_s": round(step_s, 1),
                   "total_s": round(state["total"], 1), "drift_db": round(psnr(small, comp), 2),
                   "change_db": round(psnr(shrink(img, state["prev"].shape[1::-1]), state["prev"]), 2),
                   "noise": round(noise_sigma(img), 2), "noise_720": round(noise_sigma(small), 2),
                   "detail": round(prog.detail(img), 2), "detail_720": round(prog.detail(small), 2)}
            run["rows"].append(row)
            print(json.dumps(row), flush=True)
            state["prev"] = img
            state["t"] = time.perf_counter()

        growth.grow(model, detail_positive, positive, vae, composition["samples"], run["target"], run["mode"],
                    run["n"], run["n"], sigma_start=0.5, size_shape=1.0, anchor=1.0, anchor_shape=2.0, cfg=1.0,
                    seed=prog.SEED + 1, tile_px=1024, overlap_px=256, tile_batch=4, upscale_method="lanczos",
                    upscale_model=None, on_step=report)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
panels = (("drift_db", "drift vs composition (PSNR at 1280x720, dB)"),
          ("change_db", "change per step (PSNR at previous size, dB)"),
          ("total_s", "cumulative growth time (s)"),
          ("noise", "noise, full res (Immerkaer sigma, 8-bit)"),
          ("noise_720", "noise at 1280x720 (stop criterion)"),
          ("detail_720", "detail at 1280x720 (mean |Laplacian|)"))
for ax, (key, title) in zip(axes.flat, panels):
    for run in RUNS:
        ax.plot([r["mp"] for r in run["rows"]], [r[key] for r in run["rows"]], marker="o", label=run["label"])
    ax.set_title(title)
    ax.set_xlabel("output megapixels")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Resolution growth from 1280x720 (Krea2 int4, 2 GPUs)")
fig.tight_layout()
plot = os.path.join(OUT, f"metrics_{'__'.join(r['label'] for r in RUNS)}.png")
fig.savefig(plot, dpi=110)
print(json.dumps({"done": True, "plot": plot}), flush=True)
