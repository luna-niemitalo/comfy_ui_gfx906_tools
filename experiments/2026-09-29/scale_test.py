"""How far does resolution growth hold up? (Luna: consistent 720p -> 2160p; find where it
falls apart, report between every step.) One growth run from the 1280x720 composition to
TARGET with ~1.2x per step (the 4K run's stride), through the ResolutionGrowth node's own
grow(). After every step: decode, save scale_<k>_<w>x<h>.png, print one JSON line:
  drift_db     PSNR of the step image shrunk to 1280x720 vs the composition (layout / colour hold)
  change_db    PSNR of the step image shrunk to the previous size vs the previous image
  noise        Immerkaer single-image noise sigma (8-bit units; also counts fine texture)
  detail       mean |Laplacian| per pixel (falls with resolution by itself)
  detail_720   the same on the image shrunk to 1280x720 (content structure at a fixed scale)
Ends with scale_metrics.png (all against output megapixels).

Usage: scale_test.py [target_w target_h steps]   (default 7680 4320 10)"""
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
TARGET_W, TARGET_H, STEPS = (int(a) for a in sys.argv[1:4]) if len(sys.argv) == 4 else (7680, 4320, 10)
OUT = os.path.join(prog.OUT, "scale")
os.makedirs(OUT, exist_ok=True)


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
rows = []
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
    Image.fromarray(comp).save(os.path.join(OUT, "scale_0_1280x720.png"))
    state = {"prev": comp, "t": time.perf_counter()}

    def report(k, size, sigma, x0):
        step_s = time.perf_counter() - state["t"]
        img = decode(vae, x0)
        Image.fromarray(img).save(os.path.join(OUT, f"scale_{k + 1}_{size[0]}x{size[1]}.png"))
        prev = state["prev"]
        row = {"step": k + 1, "size": f"{size[0]}x{size[1]}", "mp": round(size[0] * size[1] / 1e6, 2),
               "sigma": round(sigma, 3), "step_s": round(step_s, 1),
               "drift_db": round(psnr(shrink(img, prog.NATIVE), comp), 2),
               "change_db": round(psnr(shrink(img, prev.shape[1::-1]), prev), 2),
               "noise": round(noise_sigma(img), 2), "detail": round(prog.detail(img), 2),
               "detail_720": round(prog.detail(shrink(img, prog.NATIVE)), 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        state["prev"] = img
        state["t"] = time.perf_counter()

    growth.grow(model, detail_positive, positive, vae, composition["samples"], (TARGET_W, TARGET_H), STEPS,
                sigma_start=0.5, size_shape=1.0, anchor=1.0, anchor_shape=2.0, cfg=1.0, seed=prog.SEED + 1,
                tile_px=1024, overlap_px=256, tile_batch=4, upscale_method="lanczos", upscale_model=None,
                on_step=report)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
mp = [r["mp"] for r in rows]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, keys, title in ((axes[0, 0], ["drift_db"], "drift vs composition (PSNR at 1280x720, dB)"),
                        (axes[0, 1], ["change_db"], "change per step (PSNR at previous size, dB)"),
                        (axes[1, 0], ["noise"], "noise estimate (Immerkaer sigma, 8-bit)"),
                        (axes[1, 1], ["detail", "detail_720"], "detail (mean |Laplacian|)")):
    for key in keys:
        ax.plot(mp, [r[key] for r in rows], marker="o", label=key)
    ax.set_title(title)
    ax.set_xlabel("output megapixels")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"Resolution growth 1280x720 -> {TARGET_W}x{TARGET_H}, {STEPS} steps (Krea2 int4, 2 GPUs)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "scale_metrics.png"), dpi=110)
print(json.dumps({"done": True, "plot": os.path.join(OUT, "scale_metrics.png")}), flush=True)
