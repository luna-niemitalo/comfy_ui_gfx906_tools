"""2560x1440, same seed, through the real Gfx906FusedAttention node: int4/u8
dense vs block-sparse (tau 0.995). Saves PNGs to develop/out/block_sparse and
prints pixel differences and wall time per render."""
import json, os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
comfy.cli_args.args.whitelist_custom_nodes = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU"]
import logging  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

SCRATCH = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(SCRATCH, "capture_qkv.py")).read().split('PROMPT = (')[1].split(')\n')[0]
PROMPT = eval("(" + PROMPT + ")")
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/block_sparse"
os.makedirs(OUT, exist_ok=True)
VARIANTS = [("dense", 0.0), ("sparse_0.995", 0.995)]

logging.getLogger().setLevel(logging.INFO)
bootstrap.ensure_custom_nodes()
resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": PROMPT, "latent": {"width": 2560, "height": 1440}})
images = {}
with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    for name, tau in VARIANTS:
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=tau, max_missed=0.02)
        t0 = time.perf_counter()
        img = pipeline.run_with_loaders(resolved, model, clip, vae)
        seconds = time.perf_counter() - t0
        images[name] = (img[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(images[name]).save(os.path.join(OUT, f"{name}.png"))
        print(json.dumps({"rendered": name, "seconds": round(seconds, 1)}), flush=True)
base = images["dense"].astype(np.float32)
for name in images:
    diff = np.abs(images[name].astype(np.float32) - base)
    print(json.dumps({"variant": name, "mean_abs_diff": round(float(diff.mean()), 2),
                      "p99_abs_diff": round(float(np.percentile(diff, 99)), 1),
                      "psnr_db": round(float(10 * np.log10(255 ** 2 / max((diff ** 2).mean(), 1e-9))), 1)}), flush=True)
