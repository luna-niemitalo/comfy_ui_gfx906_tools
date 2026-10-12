"""Persistent block-sparse calibration (Gfx906FusedAttention calibration="persistent"):
does it pay, and does it hold quality? 2560x1440, 8 steps, int4/u8, one GPU.

Sequence (ComfyUI's user dir pointed at a fresh scratch dir, so the real cache is untouched):
  learn_1..3   persistent, sparse_tau 0.995, seeds 1..3 (dense while learning, + measurement)
  then alternating, seed 0: dense, cached, dense, cached, dense, cached, per_run
Alternating because thermal drift is +-3% (single runs lie). Prints seconds per render and
PSNR of each seed-0 render vs the first dense one; saves PNGs to develop/out/calib_cache/.

Usage: calib_cache_bench.py   (run with HIP_VISIBLE_DEVICES to pick the GPU)"""
import json, os, shutil, sys, time
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
import folder_paths  # noqa: E402
import pipeline  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = eval("(" + open(os.path.join(HERE, "capture_qkv.py")).read().split('PROMPT = (')[1].split(')\n')[0] + ")")
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/calib_cache"
USER_DIR = os.path.join(OUT, "user")
shutil.rmtree(USER_DIR, ignore_errors=True)
os.makedirs(USER_DIR)
folder_paths.set_user_directory(USER_DIR)
SEQUENCE = ([(f"learn_{s}", "persistent", 0.995, s) for s in (1, 2, 3)]
            + [(f"{kind}_{i}", mode, tau, 0) for i in (1, 2, 3)
               for kind, mode, tau in (("dense", "persistent", 0.0), ("cached", "persistent", 0.995))]
            + [("per_run", "per_run", 0.995, 0)])

logging.getLogger().setLevel(logging.INFO)
bootstrap.ensure_custom_nodes()
resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": PROMPT, "latent": {"width": 2560, "height": 1440}})
seed_key = next(k for k in ("seed", "noise_seed") if k in resolved["sampler"])
base_seed = resolved["sampler"][seed_key]
reference = None


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return round(float(10 * np.log10(255 ** 2 / max(mse, 1e-9))), 1)


with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    for name, mode, tau, seed in SEQUENCE:
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=tau, max_missed=0.02, calibration=mode)
        resolved["sampler"][seed_key] = base_seed + seed
        t0 = time.perf_counter()
        img = pipeline.run_with_loaders(resolved, model, clip, vae)
        seconds = time.perf_counter() - t0
        arr = (img[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT, f"{name}.png"))
        row = {"render": name, "seconds": round(seconds, 1)}
        if seed == 0:
            reference = arr if reference is None else reference
            row["psnr_vs_dense_1"] = psnr(arr, reference)
        print(json.dumps(row), flush=True)
print(json.dumps({"cache_file": os.listdir(os.path.join(USER_DIR, "__gfx906_int4"))}), flush=True)
