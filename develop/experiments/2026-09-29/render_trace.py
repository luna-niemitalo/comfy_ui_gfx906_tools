"""Where does the splotchiness come from: same prompt + seed, resolution x
attention path. Variants: quad (ComfyUI backend, reference), fused fp16/fp16
(kernel path, full precision), fused int8/u8, fused int4/u8 with and without
the P.V tile skip. Saves PNGs to develop/out/attn_trace/<WxH>_<variant>.png and
prints a fine-detail measure (mean |Laplacian| of luma; lower = smoother /
splotchier) per image.

Usage: render_trace.py WxH [WxH ...]
"""
import json, os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
comfy.cli_args.args.whitelist_custom_nodes = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

SCRATCH = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(SCRATCH, "capture_qkv.py")).read().split('PROMPT = (')[1].split(')\n')[0]
PROMPT = eval("(" + PROMPT + ")")
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/attn_trace"
os.makedirs(OUT, exist_ok=True)
# (name, None = no node, else Gfx906FusedAttention kwargs)
VARIANTS = [("quad", None),
            ("fused_fp16_fp16", dict(qk="fp16", pv="fp16", skip_below=0.0)),
            ("fused_int8_u8", dict(qk="int8", pv="u8", skip_below=0.0)),
            ("fused_int4_u8_noskip", dict(qk="int4", pv="u8", skip_below=0.0)),
            ("fused_int4_u8", dict(qk="int4", pv="u8", skip_below=1e-4))]


def detail(img):
    luma = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = 4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    return float(np.abs(lap).mean())


bootstrap.ensure_custom_nodes()
for res in sys.argv[1:]:
    w, h = (int(x) for x in res.split("x"))
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": PROMPT, "latent": {"width": w, "height": h}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        for name, node in VARIANTS:
            model = base_model
            if node is not None:
                (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, sparse_tau=0.0,
                                              max_missed=0.02, **node)
            t0 = time.perf_counter()
            img = pipeline.run_with_loaders(resolved, model, clip, vae)
            arr = (img[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(OUT, f"{res}_{name}.png"))
            print(json.dumps({"res": res, "variant": name, "seconds": round(time.perf_counter() - t0, 1),
                              "detail": round(detail(arr), 2)}), flush=True)
        del base_model, clip, vae
