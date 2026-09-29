"""ResolutionGrowth nodes (custom_nodes/ComfyUI-ResolutionGrowth) end to end, headless:
  fixed        "Resolution Growth", geometric 6 steps to FIXED_TARGET
  progressive  "Resolution Growth (Progressive)", output PROGRESSIVE_OUT, with a simulated Stop
               press (stop.py's request set, as the web button's POST would) after STOP_AFTER_S
Saves node_<name>.png (+ schedule plots) to develop/out/progressive/nodes/, prints one JSON
line per node; the progressive report is printed as it streams.

Usage: node_test.py [fixed] [progressive]   (default both)"""
import asyncio, json, os, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402
import tiled_test as tt  # noqa: E402

import nodes  # noqa: E402
# progressive.py loads its own custom-node whitelist at import; add the nodes under test
asyncio.run(nodes.load_custom_node(os.path.join(os.path.dirname(nodes.__file__), "custom_nodes",
                                                "ComfyUI-ResolutionGrowth")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

stop = sys.modules[next(name for name in sys.modules if name.endswith("ComfyUI-ResolutionGrowth.stop"))]
FIXED_TARGET = (2560, 1440)
PROGRESSIVE_OUT = (2560, 1440)
STOP_AFTER_S = float(os.environ.get("NODE_TEST_STOP_AFTER_S", "45"))  # 0: no Stop, run to convergence
OUT = os.path.join(prog.OUT, "nodes")
os.makedirs(OUT, exist_ok=True)
WHICH = sys.argv[1:] or ["fixed", "progressive"]


def save(images, name):
    arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, name))
    return arr


GROWTH = dict(sigma_start=0.5, sigma_end=0.0, anchor=1.0, anchor_shape=2.0, cfg=1.0, seed=prog.SEED + 1,
              tile_size=1024, tile_overlap=256, tile_batch=4, upscale_method="lanczos", upscale_path="vae")

resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                             "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                  sparse_tau=0.0, max_missed=0.02)
    (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device="auto", min_image_tokens=2048)
    (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
    (detail,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=tt.DETAIL_PROMPT)
    (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
    (composition,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                        steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                        positive=positive, negative=positive, latent_image=latent,
                                        start_at_step=0, end_at_step=8, return_with_leftover_noise="disable")
    common = dict(model=model, positive=positive, negative=positive, vae=vae, latent=composition,
                  detail_positive=detail, **GROWTH)

    if "fixed" in WHICH:
        t0 = time.perf_counter()
        (grown, schedule) = pipeline.call_node(
            "ResolutionGrowth", width=FIXED_TARGET[0], height=FIXED_TARGET[1], growth_mode="geometric",
            growth_steps=6, step_px=256, size_shape=1.0, oversample=1.0, **common)
        seconds = time.perf_counter() - t0
        save(schedule, "fixed_schedule.png")
        (images,) = pipeline.call_node("VAEDecodeTiled", samples=grown, vae=vae, tile_size=512, overlap=64,
                                       temporal_size=64, temporal_overlap=8)
        arr = save(images, "fixed.png")
        print(json.dumps({"node": "fixed", "size": list(arr.shape[1::-1]), "seconds": round(seconds, 1),
                          "detail": round(prog.detail(arr), 2)}), flush=True)

    if "progressive" in WHICH:
        timer = threading.Timer(STOP_AFTER_S, lambda: stop._requests.add("7"))
        t0 = time.perf_counter()
        if STOP_AFTER_S > 0:
            timer.start()
        (image, grown, schedule, report) = pipeline.call_node(
            "ResolutionGrowthProgressive", width=PROGRESSIVE_OUT[0], height=PROGRESSIVE_OUT[1], step_px=256,
            max_oversample=1.5, converge_db=35.0, unique_id="7", **common)
        timer.cancel()
        seconds = time.perf_counter() - t0
        save(schedule, "progressive_schedule.png")
        arr = save(image, "progressive.png")
        print(json.dumps({"node": "progressive", "size": list(arr.shape[1::-1]), "latent": list(grown["samples"].shape),
                          "seconds": round(seconds, 1), "last_report_line": report.splitlines()[-1],
                          "stop_requests_left": sorted(stop._requests)}), flush=True)
