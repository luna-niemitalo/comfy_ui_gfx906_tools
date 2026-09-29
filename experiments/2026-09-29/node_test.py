"""ResolutionGrowth node (custom_nodes/ComfyUI-ResolutionGrowth) end to end: same composition,
schedule and noise seed as grow_test.py n6_3840 --split, so the result should match
develop/out/progressive/grow_n6_3840_split.png closely (the node lets comfy apply the
noise scaling that grow_test did by hand). Saves node_n6_3840.png + node_schedule.png."""
import asyncio, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402
import tiled_test as tt  # noqa: E402

import nodes  # noqa: E402
# progressive.py loads its own custom-node whitelist at import; add the node under test
asyncio.run(nodes.load_custom_node(os.path.join(os.path.dirname(nodes.__file__), "custom_nodes",
                                                "ComfyUI-ResolutionGrowth")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

TARGET = (3840, 2160)


def save(images, name):
    arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(prog.OUT, name))
    return arr


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
    t0 = time.perf_counter()
    (composition,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                        steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                        positive=positive, negative=positive, latent_image=latent,
                                        start_at_step=0, end_at_step=8, return_with_leftover_noise="disable")
    compose_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    (grown, schedule) = pipeline.call_node(
        "ResolutionGrowth", model=model, positive=positive, negative=positive, vae=vae, latent=composition,
        width=TARGET[0], height=TARGET[1], growth_steps=6, sigma_start=0.5, size_shape=1.0, anchor=1.0,
        anchor_shape=2.0, cfg=1.0, seed=prog.SEED + 1, tile_size=1024, tile_overlap=256, tile_batch=4,
        upscale_method="lanczos", detail_positive=detail)
    grow_s = time.perf_counter() - t0
    save(schedule, "node_schedule.png")
    (images,) = pipeline.call_node("VAEDecodeTiled", samples=grown, vae=vae, tile_size=512, overlap=64,
                                   temporal_size=64, temporal_overlap=8)
    arr = save(images, "node_n6_3840.png")
    ref = np.asarray(Image.open(os.path.join(prog.OUT, "grow_n6_3840_split.png"))).astype(np.float32)
    diff = np.abs(arr.astype(np.float32) - ref)
    psnr = 10 * np.log10(255 ** 2 / max(float((diff ** 2).mean()), 1e-9))
    print(json.dumps({"compose_s": round(compose_s, 1), "grow_s": round(grow_s, 1),
                      "detail": round(prog.detail(arr), 2), "vs_grow_test_psnr_db": round(psnr, 1),
                      "vs_grow_test_mean_abs": round(float(diff.mean()), 2)}), flush=True)
