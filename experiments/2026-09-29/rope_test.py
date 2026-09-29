"""RoPE position interpolation against the mirroring/afterimage artifacts at
2560x1440 (positions beyond the ~1 MP training range; Krea2 RoPE theta 1000).
Krea2's `post_input` patch rescales the image position ids (rows, cols) by
  pi   : s = sqrt(NATIVE_TOKENS / image tokens)            (constant)
  dype : s ** (sigma ** DYPE_POWER)  -- full interpolation at high noise,
         true positions as sigma -> 0 (DyPE-style schedule)
Variants: direct 8-step render at the target, or the progressive.py
hires_s0.5_anchor refinement from a 1280x720 composition.

Usage: rope_test.py <variant> [...]  (see VARIANTS; "all")
Saves develop/out/progressive/rope_<variant>.png; prints time + detail."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402  (bootstrap, loaders, helpers; its main loop is guarded)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

NATIVE_TOKENS = 64 * 64  # 1024x1024 / (8 * 2)^2
DYPE_POWER = 2.0
TARGET = (2560, 1440)
VARIANTS = {
    "direct_pi": ("direct", "pi"),
    "direct_dype": ("direct", "dype"),
    "hires_anchor_pi": ("hires", "pi"),
    "hires_anchor_dype": ("hires", "dype"),
}


def rope_patch(mode):
    def patch(args):
        ids = args["img_ids"]
        tokens = ids.shape[1]
        s = min(1.0, math.sqrt(NATIVE_TOKENS / tokens))
        if mode == "dype":
            sigma = float(args["transformer_options"]["sigmas"].flatten()[0])
            s = s ** (sigma ** DYPE_POWER)
        if s < 1.0:
            ids = ids.clone()
            ids[..., 1:] *= s
        return {**args, "img_ids": ids}
    return patch


names = list(VARIANTS) if sys.argv[1:] == ["all"] else sys.argv[1:]
resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                             "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    (fused,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                  sparse_tau=0.0, max_missed=0.02)
    (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
    for name in names:
        kind, mode = VARIANTS[name]
        model = fused.clone()
        model.set_model_patch(rope_patch(mode), "post_input")
        t0 = time.perf_counter()
        if kind == "direct":
            (latent,) = pipeline.call_node("EmptyLatentImage", width=TARGET[0], height=TARGET[1], batch_size=1)
            (latent,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                           steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                           positive=positive, negative=positive, latent_image=latent, start_at_step=0,
                                           end_at_step=8, return_with_leftover_noise="disable")
        else:
            # composition at native size without the patch (in range anyway), then the patched refinement
            (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
            (latent,) = pipeline.call_node("KSamplerAdvanced", model=fused, add_noise="enable", noise_seed=prog.SEED,
                                           steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                           positive=positive, negative=positive, latent_image=latent, start_at_step=0,
                                           end_at_step=8, return_with_leftover_noise="disable")
            w, h, sigma_start, steps, anchor, r0, r1 = prog.VARIANTS["hires_s0.5_anchor"][0]
            ref = latent["samples"]
            up = prog.resize(ref, (h // 8, w // 8), "bicubic")
            ref_model = model.model.process_latent_in(ref.to(model.load_device)).float()
            (noise,) = pipeline.call_node("RandomNoise", noise_seed=prog.SEED + 1)
            (guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=positive)
            (latent, _) = pipeline.call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                                             sampler=prog.anchored_sampler(ref_model, anchor, (r0, r1), sigma_start),
                                             sigmas=prog.stage_sigmas(model, sigma_start, steps),
                                             latent_image={"samples": up})
        (images,) = pipeline.call_node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=512, overlap=64,
                                       temporal_size=64, temporal_overlap=8)
        arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(prog.OUT, f"rope_{name}.png"))
        print(json.dumps({"variant": name, "total_s": round(time.perf_counter() - t0, 1),
                          "detail": round(prog.detail(arr), 2)}), flush=True)
