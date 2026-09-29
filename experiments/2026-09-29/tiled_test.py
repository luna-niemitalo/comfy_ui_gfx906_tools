"""Tiled detail pass: refine the upscaled composition in overlapping ~1 MP
tiles so no attention call spans more token distance than the model was
trained on (RoPE only sees relative positions, so a tile's offset doesn't
matter). Every step: cut x into tiles, run them as ONE batch (the split
node's batch path would spread it over both GPUs), blend the denoised
predictions with feathered weights (MultiDiffusion), anchor the low band to
the composition (progressive.py's ILVR anchor).

Usage: tiled_test.py <variant> [...]  (see VARIANTS; "all")
Saves develop/out/progressive/tiled_<variant>.png; prints time + detail."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
import comfy.samplers  # noqa: E402
import pipeline  # noqa: E402

TARGET = (2560, 1440)
TILE_BATCH = 3  # tiles per model call: 6 x 4096 tokens in one call OOMed at 2560x1440 (16 GB)
FULL_STEPS = 8
# Host sync after every tile batch: with 5+ batches (3840x2160, 15 tiles) queued without one,
# ROCm 7.2's rocr::core::Runtime::AsyncEventsLoop segfaults on a released signal (gdb on the
# core dump); 2-3 batches never did. Syncing caps the queue at one batch and cost nothing
# measurable (4K step 58.2 s vs 59.2 s fully serialized). TILED_SYNC_PER_BATCH=0 disables it.
SYNC_PER_BATCH = os.environ.get("TILED_SYNC_PER_BATCH", "1") == "1"
# name: (tile px, overlap px, start, steps, anchor, release_from, release_to, upscale)
#   start float: composition runs all FULL_STEPS, detail pass re-noises to sigma `start`
#                with `steps` steps of its own schedule.
#   start "cN":  composition runs N of FULL_STEPS (clean estimate at step N), detail pass
#                continues the same FULL_STEPS schedule from step N (`steps` ignored).
#   start "cNn": as cN, but the detail pass keeps the composition's own noise at step N:
#                x_N is split into clean estimate (pixel-upscaled) and noise (white 2x
#                upsample, consistent with the pre-pass noise), recombined at sigma_N.
VARIANTS = {
    "t1024_o256_s0.5_anchor": (1024, 256, 0.5, 4, 1.0, 0.6, 1.0, "latent"),
    "t1024_o256_s0.35_anchor": (1024, 256, 0.35, 4, 1.0, 0.6, 1.0, "latent"),
    "t1024_o256_s0.5_anchor_pixel": (1024, 256, 0.5, 4, 1.0, 0.6, 1.0, "pixel"),
    "t1024_o256_s0.5_noanchor": (1024, 256, 0.5, 4, 0.0, 0.0, 0.0, "latent"),
    "c6_pixel_anchor": (1024, 256, "c6", None, 1.0, 0.0, 1.0, "pixel"),
    "c6_pixel_noanchor": (1024, 256, "c6", None, 0.0, 0.0, 0.0, "pixel"),
    "c6_esrgan_noanchor": (1024, 256, "c6", None, 0.0, 0.0, 0.0, "RealESRGAN_x4plus.pth"),
    "c6n_pixel_noanchor": (1024, 256, "c6n", None, 0.0, 0.0, 0.0, "pixel"),
    # Luna: only s0.5_anchor_pixel (finished composition, anchor held) had no duplicate
    # figures; c6 had the best texture. Finished composition + 2 steps, anchor on step 1 only.
    "f8_pixel_s0.5_2step_hold": (1024, 256, 0.5, 2, 1.0, 0.0, 0.01, "pixel"),
    "f8_esrgan_s0.5_2step_hold": (1024, 256, 0.5, 2, 1.0, 0.0, 0.01, "RealESRGAN_x4plus.pth"),
    "f8_pixel_s0.5_2step_hold_detailprompt": (1024, 256, 0.5, 2, 1.0, 0.0, 0.01, "pixel"),
}
# Tiles take the whole prompt as their own: "a lone traveller ... small in the lower left
# third" makes every tile draw one (Luna saw ghost duplicates). These variants give the
# tile pass a subject-free prompt (materials, palette, brushwork of prog.PROMPT only).
DETAIL_PROMPT_VARIANTS = {"f8_pixel_s0.5_2step_hold_detailprompt"}
DETAIL_PROMPT = ("A painterly landscape at dusk, stone, still water, mist, reeds and wet stones, layered storm "
                 "clouds, visible oil brushwork and impasto texture, canvas grain in flat areas, muted umber, "
                 "slate blue and rust palette.")


def tile_starts(size, tile, overlap):
    """Start offsets covering [0, size) with tiles of `tile` overlapping >= `overlap`."""
    if size <= tile:
        return [0]
    n = -(-(size - overlap) // (tile - overlap))
    # even offsets: Krea2 packs the latent into 2x2 patches
    return [2 * round(i * (size - tile) / (n - 1) / 2) for i in range(n)]


def feather(tile_h, tile_w, overlap, device):
    """Weights ramping linearly over `overlap` at each edge (never 0, so every pixel is covered)."""
    def ramp(n):
        r = torch.ones(n, device=device)
        k = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
        r[:overlap] = torch.minimum(r[:overlap], k)
        r[-overlap:] = torch.minimum(r[-overlap:], k.flip(0))
        return r
    return ramp(tile_h)[:, None] * ramp(tile_w)[None, :]


def tiled_sampler(ref_model_space, anchor, release, sigma_start, tile, overlap):
    """er_sde whose denoiser runs x in tiles (one batch) and blends; low band anchored."""
    def weight(sigma):
        progress = 1.0 - float(sigma) / sigma_start
        if progress <= release[0]:
            return anchor
        if progress >= release[1]:
            return 0.0
        return anchor * (release[1] - progress) / (release[1] - release[0])

    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        H, W = x.shape[-2:]
        # an axis shorter than a tile is one full-length tile (grow_test.py's early sizes)
        th, tw = min(tile, H), min(tile, W)
        ys, xs = tile_starts(H, th, overlap), tile_starts(W, tw, overlap)
        wts = feather(th, tw, overlap, x.device)

        def denoise(x, sigma, **kwargs):
            tiles = [x[..., y:y + th, c:c + tw] for y in ys for c in xs]
            d_parts = []
            for i in range(0, len(tiles), TILE_BATCH):
                d_parts.append(model(torch.cat(tiles[i:i + TILE_BATCH], 0),
                                     sigma.repeat(len(tiles[i:i + TILE_BATCH])), **kwargs))
                if SYNC_PER_BATCH:
                    torch.cuda.synchronize()
            d_tiles = torch.cat(d_parts, 0)
            acc = torch.zeros_like(x, dtype=torch.float32)
            norm = torch.zeros(x.shape[-2:], device=x.device)
            for i, (y, c) in enumerate((y, c) for y in ys for c in xs):
                acc[..., y:y + th, c:c + tw] += d_tiles[i:i + 1].float() * wts
                norm[y:y + th, c:c + tw] += wts
            d = (acc / norm).to(x.dtype)
            w = weight(sigma.flatten()[0])
            if w == 0.0:
                return d
            low = prog.resize(d.float(), ref_model_space.shape[-2:], "area")
            return d + w * prog.resize(ref_model_space - low, d.shape[-2:], "bicubic").to(d.dtype)
        denoise.inner_model = model.inner_model
        return kds.sample_er_sde(denoise, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    return comfy.samplers.KSAMPLER(sample)


def pixel_upscale(latent, vae, w, h, upscale_model=None):
    """Decode, upscale the image (Lanczos, or a model upscaler then Lanczos down to
    the target), encode: an on-manifold upscaled latent."""
    (img,) = pipeline.call_node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=512, overlap=64,
                                temporal_size=64, temporal_overlap=8)
    if upscale_model is not None:
        (model,) = pipeline.call_node("UpscaleModelLoader", model_name=upscale_model)
        (img,) = pipeline.call_node("ImageUpscaleWithModel", upscale_model=model, image=img)
    (img,) = pipeline.call_node("ImageScale", image=img, upscale_method="lanczos", width=w, height=h, crop="disabled")
    (enc,) = pipeline.call_node("VAEEncodeTiled", pixels=img, vae=vae, tile_size=512, overlap=64,
                                temporal_size=64, temporal_overlap=8)
    return enc["samples"]


def white_upsample_2x(eps, generator):
    """2x nearest-style noise upsample that stays white N(0,1): each child is
    parent/2 + (z - mean of its 2x2 block), so children are uncorrelated with
    variance 1/4 + 3/4, and each block's sum / 2 is the parent again."""
    *lead, h, w = eps.shape
    z = torch.randn(*lead, h, 2, w, 2, generator=generator).to(eps)
    z = z - z.mean(dim=(-3, -1), keepdim=True)
    return (eps[..., :, None, :, None] / 2 + z).reshape(*lead, 2 * h, 2 * w)


if __name__ == "__main__":
    names = list(VARIANTS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        (guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=positive)
        (detail_positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=DETAIL_PROMPT)
        (detail_guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=detail_positive)
        (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
        full_sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", FULL_STEPS)
        compositions = {}  # (composition steps, keep noise) -> (clean latent dict, noise or None, seconds)
        latent_in = lambda x: model.model.process_latent_in(x.to(model.load_device)).float()  # noqa: E731
        latent_out = model.model.process_latent_out

        def composition(end, keep_noise):
            if (end, keep_noise) in compositions:
                return compositions[(end, keep_noise)]
            t0 = time.perf_counter()
            # leftover "disable" = force full denoise: the last of `end` steps jumps to sigma 0
            (native,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable",
                                           noise_seed=prog.SEED, steps=FULL_STEPS, cfg=1.0, sampler_name="er_sde",
                                           scheduler="simple", positive=positive, negative=positive,
                                           latent_image=latent, start_at_step=0, end_at_step=end,
                                           return_with_leftover_noise="enable" if keep_noise else "disable")
            eps = None
            if keep_noise:
                # comfy's flow sampling exports leftover noise as x / (1 - sigma) (inverse_noise_scaling)
                sigma = float(full_sigmas[end])
                x_t = (1 - sigma) * latent_in(native["samples"])
                (euler,) = pipeline.call_node("KSamplerSelect", sampler_name="euler")
                (off,) = pipeline.call_node("DisableNoise")
                (native, _) = pipeline.call_node("SamplerCustomAdvanced", noise=off, guider=guider, sampler=euler,
                                                 sigmas=torch.tensor([sigma, 0.0]), latent_image=native)
                eps = (x_t - (1 - sigma) * latent_in(native["samples"])) / sigma
                print(f"composition noise at sigma {sigma:.3f}: std {eps.std().item():.3f}", flush=True)
            compositions[(end, keep_noise)] = (native, eps, time.perf_counter() - t0)
            return compositions[(end, keep_noise)]

        w, h = TARGET
        for name in names:
            tile, overlap, start, steps, anchor, r0, r1, upscale = VARIANTS[name]
            continues = isinstance(start, str)
            keep_noise = continues and start.endswith("n")
            end = int(start[1:].rstrip("n")) if continues else FULL_STEPS
            sigmas = full_sigmas[end:] if continues else prog.stage_sigmas(model, start, steps)
            native, eps, native_s = composition(end, keep_noise)
            ref = native["samples"]
            ref_model = latent_in(ref)
            t1 = time.perf_counter()
            if upscale == "latent":
                up = prog.resize(ref, (h // 8, w // 8), "bicubic")
            else:
                up = pixel_upscale(native, vae, w, h, None if upscale == "pixel" else upscale)
            (noise,) = pipeline.call_node("RandomNoise", noise_seed=prog.SEED + 1)
            if keep_noise:
                sigma = float(sigmas[0])
                eps_up = white_upsample_2x(eps, torch.Generator().manual_seed(prog.SEED + 1))
                x_t = (1 - sigma) * latent_in(up) + sigma * eps_up
                up = latent_out(x_t / (1 - sigma))
                (noise,) = pipeline.call_node("DisableNoise")
            sampler = tiled_sampler(ref_model, anchor, (r0, r1), float(sigmas[0]), tile // 8, overlap // 8)
            (out, _) = pipeline.call_node("SamplerCustomAdvanced", noise=noise, sampler=sampler,
                                          guider=detail_guider if name in DETAIL_PROMPT_VARIANTS else guider,
                                          sigmas=sigmas, latent_image={"samples": up})
            refine_s = time.perf_counter() - t1
            (images,) = pipeline.call_node("VAEDecodeTiled", samples=out, vae=vae, tile_size=512, overlap=64,
                                           temporal_size=64, temporal_overlap=8)
            arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(prog.OUT, f"tiled_{name}.png"))
            print(json.dumps({"variant": name, "native_s": round(native_s, 1), "refine_s": round(refine_s, 1),
                              "detail": round(prog.detail(arr), 2)}), flush=True)
