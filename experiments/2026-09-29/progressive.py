"""Progressive upscale prototype: compose at native size, then refine up to
the target in stages. Each stage: upscale the previous result, re-noise to
sigma_start, run `steps` steps, and anchor the low-frequency band of every
denoised prediction to the previous stage's image with strength
w(progress) = anchor until release[0], linear to 0 at release[1]
(ILVR-style: x0 <- x0 + w * up(ref - down(x0))) -- the model fills in detail
with full attention but can't recompose.

Usage: progressive.py <variant> [...]   (names from VARIANTS; "all")
Saves develop/out/progressive/<variant>.png; prints time + detail."""
import json, os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
comfy.cli_args.args.whitelist_custom_nodes = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
import comfy.samplers  # noqa: E402
import pipeline  # noqa: E402

SCRATCH = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(SCRATCH, "capture_qkv.py")).read().split('PROMPT = (')[1].split(')\n')[0]
PROMPT = eval("(" + PROMPT + ")")
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/progressive"
os.makedirs(OUT, exist_ok=True)
SEED = 914872876216533
NATIVE = (1280, 720)
# stage: (width, height, sigma_start, steps, anchor, release_from, release_to)
VARIANTS = {
    "native": [],
    "hires_s0.5": [(2560, 1440, 0.5, 4, 0.0, 0.0, 0.0)],
    "hires_s0.5_anchor": [(2560, 1440, 0.5, 4, 1.0, 0.6, 1.0)],
    "hires_s0.7_anchor": [(2560, 1440, 0.7, 5, 1.0, 0.5, 0.9)],
    "progressive_2stage": [(1792, 1008, 0.55, 3, 1.0, 0.5, 1.0), (2560, 1440, 0.45, 3, 1.0, 0.5, 1.0)],
}


def detail(img):
    luma = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = 4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    return float(np.abs(lap).mean())


def resize(x, hw, mode):
    """Spatial resize of a 4-D or 5-D ([B, C, T, H, W], Qwen-Image VAE) latent."""
    lead = x.shape[:-2]
    y = F.interpolate(x.reshape(-1, 1, *x.shape[-2:]).float(), size=hw, mode=mode)
    return y.reshape(*lead, *hw).to(x.dtype)


def stage_sigmas(model, sigma_start, steps):
    """steps + 1 sigmas from sigma_start to 0, taken evenly (in index) from the
    model's own 'simple' schedule so the spacing follows its shift."""
    full = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", 1000)
    start = int((full > sigma_start).sum())
    idx = torch.linspace(start, len(full) - 1, steps + 1).round().long()
    return full[idx]


def anchored_sampler(ref_model_space, anchor, release, sigma_start):
    """KSAMPLER around er_sde whose denoiser output has its low band pulled
    to ref (model latent space, lower resolution)."""
    def weight(sigma):
        progress = 1.0 - float(sigma) / sigma_start
        if progress <= release[0]:
            return anchor
        if progress >= release[1]:
            return 0.0
        return anchor * (release[1] - progress) / (release[1] - release[0])

    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        def denoise(x, sigma, **kwargs):
            d = model(x, sigma, **kwargs)
            w = weight(sigma.flatten()[0])
            if w == 0.0:
                return d
            low = resize(d.float(), ref_model_space.shape[-2:], "area")
            return d + w * resize(ref_model_space - low, d.shape[-2:], "bicubic").to(d.dtype)
        denoise.inner_model = model.inner_model  # er_sde reads model_sampling through it
        return kds.sample_er_sde(denoise, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    return comfy.samplers.KSAMPLER(sample)


bootstrap.ensure_custom_nodes()
names = list(VARIANTS) if sys.argv[1:] == ["all"] else sys.argv[1:]
resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": PROMPT,
                             "latent": {"width": NATIVE[0], "height": NATIVE[1]}})
with torch.inference_mode():
    base_model, clip, vae = pipeline.build_loaders(resolved)
    (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                  sparse_tau=0.0, max_missed=0.02)
    (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=PROMPT)
    (guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=positive)
    for name in names:
        t0 = time.perf_counter()
        (latent,) = pipeline.call_node("EmptyLatentImage", width=NATIVE[0], height=NATIVE[1], batch_size=1)
        (latent,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=SEED, steps=8,
                                       cfg=1.0, sampler_name="er_sde", scheduler="simple", positive=positive,
                                       negative=positive, latent_image=latent, start_at_step=0, end_at_step=8,
                                       return_with_leftover_noise="disable")
        timings = {"native_s": round(time.perf_counter() - t0, 1)}
        for i, (w, h, sigma_start, steps, anchor, r0, r1) in enumerate(VARIANTS[name]):
            t1 = time.perf_counter()
            ref = latent["samples"]
            up = resize(ref, (h // 8, w // 8), "bicubic")
            ref_model = model.model.process_latent_in(ref.to(model.load_device)).float()
            (noise,) = pipeline.call_node("RandomNoise", noise_seed=SEED + 1 + i)
            sampler = anchored_sampler(ref_model, anchor, (r0, r1), sigma_start)
            sigmas = stage_sigmas(model, sigma_start, steps)
            (latent, _) = pipeline.call_node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler,
                                             sigmas=sigmas, latent_image={"samples": up})
            timings[f"stage{i}_{w}x{h}_s"] = round(time.perf_counter() - t1, 1)
        (images,) = pipeline.call_node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=512, overlap=64,
                                       temporal_size=64, temporal_overlap=8)
        arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT, f"{name}.png"))
        print(json.dumps({"variant": name, "total_s": round(time.perf_counter() - t0, 1), **timings,
                          "size": list(arr.shape[:2]), "detail": round(detail(arr), 2)}), flush=True)
