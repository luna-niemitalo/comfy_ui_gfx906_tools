"""Resolution growth: refine a composed latent up to a larger size in many small steps,
treating resolution as one more dimension of the same image. Every step:
  up      = pixel-space resize of the previous clean estimate to this step's size
  x0      = one tiled denoise of `up` noised to this step's sigma -- tiles at the model's
            native size, so attention never spans more positions than it was trained on;
            low band anchored to the composition: x0 += anchor * up(ref - down(x0))
sigma and anchor are functions of f, how far the step starts along the growth in log-area
(0 at the composition, 1 at the end size), so any step placement (geometric, linear) sees
the same noise at the same size:
  sigma(f)  = the model's own "simple" schedule, index-linear from sigma_start to sigma_end
  anchor(f) = anchor * (1 - f) ** anchor_shape
Every default comes from develop/experiments/2026-09-29 (grow_test, tiled_test, scale_test)
on Krea2 int4 at up to 7680x4320."""
import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import comfy.utils
import latent_preview
import nodes

VAE_TILE_PX, VAE_OVERLAP_PX = 512, 64


class GrowthError(ValueError):
    """Bad node input; message names the input with expected and actual values."""


# ---------------------------------------------------------------------------------------------
# schedule


def snap_px(vae):
    """Pixel grid every size is rounded to: 2x the VAE compression (2x2-patchifying DiTs such
    as Flux / Qwen-Image / Krea2 need latent sizes divisible by 2), at least 16."""
    return max(16, 2 * vae.spacial_compression_decode())


def snap(v, grid):
    return grid * max(1, round(v / grid))


def geometric_sizes(start, end, steps, shape, grid):
    """`steps` sizes, per axis start * (end / start) ** ((k / steps) ** shape), k = 1..steps;
    shape > 1 grows late, < 1 early."""
    return [tuple(snap(s * (e / s) ** ((k / steps) ** shape), grid) for s, e in zip(start, end))
            for k in range(1, steps + 1)]


def linear_sizes(start, end, step_px, grid):
    """+step_px of width per step (height in proportion) until `end`: the relative change per
    step shrinks as the image grows (Luna's "grid extension")."""
    steps = max(1, -(-(end[0] - start[0]) // step_px))
    return [tuple(snap(s + (e - s) * k / steps, grid) for s, e in zip(start, end)) for k in range(1, steps + 1)]


def start_fractions(start, sizes):
    """f per step: 0 for the first, then the log-area fraction of the previous step's size."""
    area = lambda s: s[0] * s[1]  # noqa: E731
    total = math.log(area(sizes[-1]) / area(start))
    return [0.0] + [math.log(area(s) / area(start)) / total for s in sizes[:-1]]


def sigma_curve(model_sampling, sigma_start, sigma_end, fractions):
    full = comfy.samplers.calculate_sigmas(model_sampling, "simple", 1000)
    first = int((full > sigma_start).sum())
    last = int((full > sigma_end).sum()) if sigma_end > 0 else len(full) - 1
    return [float(full[round(first + f * (last - first))]) for f in fractions]


def make_schedule(model, start, sizes, sigma_start, sigma_end, anchor, anchor_shape):
    fractions = start_fractions(start, sizes)
    return {"sizes": sizes,
            "sigmas": sigma_curve(model.get_model_object("model_sampling"), sigma_start, sigma_end, fractions),
            "anchors": [anchor * (1 - f) ** anchor_shape for f in fractions]}


def describe(start, schedule):
    return ", ".join(f"{w}x{h} s{s:.3f} a{a:.2f}" for (w, h), s, a in
                     zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"]))


# ---------------------------------------------------------------------------------------------
# one growth step


def resize(x, hw, mode):
    """interpolate over the last two dims of a 4-D [B,C,H,W] or 5-D [B,C,T,H,W] latent."""
    if x.ndim == 4:
        return F.interpolate(x, size=hw, mode=mode)
    b, c, t = x.shape[:3]
    frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, *x.shape[-2:])
    return F.interpolate(frames, size=hw, mode=mode).reshape(b, t, c, *hw).permute(0, 2, 1, 3, 4)


def tile_starts(size, tile, overlap):
    """Start offsets covering [0, size) with tiles of `tile` overlapping >= `overlap`."""
    if size <= tile:
        return [0]
    n = -(-(size - overlap) // (tile - overlap))
    # even offsets: 2x2-patchified latents
    return [2 * round(i * (size - tile) / (n - 1) / 2) for i in range(n)]


def feather(tile_h, tile_w, overlap, device):
    """Weights ramping linearly over `overlap` at each edge (never 0, so every pixel counts)."""
    def ramp(n):
        r = torch.ones(n, device=device)
        if overlap:
            k = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
            r[:overlap] = torch.minimum(r[:overlap], k)
            r[-overlap:] = torch.minimum(r[-overlap:], k.flip(0))
        return r
    return ramp(tile_h)[:, None] * ramp(tile_w)[None, :]


def tiled_x0_sampler(ref, anchor, tile, overlap, tile_batch):
    """KSAMPLER whose one step returns the model's clean estimate at sigmas[0], computed over
    overlapping tiles (feathered blend, MultiDiffusion), low band pulled to `ref` (the
    composition, model space)."""
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}
        b = x.shape[0]
        H, W = x.shape[-2:]
        th, tw = min(tile, H), min(tile, W)  # an axis shorter than a tile is one tile
        spans = [(y, c) for y in tile_starts(H, th, overlap) for c in tile_starts(W, tw, overlap)]
        weights = feather(th, tw, min(overlap, th // 2, tw // 2), x.device)
        sigma = sigmas[0] * x.new_ones(b)

        acc = torch.zeros_like(x, dtype=torch.float32)
        norm = torch.zeros(x.shape[-2:], device=x.device)
        for i in range(0, len(spans), tile_batch):
            comfy.model_management.throw_exception_if_processing_interrupted()
            batch = spans[i:i + tile_batch]
            tiles = torch.cat([x[..., y:y + th, c:c + tw] for y, c in batch], 0)
            d = model(tiles, sigma.repeat(len(batch)), **extra_args).float()
            # Host sync per tile batch: with 5+ batches queued unsynced (3840x2160, 15 tiles)
            # ROCm 7.2's rocr AsyncEventsLoop segfaults on a released signal; 2-3 never did.
            # Cost unmeasurable (58.2 vs 59.2 s fully serialized).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for j, (y, c) in enumerate(batch):
                acc[..., y:y + th, c:c + tw] += d[j * b:(j + 1) * b] * weights
                norm[y:y + th, c:c + tw] += weights
        x0 = (acc / norm).to(x.dtype)
        if anchor > 0:
            low = resize(x0.float(), ref.shape[-2:], "area")
            x0 = x0 + (anchor * resize(ref - low, x0.shape[-2:], "bicubic")).to(x0.dtype)
        return x0
    return comfy.samplers.KSAMPLER(sample)


def no_miopen():
    """MIOpen builds kernels per new conv shape (~8 s per new VAE size on gfx906) and growth
    sizes rarely repeat; ROCm only -- on CUDA, cuDNN has no such cost."""
    if torch.version.hip:
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


def decode(vae, samples):
    """IMAGE [B, H, W, C]; one latent at a time (video VAEs read a batch as frames)."""
    images = []
    with no_miopen():
        for i in range(samples.shape[0]):
            (image,) = nodes.VAEDecodeTiled().decode(vae, {"samples": samples[i:i + 1]}, VAE_TILE_PX, VAE_OVERLAP_PX)
            images.append(image.reshape(-1, *image.shape[-3:])[:1])
    return torch.cat(images, 0)


def encode(vae, images):
    latents = []
    with no_miopen():
        for i in range(images.shape[0]):
            (latent,) = nodes.VAEEncodeTiled().encode(vae, images[i:i + 1], VAE_TILE_PX, VAE_OVERLAP_PX)
            latents.append(latent["samples"])
    return torch.cat(latents, 0)


def resize_image(images, size, method, upscale_model=None):
    if upscale_model is not None:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        images = ImageUpscaleWithModel.execute(upscale_model, images)[0]
    (images,) = nodes.ImageScale().upscale(images, method, size[0], size[1], "disabled")
    return images


def pixel_upscale(vae, samples, size, method, upscale_model):
    """Decode, resize the image, encode: an on-manifold latent at `size` (a bicubic latent
    upscale smears -- Luna saw it)."""
    return encode(vae, resize_image(decode(vae, samples), size, method, upscale_model))


class Grower:
    """Runs growth steps for one sampling call: model, conditioning, reference and settings."""

    def __init__(self, model, positive, negative, vae, composition, cfg, seed, tile_px, overlap_px, tile_batch,
                 upscale_method, upscale_model):
        self.model, self.vae = model, vae
        self.guider = comfy.samplers.CFGGuider(model)
        self.guider.set_conds(positive, negative)
        self.guider.set_cfg(cfg)
        self.ref = model.model.process_latent_in(composition.to(model.load_device)).float()
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        compression = vae.spacial_compression_decode()
        self.tile = 2 * max(1, tile_px // compression // 2)
        self.overlap = 2 * (overlap_px // compression // 2)
        self.tile_batch = tile_batch
        self.upscale_method, self.upscale_model = upscale_method, upscale_model

    def step(self, k, x0, size, sigma, anchor):
        comfy.model_management.throw_exception_if_processing_interrupted()
        up = pixel_upscale(self.vae, x0, size, self.upscale_method, self.upscale_model)
        noise = torch.randn(up.shape, generator=self.generator, dtype=torch.float32)
        # comfy scales the noise into the clean latent for this model type (noise_scaling)
        out = self.guider.sample(noise, up, tiled_x0_sampler(self.ref, anchor, self.tile, self.overlap, self.tile_batch),
                                 torch.tensor([sigma, 0.0]), seed=self.seed + k, disable_pbar=True)
        return out.to(x0.device)

    def preview_latent(self, x0):
        return self.model.model.process_latent_in(x0.to(self.model.load_device))


# ---------------------------------------------------------------------------------------------
# convergence metrics (progressive mode)


def psnr_db(a, b):
    """PSNR between two IMAGE tensors in [0, 1] of the same shape."""
    mse = float(((a.float() - b.float()) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def noise_sigma(image):
    """Immerkaer (1996) single-image noise sigma on luma, 8-bit units (counts fine texture too)."""
    y = (image[0].float().cpu().numpy() * 255) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    r = (y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:] - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1]
         - 2 * y[1:-1, 2:] + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:])
    return float(math.sqrt(math.pi / 2) * np.abs(r).mean() / 6)
