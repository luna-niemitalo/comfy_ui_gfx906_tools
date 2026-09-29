"""Resolution growth: refine a composed latent up to a target size in many small steps,
treating resolution as one more dimension of the same image. Step k of n:
  size_k  = start * (target / start) ** ((k / n) ** size_shape)   per axis, SNAP_PX grid
  x0_k    = one tiled denoise of (pixel_upscale(x0_{k-1}, size_k) noised to sigma_k)
            -- tiles at the model's native size, so attention never spans more positions
            than it was trained on; low band anchored to the composition with weight
            anchor * (1 - (k - 1) / n) ** anchor_shape
sigma_1..n: the model's own "simple" schedule from sigma_start towards 0.
Every default comes from develop/experiments/2026-09-29 (grow_test.py, tiled_test.py) on
Krea2 at 2560x1440 and 3840x2160."""
import contextlib

import torch
import torch.nn.functional as F

import comfy.samplers
import latent_preview
import nodes

SNAP_PX = 16  # 8x VAE x 2x2 patchify (Flux / Qwen-Image / Krea2); also a multiple of SD's 8
VAE_TILE_PX, VAE_OVERLAP_PX = 512, 64


def growth_sizes(start, target, steps, shape):
    """`steps` (width, height) pairs from just above `start` to `target`, geometric per axis;
    shape > 1 grows late, < 1 grows early."""
    sizes = []
    for k in range(1, steps + 1):
        f = (k / steps) ** shape
        sizes.append(tuple(SNAP_PX * max(1, round(s * (t / s) ** f / SNAP_PX)) for s, t in zip(start, target)))
    return sizes


def growth_sigmas(model_sampling, sigma_start, steps):
    """`steps` sigmas from sigma_start towards 0, evenly spaced in index along the model's own
    'simple' schedule so the spacing follows its shift."""
    full = comfy.samplers.calculate_sigmas(model_sampling, "simple", 1000)
    first = int((full > sigma_start).sum())
    index = torch.linspace(first, len(full) - 1, steps + 1).round().long()
    return full[index][:-1]


def anchor_weights(anchor, shape, steps):
    return [anchor * (1 - k / steps) ** shape for k in range(steps)]


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
        k = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
        r[:overlap] = torch.minimum(r[:overlap], k)
        r[-overlap:] = torch.minimum(r[-overlap:], k.flip(0))
        return r
    return ramp(tile_h)[:, None] * ramp(tile_w)[None, :]


def tiled_x0_sampler(ref, anchor, tile, overlap, tile_batch):
    """KSAMPLER whose one step returns the model's clean estimate at sigmas[0], computed over
    overlapping tiles (feathered blend, MultiDiffusion) with its low band pulled to `ref`
    (the composition, model space): x0 += anchor * up(ref - down(x0))."""
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}
        b = x.shape[0]
        H, W = x.shape[-2:]
        th, tw = min(tile, H), min(tile, W)  # an axis shorter than a tile is one tile
        spans = [(y, c) for y in tile_starts(H, th, overlap) for c in tile_starts(W, tw, overlap)]
        weights = feather(th, tw, overlap, x.device)
        sigma = sigmas[0] * x.new_ones(b)

        acc = torch.zeros_like(x, dtype=torch.float32)
        norm = torch.zeros(x.shape[-2:], device=x.device)
        for i in range(0, len(spans), tile_batch):
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
        if callback is not None:
            callback({"x": x, "i": 0, "sigma": sigmas[0], "sigma_hat": sigmas[0], "denoised": x0})
        return x0
    return comfy.samplers.KSAMPLER(sample)


def no_miopen():
    """MIOpen builds kernels per new conv shape (~8 s per new VAE size on gfx906) and growth
    sizes never repeat; ROCm only -- on CUDA, cuDNN has no such cost."""
    if torch.version.hip:
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


def pixel_upscale(vae, samples, size, method, upscale_model):
    """Decode, resize the image (optionally through a model upscaler first), encode: an
    on-manifold latent at `size` (a bicubic latent upscale smears, Luna saw it). One image
    at a time: video VAEs would read a batch of images as frames."""
    w, h = size
    out = []
    with no_miopen():
        for i in range(samples.shape[0]):
            (image,) = nodes.VAEDecodeTiled().decode(vae, {"samples": samples[i:i + 1]}, VAE_TILE_PX, VAE_OVERLAP_PX)
            if upscale_model is not None:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                image = ImageUpscaleWithModel.execute(upscale_model, image)[0]
            (image,) = nodes.ImageScale().upscale(image, method, w, h, "disabled")
            (latent,) = nodes.VAEEncodeTiled().encode(vae, image, VAE_TILE_PX, VAE_OVERLAP_PX)
            out.append(latent["samples"])
    return torch.cat(out, 0)


def grow(model, positive, negative, vae, samples, target, steps, sigma_start, size_shape, anchor,
         anchor_shape, cfg, seed, tile_px, overlap_px, tile_batch, upscale_method, upscale_model):
    """Returns (latent samples at target, schedule dict). `samples`: clean composition latent."""
    compression = vae.spacial_compression_decode()
    start = (samples.shape[-1] * compression, samples.shape[-2] * compression)
    schedule = {
        "sizes": growth_sizes(start, target, steps, size_shape),
        "sigmas": [float(s) for s in growth_sigmas(model.get_model_object("model_sampling"), sigma_start, steps)],
        "anchors": anchor_weights(anchor, anchor_shape, steps),
    }
    print(f"[Resolution Growth] {start[0]}x{start[1]} -> {target[0]}x{target[1]}: "
          + ", ".join(f"{w}x{h} s{s:.3f} a{a:.2f}" for (w, h), s, a in
                      zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"])), flush=True)

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    ref = model.model.process_latent_in(samples.to(model.load_device)).float()
    generator = torch.Generator().manual_seed(seed)
    preview = latent_preview.prepare_callback(model, steps)
    tile, overlap = tile_px // compression, overlap_px // compression

    x0 = samples
    for k, (size, sigma, weight) in enumerate(zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"])):
        up = pixel_upscale(vae, x0, size, upscale_method, upscale_model)
        noise = torch.randn(up.shape, generator=generator, dtype=torch.float32)
        # comfy scales noise into the clean latent for this model type (noise_scaling at sigma)
        x0 = guider.sample(noise, up, tiled_x0_sampler(ref, weight, tile, overlap, tile_batch),
                           torch.tensor([sigma, 0.0]), seed=seed + k, disable_pbar=True)
        x0 = x0.to(samples.device)
        preview(k, model.model.process_latent_in(x0.to(model.load_device)), None, steps)
    return x0, schedule
