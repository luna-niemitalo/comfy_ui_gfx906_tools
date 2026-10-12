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
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy.cli_args import args
import comfy.samplers
import comfy.utils
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


def tiled_x0_sampler(ref, anchor, tile, overlap, tile_batch, timing):
    """KSAMPLER whose one step returns the model's clean estimate at sigmas[0], computed over
    overlapping tiles (feathered blend, MultiDiffusion), low band pulled to `ref` (the
    composition, model space). Writes timing["seconds"] and timing["tiles"].
    (Skipping tiles whose region barely changed the step before was tried: on linear 4K every
    tile changed > 0.5x the largest change, nothing to skip -- 2026-09-29, scale_test %skip.)"""
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        t0 = time.perf_counter()
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
            for n, (y, c) in enumerate(batch):
                acc[..., y:y + th, c:c + tw] += d[n * b:(n + 1) * b] * weights
                norm[y:y + th, c:c + tw] += weights
        x0 = (acc / norm).to(x.dtype)
        timing["tiles"] = len(spans)
        if anchor > 0:
            low = resize(x0.float(), ref.shape[-2:], "area")
            x0 = x0 + (anchor * resize(ref - low, x0.shape[-2:], "bicubic")).to(x0.dtype)
        timing["seconds"] = time.perf_counter() - t0
        return x0
    return comfy.samplers.KSAMPLER(sample)


def no_miopen():
    """MIOpen builds kernels per new conv shape (~8 s per new VAE size on gfx906) and growth
    sizes rarely repeat; ROCm only -- on CUDA, cuDNN has no such cost."""
    if torch.version.hip:
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


@contextlib.contextmanager
def on_vae_device(vae):
    """VAE outputs stay on the VAE's GPU (ComfyUI's intermediate device is the host unless
    --gpu-only); restored afterwards."""
    previous, vae.output_device = vae.output_device, vae.device
    try:
        yield
    finally:
        vae.output_device = previous


def decode(vae, samples):
    """IMAGE [B, H, W, C] on the VAE's device; one latent at a time (video VAEs read a batch as
    frames)."""
    images = []
    with no_miopen(), on_vae_device(vae):
        for i in range(samples.shape[0]):
            (image,) = nodes.VAEDecodeTiled().decode(vae, {"samples": samples[i:i + 1]}, VAE_TILE_PX, VAE_OVERLAP_PX)
            images.append(image.reshape(-1, *image.shape[-3:])[:1])
    return torch.cat(images, 0)


def encode(vae, images):
    latents = []
    with no_miopen(), on_vae_device(vae):
        for i in range(images.shape[0]):
            (latent,) = nodes.VAEEncodeTiled().encode(vae, images[i:i + 1], VAE_TILE_PX, VAE_OVERLAP_PX)
            latents.append(latent["samples"])
    return torch.cat(latents, 0)


def lanczos_matrix(n_in, n_out, device, a=3):
    """[n_out, n_in] Lanczos-a resampling weights; kernel widened by the downscale factor
    (antialias, as PIL), edge taps clamped (replicate), rows normalised."""
    scale = n_out / n_in
    support = a / min(scale, 1.0)
    centers = (torch.arange(n_out, device=device, dtype=torch.float64) + 0.5) / scale - 0.5
    offsets = torch.arange(-math.ceil(support), math.ceil(support) + 1, device=device, dtype=torch.float64)
    taps = centers.floor()[:, None] + offsets[None, :]
    d = (taps - centers[:, None]) * min(scale, 1.0)
    w = torch.sinc(d) * torch.sinc(d / a) * (d.abs() < a)
    w = w / w.sum(1, keepdim=True)
    m = torch.zeros(n_out, n_in, device=device, dtype=torch.float64)
    m.scatter_add_(1, taps.clamp(0, n_in - 1).long(), w)
    return m.float()


def resize_image(images, size, method, upscale_model=None):
    """IMAGE resize on the images' own device. lanczos: separable float Lanczos-3 as two
    matmuls (ComfyUI's goes through 8-bit PIL on the host; same result to 53 dB,
    upscale_bench.py); other methods: torch interpolate with antialias."""
    if upscale_model is not None:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        images = ImageUpscaleWithModel.execute(upscale_model, images)[0]
    x = images.movedim(-1, 1).float()
    w, h = size
    if method == "lanczos":
        x = lanczos_matrix(x.shape[-2], h, x.device) @ x @ lanczos_matrix(x.shape[-1], w, x.device).T
    else:
        x = F.interpolate(x, size=(h, w), mode=method, antialias=method in ("bilinear", "bicubic"))
    return x.clamp(0, 1).movedim(1, -1)


def pixel_upscale(vae, samples, size, method, upscale_model):
    """Decode, resize the image, encode: an on-manifold latent at `size` (a bicubic latent
    upscale smears -- Luna saw it)."""
    return encode(vae, resize_image(decode(vae, samples), size, method, upscale_model))


# how each step moves the previous clean estimate to the next size. Linear 4K growth (scale_test
# +path): tiny_vae 212 vs 219 s, detail at 720p 9.10 vs 9.43. A latent-space resize was tested
# and dropped (Luna): noise stuck at ~1.7 instead of falling, output never converged.
UPSCALE_PATHS = ["vae", "tiny_vae"]


class Grower:
    """Runs growth steps for one sampling call: model, conditioning, reference and settings.
    Keeps the last step's decoded image, so a caller that decodes a result (Progressive's
    convergence check) and the next step's upscale share one decode."""

    def __init__(self, model, positive, negative, vae, composition, cfg, seed, tile_px, overlap_px, tile_batch,
                 upscale_method, upscale_model, upscale_path="vae", tiny_vae=None):
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
        self.upscale_path = upscale_path
        self.step_vae = tiny_vae if upscale_path == "tiny_vae" else vae
        self.decoded = None  # (latent tensor id, IMAGE) of the last decode by step_vae
        self.timing = {}

    def decode_step(self, x0):
        """x0 decoded by the step VAE, reusing the last decode of the same latent."""
        if self.decoded is None or self.decoded[0] is not x0:
            self.decoded = (x0, decode(self.step_vae, x0))
        return self.decoded[1]

    def upscale(self, x0, size):
        image = resize_image(self.decode_step(x0), size, self.upscale_method, self.upscale_model)
        return encode(self.step_vae, image)

    def step(self, k, x0, size, sigma, anchor):
        comfy.model_management.throw_exception_if_processing_interrupted()
        t0 = time.perf_counter()
        up = self.upscale(x0, size)
        # no sync here: the upscale's tail on the VAE's GPU overlaps the denoise start (syncing
        # for exact timing cost 12 s of 219 on linear 4K) -- upscale_s is a lower bound
        t1 = time.perf_counter()
        noise = torch.randn(up.shape, generator=self.generator, dtype=torch.float32)
        inner = {}
        sampler = tiled_x0_sampler(self.ref, anchor, self.tile, self.overlap, self.tile_batch, inner)
        # comfy scales the noise into the clean latent for this model type (noise_scaling)
        out = self.guider.sample(noise, up, sampler, torch.tensor([sigma, 0.0]), seed=self.seed + k, disable_pbar=True)
        t2 = time.perf_counter()
        self.timing = {"upscale_s": t1 - t0, "denoise_s": inner["seconds"], "overhead_s": t2 - t1 - inner["seconds"],
                       "tiles": inner["tiles"]}
        return out.to(x0.device)

    def preview(self, pbar, k, total, x0):
        """Show step k in the UI: the step VAE's decode of x0 (cached -- the next step's upscale
        reuses it), shrunk to --preview-size on the GPU, handed to the progress bar as ComfyUI's
        own previewers do. Not the TAE previewer: at full size it OOMed next to the UNet (2.2-2.9
        GB per conv at ~2K), and any latent-space shrink before it decodes garbage (area:
        streaks, subsampling: checkerboard -- Luna's previews, 2026-09-29)."""
        image = self.decode_step(x0)[:1]
        h, w = image.shape[1:3]
        scale = min(1.0, args.preview_size / max(h, w))
        if scale < 1:
            image = resize_image(image, (max(1, round(w * scale)), max(1, round(h * scale))), "lanczos")
        arr = (image[0].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        pbar.update_absolute(k + 1, total, ("JPEG", Image.fromarray(arr), args.preview_size))


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
