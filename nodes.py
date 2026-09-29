import numpy as np
import torch
from PIL import Image, ImageDraw

from .growth import SNAP_PX, grow

PLOT_W, PLOT_H, PLOT_PAD = 768, 320, 36
CURVE_COLORS = {"size": (80, 170, 255), "sigma": (255, 150, 60), "anchor": (120, 220, 120)}


def schedule_plot(schedule, start, target):
    """IMAGE of the three curves over growth steps, each normalised to its own range:
    size (fraction of the way from start to target width), sigma, anchor weight."""
    steps = len(schedule["sizes"])
    image = Image.new("RGB", (PLOT_W, PLOT_H), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    x_of = lambda k: PLOT_PAD + (PLOT_W - 2 * PLOT_PAD) * k / steps  # noqa: E731
    y_of = lambda v: PLOT_H - PLOT_PAD - (PLOT_H - 2 * PLOT_PAD) * v  # noqa: E731
    draw.rectangle((PLOT_PAD, PLOT_PAD, PLOT_W - PLOT_PAD, PLOT_H - PLOT_PAD), outline=(70, 70, 80))

    span = max(target[0] - start[0], 1)
    sigma_max = max(schedule["sigmas"][0], 1e-6)
    anchor_max = max(max(schedule["anchors"]), 1e-6)
    curves = {
        "size": [0.0] + [(w - start[0]) / span for w, _ in schedule["sizes"]],
        "sigma": [s / sigma_max for s in schedule["sigmas"]] + [0.0],
        "anchor": [a / anchor_max for a in schedule["anchors"]] + [0.0],
    }
    for name, values in curves.items():
        points = [(x_of(k), y_of(v)) for k, v in enumerate(values)]
        draw.line(points, fill=CURVE_COLORS[name], width=3)
        for p in points:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=CURVE_COLORS[name])
    legend = (f"size {start[0]}x{start[1]} -> {target[0]}x{target[1]}   "
              f"sigma {schedule['sigmas'][0]:.2f} -> 0   anchor {max(schedule['anchors']):.2f} -> 0")
    draw.text((PLOT_PAD, 10), legend, fill=(220, 220, 220))
    for i, name in enumerate(CURVE_COLORS):
        draw.text((PLOT_PAD + 8 + 80 * i, PLOT_H - PLOT_PAD + 10), name, fill=CURVE_COLORS[name])
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


class ResolutionGrowth:
    """Grows a composed latent to a target size in many small steps: each step upscales the
    last clean estimate a little in pixel space, re-noises it to that step's sigma and runs
    one tiled denoise at the model's native tile size, anchored to the composition.
    Feed it a finished composition at the model's native size (e.g. a KSampler at ~1 MP)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "Used for the tiles unless detail_positive is given."}),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT", {"tooltip": "Finished composition at the model's native size."}),
                "width": ("INT", {"default": 3840, "min": 256, "max": 16384, "step": SNAP_PX}),
                "height": ("INT", {"default": 2160, "min": 256, "max": 16384, "step": SNAP_PX}),
                "growth_mode": (["geometric", "linear"], {"default": "geometric", "tooltip": "geometric: growth_steps steps, each the same ratio (shaped by size_shape). linear: +step_px of width per step, so steps get relatively smaller as the image grows."}),
                "growth_steps": ("INT", {"default": 6, "min": 1, "max": 64, "tooltip": "geometric mode: one tiled denoise per step; more, smaller steps looked better (6 beat 4 beat one 2x jump)."}),
                "step_px": ("INT", {"default": 256, "min": SNAP_PX, "max": 4096, "step": SNAP_PX, "tooltip": "linear mode: width added per step (height in proportion)."}),
                "sigma_start": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Noise level of the first growth step, in the model's sigma units (flow models: 0-1). Later steps follow the model's own schedule down towards 0."}),
                "size_shape": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "geometric mode size curve: 1 = even growth, > 1 = grow late (more steps near the start size), < 1 = grow early."}),
                "anchor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How strongly each step's low frequencies are pulled back to the composition. Keeps layout, stops tiles inventing duplicate subjects."}),
                "anchor_shape": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Anchor fade with size: weight = anchor * (1 - f) ** shape, f = how far the step starts along the growth (log-area, 0..1). Higher = released sooner."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": SNAP_PX, "tooltip": "Pixels; keep at the model's training size so attention never spans more positions than trained."}),
                "tile_overlap": ("INT", {"default": 256, "min": 0, "max": 2048, "step": SNAP_PX}),
                "tile_batch": ("INT", {"default": 3, "min": 1, "max": 64, "tooltip": "Tiles per model call. With the Gfx906 two-GPU split node, use an even number (4 = 2 per GPU)."}),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact"], {"default": "lanczos"}),
            },
            "optional": {
                "detail_positive": ("CONDITIONING", {"tooltip": "Prompt for the tiles: describe materials / texture / style, not the subject or its placement (every tile would draw it)."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Runs before the resize in every step (RealESRGAN tested worse than plain Lanczos)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "schedule")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "Resolution Growth"

    def sample(self, model, positive, negative, vae, latent, width, height, growth_mode, growth_steps, step_px,
               sigma_start, size_shape, anchor, anchor_shape, cfg, seed, tile_size, tile_overlap, tile_batch,
               upscale_method, detail_positive=None, upscale_model=None):
        samples, schedule = grow(model, detail_positive if detail_positive is not None else positive, negative,
                                 vae, latent["samples"], (width, height), growth_mode, growth_steps, step_px,
                                 sigma_start, size_shape, anchor, anchor_shape, cfg, seed, tile_size, tile_overlap,
                                 tile_batch, upscale_method, upscale_model)
        compression = vae.spacial_compression_decode()
        start = (latent["samples"].shape[-1] * compression, latent["samples"].shape[-2] * compression)
        return ({"samples": samples}, schedule_plot(schedule, start, (width, height)))


NODE_CLASS_MAPPINGS = {"ResolutionGrowth": ResolutionGrowth}
NODE_DISPLAY_NAME_MAPPINGS = {"ResolutionGrowth": "Resolution Growth"}
