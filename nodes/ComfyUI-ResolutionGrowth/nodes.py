import numpy as np
import torch
from PIL import Image, ImageDraw

import comfy.model_management
import comfy.utils

from . import growth, stop

PLOT_W, PLOT_H, PLOT_PAD = 768, 320, 36
CURVE_COLORS = {"size": (80, 170, 255), "sigma": (255, 150, 60), "anchor": (120, 220, 120)}
UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]


def schedule_plot(schedule, start, done_steps=None):
    """IMAGE of the three curves over growth steps, each normalised to its own range: size
    (fraction of the way from start to the last width), sigma, anchor weight. Steps after
    `done_steps` (progressive mode stopped early) are drawn dimmed."""
    steps = len(schedule["sizes"])
    done = steps if done_steps is None else done_steps
    image = Image.new("RGB", (PLOT_W, PLOT_H), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    x_of = lambda k: PLOT_PAD + (PLOT_W - 2 * PLOT_PAD) * k / steps  # noqa: E731
    y_of = lambda v: PLOT_H - PLOT_PAD - (PLOT_H - 2 * PLOT_PAD) * v  # noqa: E731
    draw.rectangle((PLOT_PAD, PLOT_PAD, PLOT_W - PLOT_PAD, PLOT_H - PLOT_PAD), outline=(70, 70, 80))

    end = schedule["sizes"][-1]
    span = max(end[0] - start[0], 1)
    sigma_max = max(schedule["sigmas"][0], 1e-6)
    anchor_max = max(max(schedule["anchors"]), 1e-6)
    curves = {
        "size": [0.0] + [(w - start[0]) / span for w, _ in schedule["sizes"]],
        "sigma": [s / sigma_max for s in schedule["sigmas"]] + [schedule["sigmas"][-1] / sigma_max],
        "anchor": [a / anchor_max for a in schedule["anchors"]] + [0.0],
    }
    for name, values in curves.items():
        color = CURVE_COLORS[name]
        dim = tuple(c // 3 for c in color)
        points = [(x_of(k), y_of(v)) for k, v in enumerate(values)]
        for k in range(len(points) - 1):
            draw.line(points[k:k + 2], fill=color if k < done else dim, width=3)
        for k, p in enumerate(points):
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color if k <= done else dim)
    legend = (f"size {start[0]}x{start[1]} -> {end[0]}x{end[1]}   sigma {schedule['sigmas'][0]:.2f} -> "
              f"{schedule['sigmas'][-1]:.2f}   anchor {max(schedule['anchors']):.2f} -> 0   steps {done}/{steps}")
    draw.text((PLOT_PAD, 10), legend, fill=(220, 220, 220))
    for i, name in enumerate(CURVE_COLORS):
        draw.text((PLOT_PAD + 8 + 80 * i, PLOT_H - PLOT_PAD + 10), name, fill=CURVE_COLORS[name])
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


# ---------------------------------------------------------------------------------------------
# input validation (the node boundary; everything past it trusts these)


def check(ok, message):
    if not ok:
        raise growth.GrowthError(message)


def composition_size(latent, vae):
    """IN, critical: the composition latent. Returns its size in pixels."""
    samples = latent["samples"]
    check(samples.ndim in (4, 5), f"latent: expected 4-D [B,C,H,W] or 5-D [B,C,T,H,W] samples, got shape {tuple(samples.shape)}")
    check(samples.ndim == 4 or samples.shape[2] == 1, f"latent: expected a single frame (T = 1), got T = {samples.shape[2]}")
    compression = vae.spacial_compression_decode()
    return samples.shape[-1] * compression, samples.shape[-2] * compression


def check_size(name, size, grid):
    check(size[0] % grid == 0 and size[1] % grid == 0,
          f"{name}: expected multiples of {grid} px (this VAE's grid), got {size[0]}x{size[1]}")


def check_common(start, end, grid, sigma_start, sigma_end, tile_size, tile_overlap):
    check(end[0] >= start[0] and end[1] >= start[1],
          f"growth end size: expected >= the composition {start[0]}x{start[1]}, got {end[0]}x{end[1]}")
    check(sigma_end < sigma_start, f"sigma_end: expected < sigma_start ({sigma_start}), got {sigma_end}")
    check(2 * tile_overlap < tile_size, f"tile_overlap: expected < tile_size / 2 ({tile_size // 2}), got {tile_overlap}")


# ---------------------------------------------------------------------------------------------
# shared inputs

def common_inputs():
    return {
        "model": ("MODEL",),
        "positive": ("CONDITIONING", {"tooltip": "Used for the tiles unless detail_positive is given."}),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "latent": ("LATENT", {"tooltip": "Finished composition at the model's native size (e.g. a KSampler at ~1 MP)."}),
    }


def growth_inputs():
    return {
        "sigma_start": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Noise level of the first growth step, in the model's sigma units (flow models: 0-1). Later steps follow the model's own schedule down to sigma_end."}),
        "sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Noise floor of the last steps. 0 = settle fully. Above 0 every step keeps re-inventing a little (worth trying far above native, where pure refinement starts to look like AI upscaling)."}),
        "anchor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How strongly each step's low frequencies are pulled back to the composition. Keeps layout, stops tiles inventing duplicate subjects."}),
        "anchor_shape": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": "Anchor fade with size: weight = anchor * (1 - f) ** shape, f = how far the step starts along the growth (log-area, 0..1). Higher = released sooner."}),
        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip": "Pixels; keep at the model's training size so attention never spans more positions than trained."}),
        "tile_overlap": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 16}),
        "tile_batch": ("INT", {"default": 3, "min": 1, "max": 64, "tooltip": "Tiles per model call. With the Gfx906 two-GPU split node, use an even number (4 = 2 per GPU)."}),
        "upscale_method": (UPSCALE_METHODS, {"default": "lanczos", "tooltip": "Pixel-space resize between steps (Lanczos looked best)."}),
        "upscale_path": (growth.UPSCALE_PATHS, {"default": "vae", "tooltip": "How each step reaches the next size. vae: decode, resize, encode with the full VAE (reference). tiny_vae: the same through tiny_vae (linear 4K: 3% faster, slightly less detail)."}),
    }


def optional_inputs():
    return {
        "detail_positive": ("CONDITIONING", {"tooltip": "Prompt for the tiles: materials, texture, style -- not the subject or its placement (every tile would draw it)."}),
        "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Runs before the resize in every step (RealESRGAN tested worse than plain Lanczos)."}),
        "tiny_vae": ("VAE", {"tooltip": "For upscale_path tiny_vae: a tiny VAE for the same latent space (Krea2 / Wan 2.1 latents: lighttaew2_1). The final output is still decoded with the full vae."}),
    }


def make_grower(model, positive, negative, vae, latent, kw, detail_positive, upscale_model, tiny_vae):
    check(kw["upscale_path"] != "tiny_vae" or tiny_vae is not None,
          "tiny_vae: expected a VAE connected when upscale_path is tiny_vae, got nothing")
    return growth.Grower(model, detail_positive if detail_positive is not None else positive, negative, vae,
                         latent["samples"], kw["cfg"], kw["seed"], kw["tile_size"], kw["tile_overlap"],
                         kw["tile_batch"], kw["upscale_method"], upscale_model, kw["upscale_path"], tiny_vae)


# ---------------------------------------------------------------------------------------------


class ResolutionGrowth:
    """Grows a composed latent to a fixed target size in small steps: each step upscales the
    last clean estimate a little in pixel space, re-noises it to that step's sigma and runs one
    tiled denoise at the model's native tile size, anchored to the composition."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                **common_inputs(),
                "width": ("INT", {"default": 3840, "min": 256, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 2160, "min": 256, "max": 16384, "step": 16}),
                "growth_mode": (["geometric", "linear"], {"default": "geometric", "tooltip": "geometric: growth_steps steps of equal ratio (faster). linear: +step_px of width per step (slower, cleaner)."}),
                "growth_steps": ("INT", {"default": 6, "min": 1, "max": 64, "tooltip": "geometric mode: one tiled denoise per step."}),
                "step_px": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16, "tooltip": "linear mode: width added per step (height in proportion)."}),
                "size_shape": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "geometric mode: 1 = even growth, > 1 = grow late, < 1 = grow early."}),
                "oversample": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Supersampling: grow to width/height x this, then Lanczos down to width/height."}),
                **growth_inputs(),
            },
            "optional": optional_inputs(),
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "schedule")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "Resolution Growth"

    def sample(self, model, positive, negative, vae, latent, width, height, growth_mode, growth_steps, step_px,
               size_shape, oversample, detail_positive=None, upscale_model=None, tiny_vae=None, **kw):
        grid = growth.snap_px(vae)
        start = composition_size(latent, vae)
        check_size("width/height", (width, height), grid)
        end = (growth.snap(width * oversample, grid), growth.snap(height * oversample, grid))
        check_common(start, end, grid, kw["sigma_start"], kw["sigma_end"], kw["tile_size"], kw["tile_overlap"])

        if growth_mode == "linear":
            sizes = growth.linear_sizes(start, end, step_px, grid)
        else:
            sizes = growth.geometric_sizes(start, end, growth_steps, size_shape, grid)
        schedule = growth.make_schedule(model, start, sizes, kw["sigma_start"], kw["sigma_end"], kw["anchor"],
                                        kw["anchor_shape"])
        print(f"[Resolution Growth] {start[0]}x{start[1]} -> {end[0]}x{end[1]}: {growth.describe(start, schedule)}",
              flush=True)

        grower = make_grower(model, positive, negative, vae, latent, kw, detail_positive, upscale_model, tiny_vae)
        pbar = comfy.utils.ProgressBar(len(sizes))
        x0 = latent["samples"]
        for k, (size, sigma, anchor) in enumerate(zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"])):
            x0 = grower.step(k, x0, size, sigma, anchor)
            if k + 1 < len(sizes):  # the last step's decode would be extra work: the output shows it
                grower.preview(pbar, k, len(sizes), x0)
            else:
                pbar.update_absolute(k + 1, len(sizes))
        if end != (width, height):
            x0 = growth.pixel_upscale(vae, x0, (width, height), "lanczos", None)
        return ({"samples": x0.to(comfy.model_management.intermediate_device())}, schedule_plot(schedule, start))


class ResolutionGrowthProgressive:
    """Cycles-style: the output size is fixed; the internal image grows from the composition by
    +step_px per step -- upscaled to the output at first, supersampled once past it -- until the
    output converges (a step changes it less than converge_db), the internal size reaches
    output x max_oversample, or the Stop button is pressed. Returns the output-size image of
    the last finished step, its latent, and a per-step report."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                **common_inputs(),
                "width": ("INT", {"default": 3840, "min": 256, "max": 16384, "step": 16, "tooltip": "Output width."}),
                "height": ("INT", {"default": 2160, "min": 256, "max": 16384, "step": 16, "tooltip": "Output height."}),
                "step_px": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16, "tooltip": "Internal width added per step (height in proportion)."}),
                "max_oversample": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Largest internal size, as a multiple of the output. The noise schedule spans the composition to this size. Far past ~1.3x details began to look AI-upscaled in tests."}),
                "converge_db": ("FLOAT", {"default": 35.0, "min": 10.0, "max": 80.0, "step": 0.5, "tooltip": "Stop once a step changes the output image less than this (PSNR vs the previous step, dB), checked only after the internal size has reached the output. 4K test: 25 dB at 2.5K internal, 35 dB at ~4800, 41 dB at ~5500."}),
                **growth_inputs(),
            },
            "optional": optional_inputs(),
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "latent", "schedule", "report")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    TITLE = "Resolution Growth (Progressive)"

    def sample(self, model, positive, negative, vae, latent, width, height, step_px, max_oversample, converge_db,
               detail_positive=None, upscale_model=None, tiny_vae=None, unique_id=None, **kw):
        grid = growth.snap_px(vae)
        start = composition_size(latent, vae)
        output = (width, height)
        check_size("width/height", output, grid)
        check(step_px % grid == 0, f"step_px: expected a multiple of {grid}, got {step_px}")
        cap = (growth.snap(width * max_oversample, grid), growth.snap(height * max_oversample, grid))
        check_common(start, cap, grid, kw["sigma_start"], kw["sigma_end"], kw["tile_size"], kw["tile_overlap"])

        sizes = growth.linear_sizes(start, cap, step_px, grid)
        schedule = growth.make_schedule(model, start, sizes, kw["sigma_start"], kw["sigma_end"], kw["anchor"],
                                        kw["anchor_shape"])
        print(f"[Resolution Growth Progressive] {start[0]}x{start[1]} -> up to {cap[0]}x{cap[1]}, output "
              f"{width}x{height}, converge at {converge_db} dB: {growth.describe(start, schedule)}", flush=True)

        stop.clear(unique_id)
        grower = make_grower(model, positive, negative, vae, latent, kw, detail_positive, upscale_model, tiny_vae)
        pbar = comfy.utils.ProgressBar(len(sizes))
        x0, image, report, reason = latent["samples"], None, [], "reached max_oversample"
        for k, (size, sigma, anchor) in enumerate(zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"])):
            x0 = grower.step(k, x0, size, sigma, anchor)
            grower.preview(pbar, k, len(sizes), x0)
            # the step VAE's decode, shared with the next step's upscale (Grower caches it)
            previous, image = image, growth.resize_image(grower.decode_step(x0), output, "lanczos")
            change = None if previous is None else growth.psnr_db(image, previous)
            t = grower.timing
            line = (f"step {k + 1}: internal {size[0]}x{size[1]}, sigma {sigma:.3f}, anchor {anchor:.2f}, "
                    f"change {'-' if change is None else f'{change:.1f} dB'}, noise {growth.noise_sigma(image):.2f}, "
                    f"upscale {t['upscale_s']:.1f} s, denoise {t['denoise_s']:.1f} s, overhead {t['overhead_s']:.1f} s")
            report.append(line)
            print(f"[Resolution Growth Progressive] {line}", flush=True)
            if stop.requested(unique_id):
                reason = "Stop button"
                break
            if change is not None and size[0] >= width and change >= converge_db:
                reason = f"converged ({change:.1f} >= {converge_db} dB)"
                break
        stop.clear(unique_id)
        report.append(f"stopped: {reason} after {len(report)} of {len(sizes)} steps")
        print(f"[Resolution Growth Progressive] {report[-1]}", flush=True)
        if grower.step_vae is not vae:  # the steps ran on tiny_vae; the output gets the full one
            image = growth.resize_image(growth.decode(vae, x0), output, "lanczos")
        out_device = comfy.model_management.intermediate_device()
        return (image.to(out_device), {"samples": growth.encode(vae, image).to(out_device)},
                schedule_plot(schedule, start, len(report) - 1), "\n".join(report))


NODE_CLASS_MAPPINGS = {"ResolutionGrowth": ResolutionGrowth, "ResolutionGrowthProgressive": ResolutionGrowthProgressive}
NODE_DISPLAY_NAME_MAPPINGS = {"ResolutionGrowth": "Resolution Growth",
                              "ResolutionGrowthProgressive": "Resolution Growth (Progressive)"}
