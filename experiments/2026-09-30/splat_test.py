"""Scattered-token sampling for fixed-cost resolution growth (Luna's idea): every model call
gets a fixed token budget drawn from anywhere in the image, not a window. Tokens keep their
true (row, col) RoPE ids; they are packed into a 1-token-high strip latent and a `post_input`
patch rewrites the ids. Optional clean context: the 1-token ring around the active tokens,
passed as Krea2 reference tokens at t=0 (`index_timestep_zero`), ids set to the same image
(index 0).

Part A -- does a token predict well without its neighbours? Native 1024x576 composition,
  re-noised to sigma (fixed noise). x0 prediction of token subsets (fraction f, splats of kxk
  tokens, spread by random sequential placement; + a contiguous window of the same size) vs
  the full-token prediction, on the active tokens. Scale: `spread` = the same full prediction
  with other noise (natural sample spread); `perm` = all tokens, shuffled strip order (must
  be ~0: validates the gather/ids path); `comfy` = direct call vs ComfyUI's sampler path.
Part B -- blending: a 2x Lanczos-degraded composition refined for P passes' worth of tokens
  (equal compute to P full passes). Each call: FRACTION of all tokens as kxk splats, placed
  greedily at the lowest splat-mean spp (1-token gap, jittered order); sigma from the active
  footprint's mean spp (SIGMA0 -> SIGMA1 geometric over P passes); write-back
  x0 <- lerp(x0, pred, w) and spp += w with a per-latent-pixel kernel w over the splat.
  Noise `shared` = one field per integer spp level (overlapping splats see the same noise),
  `fresh` = new noise per call. Reference: P full-image passes, same sigmas and noise levels.
  Metrics: PSNR vs reference (and a reference with other noise = spread), detail, seam index
  (pixel gradient across 16 px token borders / elsewhere; the reference sits near 1).

Part C -- polish only: B's first C_PRE_PASSES full passes set the structure, then C_SCHEDULE's
  passes' worth of splat tokens (low sigma); context ring widths 0/1/2, splats 3x3/4x4.

Usage: splat_test.py [a] [b] [c]   (default a b). Saves develop/out/splat/*.png; JSON lines."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "2026-09-29"))
import progressive as prog  # noqa: E402  (bootstrap, custom-node whitelist, PROMPT, SEED, detail)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.samplers  # noqa: E402
import pipeline  # noqa: E402

OUT = "/media/luna/work/tinker/ComfyUI/develop/out/splat"
os.makedirs(OUT, exist_ok=True)
NATIVE = (1024, 576)
A_SIGMAS = (0.3, 0.6)
A_FRACTIONS = (0.5, 0.25, 0.1)
A_SPLATS = (1, 2, 3, 4)
PASSES = 4
SIGMA0, SIGMA1 = 0.5, 0.1
FRACTION = 0.25
# kernel: (inner radius, rim floor) in units of the splat half-side; None = box
KERNELS = {"box": None, "cos": (0.0, 0.0), "flat25": (0.5, 0.25), "flat50": (0.5, 0.5)}
# (splat side in tokens, kernel, noise, context ring width in tokens)
B_VARIANTS = [(2, kern, noise, ctx) for kern in KERNELS for noise in ("shared", "fresh") for ctx in (0, 1)]
B_VARIANTS += [(3, "flat25", "shared", 0), (3, "flat25", "shared", 1)]
# Part C: polish only -- B's first 2 full passes (sigma 0.5, 0.29) set the structure, then 2 passes'
# worth of tokens at sigma 0.2 -> 0.1 (B's full schedule continues 0.171, 0.1).
C_PRE_PASSES = 2
C_SCHEDULE = (0.2, 0.1, 2)
C_VARIANTS = [(4, "flat25", "shared", 0), (4, "flat25", "shared", 1), (4, "flat25", "shared", 2),
              (3, "flat25", "shared", 1), (3, "flat25", "shared", 2), (4, "box", "shared", 1)]


def tokens_of(x):
    """[B, C, T, H, W] latent -> [B, C, T, h*w, 2, 2] tokens, Krea2's patchify order."""
    b, c, t, H, W = x.shape
    return x.reshape(b, c, t, H // 2, 2, W // 2, 2).permute(0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, -1, 2, 2)


def untokens(tok, h, w):
    b, c, t = tok.shape[:3]
    return tok.reshape(b, c, t, h, w, 2, 2).permute(0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, 2 * h, 2 * w)


def strip(x, idx):
    """Tokens idx of x packed side by side: [B, C, T, 2, 2n]; strip token j is x's token idx[j]."""
    tok = tokens_of(x)[:, :, :, idx]
    b, c, t, n = tok.shape[:4]
    return tok.permute(0, 1, 2, 4, 3, 5).reshape(b, c, t, 2, 2 * n)


def unstrip(s):
    b, c, t, _, w2 = s.shape
    return s.reshape(b, c, t, 2, w2 // 2, 2).permute(0, 1, 2, 4, 3, 5)


class Predictor:
    """x0 predictions for token subsets through BaseModel.apply_model (model space)."""

    def __init__(self, model, positive, shape):
        comfy.model_management.load_models_gpu([model])
        self.model = model
        self.device = model.load_device
        dtype = model.model.get_dtype_inference()
        self.text = model.model.diffusion_model.preprocess_text(positive[0][0].to(self.device, dtype))
        self.options = model.model_options.get("transformer_options", {})
        self.h, self.w = shape[-2] // 2, shape[-1] // 2

    def rc(self, idx):
        return (idx // self.w).float(), (idx % self.w).float()

    def __call__(self, x_t, sigma, idx, ctx_idx=None, ctx_src=None, replica=None):
        """replica: (diffusion model copy, device) -- the call runs there, through the same steps as
        BaseModel._apply_model (splat4k.py: a second call in parallel on the row-split's replica)."""
        if replica is None:
            comfy.model_management.load_models_gpu([self.model])  # a VAE decode in between may have evicted it
        rows, cols = self.rc(idx)
        n = len(idx)
        kwargs = {}
        if ctx_idx is not None and len(ctx_idx):
            kwargs = {"ref_latents": [strip(ctx_src, ctx_idx)], "ref_latents_method": "index_timestep_zero"}
            crows, ccols = self.rc(ctx_idx)

        def post_input(args):
            ids = args["img_ids"].clone()
            ids[:, :n, 1] = rows.to(ids.device)
            ids[:, :n, 2] = cols.to(ids.device)
            if kwargs:
                ids[:, n:, 0] = 0  # same image, not "reference image 1"
                ids[:, n:, 1] = crows.to(ids.device)
                ids[:, n:, 2] = ccols.to(ids.device)
            return {**args, "img_ids": ids}

        patches = dict(self.options.get("patches", {}))
        patches["post_input"] = list(patches.get("post_input", [])) + [post_input]
        options = {**self.options, "patches": patches}
        s = torch.full((x_t.shape[0],), float(sigma), device=self.device)
        if replica is None:
            out = self.model.model.apply_model(strip(x_t, idx), s, c_crossattn=self.text, transformer_options=options,
                                               **kwargs)
            return unstrip(out.float())
        dm, device = replica
        sampling = self.model.model.model_sampling
        dtype = self.model.model.get_dtype_inference()
        x = strip(x_t, idx).to(device)
        s = s.to(device)
        xc = sampling.calculate_input(s, x).to(dtype)
        t = sampling.timestep(s).float()
        if kwargs:
            kwargs["ref_latents"] = [r.to(device, dtype) for r in kwargs["ref_latents"]]
        out = dm._forward(xc, t, self.text.to(device), None, kwargs.pop("ref_latents", None),
                          {**options, "prefetch_dynamic_vbars": False}, **kwargs)
        return unstrip(sampling.calculate_denoised(s, out.float(), x).to(self.device))

    def full(self, x_t, sigma):
        comfy.model_management.load_models_gpu([self.model])
        s = torch.full((x_t.shape[0],), float(sigma), device=self.device)
        return self.model.model.apply_model(x_t, s, c_crossattn=self.text,
                                            transformer_options=self.options).float()


def splat_tokens(origins, k, w):
    return [(r + i) * w + (c + j) for r, c in origins for i in range(k) for j in range(k)]


def place_splats(score, k, gap, budget, rng, jitter):
    """Greedy: highest score first (+ jitter for spread), no two splats closer than `gap` tokens."""
    order = np.argsort(-(score + rng.uniform(0, jitter, score.shape)), axis=None)
    blocked = np.zeros(score.shape, dtype=bool)
    reach = k + gap
    origins = []
    for flat in order:
        r, c = divmod(int(flat), score.shape[1])
        if blocked[r, c]:
            continue
        origins.append((r, c))
        blocked[max(0, r - reach + 1):r + reach, max(0, c - reach + 1):c + reach] = True
        if len(origins) * k * k >= budget:
            break
    return origins


def ring(idx, h, w, device, width):
    """Tokens within `width` tokens (Chebyshev) of idx, excluding idx."""
    mask = torch.zeros(h * w, device=device)
    mask[idx] = 1
    grown = F.max_pool2d(mask.view(1, 1, h, w), 2 * width + 1, stride=1, padding=width).flatten()
    return torch.nonzero((grown > 0) & (mask == 0)).flatten()


def kernel(k, name, device):
    """Write-back weight over a kxk-token splat (2k x 2k latent pixels)."""
    if KERNELS[name] is None:
        return torch.ones(2 * k, 2 * k, device=device)
    inner, floor = KERNELS[name]
    centre = torch.arange(2 * k, device=device, dtype=torch.float32) + 0.5 - k
    r = torch.sqrt(centre[:, None] ** 2 + centre[None, :] ** 2) / k
    fall = torch.cos(0.5 * math.pi * ((r - inner) / (1 - inner)).clamp(0, 1)) ** 2
    return torch.where(r <= inner, torch.ones_like(r), floor + (1 - floor) * fall)


def sigma_of(spp, schedule):
    s0, s1, passes = schedule
    return s0 * (s1 / s0) ** (min(spp, passes - 1) / (passes - 1))


def rms_on(diff, idx):
    return float(tokens_of(diff)[:, :, :, idx].pow(2).mean().sqrt())


def comfy_path(model, positive, latent, eps, sigma):
    """The same one-call x0 prediction through SamplerCustomAdvanced (process_latent_in/out)."""
    class FixedNoise:
        seed = 0

        def generate_noise(self, input_latent):
            return eps.cpu()

    def one_call(model_fn, x, sigmas, extra_args=None, callback=None, disable=None):
        return model_fn(x, sigmas[0] * x.new_ones(x.shape[0]), **(extra_args or {}))

    (guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=positive)
    (out, _) = pipeline.call_node("SamplerCustomAdvanced", noise=FixedNoise(), guider=guider,
                                  sampler=comfy.samplers.KSAMPLER(one_call),
                                  sigmas=torch.tensor([sigma, 0.0]), latent_image=latent)
    return out["samples"]


def part_a(pred, model, positive, latent, x0):
    h, w = pred.h, pred.w
    n_tok = h * w
    everything = torch.arange(n_tok, device=pred.device)
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(prog.SEED + 10)
    eps = torch.randn(x0.shape, generator=g).to(pred.device)
    eps2 = torch.randn(x0.shape, generator=g).to(pred.device)
    for sigma in A_SIGMAS:
        x_t = (1 - sigma) * x0 + sigma * eps
        full = pred.full(x_t, sigma)
        spread = rms_on(pred.full((1 - sigma) * x0 + sigma * eps2, sigma) - full, everything)
        report = {"part": "a", "sigma": sigma, "spread": round(spread, 5)}
        comfy_out = model.model.process_latent_in(comfy_path(model, positive, latent, eps, sigma).to(pred.device))
        report["comfy_vs_direct"] = round(rms_on(comfy_out.float() - full, everything) / spread, 4)
        perm = everything[torch.randperm(n_tok, generator=torch.Generator().manual_seed(1)).to(pred.device)]
        tok = tokens_of(full).clone()
        tok[:, :, :, perm] = pred(x_t, sigma, perm)
        report["perm"] = round(rms_on(untokens(tok, h, w) - full, everything) / spread, 4)
        print(json.dumps(report), flush=True)
        for f in A_FRACTIONS:
            side_h = max(1, round(h * math.sqrt(f)))
            side_w = max(1, round(n_tok * f / side_h))
            r0, c0 = (h - side_h) // 2, (w - side_w) // 2
            window = torch.tensor([(r0 + i) * w + c0 + j for i in range(side_h) for j in range(side_w)],
                                  device=pred.device)
            configs = [("window", window)]
            for k in A_SPLATS:
                origins = place_splats(np.zeros((h - k + 1, w - k + 1)), k, 0, f * n_tok, rng, 1.0)
                configs.append((f"splat{k}", torch.tensor(splat_tokens(origins, k, w), device=pred.device)))
            for name, idx in configs:
                for ctx in (False, True):
                    if ctx and name == "window":
                        continue
                    ctx_idx = ring(idx, h, w, pred.device, 1) if ctx else None
                    t0 = time.perf_counter()
                    p = pred(x_t, sigma, idx, ctx_idx, x0)
                    seconds = time.perf_counter() - t0
                    err = float((p - tokens_of(full)[:, :, :, idx]).pow(2).mean().sqrt())
                    print(json.dumps({"part": "a", "sigma": sigma, "fraction": f, "config": name, "context": ctx,
                                      "tokens": len(idx), "ctx_tokens": 0 if ctx_idx is None else len(ctx_idx),
                                      "err_over_spread": round(err / spread, 4), "s": round(seconds, 2)}),
                          flush=True)


def noise_levels(shape, seed, device):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randn(shape, generator=g) for _ in range(PASSES + 4)]).to(device)


def run_reference(pred, start, levels, schedule, passes=None):
    """Full-image passes; `passes` < schedule's stops early (Part C's pre-polish state)."""
    x0 = start.clone()
    for p in range(schedule[2] if passes is None else passes):
        sigma = sigma_of(p, schedule)
        x0 = pred.full((1 - sigma) * x0 + sigma * levels[p], sigma)
    return x0


def run_splats(pred, start, levels, k, kern, noise, ctx, seed, schedule=(SIGMA0, SIGMA1, PASSES)):
    """ctx = clean context ring width in tokens (0 = none)."""
    h, w = pred.h, pred.w
    n_tok = h * w
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    wk = kernel(k, kern, pred.device)
    x0 = start.clone()
    spp = torch.zeros(start.shape[-2:], device=pred.device)
    used = calls = 0
    while used < schedule[2] * n_tok:
        footprint_mean = F.avg_pool2d(spp[None, None], 2 * k, stride=2)[0, 0]
        origins = place_splats(-footprint_mean.cpu().numpy(), k, 1, FRACTION * n_tok, rng, 0.05)
        idx = torch.tensor(splat_tokens(origins, k, w), device=pred.device)
        weight = torch.zeros_like(spp)
        for r, c in origins:
            weight[2 * r:2 * r + 2 * k, 2 * c:2 * c + 2 * k] = wk
        sigma = sigma_of(float(spp[weight > 0].mean()), schedule)
        if noise == "shared":
            level = spp.floor().long().clamp(max=len(levels) - 1)
            eps = torch.gather(levels, 0, level.expand(1, *start.shape)).squeeze(0)
        else:
            eps = torch.randn(start.shape, generator=g).to(pred.device)
        x_t = (1 - sigma) * x0 + sigma * eps
        ctx_idx = ring(idx, h, w, pred.device, ctx) if ctx else None
        tok = tokens_of(x0).clone()
        tok[:, :, :, idx] = pred(x_t, sigma, idx, ctx_idx, x0)
        x0 = x0 + weight * (untokens(tok, h, w) - x0)
        spp += weight
        used += len(idx)
        calls += 1
    return x0, {"calls": calls, "spp_min": round(float(spp.min()), 2), "spp_mean": round(float(spp.mean()), 2)}


def seam_index(arr):
    luma = arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx = np.abs(np.diff(luma, axis=1))
    gy = np.abs(np.diff(luma, axis=0))
    border_x = (np.arange(gx.shape[1]) % 16) == 15
    border_y = (np.arange(gy.shape[0]) % 16) == 15
    return float(0.5 * (gx[:, border_x].mean() / gx[:, ~border_x].mean()
                        + gy[border_y].mean() / gy[~border_y].mean()))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / mse))


def decode(model, vae, x0):
    samples = model.model.process_latent_out(x0).cpu()
    (images,) = pipeline.call_node("VAEDecodeTiled", samples={"samples": samples}, vae=vae, tile_size=512,
                                   overlap=64, temporal_size=64, temporal_overlap=8)
    return (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def degraded_start(model, vae, x0, pred):
    """The composition through 2x Lanczos down + up in pixels: an 'upscaled' input to refine."""
    arr = decode(model, vae, x0)
    img = Image.fromarray(arr)
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS).resize(img.size, Image.LANCZOS)
    small.save(os.path.join(OUT, "b_start.png"))
    pixels = torch.from_numpy(np.asarray(small)).float()[None] / 255
    (latent,) = pipeline.call_node("VAEEncode", pixels=pixels, vae=vae)
    return model.model.process_latent_in(latent["samples"].to(pred.device)).float()


def part_b(pred, model, vae, x0):
    start = degraded_start(model, vae, x0, pred)
    levels = noise_levels(start.shape, prog.SEED + 20, pred.device)
    t0 = time.perf_counter()
    schedule = (SIGMA0, SIGMA1, PASSES)
    ref = decode(model, vae, run_reference(pred, start, levels, schedule))
    Image.fromarray(ref).save(os.path.join(OUT, "b_reference.png"))
    spread_img = decode(model, vae, run_reference(pred, start, noise_levels(start.shape, prog.SEED + 21, pred.device),
                                                  schedule))
    Image.fromarray(spread_img).save(os.path.join(OUT, "b_reference_seed2.png"))
    print(json.dumps({"part": "b", "variant": "reference", "s": round(time.perf_counter() - t0, 1),
                      "psnr_seed2": round(psnr(spread_img, ref), 2), "detail": round(prog.detail(ref), 2),
                      "seam": round(seam_index(ref), 3), "seam_seed2": round(seam_index(spread_img), 3)}), flush=True)
    for k, kern, noise, ctx in B_VARIANTS:
        name = f"k{k}_{kern}_{noise}{'_ctx' if ctx else ''}"
        t0 = time.perf_counter()
        x, stats = run_splats(pred, start, levels, k, kern, noise, ctx, seed=7)
        seconds = time.perf_counter() - t0
        arr = decode(model, vae, x)
        Image.fromarray(arr).save(os.path.join(OUT, f"b_{name}.png"))
        print(json.dumps({"part": "b", "variant": name, "s": round(seconds, 1), **stats,
                          "psnr_ref": round(psnr(arr, ref), 2), "detail": round(prog.detail(arr), 2),
                          "seam": round(seam_index(arr), 3)}), flush=True)


def part_c(pred, model, vae, x0):
    start = degraded_start(model, vae, x0, pred)
    pre = run_reference(pred, start, noise_levels(start.shape, prog.SEED + 20, pred.device), (SIGMA0, SIGMA1, PASSES),
                        passes=C_PRE_PASSES)
    pre_img = decode(model, vae, pre)
    Image.fromarray(pre_img).save(os.path.join(OUT, "c_start.png"))
    levels = noise_levels(start.shape, prog.SEED + 30, pred.device)
    t0 = time.perf_counter()
    ref = decode(model, vae, run_reference(pred, pre, levels, C_SCHEDULE))
    Image.fromarray(ref).save(os.path.join(OUT, "c_reference.png"))
    spread_img = decode(model, vae, run_reference(pred, pre, noise_levels(start.shape, prog.SEED + 31, pred.device),
                                                  C_SCHEDULE))
    Image.fromarray(spread_img).save(os.path.join(OUT, "c_reference_seed2.png"))
    print(json.dumps({"part": "c", "variant": "reference", "s": round(time.perf_counter() - t0, 1),
                      "psnr_seed2": round(psnr(spread_img, ref), 2), "psnr_start": round(psnr(pre_img, ref), 2),
                      "detail": round(prog.detail(ref), 2), "detail_start": round(prog.detail(pre_img), 2)}),
          flush=True)
    for k, kern, noise, ctx in C_VARIANTS:
        name = f"k{k}_{kern}_{noise}_ring{ctx}"
        t0 = time.perf_counter()
        x, stats = run_splats(pred, pre, levels, k, kern, noise, ctx, seed=7, schedule=C_SCHEDULE)
        seconds = time.perf_counter() - t0
        arr = decode(model, vae, x)
        Image.fromarray(arr).save(os.path.join(OUT, f"c_{name}.png"))
        print(json.dumps({"part": "c", "variant": name, "s": round(seconds, 1), **stats,
                          "psnr_ref": round(psnr(arr, ref), 2), "detail": round(prog.detail(arr), 2)}), flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["a", "b"]
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": NATIVE[0], "height": NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        (latent,) = pipeline.call_node("EmptyLatentImage", width=NATIVE[0], height=NATIVE[1], batch_size=1)
        (latent,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                       steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple", positive=positive,
                                       negative=positive, latent_image=latent, start_at_step=0, end_at_step=8,
                                       return_with_leftover_noise="disable")
        pred = Predictor(model, positive, latent["samples"].shape)
        x0 = model.model.process_latent_in(latent["samples"].to(pred.device)).float()
        if x0.ndim != 5 or x0.shape[2] != 1:
            raise ValueError(f"expected a [B, C, 1, H, W] Krea2 latent, got {tuple(x0.shape)}")
        Image.fromarray(decode(model, vae, x0)).save(os.path.join(OUT, "composition.png"))
        if "a" in parts:
            part_a(pred, model, positive, latent, x0)
        if "b" in parts:
            part_b(pred, model, vae, x0)
        if "c" in parts:
            part_c(pred, model, vae, x0)
