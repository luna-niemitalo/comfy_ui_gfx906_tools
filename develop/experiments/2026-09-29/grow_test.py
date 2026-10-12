"""Continuous resolution growth (Luna: "the bigger resolution is just a different
dimension on the same image"; instead of 2x/3x jumps, every step runs a little larger).
Compose at NATIVE (8 steps, clean), then for k = 1..n:
  size_k  = native * ratio ** ((k / n) ** SIZE_SHAPE)     (geometric growth, 16 px grid)
  x_k     = (1 - sigma_k) * pixel_up(x0_{k-1}, size_k) + sigma_k * fresh noise
  x0_k    = one tiled denoise at sigma_k (tiled_test.py: 1024 px tiles, overlap 256,
            subject-free DETAIL_PROMPT), low band anchored to the composition with
            weight ANCHOR * (1 - (k - 1) / n) ** ANCHOR_SHAPE
sigma_1..n from the model's simple schedule between SIGMA_START and 0 (progressive.py).
Winner to beat at 2560x1440: tiled_f8_pixel_s0.5_2step_hold_detailprompt (78 s);
at 3840x2160: pid_decode_4step_bfloat16 (4096x2304, ~160 s).

Usage: grow_test.py [--split] <variant> [...]  -> develop/out/progressive/grow_<variant>[_split].png
  --split: Gfx906RowSplitKrea2 on the model -- tile batches split across both GPUs (by batch,
           bit-identical), the native composition by image rows."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402
import tiled_test as tt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

TILE_PX, OVERLAP_PX = 1024, 256  # tiled_test.py's winning tile setup
SIGMA_START = 0.5
SIZE_SHAPE = 1.0    # > 1: grow late, < 1: grow early
ANCHOR = 1.0
ANCHOR_SHAPE = 2.0  # anchor fades as (1 - (k - 1) / n) ** shape
# name: (target w, target h, growth steps)
VARIANTS = {
    "n4_2560": (2560, 1440, 4),
    "n6_2560": (2560, 1440, 6),
    "n6_3840": (3840, 2160, 6),
    # repro: n6_3840 segfaults (libhsa-runtime64, same ip, 1 and 2 GPUs) entering its
    # 3840x2160 step; this reaches that size in one step
    "n1_3840": (3840, 2160, 1),
    "n2_3840": (3840, 2160, 2),  # repro: 2560x1440-ish step then the 3840x2160 step
}


def log_model_loads():
    """Repro instrumentation: comfy sizes its load from the WHOLE latent (not our tiles) --
    log the estimate, the partial-load state and where an int4 weight lives, per load."""
    import comfy.model_management as mm
    load = mm.load_models_gpu

    def logged(models, memory_required=0, *args, **kwargs):
        out = load(models, memory_required, *args, **kwargs)
        for lm in mm.current_loaded_models:
            m = lm.model
            if not hasattr(m.model, "diffusion_model"):
                continue
            w = next(p for n, p in m.model.diffusion_model.named_parameters() if "blocks.0" in n)
            print(f"load: required {memory_required / 2**30:.1f} GiB, loaded {m.loaded_size() / 2**30:.2f} "
                  f"of {m.model_size() / 2**30:.2f} GiB, lowvram {getattr(m.model, 'model_lowvram', '?')}, "
                  f"blocks.0 weight on {w.device}; free cuda:0 {mm.get_free_memory(torch.device('cuda:0')) / 2**30:.1f} GiB",
                  flush=True)
        return out
    mm.load_models_gpu = logged


def growth_sizes(target, n):
    native_w, native_h = prog.NATIVE
    ratio = target[0] / native_w
    sizes = []
    for k in range(1, n + 1):
        f = ratio ** ((k / n) ** SIZE_SHAPE)
        sizes.append((16 * round(native_w * f / 16), 16 * round(native_h * f / 16)))
    return sizes


if __name__ == "__main__":
    args = sys.argv[1:]
    split = "--split" in args
    args = [a for a in args if a != "--split"]
    names = list(VARIANTS) if args == ["all"] else args
    if split:
        tt.TILE_BATCH = 4  # 2 tiles per GPU per call (3 on one GPU was the single-GPU limit)
    log_model_loads()
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        if split:
            (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device="auto", min_image_tokens=2048)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        (detail_positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=tt.DETAIL_PROMPT)
        (detail_guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=detail_positive)
        (off,) = pipeline.call_node("DisableNoise")
        latent_in = lambda x: model.model.process_latent_in(x.to(model.load_device)).float()  # noqa: E731
        latent_out = model.model.process_latent_out

        (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
        t0 = time.perf_counter()
        (composition,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                            steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                            positive=positive, negative=positive, latent_image=latent,
                                            start_at_step=0, end_at_step=8, return_with_leftover_noise="disable")
        native_s = time.perf_counter() - t0
        ref_model = latent_in(composition["samples"])

        for name in names:
            w, h, n = VARIANTS[name]
            sizes = growth_sizes((w, h), n)
            sigmas = prog.stage_sigmas(model, SIGMA_START, n)[:-1]
            generator = torch.Generator().manual_seed(prog.SEED + 1)
            print(f"{name}: sizes {sizes}, sigmas {[round(float(s), 3) for s in sigmas]}", flush=True)
            t1 = time.perf_counter()
            x0 = composition
            for k, ((sw, sh), sigma) in enumerate(zip(sizes, sigmas), start=1):
                sigma = float(sigma)
                ts = time.perf_counter()
                # growth sizes never repeat, so MIOpen's per-shape build (~8 s per new size,
                # vae_shape_test.py) never pays off here: plain im2col + rocBLAS for these VAE
                # calls only (Luna: MIOpen off in the growth path; the rest keeps it)
                with torch.backends.cudnn.flags(enabled=False):
                    up = latent_in(tt.pixel_upscale(x0, vae, sw, sh))
                # repro: the 3840x2160 step segfaults in libhsa-runtime with async launches but
                # passes with AMD_SERIALIZE_KERNEL=3; the VAE runs on cuda:1 (recipe), the UNet on
                # cuda:0 -- sync BOTH before the denoise to test the cross-device hand-off
                for d in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(d)
                upscale_s = time.perf_counter() - ts
                noise = torch.randn(up.shape, generator=generator).to(up)
                x_t = (1 - sigma) * up + sigma * noise
                anchor = ANCHOR * (1 - (k - 1) / n) ** ANCHOR_SHAPE
                # one er_sde step from sigma to 0 = the model's clean estimate at this size
                sampler = tt.tiled_sampler(ref_model, anchor, (0.0, 0.01), sigma, TILE_PX // 8, OVERLAP_PX // 8)
                # comfy's flow sampling scales the input latent by (1 - sigma) (noise_scaling)
                (x0, _) = pipeline.call_node("SamplerCustomAdvanced", noise=off, guider=detail_guider,
                                             sampler=sampler, sigmas=torch.tensor([sigma, 0.0]),
                                             latent_image={"samples": latent_out(x_t / (1 - sigma))})
                torch.cuda.synchronize()
                print(f"  step {k} {sw}x{sh} sigma {sigma:.3f}: upscale {upscale_s:.1f} s, "
                      f"denoise {time.perf_counter() - ts - upscale_s:.1f} s", flush=True)
            grow_s = time.perf_counter() - t1
            (images,) = pipeline.call_node("VAEDecodeTiled", samples=x0, vae=vae, tile_size=512, overlap=64,
                                           temporal_size=64, temporal_overlap=8)
            arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(prog.OUT, f"grow_{name}{'_split' if split else ''}.png"))
            print(json.dumps({"variant": name, "split": split, "native_s": round(native_s, 1), "grow_s": round(grow_s, 1),
                              "detail": round(prog.detail(arr), 2)}), flush=True)
