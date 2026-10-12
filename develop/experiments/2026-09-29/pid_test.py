"""PiD (NVIDIA PixelDiT latent->pixel decoder, qwenimage 2kto4k) on Krea2 latents:
Krea2 decodes with the Qwen-Image VAE, so the qwenimage PiD reads its latents as-is.
Compose at 1024x576 (8 steps), then PiD Decode 4x -> 4096x2304 with no Krea2 pass above
native size (no RoPE extrapolation at all). Reference: VAE decode + Lanczos 4x.
Upstream ships bf16 only; comfy's loader picks the dtype for this GPU.

Two processes, because PiD's "unload comfy models" does not free the int4 Krea2 model
and the Gemma TE then OOMs on the same 16 GB GPU:
  pid_test.py compose   -> latent to develop/out/progressive/pid_base_latent.pt + reference
  pid_test.py decode    -> PiD only (no Krea2 loaded) -> pid_decode_<steps>step.png
Prints times + detail."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402

import asyncio  # noqa: E402
import nodes  # noqa: E402
# progressive.py already loaded its whitelist (Krea2 path only) at import; add PiD on top
asyncio.run(nodes.load_custom_node(os.path.join(os.path.dirname(nodes.__file__), "custom_nodes", "ComfyUI-PiD")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import pipeline  # noqa: E402

BASE = (1024, 576)
LATENT_FILE = os.path.join(prog.OUT, "pid_base_latent.pt")


def save(images, name, seconds):
    arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(prog.OUT, f"pid_{name}.png"))
    print(json.dumps({"variant": name, "size": list(arr.shape[1::-1]), "seconds": round(seconds, 1),
                      "detail": round(prog.detail(arr), 2)}), flush=True)


def compose():
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": BASE[0], "height": BASE[1]}})
    base_model, clip, vae = pipeline.build_loaders(resolved)
    (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                  sparse_tau=0.0, max_missed=0.02)
    (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
    (latent,) = pipeline.call_node("EmptyLatentImage", width=BASE[0], height=BASE[1], batch_size=1)
    t0 = time.perf_counter()
    (latent,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                   steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple", positive=positive,
                                   negative=positive, latent_image=latent, start_at_step=0, end_at_step=8,
                                   return_with_leftover_noise="disable")
    print(json.dumps({"composition_s": round(time.perf_counter() - t0, 1)}), flush=True)
    torch.save(latent["samples"].cpu(), LATENT_FILE)
    t0 = time.perf_counter()
    (images,) = pipeline.call_node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=512, overlap=64,
                                   temporal_size=64, temporal_overlap=8)
    (images,) = pipeline.call_node("ImageScale", image=images, upscale_method="lanczos", width=BASE[0] * 4,
                                   height=BASE[1] * 4, crop="disabled")
    save(images, "vae_lanczos4x", time.perf_counter() - t0)


def force_pid_dtype(dtype):
    """comfy lists PixelDiT as bf16|fp32 only, so gfx906 (no fast bf16) gets fp32 and the
    4096x2304 per-pixel adaLN tensors OOM 16 GB. fp16 (Luna OK'd) runs (110 s) but gives
    an all-NaN image (overflow -- why upstream ships bf16 only); bf16 has fp32's range
    and fp16's memory. Wrap the node's loader instead of editing the node."""
    import comfy.sd
    load = comfy.sd.load_diffusion_model

    def load_forced(path, model_options={}):
        model = load(path, model_options={**model_options, "dtype": dtype})
        print(f"PiD diffusion model dtype: {model.model.get_dtype()}; {vram()}", flush=True)
        return model
    comfy.sd.load_diffusion_model = load_forced


def vram():
    return ", ".join(f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.2f} GiB"
                     for i in range(torch.cuda.device_count()))


def gemma_on_second_gpu():
    """The Gemma TE (5.2 GB) lands on cuda:0 next to PiD and stays there: move it to cuda:1
    via ComfyUI-MultiGPU's text-encoder device (it patches mm.text_encoder_device)."""
    import comfy.sd
    # by name: torch.ops-style namespaces answer hasattr() for anything
    multigpu = next(m for name, m in list(sys.modules.items())
                    if name.endswith("ComfyUI-MultiGPU") and hasattr(m, "set_current_text_encoder_device"))
    multigpu.set_current_text_encoder_device(torch.device("cuda:1"))
    load_clip = comfy.sd.load_clip

    def load_clip_logged(*args, **kwargs):
        clip = load_clip(*args, **kwargs)
        print(f"PiD TE loaded on {clip.patcher.load_device}; {vram()}", flush=True)
        return clip
    comfy.sd.load_clip = load_clip_logged


def decode(steps, dtype):
    force_pid_dtype(dtype)
    gemma_on_second_gpu()
    latent = {"samples": torch.load(LATENT_FILE)}
    t0 = time.perf_counter()
    (images,) = pipeline.call_node("PiDDecode", latent=latent, backbone="qwenimage", pid_ckpt_type="2kto4k",
                                   pid_steps=steps, scale=0, cfg_scale=1.0, sigma=0.0, seed=0,
                                   auto_download=False, model_precision="bf16", unload_comfy_before_pid=True,
                                   aggressive_cleanup=True, caption=prog.PROMPT)
    arr = images[0].float()
    print(f"output: nan {arr.isnan().float().mean().item():.3f}, min {arr.nan_to_num().min().item():.3f}, "
          f"max {arr.nan_to_num().max().item():.3f}", flush=True)
    save(images, f"decode_{steps}step_{str(dtype).split('.')[-1]}", time.perf_counter() - t0)


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
with torch.inference_mode():
    if sys.argv[1:] == ["compose"]:
        compose()
    elif len(sys.argv) == 3 and sys.argv[1] == "decode" and sys.argv[2] in DTYPES:
        decode(4, DTYPES[sys.argv[2]])
    else:
        sys.exit(f"usage: pid_test.py compose | decode {{{'|'.join(DTYPES)}}}, got {sys.argv[1:]}")
