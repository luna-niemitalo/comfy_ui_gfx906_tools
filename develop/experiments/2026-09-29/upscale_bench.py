"""Growth-step upscale: faster paths (Luna's list, 2026-09-29). One GPU, VAEs only.
Inputs: real content -- step images of the 4K geometric growth run (scale study), encoded
with the full VAE; each is upscaled to the NEXT step's size by:
  A cpu        current node path as the harness runs it: tiled decode to host, ComfyUI
               lanczos (8-bit PIL, CPU), tiled encode
  B gpu        tiled decode kept on the GPU, torch separable Lanczos-3 (float), tiled encode
  C tae        lighttaew2_1 decode + encode (tiny VAE), torch Lanczos
  D tae_dec    tiny decode, full encode
  E tae_enc    full decode, tiny encode
  F lat_bicubic   bicubic resize of the latent itself (no VAE)
  G lat_lanczos   Lanczos resize of the latent itself (no VAE)
Quality proxy: every result decoded with the full VAE vs B's (the float ideal), PSNR at
the target size. Proxy only: the next growth step re-denoises; winners get a growth run.
MIOpen off throughout (as the node's growth path). Prints one JSON line per method x size.

Usage: upscale_bench.py [device]   (default cuda:1)"""
import json, math, os, sys, time
sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/develop/harness")
import bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402

DEVICE = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
RUN = "/media/luna/work/tinker/ComfyUI/develop/out/progressive/scale/geometric_3840x2160_6"
# (input step image, next step's size): a mid and the top 4K transition of that run
TRANSITIONS = [("step_3_2224x1248.png", (2656, 1504)), ("step_5_3200x1792.png", (3840, 2160))]
TAE_PATH = "/media/luna/models/VAE-approx/lighttaew2_1.safetensors"
torch.cuda.set_device(DEVICE)
torch.backends.cudnn.enabled = False  # the node's growth path runs its VAE calls without MIOpen


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


def lanczos(x, size):
    """x [..., H, W] float -> [..., h, w]: separable, both axes as matmuls."""
    w, h = size
    my = lanczos_matrix(x.shape[-2], h, x.device)
    mx = lanczos_matrix(x.shape[-1], w, x.device)
    return my @ x.float() @ mx.T


def lanczos_image(images, size):
    """IMAGE [B, H, W, C] -> resized, channels through the separable filter."""
    return lanczos(images.movedim(-1, 1), size).movedim(1, -1).clamp(0, 1)


def decode(vae, samples, gpu):
    vae.output_device = vae.device if gpu else torch.device("cpu")
    (images,) = nodes.VAEDecodeTiled().decode(vae, {"samples": samples}, 512, 64)
    return images.reshape(-1, *images.shape[-3:])[:1]


def encode(vae, images, gpu):
    vae.output_device = vae.device if gpu else torch.device("cpu")
    (latent,) = nodes.VAEEncodeTiled().encode(vae, images, 512, 64)
    return latent["samples"].to(DEVICE)


def lat_resize(samples, size, mode):
    """5-D [B,C,T,H,W] latent -> the latent grid of `size` (8x compression)."""
    hw = (size[1] // 8, size[0] // 8)
    x = samples.to(DEVICE).float()[:, :, 0]
    x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False) if mode == "bicubic" else lanczos(x, size=hw[::-1])
    return x.unsqueeze(2)


def psnr(a, b):
    mse = float(((a.float().to(DEVICE) - b.float().to(DEVICE)) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


full = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise(
    "vae", "qwenImageVAE_qwenImageVAE.safetensors")), device=DEVICE, dtype=torch.float32)
tae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(TAE_PATH), device=DEVICE, dtype=torch.float32)

METHODS = {
    "A_cpu": lambda s, size: encode(full, nodes.ImageScale().upscale(decode(full, s, False), "lanczos", *size, "disabled")[0], False),
    "B_gpu": lambda s, size: encode(full, lanczos_image(decode(full, s, True), size), True),
    "C_tae": lambda s, size: encode(tae, lanczos_image(decode(tae, s, True), size), True),
    "D_tae_dec": lambda s, size: encode(full, lanczos_image(decode(tae, s, True), size), True),
    "E_tae_enc": lambda s, size: encode(tae, lanczos_image(decode(full, s, True), size), True),
    "F_lat_bicubic": lambda s, size: lat_resize(s, size, "bicubic"),
    "G_lat_lanczos": lambda s, size: lat_resize(s, size, "lanczos"),
}

with torch.inference_mode():
    for name, size in TRANSITIONS:
        image = torch.from_numpy(np.asarray(Image.open(os.path.join(RUN, name)), dtype=np.float32) / 255.0)[None]
        samples = encode(full, image.to(DEVICE), True)
        results = {}
        for method, fn in METHODS.items():
            fn(samples, size)  # warm-up: first use of this shape
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results[method] = fn(samples, size)
            torch.cuda.synchronize()
            results[method + "_s"] = time.perf_counter() - t0
        reference = decode(full, results["B_gpu"], True)
        for method in METHODS:
            quality = psnr(decode(full, results[method], True), reference) if method != "B_gpu" else float("inf")
            print(json.dumps({"input": name, "to": f"{size[0]}x{size[1]}", "method": method,
                              "seconds": round(results[method + "_s"], 3),
                              "psnr_vs_B_db": None if math.isinf(quality) else round(quality, 2)}), flush=True)
