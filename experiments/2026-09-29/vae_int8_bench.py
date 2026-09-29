"""Needs the int4 extension at d21d717 (the int8 path was removed in 5f98c0d: fp16 won).
The int4 extension's int8 VAE path (vae.patch_vae) end to end, against the fp32 VAE and an
fp16 VAE, all with MIOpen off (the growth path's setting), tiled 512/64 like the growth
node. Per size and variant: decode / encode seconds (best of 2 after a warm-up), peak
allocated GB, PSNR vs fp32 (decode of the fp32 latent; encode+fp32-decode; both).

Usage: vae_int8_bench.py <device> <image.png> [WxH ...]   (image resized to each size)"""
import importlib, json, math, os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/custom_nodes")
int8_vae = importlib.import_module("ComfyUI-Gfx906Int4.vae")

VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"
TILE = dict(tile_x=512 // 8, tile_y=512 // 8, overlap=64 // 8)
TILE_PX = dict(tile_x=512, tile_y=512, overlap=64)

torch.backends.cudnn.enabled = False
device = torch.device(sys.argv[1])
torch.cuda.set_device(device)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME))
fp32 = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float32)
int8, count = int8_vae.patch_vae(fp32)
fp16 = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float16)
int8_fp16, _ = int8_vae.patch_vae(fp16)
print(json.dumps({"int8_convs_patched": count}), flush=True)


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def best(fn):
    out, _, _ = timed(fn)
    runs = [timed(fn) for _ in range(2)]
    return out, min(r[1] for r in runs), max(r[2] for r in runs)


def to_u8(img):
    return (img.reshape(-1, *img.shape[-3:])[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return round(10 * math.log10(255 ** 2 / max(mse, 1e-9)), 2)


src_img = Image.open(sys.argv[2]).convert("RGB")
with torch.inference_mode():
    for res in sys.argv[3:] or ["2560x1440"]:
        w, h = (int(v) for v in res.split("x"))
        image = torch.from_numpy(np.asarray(src_img.resize((w, h), Image.LANCZOS))).float().div(255)[None]
        ref_latent = fp32.encode_tiled(image, **TILE_PX)
        ref = None
        for name, vae in (("fp32", fp32), ("int8", int8), ("fp16", fp16), ("int8_fp16", int8_fp16)):
            dec, dec_s, dec_gb = best(lambda: vae.decode_tiled(ref_latent, **TILE))
            lat, enc_s, enc_gb = best(lambda: vae.encode_tiled(image, **TILE_PX))
            dec = to_u8(dec)
            row = {"res": res, "vae": name, "decode_s": round(dec_s, 2), "encode_s": round(enc_s, 2),
                   "decode_peak_gb": round(dec_gb, 2), "encode_peak_gb": round(enc_gb, 2)}
            if ref is None:
                ref = dec
            else:
                row.update(decode_db=psnr(dec, ref), encode_db=psnr(to_u8(fp32.decode_tiled(lat, **TILE)), ref),
                           latent_rel_err=round(float((lat.float() - ref_latent).norm() / ref_latent.norm()), 4))
            print(json.dumps(row), flush=True)
