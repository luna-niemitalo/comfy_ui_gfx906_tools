"""Is the fp16 Qwen-Image VAE safe with MIOpen off (rocBLAS convs)? Per image (native size,
cropped to a multiple of 8) plus two synthetic stress images (hard black/white checker,
saturated colour noise): decode of the fp32 latent and encode, fp16 vs fp32 -- PSNR, max
abs pixel error, non-finite values. Tiled 512/64 like the growth node.

Usage: vae_fp16_check.py <device> <image.png> [...]"""
import json, math, os, sys
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

VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"
TILE = dict(tile_x=512 // 8, tile_y=512 // 8, overlap=64 // 8)
TILE_PX = dict(tile_x=512, tile_y=512, overlap=64)

torch.backends.cudnn.enabled = False
device = torch.device(sys.argv[1])
torch.cuda.set_device(device)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME))
fp32 = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float32)
fp16 = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float16)


def stress(kind):
    rng = np.random.default_rng(0)
    if kind == "checker":
        y, x = np.mgrid[:1440, :2560]
        return (((y // 2 + x // 2) % 2) * 255).astype(np.uint8)[..., None].repeat(3, -1)
    return (rng.integers(0, 2, (1440, 2560, 3)) * 255).astype(np.uint8)


def compare(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    finite = bool(torch.isfinite(b).all())
    d = (a - b).abs()
    mse = float((d ** 2).mean())
    return finite, round(10 * math.log10(1 / max(mse, 1e-12)), 1), round(float(d.max()), 4)


sources = [(os.path.basename(p), np.asarray(Image.open(p).convert("RGB"))) for p in sys.argv[2:]]
sources += [("stress_checker", stress("checker")), ("stress_colour_noise", stress("noise"))]
with torch.inference_mode():
    for name, arr in sources:
        h, w = arr.shape[0] // 8 * 8, arr.shape[1] // 8 * 8
        image = torch.from_numpy(arr[:h, :w].copy()).float().div(255)[None]
        lat32 = fp32.encode_tiled(image, **TILE_PX)
        lat16 = fp16.encode_tiled(image, **TILE_PX)
        dec32 = fp32.decode_tiled(lat32, **TILE)
        dec16 = fp16.decode_tiled(lat32, **TILE)
        enc_ok = bool(torch.isfinite(lat16.float()).all())
        dec_ok, dec_db, dec_max = compare(dec32, dec16)
        print(json.dumps({"image": name, "size": f"{w}x{h}", "encode_finite": enc_ok,
                          "latent_rel_err": round(float((lat16.float().cpu() - lat32.float().cpu()).norm() / lat32.float().cpu().norm()), 4),
                          "decode_finite": dec_ok, "decode_db": dec_db, "decode_max_abs": dec_max}), flush=True)
