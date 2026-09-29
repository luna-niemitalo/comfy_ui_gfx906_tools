"""Can the Qwen-Image (Wan 2.1) VAE's 3x3 convs run on the int4 GEMM (W4A4, 32-element
blocks along K, one fp32 scale per block)? Quality study by exact emulation in fp32:

A 3x3 conv as a GEMM has K = (tap, channel). Ordering K tap-major, every 32-block of the
im2col row is 32 channels of ONE input pixel, so quantizing x per (pixel, 32 channels)
before the conv equals quantizing the im2col matrix blockwise -- the kernel's scheme.
Weights: per (out channel, tap, 32 in channels). Optional Hadamard over each 32-channel
group of x and w (orthogonal: the conv result is unchanged before quantization).
The first conv (3 / 16 in channels) and the 96->3 output conv stay fp32 (tiny K / N).

Image: a real render (arg 2), encoded fp32 -> reference latent -> fp32 decode = reference.
Per variant: decode PSNR (int decoder on the reference latent vs the reference image),
encode (int encoder, fp32 decoder) PSNR, and both. Saves PNGs to develop/out/vae_int/.

Usage: vae_int_study.py <device> <image.png>"""
import json, math, os, sys
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"
TILE = dict(tile_x=512 // 8, tile_y=512 // 8, overlap=64 // 8)
TILE_PX = dict(tile_x=512, tile_y=512, overlap=64)
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/vae_int"
BLOCK = 32
os.makedirs(OUT, exist_ok=True)

torch.backends.cudnn.enabled = False
device = torch.device(sys.argv[1])
torch.cuda.set_device(device)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME)),
                   device=device, dtype=torch.float32)
hadamard = None


def block_quant(t, dim, bits):
    """Symmetric fake quantization in blocks of 32 along dim (the kernel's rounding)."""
    qmax = 2 ** (bits - 1) - 1
    t = t.movedim(dim, -1)
    shape = t.shape
    tb = t.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    scale = tb.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (tb / scale).round().clamp(-qmax - 1, qmax) * scale
    return q.reshape(shape).movedim(-1, dim)


def rotate(t, dim):
    t = t.movedim(dim, -1)
    shape = t.shape
    return (t.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK) @ hadamard).reshape(shape).movedim(-1, dim)


def make_conv(module, w_bits, a_bits, rot):
    original = module._conv_forward

    def conv(input, weight, bias, autopad=None, *args, **kwargs):
        if weight.dim() == 5:
            if input.shape[2] != 1 or autopad != "causal_zero" and weight.shape[2] != 1:
                return original(input, weight, bias, autopad, *args, **kwargs)
            x, w = input[:, :, 0], weight[:, :, -1]
            stride, padding, dilation = module.stride[1:], module.padding[1:], module.dilation[1:]
        else:
            x, w = input, weight
            stride, padding, dilation = module.stride, module.padding, module.dilation
        if rot:
            x, w = rotate(x, 1), rotate(w, 1)
        x = block_quant(x, 1, a_bits)
        w = block_quant(w, 1, w_bits)
        out = F.conv2d(x, w, bias, stride, padding, dilation, module.groups)
        return out.unsqueeze(2) if weight.dim() == 5 else out
    return conv


def patch(part, w_bits, a_bits, rot):
    """Patch every conv of part ('encoder' | 'decoder') with >= 32 in and >= 32 out channels;
    returns an undo function."""
    patched = []
    for m in getattr(vae.first_stage_model, part).modules():
        w = getattr(m, "weight", None)
        if w is None or w.dim() < 4 or w.shape[1] % BLOCK or w.shape[0] < BLOCK:
            continue
        m._conv_forward = make_conv(m, w_bits, a_bits, rot)
        patched.append(m)

    def undo():
        for m in patched:
            del m._conv_forward
    return undo


def to_u8(img):
    return (img.reshape(-1, *img.shape[-3:])[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return round(10 * math.log10(255 ** 2 / max(mse, 1e-9)), 2)


with torch.inference_mode():
    # 32x32 Sylvester Hadamard, normalized
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < BLOCK:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    hadamard = h / math.sqrt(BLOCK)

    src = np.asarray(Image.open(sys.argv[2]).convert("RGB"))
    image = torch.from_numpy(src).float().div(255)[None]
    ref_latent = vae.encode_tiled(image, **TILE_PX)
    ref = to_u8(vae.decode_tiled(ref_latent, **TILE))
    Image.fromarray(ref).save(os.path.join(OUT, "fp32.png"))
    print(json.dumps({"variant": "fp32", "vs_source_db": psnr(ref, src)}), flush=True)

    for name, w_bits, a_bits, rot in (("w8a8", 8, 8, False), ("w4a4", 4, 4, False), ("w4a4_had", 4, 4, True),
                                      ("w4a8_had", 4, 8, True)):
        undo = patch("decoder", w_bits, a_bits, rot)
        dec = to_u8(vae.decode_tiled(ref_latent, **TILE))
        undo()
        undo = patch("encoder", w_bits, a_bits, rot)
        lat = vae.encode_tiled(image, **TILE_PX)
        undo()
        enc = to_u8(vae.decode_tiled(lat, **TILE))
        undo_d, undo_e = patch("decoder", w_bits, a_bits, rot), patch("encoder", w_bits, a_bits, rot)
        both = to_u8(vae.decode_tiled(vae.encode_tiled(image, **TILE_PX), **TILE))
        undo_d()
        undo_e()
        Image.fromarray(dec).save(os.path.join(OUT, f"{name}_decode.png"))
        Image.fromarray(both).save(os.path.join(OUT, f"{name}_both.png"))
        lat_err = float((lat - ref_latent).norm() / ref_latent.norm())
        print(json.dumps({"variant": name, "decode_db": psnr(dec, ref), "encode_db": psnr(enc, ref),
                          "latent_rel_err": round(lat_err, 4), "both_db": psnr(both, ref),
                          "both_vs_source_db": psnr(both, src)}), flush=True)
