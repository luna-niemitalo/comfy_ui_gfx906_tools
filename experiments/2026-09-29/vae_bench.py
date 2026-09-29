"""VAE decode cost + two speedups for single images (Krea2 / Qwen-Image VAE =
Wan 2.1 VAE, 5-D latents with T=1):
  conv2d -- run a 3-D conv whose input has one frame as a 2-D conv (causal convs
            already slice the kernel to its last frame for T=1; MIOpen still
            runs that as 3-D)
  fp16   -- VAE weights/activations in fp16 instead of the loader's choice
Decodes the same random latent per resolution with the recipe's tiled settings
and reports seconds and max/mean abs difference vs the fp32 unpatched decode.

Usage: vae_bench.py <device> WxH [...]"""
import json, os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"
TILED = dict(tile_x=512 // 8, tile_y=512 // 8, overlap=64 // 8)
original_conv_forward = comfy.ops.disable_weight_init.Conv3d._conv_forward


def conv3d_as_2d(self, input, weight, bias, autopad=None, *args, **kwargs):
    single = input.shape[2] == 1 and self.padding[0] == 0 and self.stride[0] == 1 and (
        autopad == "causal_zero" or weight.shape[2] == 1)
    if not single:
        return original_conv_forward(self, input, weight, bias, autopad, *args, **kwargs)
    w2 = weight[:, :, -1]
    return F.conv2d(input[:, :, 0], w2, bias, self.stride[1:], self.padding[1:], self.dilation[1:],
                    self.groups).unsqueeze(2)


def decode(vae, latent):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = vae.decode_tiled(latent, **TILED)
    torch.cuda.synchronize()
    return img, time.perf_counter() - t0


device = torch.device(sys.argv[1])
torch.cuda.set_device(device)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME))
variants = {}
with torch.inference_mode():
    for dtype in (torch.float32, torch.float16):
        vae = comfy.sd.VAE(sd=sd, device=device, dtype=dtype)
        for res in sys.argv[2:]:
            w, h = (int(x) for x in res.split("x"))
            g = torch.Generator().manual_seed(0)
            latent = torch.randn(1, 16, 1, h // 8, w // 8, generator=g)
            for patched in (False, True):
                comfy.ops.disable_weight_init.Conv3d._conv_forward = conv3d_as_2d if patched else original_conv_forward
                decode(vae, latent[..., :64, :64])  # warm up MIOpen kernel selection for this config
                img, seconds = decode(vae, latent)
                key = (res, str(dtype).split(".")[-1], "conv2d" if patched else "conv3d")
                ref = variants.get((res, "float32", "conv3d"))
                rec = {"res": res, "dtype": key[1], "conv": key[2], "seconds": round(seconds, 2),
                       "nan": bool(torch.isnan(img).any())}
                if ref is not None:
                    d = (img.float().cpu() - ref.float()).abs()
                    rec.update(max_diff=round(d.max().item(), 4), mean_diff=round(d.mean().item(), 5))
                else:
                    variants[key] = img.cpu()
                print(json.dumps(rec), flush=True)
        comfy.ops.disable_weight_init.Conv3d._conv_forward = original_conv_forward
        del vae
        torch.cuda.empty_cache()
