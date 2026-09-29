"""comfy/ops.py Conv3d T=1 -> conv2d path: same decode as the old 3-D path?
Decodes one random 1024x576 latent through the Qwen-Image VAE with the
patched ops and with the pre-patch _conv_forward body; prints max diff + times."""
import os, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import torch  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

Conv3d = comfy.ops.disable_weight_init.Conv3d
patched = Conv3d._conv_forward


def pre_patch(self, input, weight, bias, autopad=None, *args, **kwargs):
    if autopad == "causal_zero":
        weight = weight[:, :, -input.shape[2]:, :, :]
    return torch.nn.Conv3d._conv_forward(self, input, weight, bias)


device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
torch.cuda.set_device(device)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", "qwenImageVAE_qwenImageVAE.safetensors"))
latent = torch.randn(1, 16, 1, 72, 128, generator=torch.Generator().manual_seed(0))
with torch.inference_mode():
    vae = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float32)
    out = {}
    for name, fn in (("pre_patch", pre_patch), ("patched", patched)):
        Conv3d._conv_forward = fn
        vae.decode_tiled(latent, tile_x=64, tile_y=64, overlap=8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out[name] = vae.decode_tiled(latent, tile_x=64, tile_y=64, overlap=8).float().cpu()
        torch.cuda.synchronize()
        print(f"{name}: {time.perf_counter() - t0:.2f} s", flush=True)
    Conv3d._conv_forward = patched
    print("max abs diff:", (out["patched"] - out["pre_patch"]).abs().max().item())
