"""VRAM of the Qwen-Image VAE through ComfyUI's own nodes (VAEDecodeTiled / VAEEncodeTiled
512/64 = the growth path, VAEDecode / VAEEncode = a plain workflow), MIOpen off. Per variant
(fp32, fp16, int8 = int4 extension vae.patch_vae on fp16) and call: ComfyUI's memory
estimate (what load_models_gpu frees room for), peak allocated, peak reserved (what a GPU
monitor sees, minus the context), seconds.

Usage: vae_mem_bench.py <device> WxH [...]"""
import importlib, json, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402

sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/custom_nodes")
int8_vae = importlib.import_module("ComfyUI-Gfx906Int4.vae")
VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"

torch.backends.cudnn.enabled = False
device = torch.device(sys.argv[1])
torch.cuda.set_device(device)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME))


def make(variant):
    vae = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float32 if variant == "fp32" else torch.float16)
    return int8_vae.patch_vae(vae)[0] if variant == "int8" else vae


def measure(vae, name, fn, estimate):
    mm.soft_empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc, base_res = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, {"call": name, "estimate_gb": round(estimate / 2**30, 2),
                 "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                 "before_alloc_gb": round(base_alloc / 2**30, 2), "before_reserved_gb": round(base_res / 2**30, 2),
                 "seconds": round(time.perf_counter() - t0, 2)}


with torch.inference_mode():
    for res in sys.argv[2:]:
        w, h = (int(v) for v in res.split("x"))
        image = torch.rand(1, h, w, 3)
        latent = torch.randn(1, 16, 1, h // 8, w // 8, generator=torch.Generator().manual_seed(0))
        for variant in ("fp32", "fp16", "int8"):
            vae = make(variant)
            vae.decode_tiled({"samples": latent}["samples"], tile_x=64, tile_y=64, overlap=8)  # warm-up / load
            est_dec_tile = vae.memory_used_decode(vae._tile_bounded_shape(latent.shape, 64, 64, None), vae.vae_dtype)
            est_dec = vae.memory_used_decode(latent.shape, vae.vae_dtype)
            est_enc = vae.memory_used_encode((1, 3, 1, h, w), vae.vae_dtype)
            calls = (("decode_tiled", lambda: nodes.VAEDecodeTiled().decode(vae, {"samples": latent}, 512, 64), est_dec_tile),
                     ("encode_tiled", lambda: nodes.VAEEncodeTiled().encode(vae, image, 512, 64), est_enc),
                     ("decode", lambda: nodes.VAEDecode().decode(vae, {"samples": latent}), est_dec),
                     ("encode", lambda: nodes.VAEEncode().encode(vae, image), est_enc))
            for name, fn, est in calls:
                _, row = measure(vae, name, fn, est)
                print(json.dumps({"res": res, "vae": variant, **row}), flush=True)
            del vae
            mm.unload_all_models()
            mm.soft_empty_cache()
