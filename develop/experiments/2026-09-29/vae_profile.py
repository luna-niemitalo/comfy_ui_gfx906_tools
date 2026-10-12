"""Where do the Qwen-Image (Wan 2.1) VAE's time and memory go with MIOpen off (the growth
path's setting)? Tiled 512/64 decode + encode of one image per size, fp32, on one GPU.
Prints, per size and direction: wall seconds (best of 2 after a warm-up), peak allocated
memory, and a per-layer-kind table from sync-bracketed forward hooks (hooks add sync
overhead, so the table's total exceeds the wall time; shares are what matters).

Usage: vae_profile.py <device> [fp32|int8|fp16] WxH [...]   (int8 = the int4 extension's vae.patch_vae)"""
import collections, json, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

VAE_NAME = "qwenImageVAE_qwenImageVAE.safetensors"
TILE = dict(tile_x=512 // 8, tile_y=512 // 8, overlap=64 // 8)
TILE_PX = dict(tile_x=512, tile_y=512, overlap=64)

torch.backends.cudnn.enabled = False
device = torch.device(sys.argv[1])
VARIANT = sys.argv[2]
torch.cuda.set_device(device)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", VAE_NAME)),
                   device=device, dtype=torch.float16 if VARIANT == "fp16" else torch.float32)
if VARIANT == "int8":
    import importlib
    sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/custom_nodes")
    vae, _ = importlib.import_module("ComfyUI-Gfx906Int4.vae").patch_vae(vae)
print(json.dumps({"weights_mb": round(sum(p.numel() * p.element_size() for p in vae.first_stage_model.parameters()) / 2**20, 1)}), flush=True)

stats = collections.defaultdict(lambda: [0, 0.0])
starts = {}


def key(module, inputs):
    x = inputs[0]
    kind = type(module).__name__
    if hasattr(module, "weight") and module.weight is not None and module.weight.dim() >= 4:
        w = module.weight
        t = x.shape[2] if x.dim() == 5 else "-"
        return f"{kind} {w.shape[1]}->{w.shape[0]} k{tuple(w.shape[2:])} T{t}"
    return kind


def pre(module, inputs):
    torch.cuda.synchronize()
    starts[id(module)] = (time.perf_counter(), key(module, inputs))


def post(module, inputs, output):
    torch.cuda.synchronize()
    t0, k = starts.pop(id(module))
    stats[k][0] += 1
    stats[k][1] += time.perf_counter() - t0


LEAVES = ("CausalConv3d", "Conv2d", "Conv3d", "RMS_norm", "SiLU", "Upsample", "AttentionBlock", "ZeroPad2d")
hooks = []
for m in vae.first_stage_model.modules():
    if type(m).__name__ in LEAVES:
        hooks.append(m.register_forward_pre_hook(pre))
        hooks.append(m.register_forward_hook(post))


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def remove_hooks():
    for h in hooks:
        h.remove()


with torch.inference_mode():
    for res in sys.argv[3:]:
        w, h = (int(v) for v in res.split("x"))
        latent = torch.randn(1, 16, 1, h // 8, w // 8, generator=torch.Generator().manual_seed(0))
        image = vae.decode_tiled(latent, **TILE).reshape(1, h, w, 3)
        for direction, fn in (("decode", lambda: vae.decode_tiled(latent, **TILE)),
                              ("encode", lambda: vae.encode_tiled(image, **TILE_PX))):
            stats.clear()
            timed(fn)
            layer_table = {k: (n, round(s, 3)) for k, (n, s) in sorted(stats.items(), key=lambda kv: -kv[1][1])}
            hooked_total = sum(s for _, s in stats.values())
            saved = list(hooks)
            for hk in saved:
                hk.remove()
            best = min(timed(fn)[1] for _ in range(2))
            _, _, peak = timed(fn)
            for m in vae.first_stage_model.modules():
                if type(m).__name__ in LEAVES:
                    hooks.append(m.register_forward_pre_hook(pre))
                    hooks.append(m.register_forward_hook(post))
            print(json.dumps({"res": res, "dir": direction, "seconds": round(best, 2), "peak_gb": round(peak, 2),
                              "hooked_total_s": round(hooked_total, 2)}), flush=True)
            for k, (n, s) in layer_table.items():
                print(f"   {s:8.3f} s  {100 * s / hooked_total:5.1f}%  x{n:<5} {k}", flush=True)
