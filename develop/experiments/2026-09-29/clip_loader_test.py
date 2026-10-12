"""Gfx906Int4CLIPLoader (quantize while loading) vs ComfyUI's loader + Gfx906Int4TranscodeCLIP
(the old two-node path), Qwen3-VL-4B (Krea2 text encoder), activation rotation on, one GPU.
Per path: peak allocated / reserved VRAM during load, resident after, seconds; then the
conditioning of one prompt compared between the paths (max abs diff, cosine).

Usage: clip_loader_test.py <device>"""
import importlib, json, sys, time
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402,F401
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402

sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/custom_nodes")
int4_nodes = importlib.import_module("ComfyUI-Gfx906Int4.nodes")
CLIP_NAME = "qwen3vl_4b_bf16.safetensors"
PROMPT = "ultra wide wallpaper of a lone traveller in a misty reed marsh at dawn, photorealistic, 8K"
device = torch.device(sys.argv[1])
torch.cuda.set_device(device)


def old_path():
    clip = comfy.sd.load_clip(ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP_NAME)],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.KREA2, model_options={"load_device": device})
    return int4_nodes.Gfx906Int4TranscodeCLIP().transcode(clip, True)[0]


def new_path():
    return int4_nodes.Gfx906Int4CLIPLoader().load_clip(CLIP_NAME, "krea2", str(device), True)[0]


def encode(clip):
    tokens = clip.tokenize(PROMPT)
    return clip.encode_from_tokens_scheduled(tokens)[0][0].float().cpu()


conds = {}
with torch.inference_mode():
    for name, load in (("loader", new_path), ("clip_loader+transcode", old_path)):
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        clip = load()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        row = {"path": name, "load_s": round(seconds, 1),
               "peak_alloc_gb": round((torch.cuda.max_memory_allocated() - base) / 2**30, 2),
               "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
               "resident_gb": round((torch.cuda.memory_allocated() - base) / 2**30, 2)}
        conds[name] = encode(clip)
        print(json.dumps(row), flush=True)
        del clip
a, b = conds["loader"], conds["clip_loader+transcode"]
print(json.dumps({"cond_shape": list(a.shape), "max_abs_diff": float((a - b).abs().max()),
                  "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}), flush=True)
