"""Record, for every attention call of real 2560x1440 runs, the per-head mass
the pooled block mask (tau 0.995) would drop -- the calibration measure -- to
test whether head verdicts transfer across steps, prompts and seeds.
Runs dense int4/u8 (the recording doesn't change the output).
Saves scratchpad/missed_<run>.pt: {(step, block): missed [H]}."""
import importlib, os, sys
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
comfy.cli_args.args.whitelist_custom_nodes = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU"]
import torch  # noqa: E402
import pipeline  # noqa: E402

SCRATCH = os.path.dirname(os.path.abspath(__file__))
LANDSCAPE = open(os.path.join(SCRATCH, "capture_qkv.py")).read().split('PROMPT = (')[1].split(')\n')[0]
LANDSCAPE = eval("(" + LANDSCAPE + ")")
PORTRAIT = ("Close-up studio portrait of an elderly woman with deep wrinkles and silver hair tied back, soft "
            "window light from the left, shallow depth of field, knitted wool cardigan, calm expression, "
            "photographic, fine skin texture, dark neutral background.")
CITY = ("A crowded night market street in a rainy city, neon signs in many colours reflected on wet asphalt, "
        "food stalls with steam, people with umbrellas, dense detail, cinematic wide shot, overhead cables.")
RUNS = [("landscape_s0", LANDSCAPE, None), ("portrait_s0", PORTRAIT, None), ("city_s0", CITY, None),
        ("landscape_s1", LANDSCAPE, 1)]
TAU, ROWS = 0.995, 128

bootstrap.ensure_custom_nodes()
sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/custom_nodes")
bs = importlib.import_module("ComfyUI-Gfx906Int4.block_sparse")
for run, prompt, seed in RUNS:
    recipe = {"kind": "moodykrea_int4", "prompt": prompt, "latent": {"width": 2560, "height": 1440}}
    if seed is not None:
        recipe["sampler"] = {"noise_seed": seed}
    resolved = pipeline.resolve(recipe)
    records, sigmas_seen = {}, []
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        inner = model.model_options["transformer_options"]["optimized_attention_override"]

        def recorder(func, q, k, v, heads, **kwargs):
            opts = kwargs.get("transformer_options") or {}
            block = opts.get("block_index")
            if block is not None and q.dim() == 4 and k.shape[2] >= 768:
                sigma = float(opts["sigmas"].flatten()[0])
                if not sigmas_seen or sigmas_seen[-1] != sigma:
                    sigmas_seen.append(sigma)
                kk = k[:, ::q.shape[1] // 12] if k.shape[1] == q.shape[1] else k
                keep = bs.predict_keep(q, kk, TAU, ROWS)
                records[(len(sigmas_seen) - 1, block)] = bs.missed_mass(q, kk, keep, ROWS).cpu()
            return inner(func, q, k, v, heads, **kwargs)

        model.model_options["transformer_options"]["optimized_attention_override"] = recorder
        pipeline.run_with_loaders(resolved, model, clip, vae)
        del base_model, clip, vae, model
    torch.save(records, os.path.join(SCRATCH, f"missed_{run}.pt"))
    print(f"recorded {run}: {len(records)} calls", flush=True)
