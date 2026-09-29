"""Capture real post-RoPE Q/K/V (pre-GQA expansion) from a Krea2 int4 run, for
offline attention-quantization accuracy study. Saves attn_capture/step{s}_block{b}.pt."""
import os, sys
HARNESS = "/media/luna/work/tinker/ComfyUI/develop/harness"
sys.path.insert(0, HARNESS)
import bootstrap  # noqa: E402  (sets comfy args before comfy imports)
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.disable_all_custom_nodes = True
comfy.cli_args.args.whitelist_custom_nodes = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU"]
import torch  # noqa: E402
import pipeline  # noqa: E402
from comfy.ldm.flux.math import apply_rope  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_capture_hr")
os.makedirs(OUT, exist_ok=True)
BLOCKS = {0, 9, 18, 27}
STEPS = set(range(8))
PROMPT = ("A wide painterly landscape at dusk: a lone traveller in a long weathered coat walks along a "
          "ruined stone causeway that crosses a flooded valley, broken arches reflected in still water, "
          "distant snow-capped mountains under a heavy layered sky with storm clouds breaking into a thin "
          "band of amber light on the horizon, flocks of birds, mist over the water, reeds and wet stones "
          "in the foreground, visible oil brushwork and impasto texture, canvas grain in flat areas, muted "
          "umber, slate blue and rust palette, strong contrast between the dark sky and the lit horizon, "
          "romantic sublime landscape composition, the figure small in the lower left third, epic scale.")

seen_sigmas = []

def capture(q, k, v, pe=None, attn_mask=None, extra_options=None):
    sigma = float(extra_options["sigmas"][0])
    if not seen_sigmas or seen_sigmas[-1] != sigma:
        seen_sigmas.append(sigma)
    step, block = len(seen_sigmas) - 1, extra_options["block_index"]
    if step in STEPS and block in BLOCKS:
        qr, kr = apply_rope(q, k, pe) if pe is not None else (q, k)
        torch.save({"q": qr.cpu(), "k": kr.cpu(), "v": v.cpu(), "sigma": sigma, "dtype": str(q.dtype)},
                   os.path.join(OUT, f"step{step}_block{block}.pt"))
        print(f"captured step {step} block {block} sigma {sigma:.4f} q {tuple(q.shape)} k {tuple(k.shape)} {q.dtype}", flush=True)
    return {}

bootstrap.ensure_custom_nodes()
resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": PROMPT, "latent": {"width": 2560, "height": 1440}})
resolved["conditioning"]["positive"]["text"] = PROMPT
with torch.inference_mode():
    model, clip, vae = pipeline.build_loaders(resolved)
    model = model.clone()
    model.set_model_attn1_patch(capture)
    pipeline.run_with_loaders(resolved, model, clip, vae)
print("done", sorted(os.listdir(OUT)), flush=True)
