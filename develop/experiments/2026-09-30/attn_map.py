"""What does the model think each region is? (Luna: "what is this splat supposed to be")
Krea2 is single-stream: image tokens attend to the prompt tokens in every block. An attn1 patch
(observes only; returns nothing) computes, per block, the true softmax over all keys in fp32
from the patch's q and k (after QK-norm; RoPE and GQA applied as the block does) and sums the
probability mass each image token puts on each text token, over heads and blocks.
Text tokens -> prompt words: Krea2TEModel keeps the tokens after the template prefix (second
<|im_start|> + "user\\n"); tokens are grouped into words at leading spaces / punctuation.

Run on the native composition (1280x720, 3600 image tokens) re-noised to SIGMAS (fixed noise):
the model's intent at composition scale, mapped onto the 4K image.
  out/attn_map/words_s<sigma>.png  per-word heatmaps over the image (WORDS)
  out/attn_map/label_s<sigma>.png  dominant word per image token (share among WORDS)
  out/attn_map/maps.pt             {sigma: {word: [45, 80] map}} for zoom_test

Usage: attn_map.py [base.png]   (default: splat4k A_yarn_settle_t1.0; only for the overlays)"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "2026-09-29"))
import splat_test as st  # noqa: E402  (bootstrap via progressive; Predictor)
import progressive as prog  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from comfy.ldm.flux.math import apply_rope  # noqa: E402
import pipeline  # noqa: E402

OUT = "/media/luna/work/tinker/ComfyUI/develop/out/attn_map"
os.makedirs(OUT, exist_ok=True)
BASE = "/media/luna/work/tinker/ComfyUI/develop/out/splat4k/A_yarn_settle_t1.0.png"
SIGMAS = (0.9, 0.6, 0.3)
WORDS = ["traveller", "coat", "causeway", "arches", "water", "valley", "mountains", "sky", "clouds", "light",
         "horizon", "birds", "mist", "reeds", "stones", "foreground", "figure"]
IM_START, USER, NEWLINE = 151644, 872, 198  # Krea2TEModel.encode_token_weights' template strip


class TextAttention:
    """attn1 patch: accumulates [image tokens, text tokens] attention mass over heads and blocks."""

    def __init__(self):
        self.mass = None
        self.heads = 0  # head-block evaluations summed into mass

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        txt, end = extra_options["img_slice"]
        qr, kr = apply_rope(q, k, pe)
        if kr.shape[1] != qr.shape[1]:
            kr = kr.repeat_interleave(qr.shape[1] // kr.shape[1], dim=1)
        scale = qr.shape[-1] ** -0.5
        acc = torch.zeros(end - txt, txt, device=q.device)
        for h in range(qr.shape[1]):  # one head at a time: [3600, ~4200] fp32 = 60 MB
            p = torch.softmax((qr[0, h, txt:end].float() @ kr[0, h].float().T) * scale, dim=-1)
            acc += p[:, :txt]
        self.mass = acc if self.mass is None else self.mass + acc
        self.heads += qr.shape[1]
        return {}


def prompt_words(clip, text, n_text):
    """[(word, [text token indices])] for the conditioning's text tokens."""
    tokens = clip.tokenize(text)
    pairs = next(iter(tokens.values()))[0]
    ids = [int(t[0]) for t in pairs]
    starts = [i for i, t in enumerate(ids) if t == IM_START][:2]
    end = starts[-1]
    if ids[end + 1] == USER and ids[end + 2] == NEWLINE:
        end += 3
    ids = ids[end:]
    if len(ids) != n_text:
        raise ValueError(f"text tokens after the template: expected {n_text} (conditioning length), got {len(ids)}")
    inner = getattr(clip.tokenizer, clip.tokenizer.clip)
    pieces = [inner.tokenizer.decode([t]) for t in ids]
    words = []
    for i, piece in enumerate(pieces):
        if not words or piece.startswith(" ") or not piece.strip().isalnum() or not words[-1][0][-1:].isalnum():
            words.append([piece.strip(), [i]])
        else:
            words[-1][0] += piece.strip()
            words[-1][1].append(i)
    return [(w.lower(), idx) for w, idx in words]


def heat(base, grid, colour):
    """base: RGB PIL; grid: [h, w] in [0, 1] -> base darkened, grid tinted in colour."""
    g = Image.fromarray((grid.clamp(0, 1).numpy() * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    g = np.asarray(g, dtype=np.float32)[..., None] / 255
    b = np.asarray(base, dtype=np.float32) * 0.35
    return Image.fromarray((b * (1 - g) + np.array(colour, dtype=np.float32) * g).clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else BASE
    base = Image.open(base_path).convert("RGB").resize((960, 540), Image.LANCZOS)
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
        (comp,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                     steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple", positive=positive,
                                     negative=positive, latent_image=latent, start_at_step=0, end_at_step=8,
                                     return_with_leftover_noise="disable")
        capture = TextAttention()
        probe = model.clone()
        probe.set_model_attn1_patch(capture)
        pred = st.Predictor(probe, positive, comp["samples"].shape)
        words = prompt_words(clip, prog.PROMPT, pred.text.shape[1])
        print(json.dumps({"text_tokens": pred.text.shape[1], "words": [w for w, _ in words]}), flush=True)
        x0 = probe.model.process_latent_in(comp["samples"].to(pred.device)).float()
        eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(prog.SEED + 70)).to(pred.device)
        h, w = pred.h, pred.w
        maps = {}
        for sigma in SIGMAS:
            capture.mass, capture.heads = None, 0
            pred.full((1 - sigma) * x0 + sigma * eps, sigma)
            mass = capture.mass.cpu()
            per_word = {}
            for word in WORDS:
                idx = [i for wd, ids in words if wd == word for i in ids]
                if idx:
                    per_word[word] = mass[:, idx].sum(1).view(h, w)
            maps[sigma] = per_word
            missing = [wd for wd in WORDS if wd not in per_word]
            share = torch.stack(list(per_word.values()))
            share = share / share.sum(0, keepdim=True)
            tiles = []
            for word, m in per_word.items():
                norm = (m - m.min()) / (m.max() - m.min() + 1e-12)
                tile = heat(base, norm, (255, 220, 60))
                ImageDraw.Draw(tile).text((8, 8), f"{word}  s{sigma}", fill=(255, 255, 255))
                tiles.append(tile)
            cols = 4
            sheet = Image.new("RGB", (cols * 960, math.ceil(len(tiles) / cols) * 540))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % cols) * 960, (i // cols) * 540))
            sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(os.path.join(OUT, f"words_s{sigma}.png"))
            palette = np.array([[(37 * i) % 255, (91 * i + 80) % 255, (151 * i + 40) % 255] for i in range(len(per_word))])
            label = palette[share.argmax(0).numpy()].astype(np.uint8)
            label_img = Image.blend(base, Image.fromarray(label).resize(base.size, Image.NEAREST), 0.55)
            draw = ImageDraw.Draw(label_img)
            for i, word in enumerate(per_word):
                draw.rectangle([8, 8 + 16 * i, 20, 20 + 16 * i], fill=tuple(int(c) for c in palette[i]))
                draw.text((26, 8 + 16 * i), word, fill=(255, 255, 255))
            label_img.save(os.path.join(OUT, f"label_s{sigma}.png"))
            print(json.dumps({"sigma": sigma, "words": list(per_word), "missing": missing,
                              "text_share": round(float(mass.sum(1).mean()) / capture.heads, 4)}), flush=True)
        torch.save(maps, os.path.join(OUT, "maps.pt"))
        print(json.dumps({"done": True}), flush=True)
