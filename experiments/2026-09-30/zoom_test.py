"""Zoom detailing (Luna): small or distant objects have too few pixels to exist on their own at
4K, so they pick up AI artifacts. Find them, render each one magnified at the model's native
scale, supersample it back.

Stage detect: SAM3 (native ComfyUI, /media/luna/models/sam3/sam3.pt) per concept over
  overlapping ~1 MP tiles of the 4K image (SAM3 resizes its input to 1008 px: on the whole 4K
  frame a distant rock would be a few pixels), coarse masks, cross-tile NMS.
  -> out/zoom/detections.json, mask_<concept>.png (4K union), detect_overlay.png
Stage render: detections smaller than SMALL_PX are grouped into square crops; a crop's side is
  CANVAS / m with magnification m = clamp(TARGET_PX / object size, M_MIN, M_MAX) of the object
  that opened it. Per crop: Lanczos up to CANVAS, encode, re-noise to SIGMA_Z, STEPS er_sde steps
  with the low band anchored to the crop area-shrunk by m (exactly what the 4K image resolves;
  the model only adds detail above it), concept prompt + the detail prompt (never the whole
  prompt: every crop would draw the traveller), decode, Lanczos down, blend two ways:
    region  whole crop, box feather
    mask    the concept's exclusive SAM mask (minus every other concept's), dilated HALO px and
            softened, times the box feather
  Detections whose box is covered more by other concepts' masks than by their own exclusive mask
  are dropped (Luna: mutual exclusivity of the binary masks).
  -> out/zoom/final_{region,mask}.png, crop_<n>_<concept>.png (before | region | mask, 2x nearest)

Usage: zoom_test.py detect [base.png] | render [base.png]   (default base: splat4k A_yarn_settle_t1.0)"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "2026-09-29"))
import progressive as prog  # noqa: E402  (bootstrap, custom-node whitelist, stage_sigmas, anchored_sampler)
import tiled_test as tt  # noqa: E402

import asyncio  # noqa: E402
import nodes  # noqa: E402
NODE_DIR = os.path.join(os.path.dirname(nodes.__file__), "custom_nodes", "ComfyUI-ResolutionGrowth")
asyncio.run(nodes.load_custom_node(NODE_DIR))  # growth helpers (Lanczos, tiled VAE); whitelist excludes it

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.ops  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import pipeline  # noqa: E402

growth = sys.modules[next(name for name in sys.modules if name.endswith("ComfyUI-ResolutionGrowth.growth"))]
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/zoom"
os.makedirs(OUT, exist_ok=True)
BASE = "/media/luna/work/tinker/ComfyUI/develop/out/splat4k/A_yarn_settle_t1.0.png"
SAM3_PATH = "/media/luna/models/sam3/sam3.pt"
# concept: (max detections per tile, crop phrase) -- phrases hand-written from the render prompt
CONCEPTS = {
    "rock": (300, "weathered rocks and stones half hidden in dry autumn grass on a lake shore"),
    "person": (5, "a lone traveller in a long weathered coat, seen from behind"),
    "bird": (80, "a flock of birds in flight against a dusk sky"),
}
DET_TILE, DET_OVERLAP, DET_THRESHOLD = 1080, 180, 0.4
SMALL_PX = {"rock": 200, "person": 400, "bird": 200}  # the traveller is ~200x316 px at 4K
TARGET_PX, M_MIN, M_MAX = 160, 1.5, 4.0
CANVAS = 1024
PER_CONCEPT = 4  # first detect run: causeway stones alone would fill every slot
SIGMA_Z, STEPS = 0.4, 4
ANCHOR, RELEASE = 1.0, (0.5, 1.0)
HALO = 4
# region stage (attention maps from attn_map.py)
MAPS = "/media/luna/work/tinker/ComfyUI/develop/out/attn_map/maps.pt"
MAP_SIGMA = 0.6
# region = tokens whose dominant word is a target (a share threshold of 0.3 caught 3 tokens:
# 17 words spread the shares thin)
REGION_M, REGION_COVER = 3.0, 0.3
PROMPT_INTENT = os.environ.get("ZOOM_INTENT", "1") == "1"  # 0: crop words from MAP_SIGMA only
PROMPT_RULE = os.environ.get("ZOOM_RULE", "dominant")      # dominant | mean (mean share top-3: invents)
PRESENCE = 0.1
# attention word -> phrase, copied from the render prompt; subject phrases only where the model
# itself put the word (a crop's top words), never added otherwise
WORD_PHRASES = {
    "traveller": "a lone traveller", "coat": "a long weathered coat", "figure": "a small figure",
    "causeway": "a ruined stone causeway", "arches": "broken arches", "water": "still water",
    "valley": "a flooded valley", "mountains": "distant snow-capped mountains", "sky": "a heavy layered sky",
    "clouds": "storm clouds", "light": "amber light", "horizon": "the horizon", "birds": "flocks of birds",
    "mist": "mist over the water", "reeds": "reeds", "stones": "wet stones", "foreground": "the foreground",
}
STYLE = ("visible oil brushwork and impasto texture, canvas grain in flat areas, muted umber, slate blue and "
         "rust palette")


def load_base(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr).float()[None] / 255


def detect(base_path, max_dets=None, out=OUT):
    """max_dets: {concept: max detections per tile}; default CONCEPTS' (oc_zoom.py passes its own)."""
    max_dets = max_dets or {c: n for c, (n, _) in CONCEPTS.items()}
    image = load_base(base_path)
    H, W = image.shape[1:3]
    t0 = time.perf_counter()
    # sam3.pt is Meta's original: its text projection is [512, 1024], ComfyUI's wrapper builds
    # [1024, 1024] (made for the repackaged checkpoint). SAM3's conditioning is the last hidden
    # state (return_projected_pooled=False), the projection is never used -- drop it.
    sd = comfy.utils.load_torch_file(SAM3_PATH)
    dropped = [k for k in sd if "text_projection" in k]
    for k in dropped:
        del sd[k]
    print(json.dumps({"stage": "detect", "dropped_keys": dropped}), flush=True)
    sam, sam_clip = comfy.sd.load_state_dict_guess_config(sd, output_vae=False, output_clip=True)[:2]
    from comfy_extras.nodes_sam3 import SAM3_Detect
    ys, xs = growth.tile_starts(H, DET_TILE, DET_OVERLAP), growth.tile_starts(W, DET_TILE, DET_OVERLAP)
    detections = []
    for concept, max_det in max_dets.items():
        (cond,) = pipeline.call_node("CLIPTextEncode", clip=sam_clip, text=f"{concept}:{max_det}")
        union = torch.zeros(H, W, dtype=torch.bool)
        score_map = torch.zeros(H, W)  # per pixel, the best score of this concept's masks covering it
        boxes, scores = [], []
        for y in ys:
            for x in xs:
                tile = image[:, y:y + DET_TILE, x:x + DET_TILE]
                masks, bbox_dicts = SAM3_Detect.execute(sam, tile, conditioning=cond, threshold=DET_THRESHOLD,
                                                        refine_iterations=0, individual_masks=True).args
                for m, d in zip(masks, bbox_dicts[0]):  # same order: both follow SAM3_Detect's score sort
                    hit = m.cpu() > 0.5
                    union[y:y + tile.shape[1], x:x + tile.shape[2]] |= hit
                    view = score_map[y:y + tile.shape[1], x:x + tile.shape[2]]
                    view.copy_(torch.maximum(view, hit * d["score"]))
                for d in bbox_dicts[0]:
                    boxes.append([x + d["x"], y + d["y"], x + d["x"] + d["width"], y + d["y"] + d["height"]])
                    scores.append(d["score"])
        if boxes:
            b, s = torch.tensor(boxes), torch.tensor(scores)
            for i in torchvision.ops.nms(b, s, 0.5).tolist():
                x1, y1, x2, y2 = b[i].tolist()
                detections.append({"concept": concept, "score": round(float(s[i]), 3),
                                   "box": [round(x1), round(y1), round(x2), round(y2)]})
        Image.fromarray(union.numpy().astype(np.uint8) * 255).save(os.path.join(out, f"mask_{concept}.png"))
        torch.save(score_map, os.path.join(out, f"score_{concept}.pt"))
        found = [d for d in detections if d["concept"] == concept]
        sizes = sorted(max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) for d in found)
        print(json.dumps({"stage": "detect", "concept": concept, "found": len(found),
                          "size_px_median": sizes[len(sizes) // 2] if sizes else None}), flush=True)
    with open(os.path.join(out, "detections.json"), "w") as f:
        json.dump({"base": base_path, "detections": detections}, f, indent=1)
    overlay = Image.open(base_path).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    palette = [(255, 200, 0), (255, 0, 80), (0, 200, 255), (120, 255, 80), (200, 80, 255), (255, 140, 0)]
    colours = {c: palette[i % len(palette)] for i, c in enumerate(max_dets)}
    for d in detections:
        draw.rectangle(d["box"], outline=colours[d["concept"]], width=3)
        draw.text((d["box"][0] + 4, d["box"][1] + 4), d["concept"], fill=colours[d["concept"]])
    overlay.resize((W // 2, H // 2), Image.LANCZOS).save(os.path.join(out, "detect_overlay.png"))
    print(json.dumps({"stage": "detect", "total": len(detections), "s": round(time.perf_counter() - t0, 1)}),
          flush=True)


def exclusive(masks):
    """Luna: SAM masks are binary claims -- a pixel two concepts both claim belongs to neither.
    Each concept's mask minus the union of all the others."""
    return {c: m * (1 - torch.stack([o for k, o in masks.items() if k != c]).amax(0)) for c, m in masks.items()}


def owned(score_maps):
    """exclusive() with confidence: a pixel two concepts claim goes to the higher-scoring one instead of
    neither (OC seed: the tail is 'tail' 0.88 and 'horn' 0.84, the horns 'horn' 0.87 and 'tail' 0.61 --
    plain exclusivity erased both)."""
    stack = torch.stack(list(score_maps.values()))
    best = stack.amax(0)
    return {c: ((s > 0) & (s >= best)).float() for c, s in score_maps.items()}


def unambiguous(detections, masks, excl):
    """Drops a detection whose box is covered more by the other concepts' masks than by its own
    exclusive mask (first run: 'birds' that were rocks in the water, 'persons' on rocks)."""
    kept = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        own = float(excl[d["concept"]][y1:y2, x1:x2].mean())
        other = float(torch.stack([m[y1:y2, x1:x2] for k, m in masks.items() if k != d["concept"]]).amax(0).mean())
        if own >= other:
            kept.append(d)
    return kept


def magnification(det):
    x1, y1, x2, y2 = det["box"]
    return min(M_MAX, max(M_MIN, TARGET_PX / max(x2 - x1, y2 - y1, 1)))


def plan_crops(detections, H, W):
    """Greedy: smallest objects first; a detection whose centre falls inside an open crop's inner
    half joins it, otherwise it opens a crop of side CANVAS / m around itself. Per concept the
    PER_CONCEPT crops of highest magnification (smallest objects) are kept."""
    small = [d for d in detections
             if max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) < SMALL_PX[d["concept"]]]
    small.sort(key=lambda d: max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]))
    crops = []
    for d in small:
        cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
        home = next((c for c in crops if c["concept"] == d["concept"]
                     and abs(cx - (c["x"] + c["side"] / 2)) < c["side"] / 4
                     and abs(cy - (c["y"] + c["side"] / 2)) < c["side"] / 4), None)
        if home is not None:
            home["members"] += 1
            continue
        m = magnification(d)
        side = 16 * round(CANVAS / m / 16)
        x = int(min(max(0, cx - side / 2), W - side))
        y = int(min(max(0, cy - side / 2), H - side))
        crops.append({"concept": d["concept"], "x": x, "y": y, "side": side, "m": round(CANVAS / side, 2),
                      "members": 1})
    crops.sort(key=lambda c: (-c["m"], -c["members"]))
    return [c for concept in CONCEPTS for c in [c for c in crops if c["concept"] == concept][:PER_CONCEPT]]


def box_feather(side, device):
    return growth.feather(side, side, max(1, side // 8), device)


def soften(mask, halo):
    """Dilate by halo px, then two box blurs of the same radius (roughly Gaussian)."""
    m = F.max_pool2d(mask[None, None], 2 * halo + 1, stride=1, padding=halo)
    for _ in range(2):
        m = F.avg_pool2d(m, 2 * halo + 1, stride=1, padding=halo, count_include_pad=False)
    return m[0, 0]


def load_krea(kind="moodykrea_int4"):
    """kind: recipe whose loaders to build. oc_krea2's unet chain already holds fused attention +
    the row split (and its own int4 text encoder / int8 VAE -- crops must match the base's)."""
    resolved = pipeline.resolve({"kind": kind, "prompt": prog.PROMPT, "latent": {"width": CANVAS, "height": CANVAS}})
    model, clip, vae = pipeline.build_loaders(resolved)
    if kind == "moodykrea_int4":
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device="auto", min_image_tokens=2048)
    return model, clip, vae


def zoom_crop(model, vae, cond, before, seed):
    """before: IMAGE [1, side, side, 3] (cpu) -> the same region re-rendered at CANVAS and Lanczos'd
    back to side: low band anchored to what `before` resolves (the crop area-shrunk by m)."""
    side = before.shape[1]
    latent = growth.encode(vae, growth.resize_image(before, (CANVAS, CANVAS), "lanczos"))
    ref = model.model.process_latent_in(latent.to(model.load_device)).float()
    ref = growth.resize(ref, (ref.shape[-2] * side // CANVAS, ref.shape[-1] * side // CANVAS), "area")
    (noise,) = pipeline.call_node("RandomNoise", noise_seed=seed)
    (guider,) = pipeline.call_node("BasicGuider", model=model, conditioning=cond)
    (out, _) = pipeline.call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                                  sampler=prog.anchored_sampler(ref, ANCHOR, RELEASE, SIGMA_Z),
                                  sigmas=prog.stage_sigmas(model, SIGMA_Z, STEPS), latent_image={"samples": latent})
    return growth.resize_image(growth.decode(vae, out["samples"]), (side, side), "lanczos").cpu()


def render(base_path):
    with open(os.path.join(OUT, "detections.json")) as f:
        detections = json.load(f)["detections"]
    image = load_base(base_path)
    H, W = image.shape[1:3]
    masks = {c: torch.from_numpy(np.array(Image.open(os.path.join(OUT, f"mask_{c}.png")))).float() / 255
             for c in CONCEPTS}
    excl = exclusive(masks)
    kept = unambiguous(detections, masks, excl)
    print(json.dumps({"stage": "render", "exclusive": {c: {"before": sum(d["concept"] == c for d in detections),
                                                           "kept": sum(d["concept"] == c for d in kept)}
                                                       for c in CONCEPTS}}), flush=True)
    masks = excl
    crops = plan_crops(kept, H, W)
    print(json.dumps({"stage": "render", "crops": crops}), flush=True)

    model, clip, vae = load_krea()
    conds = {c: pipeline.call_node("CLIPTextEncode", clip=clip, text=f"{phrase}, {tt.DETAIL_PROMPT}")[0]
             for c, (_, phrase) in CONCEPTS.items()}

    region, masked = image.clone(), image.clone()
    for n, c in enumerate(crops):
        t0 = time.perf_counter()
        x, y, side = c["x"], c["y"], c["side"]
        before = image[:, y:y + side, x:x + side]
        detail = zoom_crop(model, vae, conds[c["concept"]], before, prog.SEED + 60 + n)
        feather = box_feather(side, "cpu")[None, :, :, None]
        region[:, y:y + side, x:x + side] = before + feather * (detail - before)
        w = (soften(masks[c["concept"]][y:y + side, x:x + side], HALO) * box_feather(side, "cpu"))[None, :, :, None]
        masked[:, y:y + side, x:x + side] = masked[:, y:y + side, x:x + side] * (1 - w) + detail * w
        sheet = torch.cat([before, region[:, y:y + side, x:x + side], masked[:, y:y + side, x:x + side]], dim=2)
        arr = (sheet[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST).save(
            os.path.join(OUT, f"crop_{n}_{c['concept']}.png"))
        print(json.dumps({"stage": "render", "crop": n, "concept": c["concept"], "side": side, "m": c["m"],
                          "members": c["members"], "s": round(time.perf_counter() - t0, 1)}), flush=True)
    for name, img in (("region", region), ("mask", masked)):
        Image.fromarray((img[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(
            os.path.join(OUT, f"final_{name}.png"))
    print(json.dumps({"stage": "render", "done": True}), flush=True)


def region(base_path, targets):
    """Region crops from the DiT's own image->text attention (attn_map.py): tokens whose dominant
    word (share among attn_map.WORDS) is a target form the region; overlapping crops of
    side CANVAS / REGION_M cover it (kept if >= REGION_COVER inside); each crop's prompt is its own
    top words by share (WORD_PHRASES) + STYLE; changes are confined to the softened region."""
    image = load_base(base_path)
    H, W = image.shape[1:3]
    tag = "_".join(targets) + ("_intent" if PROMPT_INTENT else "") + ("_domin" if PROMPT_RULE == "dominant" else "")
    all_maps = torch.load(MAPS)
    words = list(all_maps[MAP_SIGMA])

    def shares(maps):
        s = torch.stack([maps[w] for w in words])
        return s / s.sum(0, keepdim=True)

    share = shares(all_maps[MAP_SIGMA])
    # crop words: each word's strongest share over every captured sigma (Luna: at sigma 0.9
    # 'stones' glows on the far shore -- the model meant rocks there before they dissolved)
    intent = torch.stack([shares(m) for m in all_maps.values()]).amax(0) if PROMPT_INTENT else share
    sigma_shares = [shares(m) for m in all_maps.values()] if PROMPT_INTENT else [share]
    th, tw = share.shape[1:]
    dominant = share.argmax(0)
    grid = sum((dominant == words.index(t)).float() for t in targets).clamp(max=1)
    region_mask = soften(F.interpolate(grid[None, None], size=(H, W), mode="nearest")[0, 0], 24)
    side = 16 * round(CANVAS / REGION_M / 16)
    stride = side * 3 // 4
    crops = [(x, y) for y in range(0, H - side + 1, stride) for x in range(0, W - side + 1, stride)
             if float(region_mask[y:y + side, x:x + side].mean()) >= REGION_COVER]
    print(json.dumps({"stage": "region", "targets": targets, "side": side, "m": round(CANVAS / side, 2),
                      "region_tokens": int(grid.sum()), "crops": len(crops)}), flush=True)

    model, clip, vae = load_krea()
    work = image.clone()
    for n, (x, y) in enumerate(crops):
        t0 = time.perf_counter()
        box = (slice(y * th // H, max(y * th // H + 1, (y + side) * th // H)),
               slice(x * tw // W, max(x * tw // W + 1, (x + side) * tw // W)))
        if PROMPT_RULE == "dominant":
            # Luna, valley run: crop 13 got "arches" from a thin spread of attention (mean share
            # top-3) and painted non-existent ruins; a word must own >= PRESENCE of the crop's tokens
            # counted only on the region's own tokens: crop 13 (dominance over the whole crop) still
            # got "arches" from the water strip below the shore and painted distant arches
            inside = grid[box[0], box[1]]
            owned = torch.stack([((s[:, box[0], box[1]].argmax(0)[None] == torch.arange(len(words))[:, None, None])
                                  .float() * inside).sum((1, 2)) / inside.sum().clamp(min=1)
                                 for s in sigma_shares]).amax(0)
            top = [words[i] for i in owned.argsort(descending=True)[:3].tolist() if owned[i] >= PRESENCE]
        else:
            tbox = intent[:, box[0], box[1]].mean((1, 2))
            top = [words[i] for i in tbox.argsort(descending=True)[:3].tolist() if tbox[i] >= 0.1]
        text = ", ".join(WORD_PHRASES[w] for w in top) + ", " + STYLE
        (cond,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=text)
        before = work[:, y:y + side, x:x + side].clone()
        detail = zoom_crop(model, vae, cond, before, prog.SEED + 80 + n)
        w = (region_mask[y:y + side, x:x + side] * box_feather(side, "cpu"))[None, :, :, None]
        work[:, y:y + side, x:x + side] = before + w * (detail - before)
        sheet = torch.cat([before, work[:, y:y + side, x:x + side]], dim=2)
        arr = (sheet[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST).save(
            os.path.join(OUT, f"region_{tag}_{n}.png"))
        print(json.dumps({"stage": "region", "crop": n, "x": x, "y": y, "words": top,
                          "s": round(time.perf_counter() - t0, 1)}), flush=True)
    rows = region_mask.amax(1).nonzero()
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    for name, img in (("before", image), ("after", work)):
        Image.fromarray((img[0, y0:y1].clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(
            os.path.join(OUT, f"band_{tag}_{name}.png"))
    Image.fromarray((work[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(
        os.path.join(OUT, f"final_{tag}.png"))
    print(json.dumps({"stage": "region", "done": True, "band_rows": [y0, y1]}), flush=True)


if __name__ == "__main__":
    stage, args = (sys.argv[1] if sys.argv[1:] else None), sys.argv[2:]
    if stage not in ("detect", "render", "region"):
        sys.exit(f"usage: zoom_test.py detect|render [base.png] | region [base.png] [word ...]; got {sys.argv[1:]}")
    base = args.pop(0) if args and args[0].endswith(".png") else BASE
    with torch.inference_mode():
        if stage == "detect":
            detect(base)
        elif stage == "render":
            render(base)
        else:
            region(base, args or ["valley"])
