#!/usr/bin/env python3
"""One shot of a shot project -> one image + one manifest row. The oc_krea2 recipe chain, headless:
compose at ~1 MP, optionally grow to the 4K-class target.

A shot project is a directory linked in as ComfyUI/develop/<project>/ with prompts/categories.yaml,
prompts/blocks/*.txt, shots/*.yaml and these tools in tools/ (deploy.nu links them there); this file
finds the project as the parent of the tools/ it is invoked through.

  draft  (default): compose only -> <project>/out/<image_id>_s<seed>.png, status draft
  --final         : compose + Resolution Growth -> <project>/images/<cat>/<image_id>.png, status final
  --print         : print the assembled prompts and sizes, load nothing

Keeps the pipeline loaded between shots: render_server.py (same functions).

Usage (from the ComfyUI root):
  direnv exec . python develop/<project>/tools/render.py --shots develop/<project>/shots/<file>.yaml \\
      --id <image_id> [--seed N] [--final] [--print]
"""
import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import time

import yaml

# abspath, not realpath: through deploy.nu's links the project is ComfyUI/develop/<project>
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVELOP_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(DEVELOP_DIR)
PROMPTS_DIR = os.path.join(PROJECT_DIR, "prompts")
DRAFT_DIR = os.path.join(PROJECT_DIR, "out")  # gitignored in the project repo
FINAL_DIR = os.path.join(PROJECT_DIR, "images")
MANIFEST = os.path.join(PROJECT_DIR, "manifest.sqlite")
# Repos whose state decides the pixels; each HEAD is recorded per image. develop/ and our nodes are all
# links into rvii-comfy (one repo since 2026-10-12); the project holds the shots and prompt blocks.
PROVENANCE_REPOS = {"comfyui": REPO_ROOT, "rvii_comfy": DEVELOP_DIR, "project": PROJECT_DIR}
CUSTOM_NODES = ["ComfyUI-Gfx906Int4", "ComfyUI-GGUF", "ComfyUI-MultiGPU", "ComfyUI-ResolutionGrowth"]
SHOT_KEYS = {"image_id", "entry_id", "category", "seed", "subject"}
OPTIONAL_SHOT_KEYS = {"add_blocks", "drop_blocks", "aspect", "style", "detail"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    TEXT NOT NULL,     -- <entry_id>-<nn>; several runs (seeds, draft/final) per image_id
    entry_id    TEXT NOT NULL,     -- the project entry this image illustrates
    status      TEXT NOT NULL,     -- draft | final | rejected
    created     TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    png         TEXT,              -- path relative to the ComfyUI root
    prompt      TEXT NOT NULL,
    detail_prompt TEXT,            -- growth tiles' prompt (final only)
    blocks      TEXT NOT NULL,     -- JSON list of prompt blocks, in prompt order
    recipe      TEXT NOT NULL,     -- JSON, fully resolved oc_krea2 recipe
    commits     TEXT NOT NULL,     -- JSON {repo: HEAD}
    timings     TEXT NOT NULL,     -- JSON seconds per phase
    error       TEXT
)"""


# --- shot file boundary (IN, critical: every field lands in the prompt or the manifest) ---

def load_shot(shots_path, image_id, categories):
    with open(shots_path) as f:
        shots = yaml.safe_load(f)["shots"]
    matches = [s for s in shots if s.get("image_id") == image_id]
    if len(matches) != 1:
        sys.exit(f"{shots_path}: expected exactly 1 shot with image_id {image_id}, found {len(matches)} "
                 f"(ids: {[s.get('image_id') for s in shots]})")
    shot = matches[0]
    missing = SHOT_KEYS - shot.keys()
    if missing:
        sys.exit(f"{image_id}: expected keys {sorted(SHOT_KEYS)}, missing {sorted(missing)}")
    unknown = shot.keys() - SHOT_KEYS - OPTIONAL_SHOT_KEYS
    if unknown:
        sys.exit(f"{image_id}: unknown keys {sorted(unknown)}, allowed {sorted(SHOT_KEYS | OPTIONAL_SHOT_KEYS)}")
    if shot["category"] not in categories:
        sys.exit(f"{image_id}: category expected one of {sorted(categories)}, got {shot['category']!r}")
    # The image id names its entry; a mismatch would file the image under the wrong entry.
    if not image_id.startswith(shot["entry_id"] + "-"):
        sys.exit(f"{image_id}: expected image_id to start with entry_id {shot['entry_id']!r}-")
    return shot


def load_categories():
    with open(os.path.join(PROMPTS_DIR, "categories.yaml")) as f:
        return yaml.safe_load(f)


def read_block(name):
    path = os.path.join(PROMPTS_DIR, "blocks", f"{name}.txt")
    if not os.path.isfile(path):
        sys.exit(f"prompt block {name!r}: expected {path}, not found")
    with open(path) as f:
        return f.read().strip()


# --- assembly ---

def block_names(shot, category):
    drop = set(shot.get("drop_blocks", []))
    return [b for b in category["blocks"] if b not in drop] + shot.get("add_blocks", [])


def size_for(aspect, megapixels, snap):
    w_ratio, h_ratio = aspect
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    return round(w_ratio * scale / snap) * snap, round(h_ratio * scale / snap) * snap


def assemble(shot, config):
    category = config["categories"][shot["category"]]
    names = block_names(shot, category)
    style = shot.get("style", category["style"])
    detail = shot.get("detail", category["detail"])  # growth tiles' style must match the composition's
    prompt = "\n\n".join([read_block(style), shot["subject"].strip()] + [read_block(n) for n in names])
    aspect = shot.get("aspect", category["aspect"])
    return {"prompt": prompt, "blocks": [style, "<subject>"] + names, "detail_block": detail,
            "detail_prompt": read_block(detail), "growth": category.get("growth", {}),
            "compose_size": size_for(aspect, config["compose_mp"], config["snap_px"]),
            "target_size": size_for(aspect, config["target_mp"], config["snap_px"])}


# --- run ---

def bootstrap_comfy():
    """Match run.nu's flags that change pixels or VRAM (fp16 VAE, MIOpen off, fp16 unet) before
    comfy.model_management is imported by bootstrap/nodes: it snapshots args at import."""
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(DEVELOP_DIR, "harness"))
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")  # run.nu's backstop, in case MIOpen comes back on
    import comfy.cli_args
    args = comfy.cli_args.args
    args.fp16_vae = True
    args.disable_miopen = True
    args.force_fp16 = True
    args.fp16_unet = True  # cli_args derives this from --force-fp16 at parse time, already past
    args.use_quad_cross_attention = True
    args.disable_all_custom_nodes = True
    args.whitelist_custom_nodes = CUSTOM_NODES
    import bootstrap
    bootstrap.ensure_custom_nodes()
    import comfy.model_management
    comfy.model_management.set_cudnn_benchmark()  # main.py re-applies --disable-miopen after custom nodes


def commits():
    heads = {}
    for name, path in PROVENANCE_REPOS.items():
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path,
                               capture_output=True, text=True).stdout.strip()
        heads[name] = result.stdout.strip() + ("+dirty" if dirty else "")
    return heads


def save_png(images, path):
    import numpy as np
    from PIL import Image
    arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)
    return arr.shape[1], arr.shape[0]


def render(shot, parts, seed, final, loaded=None):
    """loaded: (model, clip, vae) kept by render_server.py; None loads them here."""
    import torch
    import pipeline
    compose_w, compose_h = parts["compose_size"]
    target_w, target_h = parts["target_size"]
    resolved = pipeline.resolve({"kind": "oc_krea2", "prompt": parts["prompt"],
                                 "latent": {"width": compose_w, "height": compose_h},
                                 "sampler": {"noise_seed": seed},
                                 "growth": {**parts["growth"], "width": target_w, "height": target_h, "seed": seed}})
    timings = {}
    with torch.inference_mode():
        if loaded is None:
            t0 = time.perf_counter()
            loaded = pipeline.build_loaders(resolved)
            timings["load_s"] = time.perf_counter() - t0
        model, clip, vae = loaded

        t0 = time.perf_counter()
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=parts["prompt"])
        (negative,) = pipeline.call_node("ConditioningZeroOut", conditioning=positive)
        timings["encode_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        (latent,) = pipeline.call_node(resolved["latent"]["node"], **pipeline._kwargs(resolved["latent"]))
        sampler = pipeline._kwargs(resolved["sampler"])
        (samples,) = pipeline.call_node(resolved["sampler"]["node"], model=model, positive=positive,
                                        negative=negative, latent_image=latent, **sampler)
        timings["compose_s"] = time.perf_counter() - t0

        if final:
            t0 = time.perf_counter()
            (detail,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=parts["detail_prompt"])
            (samples, _schedule) = pipeline.call_node(
                resolved["growth"]["node"], model=model, positive=positive, negative=negative, vae=vae,
                latent=samples, detail_positive=detail, **pipeline._kwargs(resolved["growth"]))
            timings["growth_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        (images,) = pipeline.call_node(resolved["vae_decode"]["node"], samples=samples, vae=vae,
                                       **pipeline._kwargs(resolved["vae_decode"]))
        timings["decode_s"] = time.perf_counter() - t0
    if not final:
        resolved.pop("growth")
    return resolved, images, timings


def output_path(shot, seed, final):
    if final:
        return os.path.join(FINAL_DIR, shot["entry_id"].split("-")[0], f"{shot['image_id']}.png")
    # timestamped: re-rendering a seed after a prompt change must not overwrite the earlier draft
    return os.path.join(DRAFT_DIR, f"{shot['image_id']}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}.png")


def record(row):
    with sqlite3.connect(MANIFEST) as db:
        db.execute(SCHEMA)
        cursor = db.execute(f"INSERT INTO images ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                            list(row.values()))
        return cursor.lastrowid


def prepare(shots_path, image_id, seed_override, from_run=None):
    """from_run: take prompt, blocks and seed from that manifest row (an approved draft), so the
    final composes exactly what was approved even after the blocks changed. Detail prompt and
    growth settings still come from the current category."""
    config = load_categories()
    shot = load_shot(shots_path, image_id, config["categories"])
    parts = assemble(shot, config)
    seed = shot["seed"] if seed_override is None else seed_override
    if from_run is None:
        return shot, parts, seed
    with sqlite3.connect(MANIFEST) as db:
        row = db.execute("SELECT image_id, seed, prompt, blocks, recipe FROM images WHERE run_id = ?",
                         (from_run,)).fetchone()
    if row is None:
        sys.exit(f"--from-run {from_run}: no such run_id in {MANIFEST}")
    run_image_id, seed, parts["prompt"], blocks, recipe = row
    if run_image_id != image_id:
        sys.exit(f"--from-run {from_run}: expected image_id {image_id}, run has {run_image_id}")
    latent = json.loads(recipe)["latent"]
    if [latent["width"], latent["height"]] != list(parts["compose_size"]):
        sys.exit(f"--from-run {from_run}: expected compose size {parts['compose_size']}, "
                 f"run has {[latent['width'], latent['height']]}")
    parts["blocks"] = json.loads(blocks) + [f"<from_run {from_run}>"]
    return shot, parts, seed


def run_shot(shot, parts, seed, final, loaded=None):
    """Render, save the PNG, add the manifest row -> result dict."""
    t0 = time.perf_counter()
    resolved, images, timings = render(shot, parts, seed, final, loaded)
    timings["total_s"] = time.perf_counter() - t0
    path = output_path(shot, seed, final)
    width, height = save_png(images, path)
    run_id = record({
        "image_id": shot["image_id"], "entry_id": shot["entry_id"], "status": "final" if final else "draft",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "width": width, "height": height,
        "png": os.path.relpath(path, REPO_ROOT), "prompt": parts["prompt"],
        "detail_prompt": parts["detail_prompt"] if final else None, "blocks": json.dumps(parts["blocks"]),
        "recipe": json.dumps(resolved), "commits": json.dumps(commits()),
        "timings": json.dumps({k: round(v, 2) for k, v in timings.items()}), "error": None})
    return {"run_id": run_id, "png": os.path.relpath(path, REPO_ROOT), "size": [width, height],
            "timings": {k: round(v, 1) for k, v in timings.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--id", required=True, help="image_id of the shot")
    parser.add_argument("--seed", type=int, help="override the shot's seed")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--from-run", type=int, help="reuse that manifest run's prompt and seed")
    cli = parser.parse_args()

    shot, parts, seed = prepare(cli.shots, cli.id, cli.seed, cli.from_run)
    print(json.dumps({k: v for k, v in parts.items() if k not in ("prompt", "detail_prompt")}))
    if cli.print:
        print(parts["prompt"], "\n\n--- detail ---\n" + parts["detail_prompt"])
        return

    bootstrap_comfy()
    print(json.dumps(run_shot(shot, parts, seed, cli.final)), flush=True)


if __name__ == "__main__":
    main()
