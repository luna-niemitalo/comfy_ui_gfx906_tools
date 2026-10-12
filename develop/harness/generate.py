#!/usr/bin/env python3
"""One recipe (+overrides) -> one image + sidecar JSON. See develop/DESIGN.md.

Usage:
    python develop/harness/generate.py --recipe develop/recipes/moodykrea_qwenvl.yaml
    python develop/harness/generate.py --recipe develop/recipes/moodykrea_qwenvl.yaml \\
        --set prompt="character standing, front view" --set sampler.noise_seed=42
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import yaml
from PIL import Image

import bootstrap
import pipeline


def parse_set(pairs):
    return pipeline.build_nested(_split(pair) for pair in pairs)


def _split(pair):
    dotted_path, value = pair.split("=", 1)
    return dotted_path, _coerce(value)


def _coerce(value):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=bootstrap.REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def tensor_to_pil(images):
    arr = (255.0 * images[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_result(out_dir, run_id, resolved, timings, images=None, error=None, extra=None):
    """Write <run_id>.png (if images given), <run_id>.json, and append the
    same record to manifest.jsonl. The sidecar is always the fully-resolved
    node graph, not the sparse recipe — provenance needs the complete
    picture, input files don't (see develop/DESIGN.md)."""
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "run_id": run_id,
        "png": None,
        "recipe": resolved,
        "git_commit": git_commit(),
        "timings": timings,
        "error": error,
    }
    if extra:
        record.update(extra)
    if images is not None:
        png_path = os.path.join(out_dir, f"{run_id}.png")
        tensor_to_pil(images).save(png_path)
        record["png"] = png_path

    with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(out_dir, "manifest.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def run_one(recipe, cli_overrides, out_dir):
    resolved = pipeline.resolve(recipe, cli_overrides)
    bootstrap.ensure_custom_nodes()

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{recipe['kind']}"
    t0 = time.perf_counter()
    images = pipeline.run(resolved)
    timings = {"generate_s": time.perf_counter() - t0}

    return save_result(out_dir, run_id, resolved, timings, images=images)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="dotted.path=value",
                         help="override a resolved field, repeatable")
    parser.add_argument("--out-dir", default=os.path.join(bootstrap.REPO_ROOT, "develop", "out"))
    args = parser.parse_args()

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    record = run_one(recipe, parse_set(args.overrides), args.out_dir)
    print(f"wrote {record['png']}")


if __name__ == "__main__":
    main()
