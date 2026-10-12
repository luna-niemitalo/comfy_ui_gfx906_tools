#!/usr/bin/env python3
"""manifest.jsonl -> a flat summary (speed + fidelity-vs-reference per combo),
for feeding to a chart. No plotting here -- deliberately just data
extraction (numpy/PIL only, both already harness dependencies), so the
actual visualization can be built with whatever the reader prefers instead
of pinning a plotting library into this venv. See develop/DESIGN.md.

Usage:
    python develop/harness/summarize_sweep.py --manifest develop/out/manifest.jsonl \\
        --reference develop/out/<reference_run_id>.png --run-ids-from <sweep_stamp> \\
        --out develop/out/sampler_scheduler_summary.json
"""
import argparse
import json

import numpy as np
from PIL import Image


def load_image_array(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def pixel_deviation(a, b):
    """Mean absolute pixel difference in [0, 1] -- same "pixel deviation,
    cheapest v1 metric" choice develop/TODO.md already settled on for this
    harness family, not a new metric invented for this script."""
    return float(np.abs(a - b).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", required=True, help="path to the high-step reference PNG")
    parser.add_argument("--run-id-prefix", default=None,
                         help="only include manifest records whose run_id starts with this "
                              "(the sweep's timestamp stamp) -- manifest.jsonl is append-only "
                              "across every generate.py/sweep.py run ever made")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seconds", type=float, default=120.0,
                         help="discard runs slower than this -- a real, unrepresentative outlier "
                              "(stalled/contended GPU, not a genuine measurement of the config), "
                              "not a legitimate data point to average in")
    parser.add_argument("--min-deviation", type=float, default=1e-6,
                         help="discard runs at/near zero deviation -- a run compared against "
                              "itself (e.g. the reference render appearing in its own manifest) "
                              "rather than a real low-loss result")
    args = parser.parse_args()

    reference = load_image_array(args.reference)

    rows, discarded = [], []
    with open(args.manifest) as f:
        for line in f:
            record = json.loads(line)
            if args.run_id_prefix and not record["run_id"].startswith(args.run_id_prefix):
                continue
            if record.get("error") or not record.get("png"):
                continue
            sampler = record["recipe"]["sampler"]
            seconds = record["timings"]["generate_s"]
            image = load_image_array(record["png"])
            deviation = pixel_deviation(image, reference)

            row = {
                "run_id": record["run_id"],
                "sampler_name": sampler["sampler_name"],
                "scheduler": sampler["scheduler"],
                "steps": sampler["steps"],
                "seconds": seconds,
                "deviation": deviation,
                "png": record["png"],
            }
            if seconds > args.max_seconds:
                discarded.append((row["run_id"], f"seconds={seconds:.1f} > {args.max_seconds}"))
                continue
            if deviation < args.min_deviation:
                discarded.append((row["run_id"], f"deviation={deviation:.6f} < {args.min_deviation} (compared against itself?)"))
                continue
            rows.append(row)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {len(rows)} rows to {args.out}")
    if discarded:
        print(f"discarded {len(discarded)} outlier(s):")
        for run_id, reason in discarded:
            print(f"  {run_id}: {reason}")


if __name__ == "__main__":
    main()
