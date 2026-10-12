#!/usr/bin/env python3
"""One recipe + one axes file -> N images. See develop/DESIGN.md.

Each combination runs in its own subprocess (a plain generate.py call), not
in-process like an earlier version of this script did. A segfault can't be
caught by Python's try/except -- and this harness has hit real, intermittent
ones from a MultiGPU custom_node interaction (see develop/out/sweep_run*.log
for the diagnosis) -- so one crashing combination must not take the whole
sweep process down with it. This costs a model reload per combination
instead of reusing loaded weights across axis values that don't touch
loaders.*; correctness of the sweep surviving a crash matters more here than
that reload cost.

Usage:
    python develop/harness/sweep.py --sweep develop/sweeps/front_vs_foreground.yaml
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

import bootstrap


def load_axis_values(spec):
    """A `{file: path.txt}` axis is a newline-delimited prompt list — the
    one axis expected to grow by hand or by append (`echo ... >> file`)
    between runs. Every other axis is a plain inline YAML list."""
    if isinstance(spec, dict) and "file" in spec:
        with open(spec["file"]) as f:
            return [line.strip() for line in f if line.strip()]
    return spec


def load_axes(sweep_cfg):
    return {name: load_axis_values(spec) for name, spec in sweep_cfg["axes"].items()}


def _set_args(axes_values):
    args = []
    for dotted_path, value in axes_values.items():
        args += ["--set", f"{dotted_path}={value}"]
    return args


def run_sweep(recipe_path, sweep_cfg, out_dir):
    axes = load_axes(sweep_cfg)
    axis_names = list(axes.keys())
    combos = list(itertools.product(*(axes[name] for name in axis_names)))

    generate_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    records = []
    for index, combo in enumerate(combos):
        axes_values = dict(zip(axis_names, combo))
        run_id = f"{stamp}_{index:04d}"
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                [sys.executable, generate_py, "--recipe", recipe_path, "--out-dir", out_dir,
                 *_set_args(axes_values)],
                cwd=bootstrap.REPO_ROOT, capture_output=True, text=True, timeout=180,
            )
            returncode, stderr_tail = proc.returncode, proc.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            returncode, stderr_tail = "timeout", (exc.stderr or b"")[-2000:] if exc.stderr else "(no output captured before timeout)"
        elapsed = time.perf_counter() - t0

        if returncode == 0:
            # generate.py already wrote its own sidecar + manifest line;
            # read that record back rather than reconstructing it, so the
            # resolved-recipe/timings fields stay exactly what it produced.
            with open(manifest_path) as f:
                record = json.loads(f.readlines()[-1])
            record["axes"] = axes_values
        else:
            record = {
                "run_id": run_id, "png": None, "recipe": None,
                "timings": {"generate_s": elapsed},
                "error": f"subprocess {returncode} "
                         f"(signal {-returncode if isinstance(returncode, int) and returncode < 0 else 'n/a'}): {stderr_tail}",
                "axes": axes_values,
            }
            with open(manifest_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        records.append(record)
        status = record["png"] or f"ERROR: {record['error'][:200]}"
        print(f"[{index + 1}/{len(combos)}] {status}", flush=True)

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--out-dir", default=os.path.join(bootstrap.REPO_ROOT, "develop", "out"))
    args = parser.parse_args()

    with open(args.sweep) as f:
        sweep_cfg = yaml.safe_load(f)

    run_sweep(sweep_cfg["recipe"], sweep_cfg, args.out_dir)


if __name__ == "__main__":
    main()
