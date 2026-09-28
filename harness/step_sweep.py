#!/usr/bin/env python3
"""Step-count degradation sweep: for each (sampler, scheduler), render at
several step counts and measure fidelity loss vs a high-step reference. The
axis sampler_scheduler.yaml varies (sampler x scheduler at one fixed step
count) showed ~2s of noise across all combos -- no real speed signal at a
fixed step count. This is the axis that actually varies wall-clock time and
answers "which sampler/scheduler holds up with fewer steps."

steps and end_at_step must move together for KSamplerAdvanced to actually
run the requested number of denoise steps -- sweep.py's generic axes system
treats every axis independently, so a linked pair doesn't fit it. Same
subprocess-per-combination crash isolation as sweep.py, for the same reason
(the real segfault documented in develop/out/sweep_run*.log).

Usage:
    python develop/harness/step_sweep.py --recipe develop/recipes/moodykrea_int4_sampler_sweep.yaml \\
        --samplers res_multistep,dpmpp_3m_sde --schedulers simple,sgm_uniform,karras --steps 4,6,8,12
"""
import argparse
import json
import os
import subprocess
import sys
import time

import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--samplers", required=True, help="comma-separated sampler_name values")
    parser.add_argument("--schedulers", required=True, help="comma-separated scheduler values")
    parser.add_argument("--steps", required=True, help="comma-separated step counts")
    parser.add_argument("--out-dir", default=os.path.join(bootstrap.REPO_ROOT, "develop", "out"))
    args = parser.parse_args()

    samplers = args.samplers.split(",")
    schedulers = args.schedulers.split(",")
    steps_list = [int(s) for s in args.steps.split(",")]
    combos = [(s, sc, st) for s in samplers for sc in schedulers for st in steps_list]

    generate_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for index, (sampler, scheduler, steps) in enumerate(combos):
        run_id = f"{stamp}_{index:04d}"
        set_args = [
            "--set", f"sampler.sampler_name={sampler}",
            "--set", f"sampler.scheduler={scheduler}",
            "--set", f"sampler.steps={steps}",
            "--set", f"sampler.end_at_step={steps}",
        ]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, generate_py, "--recipe", args.recipe, "--out-dir", args.out_dir, *set_args],
                cwd=bootstrap.REPO_ROOT, capture_output=True, text=True, timeout=180,
            )
            returncode, stderr_tail = proc.returncode, proc.stderr[-2000:]
        except subprocess.TimeoutExpired as exc:
            returncode, stderr_tail = "timeout", ((exc.stderr or b"")[-2000:] if exc.stderr else "(no output before timeout)")
        elapsed = time.perf_counter() - t0

        if returncode == 0:
            with open(manifest_path) as f:
                record = json.loads(f.readlines()[-1])
            record["axes"] = {"sampler_name": sampler, "scheduler": scheduler, "steps": steps}
            status = record["png"]
        else:
            record = {
                "run_id": run_id, "png": None, "recipe": None,
                "timings": {"generate_s": elapsed},
                "error": f"subprocess {returncode}: {stderr_tail}",
                "axes": {"sampler_name": sampler, "scheduler": scheduler, "steps": steps},
            }
            with open(manifest_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            status = f"ERROR: {record['error'][:200]}"

        print(f"[{index + 1}/{len(combos)}] {sampler}/{scheduler}/{steps}steps  {status}", flush=True)


if __name__ == "__main__":
    main()
