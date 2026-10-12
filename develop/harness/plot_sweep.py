#!/usr/bin/env python3
"""summarize_sweep.py's summary JSON -> a ranked fidelity bar chart. See
develop/DESIGN.md.

Not a speed-vs-fidelity scatter: at a fixed step count, wall-clock seconds
across combinations varied by ~2s on a ~44s run -- noise, not signal, so
plotting it as an axis implies a tradeoff that isn't actually in the data.
The real question a low-step sweep answers is which sampler/scheduler holds
up best when forced to few steps, i.e. deviation from a high-step reference,
ranked. (A genuine speed axis would come from sweeping step count itself,
not sampler/scheduler at one fixed step count.)

Usage:
    python develop/harness/plot_sweep.py --summary develop/out/sampler_scheduler_summary.json \\
        --out develop/out/sampler_scheduler.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEDULER_COLORS = {"simple": "#2a78d6", "sgm_uniform": "#eb6834", "karras": "#1baf7a"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=None,
                         help="restrict to rows at this step count (default: the most common one, "
                              "excluding the reference render)")
    args = parser.parse_args()

    with open(args.summary) as f:
        rows = json.load(f)

    if args.steps is None:
        # The reference render (steps far above the rest, deviation==0 by
        # construction) shouldn't be mistaken for a real sweep point.
        step_counts = [r["steps"] for r in rows if r["deviation"] > 0]
        args.steps = min(step_counts) if step_counts else rows[0]["steps"]
    rows = [r for r in rows if r["steps"] == args.steps]

    samplers = sorted({r["sampler_name"] for r in rows})
    schedulers = [s for s in ("simple", "sgm_uniform", "karras") if any(r["scheduler"] == s for r in rows)]
    by_key = {(r["sampler_name"], r["scheduler"]): r["deviation"] for r in rows}

    # Rank samplers by their best (lowest-deviation) scheduler, best first.
    samplers.sort(key=lambda s: min(by_key.get((s, sch), float("inf")) for sch in schedulers))

    bar_h = 0.24
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(samplers) + 1.2))
    for i, sampler in enumerate(samplers):
        for j, scheduler in enumerate(schedulers):
            dev = by_key.get((sampler, scheduler))
            y = i + (j - (len(schedulers) - 1) / 2) * bar_h
            if dev is None:
                ax.barh(y, 0, height=bar_h * 0.9, color="none")
                ax.text(0.002, y, "crashed", va="center", fontsize=8, color="#e34948", style="italic")
                continue
            ax.barh(y, dev, height=bar_h * 0.9, color=SCHEDULER_COLORS.get(scheduler, "#999999"))
            ax.text(dev + 0.002, y, f"{dev:.3f}", va="center", fontsize=8, color="#55534c")

    ax.set_yticks(range(len(samplers)))
    ax.set_yticklabels(samplers)
    ax.invert_yaxis()
    ax.set_xlabel("mean pixel deviation from high-step reference (lower = closer)")
    ax.set_title(f"sampler / scheduler fidelity at {args.steps} steps -- ranked, best first")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SCHEDULER_COLORS[s]) for s in schedulers]
    ax.legend(handles, schedulers, title="scheduler", loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
