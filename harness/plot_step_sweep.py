#!/usr/bin/env python3
"""summarize_sweep.py's summary JSON -> a step-count degradation-curve chart:
fidelity loss vs steps, one line per sampler/scheduler. This is the real
speed/fidelity tradeoff axis -- unlike sampler x scheduler at one fixed step
count (~2s of noise, no signal), step count is what actually varies both
wall-clock time and fidelity loss together.

Usage:
    python develop/harness/plot_step_sweep.py --summary develop/out/step_sweep_summary_all.json \\
        --samplers res_multistep,dpmpp_3m_sde --out develop/out/step_sweep.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEDULER_COLORS = {"simple": "#2a78d6", "sgm_uniform": "#eb6834", "karras": "#1baf7a"}
SAMPLER_STYLES = {}  # filled in per sampler in main() -- solid vs dashed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--samplers", required=True, help="comma-separated sampler_name values to include")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.summary) as f:
        rows = json.load(f)

    samplers = args.samplers.split(",")
    rows = [r for r in rows if r["sampler_name"] in samplers]
    schedulers = sorted({r["scheduler"] for r in rows})
    line_styles = {samplers[0]: "-", samplers[1] if len(samplers) > 1 else "": "--"}

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for sampler in samplers:
        for scheduler in schedulers:
            pts = sorted(
                ((r["steps"], r["deviation"]) for r in rows if r["sampler_name"] == sampler and r["scheduler"] == scheduler),
                key=lambda p: p[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=SCHEDULER_COLORS.get(scheduler, "#999999"),
                    linestyle=line_styles.get(sampler, "-"), linewidth=2, markersize=6,
                    label=f"{sampler} / {scheduler}")

    ax.set_xlabel("steps")
    ax.set_ylabel("mean pixel deviation from high-step reference (lower = closer)")
    ax.set_title("fidelity vs. step count")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted({r["steps"] for r in rows}))

    # Two-part legend, placed outside the axes -- the data itself reaches
    # every corner of the plot (the dpmpp_3m_sde spike lands top-right),
    # so no in-axes corner is actually free.
    color_handles = [plt.Line2D([0], [0], color=SCHEDULER_COLORS[s], linewidth=2, label=s) for s in schedulers]
    style_handles = [plt.Line2D([0], [0], color="black", linestyle=line_styles[s], linewidth=2, label=s)
                      for s in samplers if s in line_styles]
    legend1 = ax.legend(handles=color_handles, title="scheduler", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, title="sampler", loc="upper left", bbox_to_anchor=(1.01, 0.62), fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
