"""Do head verdicts (missed <= bound) transfer? For each source run's step-0
verdicts, applied to every target run and step: allowed heads, violations
(allowed but missed > bound there), worst missed among allowed, and the mean
missed mass of allowed heads (what the output error follows)."""
import glob, json, os, sys
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOUND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
runs = {os.path.basename(p)[7:-3]: torch.load(p) for p in sorted(glob.glob(os.path.join(SCRATCH, "missed_*.pt")))}


def verdicts(rec, step):
    return {block: m <= BOUND for (s, block), m in rec.items() if s == step}


for src, src_rec in runs.items():
    allowed = verdicts(src_rec, 0)
    for dst, dst_rec in runs.items():
        steps = sorted({s for s, _ in dst_rec})
        row = {"calibrated_on": f"{src} step0", "applied_to": dst}
        for step in steps:
            n_allowed = n_viol = 0
            worst, mean_sum = 0.0, 0.0
            for (s, block), m in dst_rec.items():
                if s != step or block not in allowed:
                    continue
                a = allowed[block]
                n_allowed += int(a.sum())
                viol = a & (m > BOUND)
                n_viol += int(viol.sum())
                if a.any():
                    worst = max(worst, float(m[a].max()))
                    mean_sum += float(m[a].sum())
            row[f"step{step}"] = {"allowed": n_allowed, "violations": n_viol, "worst": round(worst, 3),
                                  "mean_missed": round(mean_sum / max(n_allowed, 1), 4)}
        print(json.dumps(row))
