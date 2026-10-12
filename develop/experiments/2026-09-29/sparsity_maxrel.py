"""Block-sparse rule the dense kernel can record for free: keep a (128-row,
64-key) block if its largest P relative to the row's max is >= eps (max over
the block's rows). Evaluated on the current step (oracle) and reused from the
previous step, plain and dilated by one key tile each side (tolerates drift).

Usage: sparsity_maxrel.py <device> <capture dir> <block> [...]
"""
import glob, json, os, sys
import torch

dev = torch.device(sys.argv[1])
CAP = sys.argv[2]
BLOCKS = [int(b) for b in sys.argv[3:]]
BQ, BK, ROWS = 128, 64, 2048
EPS = (1e-2, 1e-3, 1e-4)


def block_maxrel(q, k, rep):
    """[H, nqb, nkt] max over the block of P / row max (log2 domain avoided: exact exp)."""
    H, L, D = q.shape
    nkt, nqb = -(-L // BK), -(-L // BQ)
    out = torch.zeros(H, nqb, nkt, device=dev)
    for h in range(H):
        kh = k[h // rep]
        for r0 in range(0, L, ROWS):
            s = q[h, r0:r0 + ROWS] @ kh.T * D ** -0.5
            rel = torch.exp(s - s.amax(-1, keepdim=True))
            n = rel.shape[0]
            rel = torch.nn.functional.pad(rel, (0, nkt * BK - L, 0, -n % BQ))
            out[h, r0 // BQ:r0 // BQ + rel.shape[0] // BQ] = rel.reshape(-1, BQ, nkt, BK).amax((1, 3))
    return out


def masked_error(q, k, v, rep, keep):
    H, L, D = q.shape
    err, ref_sq = 0.0, 0.0
    for h in range(H):
        kh, vh = k[h // rep], v[h // rep]
        for r0 in range(0, L, ROWS):
            p = torch.softmax(q[h, r0:r0 + ROWS] @ kh.T * D ** -0.5, -1)
            ref = p @ vh
            km = keep[h, r0 // BQ:r0 // BQ + -(-p.shape[0] // BQ)].repeat_interleave(BQ, 0)[:p.shape[0]]
            pm = p * km.repeat_interleave(BK, 1)[:, :L]
            out = (pm @ vh) / pm.sum(-1, keepdim=True).clamp_min(1e-30)
            err += (out - ref).pow(2).sum().item()
            ref_sq += ref.pow(2).sum().item()
    return (err / ref_sq) ** 0.5


def dilate(keep):
    k = keep.clone()
    k[..., 1:] |= keep[..., :-1]
    k[..., :-1] |= keep[..., 1:]
    return k


prev = {}
for block in BLOCKS:
    for step in range(8):
        path = os.path.join(CAP, f"step{step}_block{block}.pt")
        if not os.path.exists(path):
            continue
        d = torch.load(path)
        q, k, v = (d[n][0].to(dev, torch.float32) for n in ("q", "k", "v"))
        rep = q.shape[0] // k.shape[0]
        mr = block_maxrel(q, k, rep)
        for eps in EPS:
            rules = {"oracle": mr >= eps}
            if (block, step - 1) in prev:
                p = prev[(block, step - 1)] >= eps
                rules["reuse+1"] = p
                rules["reuse+1_dilated"] = dilate(p)
            for rule, keep in rules.items():
                print(json.dumps({"capture": f"step{step}_block{block}", "rule": rule, "eps": eps,
                                  "sparsity": round(1 - keep.float().mean().item(), 3),
                                  "err": float(f"{masked_error(q, k, v, rep, keep):.2e}")}), flush=True)
        prev[(block, step)] = mr
        del q, k, v
        torch.cuda.empty_cache()
