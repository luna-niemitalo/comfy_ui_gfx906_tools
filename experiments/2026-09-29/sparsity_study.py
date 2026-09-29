"""Block-sparse attention study on captured Krea2 Q/K/V (fp32 emulation).

Block = 128 query rows (one fused-kernel workgroup) x 64 keys (one key tile).
A skipped block drops out of the softmax entirely (numerator and denominator),
which is what a block-sparse kernel computes.

Selection rules, per (head, query block):
  oracle   -- exact block mass (mean over the block's rows of the row's P
              mass in the tile), keep tiles in descending order until tau
              of the mass is covered
  pooled   -- SpargeAttn-style: softmax over pooled scores (mean q of the
              block . mean k of the tile), same top-tau rule; query blocks
              whose tokens are dissimilar (mean cosine to the block mean
              < SIM) keep everything, dissimilar key tiles are always kept
  reuse+d  -- the oracle map of step s applied to step s + d

Usage: sparsity_study.py <device> <capture dir> [block ...]
Prints one JSON line per (capture, rule, tau): sparsity (fraction of blocks
skipped) and output rel. error vs exact attention.
"""
import glob, json, os, sys
import torch

dev = torch.device(sys.argv[1])
CAP = sys.argv[2]
BLOCKS = [int(b) for b in sys.argv[3:]] or None
BQ, BK = 128, 64
TAUS = tuple(float(t) for t in os.environ.get("TAUS", "0.9,0.95,0.98,0.99,0.995").split(","))
SIM = 0.5
HEAD_CHUNK = 4
ROW_CHUNK = 2048


def load(path):
    d = torch.load(path)
    return (d[n][0].to(dev, torch.float32) for n in ("q", "k", "v"))


def pad_to(x, mult, dim):
    n = x.shape[dim]
    extra = -n % mult
    if extra == 0:
        return x
    shape = list(x.shape)
    shape[dim] = extra
    return torch.cat([x, x.new_zeros(shape)], dim)


def block_mass(q, k, rep):
    """[H, nqb, nkt] mean row P mass per block, exact."""
    H, L, D = q.shape
    nkt = -(-L // BK)
    nqb = -(-L // BQ)
    out = torch.zeros(H, nqb, nkt, device=dev)
    for h in range(H):
        kh = k[h // rep]
        for r0 in range(0, L, ROW_CHUNK):
            s = q[h, r0:r0 + ROW_CHUNK] @ kh.T * D ** -0.5
            p = torch.softmax(s, -1)
            p = pad_to(pad_to(p, BK, 1), BQ, 0)
            m = p.reshape(p.shape[0] // BQ, BQ, nkt, BK).sum(-1).sum(1)
            out[h, r0 // BQ:r0 // BQ + m.shape[0]] = m
    rows = torch.full((nqb,), BQ, device=dev)
    rows[-1] = L - (nqb - 1) * BQ
    return out / rows[None, :, None]


def pooled_scores(q, k, rep):
    """[H, nqb, nkt] softmax of pooled scores; plus similarity masks."""
    H, L, D = q.shape
    qb = pad_to(q, BQ, 1).reshape(H, -1, BQ, D)
    kb = pad_to(k, BK, 1).reshape(k.shape[0], -1, BK, D)
    qm, km = qb.mean(2), kb.mean(2)
    s = qm @ km.repeat_interleave(rep, 0).transpose(-1, -2) * D ** -0.5
    s[..., -1] += torch.log(torch.tensor((L - (kb.shape[1] - 1) * BK) / BK, device=dev))  # partial last tile
    p = torch.softmax(s, -1)
    q_sim = torch.nn.functional.cosine_similarity(qb, qm[:, :, None], dim=-1).mean(-1)  # [H, nqb]
    k_sim = torch.nn.functional.cosine_similarity(kb, km[:, :, None], dim=-1).mean(-1)  # [KVH, nkt]
    return p, q_sim < SIM, (k_sim < SIM).repeat_interleave(rep, 0)


def top_cdf(mass, tau):
    """keep mask [.., nkt]: largest tiles until tau of the row's mass."""
    srt, idx = mass.sort(-1, descending=True)
    cum = srt.cumsum(-1) / srt.sum(-1, keepdim=True)
    keep_sorted = (cum - srt / srt.sum(-1, keepdim=True)) < tau  # include the tile that crosses tau
    return torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)


def masked_error(q, k, v, rep, keep):
    """rel. error of block-sparse attention with keep [H, nqb, nkt] vs exact."""
    H, L, D = q.shape
    err, ref_sq = 0.0, 0.0
    for h in range(H):
        kh, vh = k[h // rep], v[h // rep]
        for r0 in range(0, L, ROW_CHUNK):
            s = q[h, r0:r0 + ROW_CHUNK] @ kh.T * D ** -0.5
            p = torch.softmax(s, -1)
            ref = p @ vh
            km = keep[h, r0 // BQ:r0 // BQ + -(-s.shape[0] // BQ)]
            km = km.repeat_interleave(BQ, 0)[:s.shape[0]].repeat_interleave(BK, 1)[:, :L]
            pm = p * km
            out = (pm @ vh) / pm.sum(-1, keepdim=True).clamp_min(1e-30)
            err += (out - ref).pow(2).sum().item()
            ref_sq += ref.pow(2).sum().item()
    return (err / ref_sq) ** 0.5


def emit(capture, rule, tau, keep, q, k, v, rep):
    rec = {"capture": capture, "rule": rule, "tau": tau, "sparsity": round(1 - keep.float().mean().item(), 3),
           "err": float(f"{masked_error(q, k, v, rep, keep):.2e}")}
    print(json.dumps(rec), flush=True)


paths = sorted(glob.glob(os.path.join(CAP, "*.pt")))
if BLOCKS is not None:
    paths = [p for p in paths if int(os.path.basename(p).split("block")[1][:-3]) in BLOCKS]
by_name = {os.path.basename(p)[:-3]: p for p in paths}
oracle_maps = {}
for name, path in sorted(by_name.items(), key=lambda kv: (int(kv[0].split("_")[1][5:]), int(kv[0].split("_")[0][4:]))):
    q, k, v = load(path)
    rep = q.shape[0] // k.shape[0]
    step, block = int(name.split("_")[0][4:]), int(name.split("_")[1][5:])
    mass = block_mass(q, k, rep)
    pooled, q_dis, k_dis = pooled_scores(q, k, rep)
    for tau in TAUS:
        keep = top_cdf(mass, tau)
        oracle_maps[(block, step, tau)] = keep
        emit(name, "oracle", tau, keep, q, k, v, rep)
        keep_p = top_cdf(pooled, tau) | q_dis[..., None] | k_dis[:, None, :]
        emit(name, "pooled", tau, keep_p, q, k, v, rep)
        for d in (1, 2):
            prev = oracle_maps.get((block, step - d, tau))
            if prev is not None:
                emit(name, f"reuse+{d}", tau, prev, q, k, v, rep)
    del q, k, v, mass, pooled
    torch.cuda.empty_cache()
