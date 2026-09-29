"""Block-sparse attention masks for the fused kernel, predicted per call and
calibrated online per (layer, head).

Prediction (SpargeAttn-style): softmax over pooled scores -- mean q of each
kernel query block . mean k of each 64-key tile -- and keep the largest tiles
until tau of the pooled mass; blocks whose tokens are dissimilar (mean
cosine to their mean < SIMILARITY) keep everything, since their mean
misrepresents them.

Calibration: the pooled prediction fails on some layers (Krea2 block 0 at
2560x1440: 42-53% output error, vs 1-4% on blocks 9/18/27; CLAUDE.md "Block
sparsity"). So the first call per (layer, device, shape) of a sampling run is
dense, and on sampled query blocks measures the exact attention mass the
prediction would drop, per head. Heads that drop more than max_missed stay
dense for the rest of the run. A rising sigma marks a new run and resets.
"""
import logging
import threading

import torch

TILE = 64
SIMILARITY = 0.5
# 8 gave the same head verdicts' quality as 16 at half the cost (79 vs 157
# ms per layer at 2560x1440; verdicts from step 0 held on steps 2 and 5).
CALIBRATION_BLOCKS = 8
# Scores for calibration are computed a few heads at a time: this many fp32
# elements per chunk (256 MiB; softmax and padding hold ~3 such tensors, and
# 3840x2160 on one GPU peaks at 12.3 of 16 GB without them).
CALIBRATION_CHUNK = 2 ** 26


def _block_means(x, block):
    """x [B, H, L, D] -> (means [B, H, n, D] fp32 over valid rows, mean cosine of
    each row to its block mean [B, H, n]; padding rows count as 0 similarity)."""
    B, H, L, D = x.shape
    n = -(-L // block)
    xb = torch.nn.functional.pad(x, (0, 0, 0, n * block - L)).reshape(B, H, n, block, D)
    counts = torch.full((n,), block, device=x.device, dtype=torch.float32)
    counts[-1] = L - (n - 1) * block
    means = xb.sum(3, dtype=torch.float32) / counts[:, None]
    dots = torch.einsum("bhnrd,bhnd->bhnr", xb, means.to(x.dtype)).float()
    norms = torch.linalg.vector_norm(xb, dim=-1, dtype=torch.float32) * torch.linalg.vector_norm(means, dim=-1)[..., None]
    similarity = (dots / norms.clamp_min(1e-30)).sum(-1) / block
    return means, similarity, counts


def _top_cdf(mass, tau):
    """keep [.., n] bool: largest entries until tau of the row's mass (the
    entry crossing tau included, so every row keeps at least one)."""
    srt, idx = mass.sort(-1, descending=True)
    before = (srt.cumsum(-1) - srt) / srt.sum(-1, keepdim=True)
    return torch.zeros_like(before, dtype=torch.bool).scatter(-1, idx, before < tau)


def predict_keep(q, k, tau, rows):
    """q [B, H, Lq, 128], k [B, KVH, Lk, 128] -> keep bool [B, H, ceil(Lq / rows), ceil(Lk / 64)]."""
    rep = q.shape[1] // k.shape[1]
    qm, q_sim, _ = _block_means(q, rows)
    km, k_sim, k_counts = _block_means(k, TILE)
    km, k_sim = km.repeat_interleave(rep, 1), k_sim.repeat_interleave(rep, 1)
    scores = qm @ km.transpose(-1, -2) * q.shape[-1] ** -0.5 + k_counts.log()  # a tile's weight ~ its key count
    keep = _top_cdf(torch.softmax(scores, -1), tau)
    return keep | (q_sim < SIMILARITY)[..., None] | (k_sim < SIMILARITY)[:, :, None, :]


def missed_mass(q, k, keep, rows):
    """Per head, the exact attention mass keep drops, averaged over
    CALIBRATION_BLOCKS evenly spaced query blocks of batch 0 -> [H] fp32."""
    H, Lq, D = q.shape[1:]
    Lk = k.shape[2]
    rep = H // k.shape[1]
    n_qb, n_kt = keep.shape[2], keep.shape[3]
    blocks = torch.linspace(0, n_qb - 1, min(CALIBRATION_BLOCKS, n_qb), device=q.device).round().long().unique()
    row_idx = (blocks[:, None] * rows + torch.arange(rows, device=q.device)).clamp_max(Lq - 1).flatten()
    q_rows = q[0, :, row_idx]
    dropped = ~keep[0][:, blocks]  # [H, S, n_kt]
    missed = torch.empty(H, device=q.device)
    chunk = max(1, CALIBRATION_CHUNK // (row_idx.numel() * Lk))
    for h0 in range(0, H, chunk):
        heads = torch.arange(h0, min(h0 + chunk, H), device=q.device)
        p = torch.softmax(q_rows[heads].float() @ k[0, heads // rep].float().transpose(-1, -2) * D ** -0.5, -1)
        p = torch.nn.functional.pad(p, (0, n_kt * TILE - Lk)).reshape(len(heads), len(blocks), rows, n_kt, TILE)
        tile_mass = p.sum(4).mean(2)
        missed[heads] = (tile_mass * dropped[heads]).sum(-1).mean(-1)
    return missed


class SparsityState:
    """Per override: which heads of each (layer, device, shape) may run
    sparse in the current sampling run."""

    def __init__(self, tau, max_missed, rows):
        self.tau, self.max_missed, self.rows = tau, max_missed, rows
        self.lock = threading.Lock()  # the two-GPU split calls from one thread per device
        self.last_sigma = {}
        self.sparse_heads = {}

    def _new_run(self, device, transformer_options):
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return
        sigma = float(sigmas.flatten()[0])
        with self.lock:
            last = self.last_sigma.get(device)
            self.last_sigma[device] = sigma
            if last is not None and sigma > last:
                self.sparse_heads = {key: v for key, v in self.sparse_heads.items() if key[1] != device}

    def keep(self, q, k, transformer_options):
        """uint8 keep mask for the kernel, or None to run dense (calibration
        call, or a model without block_index)."""
        block = transformer_options.get("block_index")
        if block is None:
            return None
        if transformer_options.get("block_type") is not None:
            block = (transformer_options["block_type"], block)
        self._new_run(q.device, transformer_options)
        key = (block, q.device, q.shape[2], k.shape[2])
        keep = predict_keep(q, k, self.tau, self.rows)
        with self.lock:
            sparse = self.sparse_heads.get(key)
        if sparse is None:
            missed = missed_mass(q, k, keep, self.rows)
            sparse = missed <= self.max_missed
            with self.lock:
                self.sparse_heads[key] = sparse
            final = keep | ~sparse[None, :, None, None]
            logging.info(f"[Gfx906 Attention] block {block}: {int(sparse.sum())}/{len(sparse)} heads sparse, "
                         f"{1 - final.float().mean().item():.0%} of blocks skipped, "
                         f"missed mass median {missed.median().item():.2%}")
            return None
        return (keep | ~sparse[None, :, None, None]).to(torch.uint8)
