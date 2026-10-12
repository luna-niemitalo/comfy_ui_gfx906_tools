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
import hashlib
import json
import logging
import os
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


class CalibrationCache:
    """Worst-seen missed mass per (layer, head), persisted across runs and restarts.

    Why worst-seen over several runs: head verdicts drift over the steps of a run more
    than across prompts. Step-0 verdicts of the same run let 3-9% of head-steps exceed 2%
    missed; the worst over all steps of 3 other runs let 0.4-2.8% through
    (2026-09-29, develop/experiments/2026-09-29/analyze_missed.py).

    File: <ComfyUI user dir>/__gfx906_int4/attention_calibration.json (a system user
    dir: never served over HTTP, untouched by model/browser cache wipes). Written
    atomically (tmp + rename) and merged with what is on disk, so two ComfyUI processes
    only ever raise values.
    Entry key: model fingerprint | image tokens | query rows per block | tau. Missed mass
    is stored rather than verdicts, so max_missed can change without relearning.
    Entry value: {"runs": learning runs seen, "missed": {layer: [worst per head]}}."""

    VERSION = 1

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.entries = self._load()
        self.dirty = False
        logging.info(f"[Gfx906 Attention] calibration cache {path}: {len(self.entries)} entries")

    def _load(self):
        # IN, non-critical: a missing, foreign or corrupt file means relearning, not failing
        try:
            with open(self.path) as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logging.warning(f"[Gfx906 Attention] calibration cache {self.path} unreadable ({e}); starting empty")
            return {}
        if not isinstance(data, dict) or data.get("version") != self.VERSION or not isinstance(data.get("entries"), dict):
            logging.warning(f"[Gfx906 Attention] calibration cache {self.path}: expected version {self.VERSION}, "
                            f"got {data.get('version') if isinstance(data, dict) else type(data).__name__}; starting empty")
            return {}
        entries = {}
        for key, entry in data["entries"].items():
            try:
                entries[key] = {"runs": int(entry["runs"]),
                                "missed": {layer: [float(v) for v in heads] for layer, heads in entry["missed"].items()}}
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                logging.warning(f"[Gfx906 Attention] calibration cache: dropping malformed entry {key!r} ({e})")
        return entries

    def entry(self, key):
        with self.lock:
            return self.entries.setdefault(key, {"runs": 0, "missed": {}})

    def count_run(self, key):
        with self.lock:
            self.entries[key]["runs"] += 1
            self.dirty = True

    def record(self, key, layer, missed):
        """Raise the stored worst missed mass of layer's heads to missed ([H] list)."""
        with self.lock:
            layers = self.entries[key]["missed"]
            old = layers.get(layer)
            layers[layer] = missed if old is None else [max(a, b) for a, b in zip(old, missed)]
            self.dirty = True

    def save(self):
        with self.lock:
            if not self.dirty:
                return
            merged = self._merge(self._load(), self.entries)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"version": self.VERSION, "entries": merged}, f)
            os.replace(tmp, self.path)
            self.entries, self.dirty = merged, False

    @staticmethod
    def _merge(disk, mine):
        merged = {key: {"runs": e["runs"], "missed": dict(e["missed"])} for key, e in disk.items()}
        for key, e in mine.items():
            m = merged.setdefault(key, {"runs": 0, "missed": {}})
            m["runs"] = max(m["runs"], e["runs"])
            for layer, heads in e["missed"].items():
                old = m["missed"].get(layer)
                m["missed"][layer] = heads if old is None else [max(a, b) for a, b in zip(old, heads)]
        return merged


_caches = {}


def calibration_cache():
    """One CalibrationCache per file per process (every patched model shares it)."""
    import folder_paths
    path = os.path.join(folder_paths.get_system_user_directory("gfx906_int4"), "attention_calibration.json")
    if path not in _caches:
        _caches[path] = CalibrationCache(path)
    return _caches[path]


def model_fingerprint(diffusion_model):
    """sha1 over the names, shapes and first 64 KiB of the 4 largest tensors: survives
    renames and moves of the file, changes with different weights. (LoRA patches are not
    included: they shift attention a little; worst-seen merging absorbs that.)"""
    tensors = sorted(list(diffusion_model.named_parameters()) + list(diffusion_model.named_buffers()),
                     key=lambda nt: (-nt[1].numel(), nt[0]))[:4]
    h = hashlib.sha1()
    for name, t in tensors:
        h.update(f"{name}{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.detach().reshape(-1).view(torch.uint8)[:65536].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


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


# Runs learned dense (every call measured, output identical to dense) before a cache entry
# is trusted: the leave-one-out test above used 3 other runs.
LEARN_RUNS = 3


class CachedSparsityState:
    """Like SparsityState, but head verdicts come from CalibrationCache: the first
    LEARN_RUNS runs of an entry run dense and record every call's missed mass; later
    runs are sparse where the worst-seen missed mass <= max_missed, with no calibration
    cost. A layer the entry has never seen is measured (dense) on first use."""

    def __init__(self, tau, max_missed, rows, diffusion_model, cache):
        self.tau, self.max_missed, self.rows = tau, max_missed, rows
        self.diffusion_model, self.cache = diffusion_model, cache
        self.lock = threading.Lock()
        self.fingerprint = None
        self.last_sigma = {}
        self.announced = set()

    def _key(self, q, k, transformer_options):
        with self.lock:
            if self.fingerprint is None:
                self.fingerprint = model_fingerprint(self.diffusion_model)
        img = transformer_options.get("img_slice")
        tokens = img[1] - img[0] if img is not None else k.shape[2]
        return f"{self.fingerprint}|img{tokens}|rows{self.rows}|tau{self.tau:g}"

    def _count_run(self, key, transformer_options):
        """A sigma above this entry's last one starts a run (per entry, not per device:
        the two split halves see the same sigma and count it once)."""
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return
        sigma = float(sigmas.flatten()[0])
        with self.lock:
            last = self.last_sigma.get(key)
            self.last_sigma[key] = sigma
        if last is None or sigma > last:
            self.cache.count_run(key)
            self.announced.discard(key)

    def keep(self, q, k, transformer_options):
        block = transformer_options.get("block_index")
        if block is None:
            return None
        last_layer = block == transformer_options.get("total_blocks", 0) - 1
        layer = str(block) if transformer_options.get("block_type") is None else f"{transformer_options['block_type']}:{block}"
        key = self._key(q, k, transformer_options)
        entry = self.cache.entry(key)
        self._count_run(key, transformer_options)
        keep = predict_keep(q, k, self.tau, self.rows)
        worst = entry["missed"].get(layer)
        if entry["runs"] <= LEARN_RUNS or worst is None:
            self.cache.record(key, layer, missed_mass(q, k, keep, self.rows).tolist())
            if last_layer:
                self.cache.save()
            return None
        sparse = torch.tensor(worst, device=q.device) <= self.max_missed
        if key not in self.announced:
            self.announced.add(key)
            n_sparse = sum(sum(v <= self.max_missed for v in heads) for heads in entry["missed"].values())
            n_heads = sum(len(heads) for heads in entry["missed"].values())
            logging.info(f"[Gfx906 Attention] cached calibration {key} (learned over {LEARN_RUNS} runs, "
                         f"{entry['runs']} seen): "
                         f"{n_sparse}/{n_heads} heads sparse")
        return (keep | ~sparse[None, :, None, None]).to(torch.uint8)
