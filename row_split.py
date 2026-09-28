"""
Split one Krea2 forward across two GPUs by latent rows (sequence
parallelism), entirely through ModelPatcher hooks -- no model code touched.

Each GPU runs Krea2's own _forward on its half of the latent rows; the
model is replicated onto the second GPU on first use. Everything in a
SingleStreamBlock is token-local except attention, so the only exchange is
each block's K/V (12 GQA heads, before repeat_interleave: ~45 MB per block
at 2560x1440, a few ms over PCIe vs ~300 ms of compute per block):

- post_input patch: offsets the rank's RoPE row ids by its first token row
- attn1 patch: applies RoPE locally (pe is the rank's own slice, but K is
  about to become full length), swaps K/V with the peer, returns pe=None.
  Both ranks carry the text tokens as queries; only rank 0 contributes
  them as keys
- the model's own tail slicing / unpatchify then yields each half's output

Rank 0 runs through the normal wrapper chain on the calling thread; rank 1
calls the replica's _forward on a worker thread. The two meet at a barrier
once per block.
"""
import copy
import threading

import torch

import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope

WRAPPER_KEY = "gfx906_row_split"
# A rank that died leaves its peer waiting at the barrier; this bounds that
# wait. Must exceed the slowest single block (~1 s at 2560x1440 today).
BARRIER_TIMEOUT_S = 300


class _KVExchange:
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT_S)
        self.slots = [None, None]

    def swap(self, rank, k, v):
        self.slots[rank] = (k, v)
        self.barrier.wait()
        peer_k, peer_v = self.slots[1 - rank]
        # A cross-device copy_ runs on the source device's current stream and
        # fences both devices' streams (ATen Copy.cu), so the peer's pending
        # k/v kernels are ordered before it without a host sync.
        peer_k = peer_k.to(k.device, non_blocking=True)
        peer_v = peer_v.to(v.device, non_blocking=True)
        # Second wait: the peer has taken our slot before either rank
        # overwrites its own on the next block.
        self.barrier.wait()
        return peer_k, peer_v


def _replicate(module, device):
    """deepcopy with every parameter/buffer pre-mapped to a copy on `device`,
    so the source device never holds a second copy of the weights."""
    memo = {}
    for p in module.parameters():
        memo[id(p)] = torch.nn.Parameter(p.detach().to(device), requires_grad=False)
    for b in module.buffers():
        memo[id(b)] = b.to(device)
    return copy.deepcopy(module, memo)


def _rank_patches(rank, row_offset, exchange):
    txt_len = {}

    def post_input(args):
        img_ids = args["img_ids"].clone()
        img_ids[..., 1] += row_offset
        txt_len["n"] = args["txt"].shape[1]
        return {"img": args["img"], "txt": args["txt"], "img_ids": img_ids, "txt_ids": args["txt_ids"]}

    def attn1(q, k, v, pe=None, attn_mask=None, extra_options=None):
        if pe is not None:
            q, k = apply_rope(q, k, pe)
        if rank == 1:
            # Both ranks carry the text tokens (the model's tail slices by a
            # txtlen taken before post_input runs, so they can't be dropped),
            # but only rank 0's text may be keys, or they'd count twice.
            k, v = k[:, :, txt_len["n"]:], v[:, :, txt_len["n"]:]
        peer_k, peer_v = exchange.swap(rank, k, v)
        return {"q": q, "k": torch.cat((k, peer_k), dim=2), "v": torch.cat((v, peer_v), dim=2), "pe": None}

    return {"post_input": [post_input], "attn1_patch": [attn1]}


def _with_patches(transformer_options, patches):
    opts = transformer_options.copy()
    merged = {k: list(v) for k, v in opts.get("patches", {}).items()}
    for name, fns in patches.items():
        merged[name] = merged.get(name, []) + fns
    opts["patches"] = merged
    return opts


def _to(value, device):
    return value.to(device) if torch.is_tensor(value) else value


class _RowSplit:
    def __init__(self, device, min_image_tokens):
        self.device = torch.device(device)
        self.min_image_tokens = min_image_tokens
        self.replica = None

    def __call__(self, executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        dm = executor.class_obj
        h, w = x.shape[-2], x.shape[-1]
        image_tokens = (h // dm.patch) * (w // dm.patch)
        # ref latents would be appended whole on both ranks (duplicated keys);
        # small images don't amortize the per-block barrier. Both run unsplit.
        if ref_latents or image_tokens < self.min_image_tokens:
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        if self.replica is None:
            self.replica = _replicate(dm, self.device)

        # Latent row where rank 1 starts: half the token rows, on a patch boundary
        split = max(-(-h // dm.patch) // 2, 1) * dm.patch
        exchange = _KVExchange()
        opts0 = _with_patches(transformer_options, _rank_patches(0, 0, exchange))
        opts1 = _with_patches(transformer_options, _rank_patches(1, split // dm.patch, exchange))
        peer = {}

        def run_peer():
            try:
                with torch.cuda.device(self.device), torch.inference_mode():
                    peer["out"] = self.replica._forward(
                        x[..., split:, :].to(self.device), timesteps.to(self.device), context.to(self.device),
                        _to(attention_mask, self.device), None, opts1,
                        **{k: _to(v, self.device) for k, v in kwargs.items()})
            except BaseException as e:
                peer["error"] = e
                exchange.barrier.abort()

        thread = threading.Thread(target=run_peer, name="gfx906-row-split")
        thread.start()
        try:
            out0 = executor(x[..., :split, :], timesteps, context, attention_mask, None, opts0, **kwargs)
        except BaseException:
            exchange.barrier.abort()
            thread.join()
            raise
        thread.join()
        if "error" in peer:
            raise peer["error"]
        return torch.cat((out0, peer["out"].to(x.device)), dim=-2)


class Gfx906RowSplitKrea2:
    """MODEL -> MODEL. Runs each Krea2 sampling step on two GPUs, each
    taking half the image rows. Replicates the diffusion model onto
    `device` on first use (int4 Krea2: ~7.5 GB there)."""

    @classmethod
    def INPUT_TYPES(s):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "model": ("MODEL",),
                "device": (devices, {"default": devices[-1]}),
                "min_image_tokens": ("INT", {"default": 2048, "min": 0, "max": 1 << 20, "tooltip": "Below this many image tokens (latent pixels / 4) the step runs on one GPU: the per-block sync costs more than it saves."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Row Split Krea2 (2 GPUs)"

    def patch(self, model, device, min_image_tokens):
        model = model.clone()
        model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _RowSplit(device, min_image_tokens))
        return (model,)
