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

Batches of 2+ skip all of the above: each rank runs half the batch
unmodified, with no exchange.
"""
import copy
import logging
import threading

import torch

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope

WRAPPER_KEY = "gfx906_row_split"
# A rank that died leaves its peer waiting at the barrier; this bounds that
# wait. Must exceed the slowest single block (~1 s at 2560x1440 today).
BARRIER_TIMEOUT_S = 300
# Working memory a rank needs beyond its weights. Measured 2560x1440, 600 text
# tokens: ~2.4 GB with 3.75 GB of other models parked on the peer (quad
# attention shrinks its chunks to fit, costing ~3%). Below this, split steps
# risk an OOM mid-block, so the step runs on one GPU instead.
PEER_HEADROOM_BYTES = int(2.5 * 2**30)
AUTO = "auto"


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


def _batch_slice(value, batch, part):
    return value[part] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch else value


def _module_bytes(module):
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


class _RowSplit:
    def __init__(self, device, min_image_tokens):
        self.requested = device
        self.min_image_tokens = min_image_tokens
        self.device = None
        self.replica = None
        self.last_plan = None

    def _log_plan(self, plan):
        if plan != self.last_plan:
            logging.info(f"[Gfx906 Split] {plan}")
            self.last_plan = plan

    def _pick_peer(self, main):
        """The GPU rank 1 runs on: the requested one, or under "auto" the
        other GPU with the most free memory. The main device comes from the
        latent, because loader nodes (e.g. MultiGPU) choose it at runtime."""
        if self.requested != AUTO:
            peer = torch.device(self.requested)
            if peer == main:
                raise ValueError(f"Gfx906 Split: second GPU must differ from the model's GPU; "
                                 f"expected a device other than {main}, got {peer} (use \"auto\")")
            return peer
        others = [torch.device("cuda", i) for i in range(torch.cuda.device_count()) if i != main.index]
        if not others:
            return None
        return max(others, key=comfy.model_management.get_free_memory)

    def _ensure_replica(self, dm, peer):
        """Replica on `peer` if it fits there (freeing ComfyUI-managed models
        on that GPU first if needed); else None and the step runs unsplit."""
        if self.device != peer:
            self.replica, self.device = None, peer
        need = PEER_HEADROOM_BYTES + (0 if self.replica is not None else _module_bytes(dm))
        if comfy.model_management.get_free_memory(peer) < need:
            comfy.model_management.free_memory(need, peer)
        free = comfy.model_management.get_free_memory(peer)
        if free < need:
            self._log_plan(f"one GPU: {peer} has {free / 2**30:.1f} GB free, a split needs {need / 2**30:.1f} GB")
            return None
        if self.replica is None:
            self.replica = _replicate(dm, peer)
        return self.replica

    def __call__(self, executor, x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        dm = executor.class_obj
        # ref latents would be appended whole on both ranks (duplicated keys)
        if ref_latents:
            self._log_plan("one GPU: reference latents are not split")
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        peer = self._pick_peer(x.device)
        if peer is None:
            self._log_plan("one GPU: no second GPU")
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        if self._ensure_replica(dm, peer) is None:
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        # batch >= 2 (several images, or cond+uncond under cfg): each GPU takes
        # half the batch, fully independent -- no per-block exchange at all
        batch = x.shape[0]
        if batch >= 2:
            half0, half1 = slice(0, -(-batch // 2)), slice(-(-batch // 2), batch)
            args0 = [_batch_slice(v, batch, half0) for v in (x, timesteps, context, attention_mask)]
            args1 = [_batch_slice(v, batch, half1) for v in (x, timesteps, context, attention_mask)]
            kwargs0 = {k: _batch_slice(v, batch, half0) for k, v in kwargs.items()}
            kwargs1 = {k: _batch_slice(v, batch, half1) for k, v in kwargs.items()}
            self._log_plan(f"batch split: {x.device} x{half0.stop} + {peer} x{batch - half0.stop}")
            return self._run_pair(executor, args0, kwargs0, transformer_options, args1, kwargs1, transformer_options, 0, None)

        # small images don't amortize the per-block barrier
        h, w = x.shape[-2], x.shape[-1]
        if (h // dm.patch) * (w // dm.patch) < self.min_image_tokens:
            self._log_plan(f"one GPU: {(h // dm.patch) * (w // dm.patch)} image tokens < min_image_tokens {self.min_image_tokens}")
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

        # Latent row where rank 1 starts: half the token rows, on a patch boundary
        split = max(-(-h // dm.patch) // 2, 1) * dm.patch
        self._log_plan(f"row split: {x.device} rows 0-{split // dm.patch} + {peer} rows "
                       f"{split // dm.patch}-{-(-h // dm.patch)} (token rows)")
        exchange = _KVExchange()
        opts0 = _with_patches(transformer_options, _rank_patches(0, 0, exchange))
        opts1 = _with_patches(transformer_options, _rank_patches(1, split // dm.patch, exchange))
        args0 = [x[..., :split, :], timesteps, context, attention_mask]
        args1 = [x[..., split:, :], timesteps, context, attention_mask]
        return self._run_pair(executor, args0, kwargs, opts0, args1, kwargs, opts1, -2, exchange)

    def _run_pair(self, executor, args0, kwargs0, opts0, args1, kwargs1, opts1, cat_dim, exchange):
        """Rank 0 through the wrapper chain here, rank 1 on the replica in a
        worker thread; outputs concatenated along cat_dim. A failing rank
        aborts the exchange barrier so its peer can't wait forever."""
        peer = {}

        def run_peer():
            try:
                with torch.cuda.device(self.device), torch.inference_mode():
                    peer["out"] = self.replica._forward(
                        *[_to(v, self.device) for v in args1], None, opts1,
                        **{k: _to(v, self.device) for k, v in kwargs1.items()})
            except BaseException as e:
                peer["error"] = e
                if exchange is not None:
                    exchange.barrier.abort()

        thread = threading.Thread(target=run_peer, name="gfx906-split")
        thread.start()
        try:
            out0 = executor(*args0, None, opts0, **kwargs0)
        except BaseException as e:
            if exchange is not None:
                exchange.barrier.abort()
            thread.join()
            # A broken barrier here only means the peer died first; its error
            # (typically an OOM on the second GPU) is the one worth showing.
            if isinstance(e, threading.BrokenBarrierError) and "error" in peer:
                raise peer["error"] from e
            raise
        thread.join()
        if "error" in peer:
            raise peer["error"]
        return torch.cat((out0, peer["out"].to(out0.device)), dim=cat_dim)


class Gfx906RowSplitKrea2:
    """MODEL -> MODEL. Runs each Krea2 sampling step on two GPUs: half the
    batch each when batch >= 2 (several images, or cond+uncond with cfg),
    otherwise half the image rows each. Replicates the diffusion model onto
    the second GPU on first use (int4 Krea2: ~7.5 GB there); steps run on
    one GPU when that GPU lacks room."""

    @classmethod
    def INPUT_TYPES(s):
        devices = [AUTO] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return {
            "required": {
                "model": ("MODEL",),
                "device": (devices, {"default": AUTO, "tooltip": "Second GPU. auto: whichever GPU the model is not on with the most free VRAM, checked every step."}),
                "min_image_tokens": ("INT", {"default": 2048, "min": 0, "max": 1 << 20, "tooltip": "Single-image steps below this many image tokens (latent pixels / 4) run on one GPU: the per-block sync costs more than it saves. Batches of 2+ always split by batch."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Split Krea2 (2 GPUs)"

    def patch(self, model, device, min_image_tokens):
        model = model.clone()
        model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _RowSplit(device, min_image_tokens))
        return (model,)
