#!/usr/bin/env python3
"""Two-GPU throughput of LLM-training-shaped work: whole transformer blocks
(4B / 8B shapes, fwd + bwd) per card alone and on both cards at once, PCIe
transfers alone / concurrent / overlapped with backward.

One process per card, each seeing only its own GPU (ROCR_VISIBLE_DEVICES), so
no HIP peer path exists: gradient exchange on this machine goes through pinned
host memory. Peer copies into card1 (the display card) hung the machine
(~/.claude/rules/gpu-lock.md). Phases line up through files in --barrier-dir.

Timing is wall clock between host syncs (CLAUDE.md, Hazards: HIP events overlap).

Usage (repo root):
    nu ~/dev/mantlecore/tools/gpu_run.nu --card 0 --card 1 --hip-on-display --timeout 1800 -- \
        direnv exec . python develop/experiments/2026-10-11/dual_bench.py --barrier-dir <empty dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bench import load_node_package, time_s  # noqa: E402

KFD = "/sys/class/kfd/kfd/topology/nodes"
CARD_BY_BUS = {0x10: "card0", 0x0D: "card1"}  # from gpu_run.nu's log: card0 = 0000:10:00.0, card1 (display) = 0000:0d:00.0

TOKENS = 4096  # one sequence of 4096 per micro-batch
LORA_RANK = 64

# hidden, q dim, kv dim, ffn, layers, vocab
CONFIGS = {
    "4B": dict(d=2560, q=4096, kv=1024, ffn=9728, layers=36, vocab=151936),   # Qwen3-4B
    "8B": dict(d=4096, q=4096, kv=1024, ffn=14336, layers=32, vocab=128256),  # Llama-3-8B
}
MODES = ["full_fp16", "lora_fp16", "lora_int4"]


def emit(**row):
    print(json.dumps(row), flush=True)


def barrier(directory, card, name, peers):
    open(os.path.join(directory, f"{name}.{card}"), "w").close()
    while not all(os.path.exists(os.path.join(directory, f"{name}.{p}")) for p in peers):
        time.sleep(0.05)


class Int4Frozen(torch.autograd.Function):
    """y = x W^T through the W4A4 kernel; dx = dy W through a second packing of W^T."""

    @staticmethod
    def forward(ctx, x, packs):
        ext, w4, w4s, wt4, wt4s = packs
        ctx.packs = packs
        flat = x.reshape(-1, x.shape[-1])
        return ext.gemm_int4(*ext.quantize_int4(flat), w4, w4s, None, torch.float16).view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, dy):
        ext, w4, w4s, wt4, wt4s = ctx.packs
        flat = dy.reshape(-1, dy.shape[-1]).contiguous()
        return ext.gemm_int4(*ext.quantize_int4(flat), wt4, wt4s, None, torch.float16).view(*dy.shape[:-1], -1), None


class Linear(torch.nn.Module):
    def __init__(self, k, n, mode, ops, device):
        super().__init__()
        w = torch.randn(n, k, device=device, dtype=torch.float16) * k**-0.5
        self.mode = mode
        if mode == "lora_int4":
            ext = ops._get_ext()
            self.packs = (ext, *ops.quantize_pack_int4(w), *ops.quantize_pack_int4(w.t().contiguous()))
        else:
            self.weight = torch.nn.Parameter(w, requires_grad=(mode == "full_fp16"))
        if mode != "full_fp16":
            self.lora_a = torch.nn.Parameter(torch.randn(LORA_RANK, k, device=device, dtype=torch.float16) * k**-0.5)
            self.lora_b = torch.nn.Parameter(torch.zeros(n, LORA_RANK, device=device, dtype=torch.float16))

    def forward(self, x):
        base = Int4Frozen.apply(x, self.packs) if self.mode == "lora_int4" else F.linear(x, self.weight)
        if self.mode == "full_fp16":
            return base
        return base + F.linear(F.linear(x, self.lora_a), self.lora_b)


def rms_norm(x, weight):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight


class Block(torch.nn.Module):
    """Llama/Qwen block without RoPE (elementwise, minor): RMSNorm, GQA causal SDPA, SwiGLU."""

    def __init__(self, c, mode, ops, device):
        super().__init__()
        self.c = c
        self.qkv = Linear(c["d"], c["q"] + 2 * c["kv"], mode, ops, device)
        self.o = Linear(c["q"], c["d"], mode, ops, device)
        self.gate_up = Linear(c["d"], 2 * c["ffn"], mode, ops, device)
        self.down = Linear(c["ffn"], c["d"], mode, ops, device)
        self.n1 = torch.nn.Parameter(torch.ones(c["d"], device=device, dtype=torch.float16))
        self.n2 = torch.nn.Parameter(torch.ones(c["d"], device=device, dtype=torch.float16))

    def forward(self, x):
        c, length = self.c, x.shape[1]
        q, k, v = self.qkv(rms_norm(x, self.n1)).split([c["q"], c["kv"], c["kv"]], dim=-1)
        q, k, v = (t.view(1, length, -1, 128).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(1, length, c["q"]))
        gate, up = self.gate_up(rms_norm(x, self.n2)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


def block_flops(c):
    """(tokens x weight params, causal attention fwd flops)."""
    linear = TOKENS * (c["d"] * (c["q"] + 2 * c["kv"]) + c["q"] * c["d"] + c["d"] * 2 * c["ffn"] + c["ffn"] * c["d"])
    attention = 2 * TOKENS * TOKENS * c["q"]  # causal: half of 4 L^2 q
    return linear, attention


def bench_blocks(card, ops, device, tag):
    for name, c in CONFIGS.items():
        for mode in MODES:
            block = Block(c, mode, ops, device)
            x = torch.randn(1, TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
            grad = torch.randn_like(x)

            def step():
                block(x).backward(grad)

            seconds = time_s(step, min_s=1.5)
            linear, attention = block_flops(c)
            # trained weights: fwd + dgrad + wgrad = 6 flop/param/token; frozen: fwd + dgrad = 4
            useful = (6 if mode == "full_fp16" else 4) * linear + 3.5 * attention
            emit(card=card, phase=tag, section="block", model=name, mode=mode, ms_per_block=round(seconds * 1e3, 2),
                 tflops=round(useful / seconds / 1e12, 2),
                 model_tokens_per_s=round(TOKENS / (seconds * c["layers"]), 1))
            del block, x, grad
            torch.cuda.empty_cache()
        bench_head(card, c, name, device, tag)


def bench_head(card, c, name, device, tag):
    """Tied embedding as LM head: logits, cross-entropy, backward into hidden and the embedding."""
    emb = torch.nn.Parameter(torch.randn(c["vocab"], c["d"], device=device, dtype=torch.float16) * 0.02)
    h = torch.randn(TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
    target = torch.randint(0, c["vocab"], (TOKENS,), device=device)

    def step():
        F.cross_entropy(F.linear(h, emb).float(), target).backward()

    emit(card=card, phase=tag, section="head", model=name, ms_per_step=round(time_s(step, min_s=1.0) * 1e3, 1))
    del emb, h, target
    torch.cuda.empty_cache()


def bench_pcie(card, device, tag, size=256 * 2**20):
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host2 = torch.empty_like(host)
    dev = torch.empty_like(host, device=device)
    dev2 = torch.empty_like(host, device=device)
    up, down = torch.cuda.Stream(device), torch.cuda.Stream(device)
    emit(card=card, phase=tag, section="pcie", op="D2H", gbps=round(size / time_s(lambda: host.copy_(dev, non_blocking=True)) / 1e9, 2))
    emit(card=card, phase=tag, section="pcie", op="H2D", gbps=round(size / time_s(lambda: dev.copy_(host, non_blocking=True)) / 1e9, 2))

    def both_ways():
        with torch.cuda.stream(down):
            host2.copy_(dev2, non_blocking=True)
        with torch.cuda.stream(up):
            dev.copy_(host, non_blocking=True)

    emit(card=card, phase=tag, section="pcie", op="D2H+H2D at once (per direction)",
         gbps=round(size / time_s(both_ways) / 1e9, 2))
    del host, host2, dev, dev2


def bench_overlap(card, ops, device, tag):
    """8B full_fp16 block fwd+bwd on the default stream while a side stream moves
    256 MiB each way per block, as an overlapped gradient exchange would."""
    c = CONFIGS["8B"]
    block = Block(c, "full_fp16", ops, device)
    x = torch.randn(1, TOKENS, c["d"], device=device, dtype=torch.float16, requires_grad=True)
    grad = torch.randn_like(x)
    size = 256 * 2**20
    host_out, host_in = (torch.empty(size, dtype=torch.uint8, pin_memory=True) for _ in range(2))
    dev_out, dev_in = (torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2))
    side = torch.cuda.Stream(device)

    def compute():
        block(x).backward(grad)

    def transfer():
        with torch.cuda.stream(side):
            host_out.copy_(dev_out, non_blocking=True)
            dev_in.copy_(host_in, non_blocking=True)

    def both():
        transfer()
        compute()

    alone_c, alone_t, together = time_s(compute, 1.5), time_s(transfer, 1.5), time_s(both, 1.5)
    emit(card=card, phase=tag, section="overlap", block_ms=round(alone_c * 1e3, 1), transfer_ms=round(alone_t * 1e3, 1),
         together_ms=round(together * 1e3, 1), hidden_fraction=round((alone_c + alone_t - together) / alone_t, 2))
    del block, x, grad


def child(args):
    card = args.child
    peers = ["card0", "card1"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    emit(card=card, section="device", name=props.name, visible=torch.cuda.device_count(),
         free_gb=round(torch.cuda.mem_get_info()[0] / 2**30, 2))
    ops, _ = load_node_package()
    sync = lambda name: barrier(args.barrier_dir, card, name, peers)

    sync("loaded")
    for solo in peers:
        if card == solo:
            bench_blocks(card, ops, device, f"solo_{solo}")
        sync(f"blocks_solo_{solo}")
    bench_blocks(card, ops, device, "both")
    sync("blocks_both")
    for solo in peers:
        if card == solo:
            bench_pcie(card, device, f"solo_{solo}")
        sync(f"pcie_solo_{solo}")
    bench_pcie(card, device, "both")
    sync("pcie_both")
    bench_overlap(card, ops, device, "both")
    sync("done")


def card_of_uuid(uuid):
    for node in os.listdir(KFD):
        props = dict(line.split() for line in open(os.path.join(KFD, node, "properties")))
        if props.get("unique_id") and f"GPU-{int(props['unique_id']):x}" == uuid:
            bus = int(props["location_id"]) // 256
            return CARD_BY_BUS[bus]
    raise SystemExit(f"dual_bench: tried to find KFD node of {uuid} -- expected one of {KFD}/*/properties -- actual: none")


def launcher(args):
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2:
        raise SystemExit(f"dual_bench: tried to read ROCR_VISIBLE_DEVICES -- expected two GPU UUIDs -- actual: {visible}")
    children = []
    for uuid in visible:
        env = dict(os.environ, ROCR_VISIBLE_DEVICES=uuid)
        children.append(subprocess.Popen([sys.executable, __file__, "--child", card_of_uuid(uuid),
                                          "--barrier-dir", args.barrier_dir], env=env))
    codes = [p.wait() for p in children]
    if any(codes):
        raise SystemExit(f"dual_bench: children exited {codes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--barrier-dir", required=True)
    parser.add_argument("--child", choices=["card0", "card1"])
    args = parser.parse_args()
    if args.child:
        child(args)
    else:
        launcher(args)


if __name__ == "__main__":
    main()
