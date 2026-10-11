# LLM training throughput, 2x Radeon VII (2026-10-11)

Scripts: `train_bench.py` (card0, primitives), `dual_bench.py` (both cards, whole
blocks), `attn_split.py` (attention vs the rest). Logs next to them. Wall-clock
timing between host syncs. One sequence of 4096 tokens per micro-batch.

## 1. Whole model, both cards (measured per card while both ran)

Step = layers x block fwd+bwd + tied LM head with cross-entropy (fwd+bwd).
Block: RMSNorm, GQA causal SDPA (32 q / 8 kv heads), SwiGLU; no RoPE.
LoRA rank 64 on every linear. `lora_int4` = frozen base through the W4A4
block-32 kernels (fwd + dgrad, second packing of W^T for dgrad).

| Model | Mode | Block ms | Step s / 4096 tok | tok/s per card | tok/s 2 cards | M tok/day | Fits 2x16 GB? |
|---|---|---|---|---|---|---|---|
| 4B (Qwen3-4B) | full fp16 | 745 | 27.8 | 147 | 294 | 25.4 | no: Adam needs ~64 GB (8-bit Adam + pipeline split: barely) |
| 4B | LoRA, fp16 base | 714 | 26.7 | 153 | 306 | 26.4 | yes, data parallel (8 GB base per card) |
| 4B | LoRA, int4 base (ours) | 657 | 24.6 | 166 | 332 | 28.7 | yes, data parallel (~4.5 GB) |
| 8B (Llama-3-8B) | full fp16 | 1063 | 35.4 | 116 | 232 | 20.0 | no |
| 8B | LoRA, fp16 base | 850 | 28.5 | 143 | 286 | 24.7 | pipeline split only (16 GB base) |
| 8B | LoRA, int4 base (ours) | 713 | 24.2 | 170 | 340 | 29.4 | yes, data parallel (~9 GB with W^T copy) |

- Solo vs both-at-once: within 1% for every row. The cards don't slow each other.
- 2-card column assumes a perfect split. Measured basis: PCIe exchange hides behind
  backward (section 3); LoRA gradients are ~0.2 GB/step. Pipeline-split rows lose
  a bubble of 1/(m+1) for m micro-batches.

## 1b. With our training attention kernel (same day, `attn_train_test.py`)

`custom_nodes/ComfyUI-Gfx906Int4/csrc/attention_train.hip` (fwd + bwd, causal, GQA;
design and per-kernel numbers in that repo's DESIGN.md) in place of SDPA. Block
times measured on card0; the LM head is section 1's. Both cards at once showed no
slowdown in section 1, so the 2-card column doubles.

Kernel v2 (backward with register-resident rows, DPP reduce-scatter; v1 block in brackets):

| Model | Mode | Block ms (v1; SDPA) | Step s / 4096 tok | tok/s per card | tok/s 2 cards | M tok/day | x SDPA |
|---|---|---|---|---|---|---|---|
| 4B | full fp16 | 236 (262; 745) | 9.5 | 432 | 864 | 74.6 | 2.9 |
| 4B | LoRA, fp16 base | 204 (230; 714) | 8.3 | 491 | 982 | 84.9 | 3.2 |
| 4B | LoRA, int4 base | 147 (172; 657) | 6.3 | 651 | 1303 | 112.5 | 3.9 |
| 8B | full fp16 | 560 (584; 1063) | 19.3 | 213 | 425 | 36.7 | 1.8 |
| 8B | LoRA, fp16 base | 348 (371; 850) | 12.5 | 328 | 656 | 56.7 | 2.3 |
| 8B | LoRA, int4 base | 210 (234; 713) | 8.1 | 508 | 1016 | 87.8 | 3.0 |

The projection in section 2 (3-4x) held. The "unexplained ~185 ms" was torch's
attention on the block's transposed views; the kernel takes them as strided input.

## 2. Where the time goes (card0, 4B/8B block, 4096 tokens)

| Model | Mode | Block | Without attention | Attention share |
|---|---|---|---|---|
| 4B | full fp16 | 745 ms | 170 ms | 77% |
| 4B | LoRA int4 | 657 ms | 69 ms | 89% |
| 8B | full fp16 | 1063 ms | 494 ms | 54% |
| 8B | LoRA int4 | 713 ms | 137 ms | 81% |

ROCm torch on gfx906 has no flash or memory-efficient SDPA ("No available
kernel"); only the math path runs: 1.2-2.6 TFLOPS fwd+bwd. Attention alone,
contiguous: 187 ms; with the block's transposed views: 391 ms. About 185 ms of
the 4B block is not yet accounted for by either part.

**Projection, not measured**: with a fused attention backward at ~8 TFLOPS
(the fused forward reaches 9-12 block-causal), attention would be ~60 ms a block:
4B full ~442 tok/s per card, 4B LoRA int4 ~730, 8B LoRA int4 ~540, i.e. 3-4x the
rows above. Upper bound: excludes the unexplained ~185 ms.

## 3. Primitives

| What | Stock (torch / rocBLAS) | Ours (ISA not exposed by torch) |
|---|---|---|
| ALU peak, card0 | fp32 FMA 11.6 TFLOPS, fp16 pk_fma 23.5 | dot2 f16->f32 24.2, dot4 i8 47.8, dot8 i4 96.0 TOPS |
| GEMM fwd / dgrad | fp16 10-17.8 TFLOPS, fp32 ~9.5-10.8, **bf16 4-5** | int8 b32 15.8-17.7 (0.76% err), int4 W4A4 b32 30-39 (13.8% err) |
| GEMM wgrad | fp16 2.9-18.1 (layout-dependent), **bf16 0.1-0.4** | int8 b32 3.4-11.2 (incl. transposes) |
| Attention fwd, L 8192 | SDPA math causal 2.4 TFLOPS | fused fp16 9.5, int8/u8 12.1, int4/u8 12.5 (block-causal) |
| Attention fwd+bwd | SDPA math 3.2-3.9 (16 heads MHA), 1.2-2.6 (GQA in block) | no backward kernel |
| AdamW fp32 | foreach 154 ms/1B params, fused 53 ms/1B (531 GB/s) | - |
| HBM copy | 761 GB/s | - |
| PCIe per card | 7.17 GB/s one way; 3.58 each way when both ways at once; same with both cards at once | no peer path used (card1 = display) |
| Exchange vs compute | 256 MiB each way per 8B block: 93-100% hidden | - |

## 4. Memory (card0, `mem_bench.py`, kernel v2, 4096 tokens)

Measured per layer:

| | 4B layer | 8B layer |
|---|---|---|
| fp16 weights (full / LoRA int4 incl. W^T copy) | 0.19 / 0.127 GiB | 0.41 / 0.265 GiB |
| activations kept fwd -> bwd | 0.553 GiB | 0.797 GiB |
| all layers, no checkpointing | 19.9 GiB (36 layers) | 25.5 GiB (32) |
| kept with checkpointing | 0.0195 GiB | 0.031 GiB |
| block time plain / checkpointed, full fp16 | 234 / 288 ms (+23%) | 553 / 665 ms (+20%) |
| same, LoRA int4 | 145 / 190 ms (+31%) | 206 / 280 ms (+36%) |
| LM head peak, plain / chunked CE (1024 rows) | 7.0 / 2.5 GiB | 5.9 / 3.0 GiB |
| LM head time, plain / chunked | 1.01 / 1.33 s | 1.40 / 1.94 s |

=> Activation checkpointing is mandatory at 4096 tokens on 16 GB, for LoRA too. Sections 1/1b's
tok/s assumed every layer plain; the realistic numbers below checkpoint as many layers as memory
forces (spare GiB / saved per layer stay plain).

Corrections to the commonly quoted table: 8-bit Adam with fp32 master is 10 B/param = 20 GB per
card under FSDP: does not fit 16 GB. fp16 weights without a master copy need stochastic rounding
in the update (no bf16 here). FSDP re-gathers 8 GB of weights per pass over PCIe 3.0 x8 (no
bridge): seconds per micro-batch; a pipeline split moves 21 MB of activations instead.

Per-card budget ~15 GiB usable. Estimates from the measured pieces (not run end to end):

| 4B path | GPU memory / card | Host RAM | tok/s 2 cards | Quality vs full FT |
|---|---|---|---|---|
| LoRA, int4 base, data parallel, ~9 of 36 layers plain | ~9.3 static+transient | - | ~1050 | LoRA + W4A4 error; convergence untested |
| Full FT, 2-stage pipeline, fp16 w+g on GPU, optimizer on host (fp32 master + 8-bit moments) | ~10.8 + ~6 plain layers | ~24 GB | ~700 (m = 16 micro-batches) | closest to real full FT |
| Full FT, pipeline, all on GPU: fp16 w+g, 8-bit m, Adam-mini v, stochastic rounding (HIP) | ~13.5, all checkpointed | - | ~660 | 8-bit / Adam-mini / SR approximations |
| LISA/BAdam, data parallel (replica + active block's states) | ~10.5 + ~8 plain layers | - | ~760 | between LoRA and full |
| GaLore, data parallel, layerwise update | ~10.5 + ~8 plain layers | - | ~725 minus SVD | between; needs SR for fp16 weights |

8B: full FT does not fit (fp16 w+g alone are 16 GB per pipeline half). LoRA int4 fits data-parallel
only all-checkpointed and tight (~14.5 GiB; ~375 tok/s per card); dropping the int4 W^T copy
(dgrad reading W transposed) halves the base.

Memory levers by size: checkpointing (-19 GiB, +23-36% on checkpointed layers) > optimizer
placement / precision (decides whether full FT fits) > chunked or fused cross-entropy (-4.5 GiB)
> int4 W^T copy (-50% of the int4 base) > fused SwiGLU recompute (the MLP holds most of the
0.55 GiB per layer: gate_up output alone is 160 MB at 4B).
Host-offload caveat: 64 GB RAM with the desktop in use and the 2026-09-28 kswapd Oops on record;
keep host state at the 8-bit-moments size (~24 GB), not fp32 Adam (48 GB).

## Caveats

- int4 W4A4 error (13.8% per GEMM, random data) and 4-bit dY in dgrad: training
  convergence untested. int8 is accurate but no faster than fp16.
- fp16 only: bf16 is emulated on gfx906 (see GEMM rows). Needs loss scaling.
- fp16 wgrad is slow on some shapes in the layout autograd picks (2.9 TFLOPS on
  1B `down`); the other layout wasn't tried.
- Card1 drove the desktop while measured (Luna using it); its numbers matched card0.
