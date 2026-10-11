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

## 5. Linears and low-precision training (full fine-tune track)

**fp16 wgrad layouts** (`wgrad_layouts.py`): at 4B shapes the autograd order already runs
15.3-19.0 TFLOPS (8B's wide matrices 6.4-11, other orders only partly better). The 4B block's
linears (~170 ms) are at ~15 TFLOPS of torch's fp16 GEMM ceiling of ~19: fp16 has ~10-20% left.
The lever is precision.

**Format study** (`quant_study.py`): real Qwen3-4B tensors (local checkpoint, 1024 tokens,
loss 2.0/token), X / W / dY of all 7 linears in layers 0, 1, 17, 34, 35; emulated GEMM rel. error,
mean over the 35 linears:

| Format | fprop | dgrad | wgrad | wgrad, mean of 8 SR samples |
|---|---|---|---|---|
| int8 b32 (ours) | 0.7% | 0.9% | 1.1% | |
| int4 b32 (ours) | 11.3% | 15.5% | 14.2% | |
| int4 b16 | 9.4% | 12.9% | 11.6% | |
| int4 b16 + 16x16 W | 13.0% | 16.9% | 11.6% | |
| int4 b16 + recipe (16x16 W, SR grads, RHT16 wgrad) | 13.0% | 19.7% | 12.0% | 8.4% |
| MXFP4 | 12.9% | 16.9% | 17.6% | |
| NVFP4 | 9.1% | 12.6% | 11.1% | |
| NVFP4 + 16x16 W | 10.6% | 14.2% | 11.1% | |
| NVFP4 + recipe | 10.6% | 17.1% | 16.7% | 10.8% |
| ternary W (absmean), int8 A | 59.1% | | | |

- int4 b16 ~= NVFP4 (within 0.5 points). NVFP4 + recipe trained 12B / 10T tokens within ~1-1.5%
  loss of FP8 (arXiv 2509.25149), so gfx906's v_dot8 int4 is in the right numeric class.
- 16x16 weight tiles (the paper's fix for W vs W^T inconsistency) cost int4 more than FP4
  (fprop 9.4 -> 13.0% vs 9.1 -> 10.6%): a uniform grid tolerates a coarser scale worse.
- SR: one sample is noisier, the mean of 8 drops below round-to-nearest (8.4% vs 11.6%) --
  unbiased, which accumulation over micro-batches exploits.
- Paper: most of the FP4 loss gap comes from quantizing forward tensors. For fine-tuning an
  existing model this matters more (fprop error perturbs the pretrained function from step 0).
- Ternary straight from a pretrained model is destroyed (59%); only distillation-style QAT
  (BitDistill, arXiv 2510.13998) recovers it, with a gap that grows with size.

**Proposed int4 training path on gfx906 (not built)**, cheapest-risk first:
1. fp16 fprop, int4 b16 dgrad + wgrad (SR on dY, RHT16 on wgrad inputs), last ~4 blocks fp16:
   forward exact, 2/3 of the linear flops at int4. Linears ~1.4x.
2. If (1) holds: int4 fprop too (or int8 fprop, 0.7%) -> linears ~1.6-1.7x.
   Needs: block-16 variant of the int4 GEMM (more scale math; current b32 kernel 36-39 TFLOPS incl.
   quantize), a fused transpose + RHT + SR quantize kernel for wgrad (the int8 wgrad with separate
   transposes ran 3.4-11 TFLOPS: the transposes, not the GEMM), fp32 wgrad accumulation.
   Gate: short fine-tune runs of Qwen3-4B against fp16, same data/seed, loss curves.

**Extending an existing architecture**: LLaMA Pro block expansion (arXiv 2401.02415): insert
zero-initialized (identity) copies of blocks, train only them, freeze the original. Fits this
hardware: frozen blocks run our int4 W4A4 fprop + dgrad (no wgrad, LoRA-int4 cost), new blocks
full fp16. 4B + 8 new blocks: ~430 tok/s per card data-parallel (estimate: 36 x 0.17 + 8 x 0.26 +
1.3 s per 4096 tokens), new-block optimizer state ~9.6 GB fp32 Adam on host. Depth up-scaling
(duplicate layers + full continued pretraining) and MoE upcycling (MLP -> experts) need full
training of the larger model: memory-bound here.

## 6. Training gate: int4 backward on real Qwen3-4B (`train_gate.py`, card0, 2026-10-12)

Step 1 of the path above, built (node repo f20cfc6) and run. Setup: layers 16-19 trained (host fp32
AdamW, lr 2e-5, warmup 10), backward quantized on layers 16-31, 2048-token sequences of comfy/ source,
comfy/ldm/ held out; val = 8 sequences. Same seed and data order per arm; 150 steps; no arm skipped
a step.

| arm | val 0 | 25 | 50 | 75 | 100 | 125 | 150 | vs fp16 @150 |
|---|---|---|---|---|---|---|---|---|
| fp16 | 1.0667 | 0.9639 | 0.9396 | 0.9204 | 0.9148 | 0.9102 | 0.9091 | -- |
| int4 b16 recipe, **real kernels** | 1.0667 | 0.9653 | 0.9405 | 0.9215 | 0.9160 | 0.9111 | 0.9102 | +0.0011 |
| int4 b16 recipe, emulated | 1.0667 | 0.9656 | 0.9405 | 0.9224 | 0.9157 | 0.9120 | 0.9116 | +0.0025 |
| int4 b32 RTN (inference kernel numerics) | 1.0667 | 0.9657 | 0.9407 | 0.9226 | 0.9158 | 0.9112 | 0.9102 | +0.0011 |
| int4 b16 RTN | 1.0667 | 0.9652 | 0.9404 | 0.9221 | 0.9156 | 0.9108 | 0.9100 | +0.0009 |
| int4 b16 SR only | 1.0667 | 0.9652 | 0.9400 | 0.9217 | 0.9161 | 0.9125 | 0.9124 | +0.0033 |

- Every int4 arm tracks fp16 within 0.001-0.003 (~1-3% of the 0.158 improvement); none diverges.
  The kernel arm matches the emulation (differences = SR sample noise).
- At this length and lr the ablations don't separate: RTN does as well as SR + RHT. SR's benefit
  (unbiased gradients) is a long-run effect; 150 steps can't show it. Keep the recipe for long runs.
- Wall time per step (whole step incl. host optimizer, not a speed measurement): fp16 11.9 s,
  kernel 14.6 s, emulated 16 s. The kernel arm is SLOWER end to end here although the isolated
  dgrad+wgrad bench was 1.15-1.34x faster -- per-call W^T requantization and the quantizers
  aren't amortized. Profile before claiming a speedup.

## Caveats

- int4 backward convergence: holds for 150 steps of a partial fine-tune (6.); long runs and
  int4 fprop untested. int8 is accurate but no faster than fp16.
- fp16 only: bf16 is emulated on gfx906 (see GEMM rows). Needs loss scaling.
- fp16 wgrad is slow on some shapes in the layout autograd picks (2.9 TFLOPS on
  1B `down`); the other layout wasn't tried.
- Card1 drove the desktop while measured (Luna using it); its numbers matched card0.
