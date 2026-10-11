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

| Model | Mode | Block ms (was) | Step s / 4096 tok | tok/s per card | tok/s 2 cards | M tok/day | x before |
|---|---|---|---|---|---|---|---|
| 4B | full fp16 | 262 (745) | 10.4 | 393 | 786 | 67.9 | 2.7 |
| 4B | LoRA, fp16 base | 230 (714) | 9.3 | 442 | 884 | 76.4 | 2.9 |
| 4B | LoRA, int4 base | 172 (657) | 7.2 | 572 | 1143 | 98.8 | 3.4 |
| 8B | full fp16 | 584 (1063) | 20.0 | 205 | 409 | 35.4 | 1.8 |
| 8B | LoRA, fp16 base | 371 (850) | 13.2 | 310 | 620 | 53.6 | 2.2 |
| 8B | LoRA, int4 base | 234 (713) | 8.8 | 465 | 930 | 80.3 | 2.7 |

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

## Caveats

- int4 W4A4 error (13.8% per GEMM, random data) and 4-bit dY in dgrad: training
  convergence untested. int8 is accurate but no faster than fp16.
- fp16 only: bf16 is emulated on gfx906 (see GEMM rows). Needs loss scaling.
- fp16 wgrad is slow on some shapes in the layout autograd picks (2.9 TFLOPS on
  1B `down`); the other layout wasn't tried.
- Card1 drove the desktop while measured (Luna using it); its numbers matched card0.
