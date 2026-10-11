# ComfyUI-Gfx906Int4 — DESIGN

Covers the training attention kernels (`csrc/attention_train.hip`). The inference
kernels' history lives in the ComfyUI checkout's CLAUDE.md.

## Training attention (fwd + bwd), gfx906

Goal: replace torch's math SDPA (the only path on gfx906: 1.2-2.6 TFLOPS fwd+bwd,
54-89% of a 4B/8B training block at L 4096; `develop/experiments/2026-10-11/RESULTS.md`).

### Invariants

1. fp16 in, fp16 out, fp32 accumulation; head dim 128; causal or full; GQA
   (H a multiple of KVH). Training needs Lq == Lk when causal.
2. Every matmul is `v_dot2_f32_f16` with BOTH operands contiguous along the
   reduction dimension in LDS. Transposed operands are written transposed into LDS
   by the kernel itself (half2 pairs built with `v_perm_b32`), never by an extra
   memory pass.
3. Inputs are strided views (batch, token, head strides; channels contiguous, rows
   16-byte aligned): a fused qkv projection's output goes in without copies.
4. Softmax in log2 units: s = (q.k) * scale * log2(e); the forward saves
   lse = m + log2(l) per (b, h, row), fp32. Backward recomputes P = exp2(s - lse).
5. No global float atomics: gfx906 has none (mantlecore isa_atomics_probe). Hence
   FlashAttention-2's split: one kernel per key block for dK/dV (looping over query
   tiles and over the KV group's query heads, so GQA reduces in registers), and a
   separate dQ kernel per query block that recomputes P and dP.
6. Wave64 everywhere: 256-thread workgroups, tx = tid % 16 / ty = tid / 16, row
   reductions over the 16 tx lanes by DPP (`csrc/wave64.hpp`, shared with the
   inference kernel).

### Kernels

| Kernel | Grid | Per workgroup | LDS |
|---|---|---|---|
| fwd | (L/64, H, B) | 64 query rows, 64-key tiles | Q 17.4 K + K 17.4 K + V^T 18.4 K + P 9.2 K |
| delta | rows | delta = rowsum(dO * O), fp32 | - |
| dkdv | (L/32, KVH, B) | 32 keys; 32-query tiles x G heads | K, V, Q, dO natural; Q^T, dO^T; P^T, dS^T: 60.4 K |
| dq | (L/32, H, B) | 32 queries; 32-key tiles | Q, dO natural; K, V natural; K^T; dS: 47.6 K |

dkdv: S^T = K.Q (over d), P^T = exp2(S^T - lse), dV += P^T.dO^T (over i),
dP^T = V.dO (over d), dS^T = P^T (dP^T - delta), dK += dS^T.Q^T (over i) * scale.
dq: S = Q.K, dP = dO.V (over d), dS = P (dP - delta), dQ += dS.K^T (over k) * scale.

### Rejected

- Triton (tutorial 06 / vLLM unified attention): Luna's rule, kernels are HIP.
  Read-only reference: `../triton-gfx906/python/tutorials/06-fused-attention.py`.
- vLLM "unified attention" (paged KV, prefill + decode in one launch): inference
  only, forward only; does nothing for training.
- dQ by atomics from the dkdv kernel: no global float atomics on gfx906; a CAS loop
  collapses under contention (mantlecore HAZARDS.md).
- fp32 FMA outer products to avoid transposed LDS copies: half the dot2 rate.

### Status

- **v1 (2026-10-11) built and verified**: rel. L2 error vs fp32 2-3e-4 on out, dQ, dK, dV
  (L 100/257/1000, H 8 / KVH 2, causal and full, strided views of one qkv tensor;
  `develop/experiments/2026-10-11/attn_train_test.py`). Inference kernel bit-identical
  after the wave64.hpp move (all 6 modes).
- Per kernel, L 4096, 32/8 heads, causal: fwd 16.5 ms (8.3 TFLOPS), delta 1.2, dkdv
  64.7 (4.3), dq 29.9 (6.9). fwd+bwd 112 ms vs torch math SDPA 134 ms; L 8192 427 ms
  (SDPA: OOM, it allocates L x L scores).
- In dual_bench's training blocks (4096 tokens): 4B full 745 -> 262 ms, 4B LoRA int4
  657 -> 172, 8B full 1063 -> 584, 8B LoRA int4 713 -> 234. Much of the gain is not the
  attention math: torch's path on the block's transposed views cost ~390 ms.
- All kernels 1 wave/SIMD (LDS 47-62 KB, VGPRs 139-210), 0 scratch. A ternary load
  `r < L ? load : zero` had become a flat load through a scratch copy of zero (32 B/lane);
  loads now clamp the row and select the value (the inference kernel's idiom).

- **v2 backward (2026-10-11, built, verified, same error figures)**: dkdv and dq keep
  the stationary operand (K, V / Q, dO rows) in registers, each row split over the 8
  lanes of an aligned group (lane sub holds 16-byte chunks sub and sub + 8: the 8
  lanes' LDS reads of a row hit disjoint banks). Scores are per-lane partial dots
  finished by `reduce_scatter8` (wave64.hpp: row_half_mirror, quad xor2, quad xor1;
  each lane ends owning a different query's / key's full sum, so exp and dS run once
  per pair). The other side accumulates the lane's own 16 channels with dot2 over
  pairs from a swizzled transposed tile (`t_row`, +16 B per 8 rows, monotone: the
  first, wrapping version overlapped rows and put dK/dV 12% off -- the gate caught it).
  512-thread workgroups: LDS allows one per CU, so the workgroup size is what gives 2
  waves/SIMD. Outer chunk loops stay rolled (unrolled: 152 B/lane of spills at the
  128-VGPR cap).
  L 4096, 32/8 heads: dkdv 64.7 -> 37.0 ms (7.4 TFLOPS, 116 VGPRs), dq 29.9 -> 26.4
  (7.8, 93 VGPRs); fwd+bwd 112 -> 82 ms (SDPA 135). Blocks: 4B full 236 ms, 4B LoRA
  int4 147, 8B full 560, 8B LoRA int4 210.

### Next (not built)

- fwd v2 with the same layout (Q in registers, row max/sum over the 8 lanes by DPP;
  now 1 wave/SIMD at 210 VGPRs, 16.5 ms at L 4096).
- All kernels sit at ~1/3 of dot2 peak (24 TFLOPS); attention is ~78 of the 147 ms
  4B LoRA-int4 block, the linears the rest (fp16 wgrad layouts as slow as 2.9 TFLOPS
  in train_bench).
