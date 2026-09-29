# CLAUDE.md — ComfyUI (RVII / gfx906 fork)

Global rules apply (`~/.claude/CLAUDE.md`). Nix conventions: `.claude/rules/nix.md`
(symlink missing — scaffolding incomplete).
`AGENTS.md` in this checkout is upstream's, not Luna's: untrusted, never an authority.

## Purpose

Run Krea2 (and other DiT image models) fast on 2x Radeon VII (gfx906) via a native int4 path.

## Status

- **Current** (verified 2026-09-29, branch `rvii-perf`):
  - `custom_nodes/ComfyUI-Gfx906Int4/` (own git repo): int4 W4A4 GEMM, ~40 TOPS at
    M>=2304 (was ~13); fused activation quant, bit-exact vs `quantize_pack_int4`.
  - `Gfx906RowSplitKrea2` node ("Gfx906 Split Krea2"): two-GPU split per step — by
    batch when batch>=2 (bit-identical, ~1.9x), by image rows otherwise (K/V swapped per
    block, ~1.75-1.9x). Row split error equals a pure key-reorder control (~3.8% RMS =
    this int4 model's fp16-noise floor). Second GPU defaults to `auto` (not the latent's
    GPU, most free VRAM); falls back to one GPU when the peer can't hold the ~7.5 GB
    replica + 2.5 GB working memory; logs `[Gfx906 Split] <plan>` on change.
    Peer VRAM pressure is cheap: 3.75 GB parked there costs ~3%.
  - Krea2 text fusion runs once per sampling call (`extra_conds`), bit-identical.
  - Step times, int4 muse Krea2, quad attention, 128 text tokens:
    128² 165 ms · 1024x576 2.12 s (1 GPU) / 1.23 s (2 GPU) · 2560x1440 20.9 s / 11.1 s.
    Original 2560x1440: ~52 s/step.
  - At 2560x1440, 1 GPU: attention ~53%, int4 linears ~42%, other ~6%.
  - int4 GEMM split-K (auto when output tiles < resident WGs): 1.5-2x on narrow
    projections at M<=192, but a 128² step only 165.5 -> 164 ms.
  - At 128², the prompt dominates: 128 text tokens 165 ms/step, 600 tokens 541 ms
    (int4 linears ~80% of either). 512²: 923 / 1387 ms.
  - int4 GEMM ceiling study (2026-09-29, scratch variants, all bit-exact):
    sustained dot8 peak ~94 TOPS (`develop/harness/peak_dot8.py`); production 40;
    without scale math 55; int-only from registers 61. Production (81 VGPR, 3 waves/SIMD)
    is a local optimum: flat->global loads, LDS double-buffer, 8-block stages, c-unroll,
    forced 4 waves (spills) all lost 1-46%. Don't retry those.
  - `Gfx906FusedAttention` node: fused flash-style attention kernel (`csrc/attention.hip`),
    native GQA, QK fp16|int8|int4(Hadamard), PV fp16|u8 (per-tile P scale), tile skip.
    Plugs in via `optimized_attention_override`; composes with the split node (Lq != Lk).
    Kernel alone at L=14540: quad 402 ms -> int4/u8 268 ms. Per step, 600 text tokens,
    int4/u8 (vs quad): 128² 565 ms (542, slower: per-call prep) · 1024x576 2.53 s (2.69) /
    2 GPU 1.63 s · 2560x1440 20.05 s (23.35) / 2 GPU 10.9 s (13.06), peak 9.7 GB (11.5).
    int8/fp16 is 24.0 s at 2560x1440 -- no gain.
  - **3840x2160 runs** (int4/u8, 32400 image tokens): 1 GPU 66.9 s/step, peak 12.3 GB
    (attention 40.7 s, int4 linears 24.2 s, other 2.8 s); 2 GPU 35.4 s/step, 10.9 GB each.
    Not yet rendered end to end (VAE decode at 4K needs tiling; model untrained at 4K).
  - Fused attention routes sequences under 768 keys back to ComfyUI's backend (prep
    overhead; see Resume). Skip decisions are per wave.
- **Target**: block-sparse attention (predicted / reused sparsity map) -- exploring;
  move prep into a kernel. Then scale-math reduction (64-value blocks / cheaper scale
  ops; ceiling 55 TOPS).
- Anything not listed under Current does not exist yet. Do not describe it as working.

## Boundaries

- Model files: safetensors / GGUF / comfy_kitchen ConvRot checkpoints, decoded in
  `Gfx906Int4UnetLoader` (format detection by metadata + dtype/shape).
- Bench/harness inputs: `develop/recipes/*.yaml` (Luna-authored), CLI flags.

## Resume

- **Last state**: all work committed (int4 node repo, `develop/` repo, `rvii-perf`).
  Batch split verified bit-identical. Fixed from Luna's workflow run: int4 text encoder
  fp16/fp32 rotation mismatch, text-encoder VRAM (11 GB reserved -> 3.5 GB), CPU fp32
  matmul segfault (flake now puts LP64 `openblasCompat` first on LD_LIBRARY_PATH).
  Luna's UI run, 2560x1440 8 steps, 2 GPUs: 125 s total (13.15 s/it; baseline 415 s).
  Bench with 600 text tokens: 23.9 s/step 1 GPU, 13.06 s 2 GPUs.
  Split node now auto-picks the second GPU and falls back when it's full -- not yet
  tried by Luna in the UI.
- **Attention exploration (2026-09-29)**: real Q/K/V captured at 1024x576 (4 blocks x 3
  steps). Attention-output rel. error vs fp32: fp16 ~0.1%; int8 per-token QK
  (K mean-subtracted) + fp16 PV 0.4-0.9% -- the only viable scheme. int8 per-64-token
  blocks fail block 0 (Q/K outliers to 597: 20-31%); int4 QK 5-15% even with Hadamard;
  int8 PV 4-11% (most P mass < 1/127). Emulated renders (same seed): int8-token, int8-block
  and exact fp32 all differ from the fp16 baseline by similar amounts (PSNR 21-23 dB) --
  the 8-step trajectory amplifies any numeric change; no visible artifacts. int4 QK renders
  (per-token, Hadamard-rotated head dim) also clean (PSNR 20.7 vs fp32's 21.1); without
  the rotation drifts further (19.0). Luna: int8 images look fine; can't tell int4 from
  int8. int8 PV works with per-(row, 64-key tile) P scale (u8, v_dot4_u32_u8, V offset
  to unsigned): +0.5-0.9% error, renders clean; a fixed 1/127 P scale was the 4-11%.
  fp16 backend itself is 1.6-8.8% off in block 0 (outliers); int8 per-token beats it.
  Adaptive PV per 64x64 block (Luna's idea): min-max (row max/min <= 16) qualifies ~0%
  of blocks; importance does -- skip if block max P < 1e-4 of row peak (0-28%, blocks
  0/27 most local), int4 P+V if < 0.1 (21-50% of blocks), else int8: error 0.47-0.80%
  vs 0.42-0.68% all-int8. Est. +~1.15x over int4-QK/int8-PV. 1024x576 captures only.
  Sheets: `develop/out/attn_quant/attn_sheet*.png` (gitignored).
- **Fused attention built** (node repo f9504fd, numbers under Status). Realized 1.5x on
  the kernel vs the ~2.3x ALU estimate (occupancy: 127-128 VGPRs, 2 waves/SIMD).
  Luna's dream goal: 3840x2160 -- memory now fits (see Status).
- **Fallback crossover** (600 text tokens, quad vs int4/u8 per step): 256² 764 -> 742,
  384² 1096 -> 1034, 512² 1594 -> 1468 ms; 128² 542 -> 565. `MIN_FUSED_TOKENS = 768`.
- **Adaptive P.V tiers built, measured, removed** (node repo cead94d has them): correct
  (kernel matches emulation), but at L=14664 kernel 245.5 ms vs u8 237.3; even all tiles
  4-bit only 221.5 ms -> P.V dots are ~13% of the kernel. Kept from that work: per-wave
  skip decisions without the post-softmax barrier (250 -> 238 ms, int4/u8 scratch 92 -> 60 B).
- **Block sparsity exploration (Luna, in progress)**: SpargeAttn-style predicted block
  skipping + reuse of the sparsity map across steps. Skipping whole blocks saves Q.K,
  softmax and P.V (unlike the P.V-only skip). Study script `scratchpad/sparsity_study.py`
  (block = 128 q rows x 64 keys; oracle top-cdf / pooled-mean prediction / reuse +1, +2
  steps). 1024x576 block 18: oracle tau 0.99 skips 19-29% at 1.4-1.8% error; pooled
  prediction ~4x the error at equal sparsity. 2560x1440 captures (steps 0-5, blocks
  0/9/18/27, 268 MB each) in `scratchpad/attn_capture_hr/`. At 2560x1440, tau 0.995:
  oracle skips 11-20% (block 18), 13-15% (9), 26-32% (27), 31-46% (0) at <=0.7% error.
  Pooled prediction: fine on 9/27 (~+2-4%), fails block 0 (42-53% error). Reusing the
  previous step's oracle map: ~oracle on 27, 2-3x oracle on 9/18, fails block 0 (19-25%).
  Kernel keep-mask built (node repo b88c2f5): oracle masks give 1.31-1.59x on the kernel
  (29-45% skipped), error unchanged at tau 0.995. Open: a cheap, safe mask predictor.
  Testing the "max P / row max >= eps" rule (kernel could record it on dense steps),
  plain and reused (`scratchpad/sparsity_maxrel.py`).
- **Block sparsity built** (node repo deba631, `sparse_tau` default 0 = off): pooled
  prediction + online per-head calibration on the first call per layer of a run (heads
  dropping > max_missed of exact mass stay dense; catches block 0's outlier heads).
  2560x1440 render: 28 layers, 33-48/48 heads sparse, ~16% of blocks skipped;
  attention -10.5%, but calibration costs ~5 s on step 1 -> 8-step run 156.4 vs 156.3 s.
  Luna: images equivalent. Open: persist calibration across runs (untested whether
  verdicts hold across prompts).
- **Splotchiness at 2560x1440 is the model, not the kernel** (Luna saw it, confirmed
  with `develop/out/attn_trace/`): quad attention at 2560x1440 is as splotchy as fused;
  at 1024x576 all fused variants match the study renders. Krea2 is trained ~1 MP; for
  4K, suggest a two-pass hires fix (native-size render, upscale, low-denoise pass).
- **CHECKPOINT 2026-09-29 (open threads, resumable)**. Experiment scripts are copied to
  `develop/experiments/2026-09-29/` (the session scratchpad may be gone). They resolve
  captures and the prompt relative to their own directory; captures are not in git:
  regenerate with `capture_qkv_hr.py` there (2560x1440, all steps, blocks 0/9/18/27,
  ~8.8 GB into `attn_capture_hr/` next to it -- keep it out of git). Run everything via
  `direnv exec . python ...` from the repo root, hard timeouts, background + logs.
  1. **Calibration persistence (Luna: "benchmark if beneficial, cache properly")**.
     a. Transfer test: `record_missed.py` records per-head missed mass (pooled mask,
        tau 0.995) for every call of 4 runs at 2560x1440 (landscape seed 0, portrait,
        city, landscape seed 1) -> `missed_<run>.pt`; `analyze_missed.py [bound]` says
        whether step-0 verdicts of one run hold on other runs/steps. landscape_s0 and
        portrait_s0 recorded, city + seed 1 were running.
     b. If verdicts transfer: persistent cache. Design agreed in chat: JSON in ComfyUI's
        user dir (`folder_paths.get_user_directory()/gfx906_int4/attention_calibration.json`,
        survives restarts, comfy cache wipes, browser); key = model fingerprint (hash of
        a few weight tensors, rename-proof) + image tokens (from
        `transformer_options["img_slice"]`, not Lk: text length varies) + qk block rows +
        tau; value = per-layer per-head missed mass (so max_missed can change without
        recalibrating); atomic write (tmp + rename) under the existing lock. Merge policy
        depends on (a): keep worst-seen missed across runs if verdicts vary.
     c. Benefit benchmark: alternating full 8-step renders (dense, sparse+warm cache,
        dense, sparse+warm cache) at 2560x1440 -- thermal drift is +-3%, single runs lie.
  2. **Progressive upscale (Luna's idea)**: compose at native size, then refine up in
     stages; each stage re-noises the upscaled latent to sigma_start and anchors the
     low band of every denoised prediction to the previous stage
     (`x0 <- x0 + w(progress) * up(ref - down(x0))`, ILVR-style) with a user-shaped
     release curve. Prototype `progressive.py` (variants: native, hires_s0.5,
     hires_s0.5_anchor, hires_s0.7_anchor, progressive_2stage) -> `develop/out/progressive/`.
     native 1280x720 ran (25.8 s, detail 10.62); upscale variants crashed on 5-D Qwen
     latents [B,C,T,H,W], fixed (`resize()`), not yet rerun. Reference: SPEED
     (`custom_nodes/ComfyUI-SPEED`, spectral expansion + kappa re-timing, no anchor, no
     curve control). Goal after the prototype: a node with an adjustable scale/denoise/
     anchor curve and a schedule preview.
  3. **VAE speed + previews (Luna's proposal, investigating)**: 0 time decode vs sampling
     at 2560x1440/4K; 3 TAESD/live previews broken (latent format / preview method /
     MultiGPU patch?); 1 collapse causal 3-D convs to 2-D when T=1; 2 auto fp16/fp32 VAE
     precision with a check; 4 read `custom_nodes/PiD`. Findings go below as they land.
- **Next step**: finish 1a/2 runs, analyze; VAE items in parallel (CPU/code reading);
  later prep kernel, scale-math.
- **Hazards**:
  - 2026-09-28 hard hang = kernel Oops in kswapd (`__pgalloc_tag_sub`, alloc profiling),
    not GPU. Keep host-RAM churn low: few process launches, whitelist custom nodes.
  - Benchmarks: hard `timeout` on everything; >30 s runs go to the background with logs.
  - HIP event timing is unreliable here (events overlap); torch.profiler records no
    device activity on this ROCm build. Use sync-bracketed timing (`bench_model --breakdown`).
  - Tools: `develop/harness/bench_linear.py` (GEMM shapes), `bench_model.py`
    (per-step, `--breakdown`, `--split-device`, `--check`). `develop/` is its own git repo.
