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
        whether step-0 verdicts of one run hold on other runs/steps. DONE, results:
        verdicts drift over STEPS more than across prompts. Current design (step 0 of the
        same run) leaks: 3-9% of allowed head-steps exceed 2%, worst 40-100%. Worst over
        all steps of 3 other runs (leave-one-out): 899-970/1344 heads allowed,
        violations 0.4-2.8%, mean missed 0.6-0.7%, one 100% outlier (landscape step 1,
        block 26 head 19). Block 0 heads 0-3, 16-19, 22 miss ~87-100% in every run.
        => persistent worst-seen cache is SAFER than the current per-run calibration.
        Proposed: learn over the first 3 runs per key (measure all layers every step,
        ~2.2 s/step at 2560x1440, output unchanged), then sparse with no calibration;
        ~1 s/step gain (~5%) at 2560x1440, break-even ~7 runs. Luna: build it AFTER the
        VAE/preview work. Priority (Luna, corrected): progressive upscale > VAE > cache.
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
     Ran (1 GPU, int4/u8 attention; direct 8 steps at 2560x1440 ~156 s, detail 15.6):
     hires_s0.5 113 s / 11.2, hires_s0.5_anchor 114 s / 10.8, hires_s0.7_anchor 135 s /
     13.1, progressive_2stage (1792x1008 s0.55x3 -> 2560x1440 s0.45x3) 119 s / 13.2.
     Luna's verdict: splotchiness gone, but mirror images / afterimages ("kaleidoscopic"),
     worse than SPEED. Likely cause: RoPE extrapolation (positions beyond the ~1 MP
     training grid; Krea2 image ids = integer (row, col) token grid, theta 1000) plus
     off-manifold bicubic latent upscale. Plan (Luna: "go ahead", this order):
     a. RoPE position interpolation via Krea2's `post_input` patch (scale img_ids
        rows/cols by sqrt(4096/tokens); DyPE-style `s**(sigma**2)`): `rope_test.py`
        (direct 8-step 2560x1440 and hires_s0.5_anchor, each pi/dype). DONE (s / detail):
        direct_pi 146 / 10.6, direct_dype 154 / 11.3, hires_anchor_pi 110 / 10.1,
        hires_anchor_dype 111 / 10.6. Luna: direct_pi no splotches but fine detail
        fragmented (reeds lose coherence: PI squeezes the high-freq neighbour bands too);
        direct_dype = best so far, some splotchiness back (true positions late), far less
        than unpatched; hires_anchor_pi splotchy + noisy, worst (bicubic latent upscale +
        anchor pinning to it). Next candidates: tune the DyPE exponent / floor; NTK/YaRN
        (squeeze only low-freq RoPE bands -- needs a patch of Krea2's rope frequencies,
        not img_ids). `yarn_test.py` (object-patches `pe_embedder`; exact vs stock at s=1):
        yarn_a1_b8 161 s / 11.1 -- Luna: immediately significantly better, noise gone,
        some splotchiness left. yarn_a1_b8_temp 164 / 11.3, yarn_a0.5_b4 165 / 12.1,
        dype_p4 166 / 12.4. Luna: all some splotchiness; dype_p4 best but some
        fragmentation, yarn_a0.5_b4 next. Queued: YaRN band split + DyPE-p4 release
        (`yarn_dype_*`, sigma handed to the embedder by a post_input patch).
     b. tiled_test.py (6 tiles of 1024 px, overlap 256, 3 tiles per call -- 6 OOMed):
        latent-upscale s0.5 anchor 130 s / 11.0 (Luna: the smear = bicubic latent
        upscale), s0.35 9.7, pixel-upscale s0.5 anchor 105 s refine / 10.4 (Luna: best so
        far, a bit too noisy), noanchor 11.7. Luna's 6/2 split (`cN`: composition 6 of 8
        steps -> clean estimate -> pixel upscale -> re-noise to sigma_6 -> 2 tiled steps):
        c6_pixel_anchor 24 + 54 s / 7.6, c6_pixel_noanchor 3.6 (likely soft).
        Queued: c6_esrgan_noanchor (RealESRGAN_x4plus), c6n_pixel_noanchor (Luna: keep
        the pre-pass noise -- split x_6 into clean (pixel-upscaled) + noise (white 2x
        upsample, block sum/2 = parent; checked std 1.000, corr 0)).
        Results: sigma_6 of the 8-step schedule is 0.513 (shift), so c6 = heavy re-noise
        + only 2 steps. Luna: c6_pixel_noanchor best texture, splotchiness gone, but ghost
        duplicate figures -- in EVERY variant after s0.5_anchor_pixel (incl. c6 anchor).
        Cause: tiles take the whole prompt ("lone traveller ... lower left third") +
        unfinished composition (c6) + anchor released early. c6n (kept noise, eps std
        0.969) 4.0, c6_esrgan 5.7.
        **Winner so far (Luna: "just phenomenal")**: `f8_pixel_s0.5_2step_hold` = full
        8-step 1280x720 composition -> pixel (Lanczos) upscale -> re-noise 0.5 -> 2 tiled
        steps, anchor 1.0 on step 1 only: 27 + 51 s = 79 s total (direct 2560x1440 ~150 s),
        detail 6.9. f8_esrgan 27 + 59 s / 8.9 -- Luna: worse than Lanczos.
        **Final pick: `f8_pixel_s0.5_2step_hold_detailprompt`** (same + subject-free tile
        prompt): Luna: removes the remaining duplicate, slightly better. 27 + 51 s.
        => node design: composition steps, scale, re-noise sigma, detail steps, anchor
        strength + hold, tile size/overlap, upscale method (Lanczos default; model
        upscaler / PiD optional), separate detail prompt (manual; later VLM-derived).
        yarn_dype hybrids: a0.5_b4_p4 164 s / 14.3, a1_b8_p4 166 s / 13.8 (not yet judged).
     e'. PiD WORKS (pid_test.py, two processes: `compose` saves the latent, `decode bf16`
        runs PiD alone -- Krea2 int4 is not freed by PiD's unload). 1024x576 -> 4096x2304:
        compose 17 s + PiD 146 s (incl. loads), detail 16.5 (Lanczos 4x: 1.4). Needs:
        bf16 forced (comfy lists bf16|fp32; fp32 OOMs, fp16 = all-NaN), Gemma TE on cuda:1
        via MultiGPU's text-encoder device (4.9 GB; on cuda:0 it OOMed). cuda:0 holds
        only PiD (2.5 GB). Luna: changes the image a lot but looks good; slightly prefers
        the f8 detail-prompt tiled result, but PiD "unmatched for 4K+".
     f. Continuous resolution growth (Luna: resolution as another dimension of the same
        image, like video frames): `grow_test.py` -- every step pixel-upscales the last
        clean estimate ~1.12-1.2x, re-noises to that step's sigma (0.5 -> 0), one tiled
        detail-prompt step, anchor fading (1 - (k-1)/n)^2. Variants n4_2560, n6_2560,
        n6_3840. tiled_sampler now takes non-square / sub-tile images.
        n4_2560 26 + 141 s / 5.9 (Luna: "kinda good, more smaller steps even if slower is
        definitely worth it"), n6_2560 26 + 200 s / 5.4. Much of the growth time was the
        VAE: MIOpen builds kernels per new conv shape (~8 s per new VAE size; FIND_MODE=FAST
        ~0 but slower steady; GEMM-only 0.9 s; MIOpen off 0 s and 1.2 s steady vs 0.39,
        vae_shape_test.py). Luna: MIOpen OFF globally -> `--disable-miopen` (7086d351,
        run.nu also MIOPEN_FIND_MODE=FAST backstop). 2 GPUs (--split, 4 tiles per call):
        ~2 s per tile per step.
        **4K crash fixed**: n6_3840 segfaulted entering the 3840x2160 step (1 and 2 GPUs,
        not concurrency, not offload). gdb on the core (flake now has gdb, 7abfcd4b):
        rocr::core::Runtime::AsyncEventsLoop null deref (ROCm 7.2 HSA runtime). Passes with
        AMD_SERIALIZE_KERNEL=3 and with a host sync per tile batch (5 batches of 3 queued
        unsynced -> crash; 2-3 batches never). tiled_test.SYNC_PER_BATCH default on; cost
        unmeasurable (58.2 vs 59.2 s serialized). Any tiled node must keep this.
        Verified: `grow_n6_3840_split.png` 3840x2160, 2 GPUs: compose 16 s + growth 115 s
        (steps: denoise 3.2 / 11.4 / 11.7 / 16.0 / 16.4 / 34.2 s, upscale 1.5-7.5 s),
        detail 4.2. **Luna: "that is clean... thus far the best, even above the PiD".**
        => the growth method is the upscale path.
     g. **Node built**: `custom_nodes/ComfyUI-ResolutionGrowth` (own git repo, afef4a7;
        standalone -- Luna: separate extension if not GPU-tied; whitelisted in run.nu).
        "Resolution Growth": model/pos/neg/vae/composition latent -> LATENT + schedule plot;
        width/height, growth_steps, sigma_start, size_shape, anchor, anchor_shape, cfg, seed,
        tile size/overlap/batch, upscale_method; optional detail_positive, upscale_model.
        Noise scaling left to comfy (CFGGuider.sample at sigmas [s, 0]) -> model-type
        agnostic. develop/experiments/2026-09-29/node_test.py: 3840x2160, 2 GPUs,
        16 + 104 s, 32.7 dB vs grow_test (direct x0 vs 1-step er_sde). Not yet tried by
        Luna in the UI. Later (int4 extension): shrink the VAE's ~5 GB for the VAE steps.
     h. Scaling study (`scale_test.py`, per-step report + `out/progressive/scale/`):
        1280x720 -> 7680x4320 in 10 steps (~1.2x each, sigma 0.5 -> 0.07), 2 GPUs, growth
        ~510 s (last step 167 s). Drift vs composition flat 24.1-24.6 dB over 1.3-33 MP;
        change per step 24 -> 38 dB; noise 1.7 -> 0.4; detail at 720p scale 11.9 -> ~10.5.
        Luna: "genuinely phenomenal", noise drop is fine (texture + fine detail kept).
        Luna's next ideas (proposed order, awaiting go): A/B additive "+256 px per side"
        steps (sigma tied to size, not step index) vs multiplicative at 4K; oversample
        (grow past target, Lanczos down = supersampling); Cycles-style open-ended mode
        (grow until a Stop button -- needs server route + JS, ComfyUI Interrupt drops
        outputs -- then settle at the final size with falling sigma).
        Luna refined Cycles mode: set target, grow to it, then keep denoising (more steps
        or growing past it) until Stop or noise threshold. Threshold on FULL-resolution
        Immerkaer noise (Luna; the 720p-shrunk value counts texture and rises), default < 1.
        Multiplicative (geometric) stays as an option -- proven. Order: A/B, oversample,
        Cycles mode.
        A/B at 4K done (node d4f196f adds growth_mode linear; sigma/anchor follow log-area
        fraction): linear 217 s vs geometric 106 s, noise 0.47 vs 0.57, detail_720 9.6 vs
        10.0. **Luna: linear better -- "massively slower, but clean AF"**; converges faster
        on noise => geometric for fixed-target, linear when a noise target is set.
        Oversample (node `oversample` input) test: geometric 8 steps to 5760x3240 -> Lanczos
        to 3840x2160 (scale_test `@WxH`), running.
        Luna's constant-output idea (queued as scale_test `linear:5760x3240:256=3840x2160`):
        output size fixed (4K); internal grows linearly from native (upscaled to output
        early, supersampled once past it); measure noise AT OUTPUT each step, stop below
        threshold. Test runs to the cap without stopping to see where a threshold should
        sit (upscaled early steps are smooth -> noise low but soft; may need a detail floor).
        Oversample result: 300 s, detail/px 4.79 (plain 4.18), noise 0.64 (texture).
        Constant-output result (linear to 5760 wide, measured at 3840x2160): noise AT
        OUTPUT rises with texture (0.18 -> 0.8), unusable as stop signal; detail at output
        climbs to 5.2 by ~3500 internal, flat after; change per step 24 -> 38 dB =
        convergence signal (Cycles-like). Luna: supersampled steps add coherence to
        micro-detail (brush strokes, grass blades); far past output (~5000+ internal) the
        details "start to resemble AI upscaling artifacts" -- idea: add noise between
        steps (sigma floor), esp. for targets far above native. **Luna: pause, commit, make
        functional robust nodes.**
     i. **Nodes (ComfyUI-ResolutionGrowth c47ce4e, README.md)**: "Resolution Growth" (fixed
        target; geometric|linear, oversample, sigma_end floor) and "Resolution Growth
        (Progressive)" (fixed output; linear internal growth up to max_oversample; stops on
        convergence -- change >= converge_db (35) once internal >= output -- the cap, or the
        "Stop (keep result)" button: POST /resolution_growth/stop, web/resolution_growth.js,
        matched to UNIQUE_ID incl. subgraph "a:b" ids). Inputs validated at the node boundary
        (checked with a stub VAE: every bad input names expected vs actual). Headless tests
        (node_test.py): fixed 2560x1440 69 s; progressive + simulated Stop after 4/10 steps,
        request cleared; progressive to convergence: 2560x1440 output stopped at step 9/10
        (internal 3584x2016, 35.5 dB), 217 s, deterministic (steps match the Stop run).
     j. Speed exploration of linear growth (Luna's list: GPU-resident upscale, one-GPU vs
        overlap, tiny VAE, latent resize; + reuse of previous results):
        - upscale_bench.py (VAEs only): host round trip + 8-bit PIL Lanczos vs all-GPU float
          Lanczos: SAME time (4.75 / 11.8 s at 2.6K / 4K) and 53 dB apart -- the full VAE's
          decode+encode (MIOpen off) is the cost, not transfers. Tiny VAE (lighttaew2_1) both
          ways 2.4x faster, 36-39 dB; tiny encode only 41-44 dB, 1.4x; latent resize 0 s, 27-29 dB.
        - Linear 4K growth, per path (scale_test +path): full VAE 219 s = denoise ~80%,
          upscale ~47 s (9.2 s at 4K), per-call sampling overhead ~0.05 s (single sampling
          call for all steps: nothing to gain). tiny_vae 212 s, detail_720 9.10 vs 9.43, drift
          24.0 vs 24.4. Latent: noise stuck ~1.7, never converged -- Luna: scrap (removed).
        - Async overlap already happens: the VAE (cuda:1) tail overlaps the denoise start; a
          sync for exact timing cost 12 s (removed).
        - Denoise cost follows tile count (~2 s/tile on 2 GPUs), linear growth climbs in
          stairs. Settled-tile skipping (skip tiles whose region changed < x of max last step,
          x = 0.3 / 0.5): never skipped a tile -- every tile changes > 0.5x max. Removed.
        Kept in the node: GPU-resident upscale with float torch Lanczos, decode shared between
        Progressive's check and the next step, upscale_path vae|tiny_vae (+ tiny_vae input).
        Remaining lever is the denoise itself (tiles x per-tile cost): e.g. fewer/larger
        tiles, or the int4/attention work. Committed (node c8a1577, develop e32bd85);
        re-tested: fixed 2560x1440 66.9 s (was 69.4), progressive Stop after 4/10 steps,
        request cleared. NOT yet tried in the UI (the JS button and route only exist with a
        running server). Later: int4 treatment (bf16 is fp32 math
        on gfx906). Earlier notes: PiD node registered via nodes.load_custom_node (the
        experiments' whitelist excludes it): compose 1024x576 -> PiDDecode qwenimage 2kto4k -> 4096x2304
        vs VAE + Lanczos 4x. PiD is per latent space (VAE), not per model. Files moved by
        Luna to /media/luna/models/{diffusion_models,text_encoders}/nvidia_pid/.
        Chain (detached): tiled_test2 -> tiled_test3 -> yarn_test2 -> pid_test,
        logs next to the scripts, rc lines in run_chain.log.
     b. Tiled detail pass at native tile size (MultiDiffusion blend of overlapping
        ~1 MP tiles per step, low band anchored; tiles split across both GPUs).
        Luna's refinement: caption the composition (or per tile) with a VLM and prune
        prompt parts not visible, or a dedicated manual/derived detail prompt.
     c. Pixel-space upscale option (decode -> Lanczos/ESRGAN -> encode) -- Luna wants it
        as a node option.
     d. Consistent noise across scales (Luna's idea ~ DemoFusion skip residual).
     e. PiD 4x decoder: Luna approved (transformers fine -- already in venv 5.17,
        huggingface_hub present, accelerate only for the optional captioner). Files
        downloading to scratchpad `pid_models/{diffusion_models,text_encoders}/nvidia_pid/`
        (pid_qwenimage_1024_to_4096_4step_bf16 2.7 GB, gemma_2_2b_it_elm_bf16 5.2 GB);
        Luna moves them to /media/luna/models. Later: int4 treatment like the UNet. Reference: SPEED
     (`custom_nodes/ComfyUI-SPEED`, spectral expansion + kappa re-timing, no anchor, no
     curve control). Goal after the prototype: a node with an adjustable scale/denoise/
     anchor curve and a schedule preview.
  3. **VAE speed + previews (Luna's proposal, investigating)**: 0 time decode vs sampling
     at 2560x1440/4K; 3 TAESD/live previews broken (latent format / preview method /
     MultiGPU patch?); 1 collapse causal 3-D convs to 2-D when T=1; 2 auto fp16/fp32 VAE
     precision with a check; 4 read `custom_nodes/PiD`. Findings:
     - Previews: nothing broken -- `run.nu` had no `--preview-method` (server default
       none) and the frontend settings have no preview entry. Added
       `--preview-method=taesd` to run.nu (Luna approved; uncommitted until checked).
       Krea2 latent format = `Wan21` (16 ch, 5-D), TAE `lighttaew2_1`; downloaded to
       scratchpad (45,274,004 B, sha256 8666e769...1a19) -- Luna moves it to
       /media/luna/models/VAE-approx/ herself (outside write scope). Without it,
       latent2rgb fallback.
     - VAE decode, tiled 512/64, steady state best of 3 (`vae_bench.py`, `vae_bench2.log`):
       fp32 conv3d / **fp32 conv2d** / fp16 conv3d / fp16 conv2d:
       1024x576 0.61 / **0.25** / 0.56 / 0.39 s; 2560x1440 3.93 / **1.65** / 3.49 / 2.47 s;
       3840x2160 9.24 / **4.04** / 7.87 / 5.64 s. conv2d = run T=1 3-D convs as 2-D
       (CausalConv3d already slices the kernel for T=1 via `autopad="causal_zero"`, but
       MIOpen still runs 3-D): BIT-EXACT vs fp32 conv3d. fp16 slower on gfx906 and max diff
       0.0035-0.0087, once 0.52 (not reproducible -- nondeterministic MIOpen fp16 pick).
       => ship fp32 + conv2d; no auto fp16 (item 2 dropped). First decode at a new size
       adds MIOpen tuning (5.51 s vs 0.61 s at 1024x576 the very first time).
     - PiD: NVIDIA PixelDiT pixel-space decoder/upscaler, native in comfy
       (`comfy/ldm/pixeldit`); `qwenimage` backbone (16 ch / 8x = Krea2's latent space)
       `2kto4k`: 1024-class latent -> 4x pixels (1024x576 -> 4096x2304) in ~4 steps.
       Needs `pid_qwenimage_1024_to_4096_4step_bf16.safetensors` + Gemma TE (downloads,
       Luna's call; files go to /media/luna/models by her) + transformers>=5 etc. via
       flake. bf16-only upstream; gfx906 needs fp16 test. Could replace both the 4K
       VAE decode and part of the progressive-upscale idea.
     - Luna: 2 GPUs -- the split node already splits batch>=2 across both.
     - progressive.py: er_sde needs `denoise.inner_model` (fixed); rerun pending.
     - **VAE int path (Luna: "implement the vae int path"), 2026-09-29, MIOpen off**:
       profile (vae_profile.py): 2560x1440 tiled decode 3.46 s / peak 2.65 GB, encode 1.98 s /
       1.7 GB (weights 0.48 GB -- the "5 GB" was never the VAE itself); 3x3 convs ~85%.
       Quality emulation (vae_int_study.py, 32-channel blocks = the kernel's scheme): W8A8
       58 dB decode / 59 encode, latent err 0.6%; W4A4 37-40 dB, 14% latent err (unusable
       for growth re-encodes); Hadamard doesn't rescue int4. Built an int8 block-scaled GEMM
       + implicit-GEMM conv (node repo d21d717, removed in 5f98c0d): GEMM 13.5-21 TOPS
       (1.65-2.45x rocBLAS fp32; N=96 layers worst), end to end decode 2.59 / encode 1.61 s.
       **ComfyUI's fp16 VAE is faster and better: 2.42 / 1.43 s, 66 dB** (with MIOpen it was
       slower -- that's why fp32 was picked). fp16 checked finite on 4 renders + 2 stress
       images, 65-82 dB (vae_fp16_check.py). run.nu now `--fp16-vae` (1e8fd548); needs
       `--disable-miopen` to stay.
       **Luna: the int VAE was also for VRAM** -> int8 restored as a VRAM option (node repo
       f30179e, "Gfx906 Int8 VAE", memory_used_* scaled 0.6 / 0.85 to measured use).
       vae_mem_bench.py, 2560x1440, reserved GB (tiled 512 = growth path): decode fp32 3.55 /
       fp16 1.82 / int8 1.19; encode 1.75 / 0.92 / 0.71; plain VAEDecode fp16 15.3, int8 12.8.
       The real VRAM hog was ComfyUI's tiled-ENCODE estimate: whole image, 10.3 GB fp16 /
       20.6 fp32 freed before every growth step's encode vs < 1 GB used. Fixed in comfy/sd.py
       (per tile like decode_tiled, a78eae83): 0.73 GB.
     - **Calibration cache built** (node repo b70a9f6, `calibration` input, default persistent;
       sparse_tau still defaults to 0 = off). calib_cache_bench.py, 2560x1440 8 steps, alternating:
       dense 160.8 / 161.8 / 162.3 s, cached 156.0 / 156.8 / 157.0 s (-3.4%, ~5.3 s/run), per_run
       158.6 s; learning runs 178-183 s (+~19 s each, 3 runs) -> break-even ~11 runs.
       996/1344 heads sparse. PSNR vs dense: cached 19.7 dB, per_run 20.9 (trajectory change, as
       every numeric change; Luna judged per_run equivalent before) -- cached NOT yet judged by
       Luna: develop/out/calib_cache/{dense_1,cached_1,per_run}.png.
     - **int4 GEMM scale math: not the bottleneck** (scale_math/bench_scale_math.py, speed-bound
       variants of the production kernel, M 14400 / 4224, Krea2 shapes): production 40-41 TOPS,
       64-value blocks 41-42, one-sided scale (per-token act or per-channel weight) 44-46,
       two-level i24 sub-scales 35 (slower), NO rescale at all 41-43. The old "55 without scale
       math" ceiling isn't reproducible in the full kernel (loads/LDS dominate). Best realistic
       +11% on linears (~4%/step) needs a lossy format change -- not pursued.
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
