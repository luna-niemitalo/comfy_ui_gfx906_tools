# Headless ComfyUI Benchmark Harness — TODO / IMPLEMENTATION SPEC

## Context / why

ComfyUI (full server + node graph + custom_nodes auto-discovery) is too flaky
and slow-iterating to be the primary dev loop for RVII (gfx906) kernel/quant
optimization work. This harness reuses ComfyUI's and ComfyUI-GGUF's actual
model-loading/dequant/attention code paths directly — same compute, same
correctness characteristics — while skipping the web server, prompt queue,
history cache, and custom_nodes import-everything-on-boot behavior.

This is the standing benchmark bed for a parallel RVII optimization project
(fused dequant kernels, quant-type selection, attention backend swaps). It
needs to produce comparable, reproducible numbers across those experiments,
not just "does it run."

**Verified against the live repo 2026-08-01.** This is not upstream ComfyUI —
this fork already has a "dynamic VRAM" model-loading path (`comfy.memory_management`,
`CoreModelPatcher`, multi-GPU deepclone) layered on top of the classic
`ModelPatcher`. That's an extra confound axis the harness must pin down
(see cli flags below), on top of the smart-memory/offload heuristics upstream
already has.

## Non-goals

- Not a UI replacement. No web server, no node graph executor, no
  workflow JSON parsing.
- Not touching the custom_nodes auto-discovery/import system at all —
  bypass it entirely rather than fixing it.
- Not reimplementing model forward-pass math. Import and call the real
  `comfy` / `ComfyUI-GGUF` classes/functions — if the harness diverges
  from what ComfyUI actually executes, the benchmark is worthless.

## Reused code (confirmed present, paths and call shapes below)

- `custom_nodes/ComfyUI-GGUF/dequant.py` — `dequantize_tensor`, `dequantize`,
  per-qtype `dequantize_blocks_*`, `is_quantized`/`is_torch_compatible`. (Note:
  it's an unpacked custom_nodes dir, not pip-installed — import via
  `sys.path.insert` on `custom_nodes/ComfyUI-GGUF` or `importlib`, since
  `custom_nodes.ComfyUI-GGUF` isn't a valid package name.)
- `ComfyUI-GGUF/ops.py` — `GGMLTensor`, `GGMLLayer`, `GGMLOps(comfy.ops.manual_cast)`,
  `move_patch_to_device`.
- `ComfyUI-GGUF/loader.py` — `gguf_sd_loader(path, handle_prefix=...)` →
  `(state_dict, extra)`; `gguf_clip_loader(path)` → state_dict.
  `IMG_ARCH_LIST`/`TXT_ARCH_LIST` for arch detection. No runtime quant-type
  override param anywhere in this file — confirms open item 7 below.
- `ComfyUI-GGUF/nodes.py::UnetLoaderGGUF.load_unet` / `CLIPLoaderGGUF.load_patcher`
  are the reference call sequences the harness should mirror directly:
  ```python
  ops = GGMLOps()
  sd, extra = gguf_sd_loader(unet_path)
  model = comfy.sd.load_diffusion_model_state_dict(
      sd, model_options={"custom_operations": ops}, metadata=extra.get("metadata", {}))
  model = GGUFModelPatcher.clone(model)          # from ComfyUI-GGUF/nodes.py

  clip = comfy.sd.load_text_encoder_state_dicts(
      clip_type=comfy.sd.CLIPType.FLUX, state_dicts=[gguf_clip_loader(p) for p in clip_paths],
      model_options={"custom_operations": GGMLOps,
                      "initial_device": comfy.model_management.text_encoder_offload_device()})
  clip.patcher = GGUFModelPatcher.clone(clip.patcher)
  ```
- VAE: no node needed. `comfy.sd.VAE` at `comfy/sd.py:474`:
  ```python
  sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
  vae = comfy.sd.VAE(sd=sd, metadata=metadata)
  vae.throw_exception_if_invalid()
  image = vae.decode(samples)   # comfy/sd.py:1055; decode_tiled at :1111
  ```
  (`comfy.sd.load_vae_patcher()` exists but returns `vae.patcher`, not the
  `VAE` object with `.decode` — build `VAE` directly as above instead.)
- Sampling: no KSampler node needed, `comfy/sample.py` wraps `comfy.samplers.sample`:
  ```python
  noise = comfy.sample.prepare_noise(latent_image, seed)
  samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler,
                                 positive, negative, latent_image, seed=seed)
  ```
  This is exactly what `nodes.py::common_ksampler` calls — reuse it, not the node class.
- `noflash-attention` (pip package `noflash-attention==1.0.1`, already
  `import noflash_attention`'d unconditionally at `main.py:228`, before
  `comfy.utils`/`execution`/`nodes` import — so import order is already safe
  in-process). See "Attention backend" below for how it actually switches.

## Config schema

One JSON/YAML file = a list of test-case objects. Each object is fully
self-describing (no implicit inheritance from prior cases) so cases can be
diffed, generated programmatically, and run in any order/subset.

```jsonc
[
  {
    "id": "flux_q4k_stock_sdpa_512",
    "model_path": "/path/to/flux-q4_k.gguf",
    "clip_paths": ["/path/to/clip_l.safetensors", "/path/to/t5xxl.gguf"],
    "vae_path": "/path/to/vae.safetensors",
    "quant_type_override": null,        // null = model_path's native quant; non-null = load a different pre-converted .gguf, NOT a runtime override (see open item 7)
    "attention_backend": "stock",       // stock | noflash | fused_custom — tier reached under noflash is a logged *output*, not selectable (see open item 3)
    "resolution": [512, 512],
    "steps": 20,
    "seeds": [1, 2, 3],
    "vae_decode": true,
    "reference_output": null,           // optional path to known-good image/tensor for correctness check
    "correctness_thresholds": {
      "max_pixel_deviation": 0.02,      // fraction, vs reference_output if provided
      "min_stddev": 1e-4,               // catches black/flat frames
      "reject_nan_inf": true
    },
    "notes": "quick iteration baseline, fits trivially in VRAM"
  },
  {
    "id": "flux_q4k_stock_sdpa_2mp",
    "model_path": "/path/to/flux-q4_k.gguf",
    "clip_paths": ["/path/to/clip_l.safetensors", "/path/to/t5xxl.gguf"],
    "vae_path": "/path/to/vae.safetensors",
    "quant_type_override": null,        // see field comment above
    "attention_backend": "stock",       // see field comment above
    "resolution": [1536, 1344],          // ~2MP, representative real-world size
    "steps": 20,
    "seeds": [1, 2, 3],
    "vae_decode": true,
    "reference_output": null,
    "correctness_thresholds": {
      "max_pixel_deviation": 0.02,
      "min_stddev": 1e-4,
      "reject_nan_inf": true
    },
    "notes": "same model/config as above but real-world resolution — catches VRAM-pressure-only failures the small case can't"
  }
]
```

Sweep generation (small-vs-large, quant-vs-quant, attention-vs-attention)
is just config manufacturing — write a small generator script that takes
axes (resolutions, quant types, attention backends) and emits the cartesian
product as one config file. Not part of the harness itself, keep separate
(`gen_sweep_config.py` producing `sweep_config.json`).

## Script: single entrypoint, `--mode` flag

`bench.py --config sweep_config.json --mode {cold,warm} --device cuda:0`

### Shared phases (both modes), per test case, all explicitly timed:

1. **import** — time to import torch + comfy core + ComfyUI-GGUF modules +
   noflash_attention. Only done once for `warm` mode (first case), repeated
   per-subprocess for `cold` mode.
2. **load** — load GGUF weights + CLIP/T5 + VAE via direct calls (no queue,
   no UI). Record wall time, VRAM after load (`torch.cuda.memory_allocated`,
   `torch.cuda.max_memory_allocated` — same API under ROCm/HIP).
3. **run** — N denoise steps at configured resolution/seed. Record
   wall time, VRAM peak during run.
4. **vae_decode** (if `vae_decode: true`) — full VAE decode to image tensor.
   Not just a memory-characteristics step — also the correctness gate:
   - reject if any NaN/Inf present
   - reject if stddev below `min_stddev` (catches flat black/noise-only output)
   - if `reference_output` provided: compare against it (pixel-wise deviation
     or perceptual hash — pick whichever is cheaper to implement first,
     pixel deviation is fine as v1), flag if beyond `max_pixel_deviation`
   - this is the "silently corrupts at larger resolution" detector the
     small-vs-large sweep exists to trigger
5. **cleanup** — explicit `del` of model/tensors, `torch.cuda.empty_cache()`,
   `gc.collect()`. Record VRAM after cleanup. This is the leak signal:
   compare against VRAM baseline (post-import, pre-load) — should return
   close to baseline; if not, delta is logged per test case for trend
   analysis across the sweep.
6. **teardown/exit** — always log a result line even on exception/crash;
   never let a bad run vanish silently (breadcrumb-first, matches existing
   foreign-data validation convention: print expected vs actual).

### Mode-specific behavior

- **cold**: one subprocess per test case (`id`). Full isolation, clean
  CUDA/HIP context per case. Best for "what's the actual number for this
  config" — no cross-contamination.
- **warm**: single process, loop over all test cases (and all seeds within
  each case) without process restart, model reloaded only when `model_path`/
  `quant_type_override`/`attention_backend` changes between consecutive
  cases (skip reload if unchanged — this itself is a thing worth measuring:
  does re-running the same loaded model with a new seed leak vs a seed
  that's been seen before, i.e. does something get cached per-seed that
  shouldn't be). This is the actual leak detector — VRAM should be flat
  across iterations of the same config; growth across seed-only changes is
  the leak signature.

### Output

Append-only JSONL, one line per (test case × seed) execution:

```jsonc
{
  "test_id": "flux_q4k_stock_sdpa_2mp",
  "seed": 2,
  "mode": "warm",
  "attention_backend": "stock",
  "quant_type": "Q4_K",
  "resolution": [1536, 1344],
  "timings": {"import_s": 0.0, "load_s": 4.2, "run_s": 38.7, "vae_decode_s": 1.9, "cleanup_s": 0.3},
  "vram": {"baseline_mb": 412, "post_load_mb": 9800, "peak_run_mb": 14200, "post_cleanup_mb": 500},
  "correctness": {"nan_inf": false, "stddev": 0.14, "reference_deviation": null, "passed": true},
  "error": null
}
```

Diffable across runs/kernel changes; this becomes the evidence for "fused
Q4_K kernel is N% faster / does not regress correctness / does not leak."

## Open items — resolved from live repo (2026-08-01)

1. **Minimal-import path.** `comfy.options` → `comfy.cli_args` must load
   first, and `comfy.cli_args.args` is populated by `argparse.parse_args()`
   at *import time*, reading `sys.argv`. Several modules (`comfy.model_management`)
   snapshot `args.*` into module globals (`vram_state`, `DISABLE_SMART_MEMORY`)
   at their own import time. **So: set `sys.argv` before the first `import comfy.*`**,
   e.g. `sys.argv = ["bench.py", "--gpu-only", "--disable-smart-memory", "--deterministic"]`.
   No need to import `execution`/`server`/`nodes`/the custom_nodes loader at all —
   only `comfy.sd`, `comfy.utils`, `comfy.model_management`, `comfy.samplers`,
   `comfy.sample`, plus the two ComfyUI-GGUF files pulled in via `sys.path`.
2. **Disabling smart memory / dynamic VRAM — resolved, three flags, set via `sys.argv` per item 1:**
   - `--gpu-only` (or `--highvram`) — keeps weights resident, no offload.
   - `--disable-smart-memory` — kills the offload-heuristic path outright.
   - `--disable-dynamic-vram` — this fork also has a *separate* dynamic-VRAM
     loading path (`comfy.memory_management`, `CoreModelPatcher`); `--gpu-only`
     already implies `enables_dynamic_vram() == False` (see `comfy/cli_args.py:283`),
     but pass it explicitly for clarity/robustness against flag changes.
3. **noflash-attention control — resolved, no tier flag exists.** It's an
   automatic OOM-driven cascade inside one SDPA patch, not three selectable
   backends:
   - `NOFLASH_DISABLE=1` env var (checked in `detect_gpu.py` before the patch
     decides whether to activate) → **use this for `attention_backend: "stock"`**.
   - `NOFLASH_FORCE_PATCH=1` / default → chunked path active whenever
     `seq_kv > FLASH_THRESHOLD` (env `NOFLASH_THRESHOLD`, default `"0"` — i.e.
     chunked is already the default-active path on this GPU).
   - Tiers 1/2/3 (`chunked_sdpa` → `chunked_sdpa_online` → `chunked_sdpa_inplace`)
     are tried in order **only on `torch.cuda.OutOfMemoryError`** from the
     previous tier (`noflash_attention/patch.py:356-393`) — there is no config
     value to force tier 2/3 directly. To benchmark them deliberately you'd
     need to shrink the auto-computed `block_m`/budget (monkeypatch
     `_auto_block_m`'s `budget_factor`) or actually starve VRAM. Revise the
     config schema: `attention_backend: stock | noflash` (env-var toggle,
     set before `import noflash_attention`), tier reached is an *output*
     metric to log, not an input.
   - `noflash_attention.enable()`/`.disable()` also exist for in-process
     toggling without re-importing (needed for **warm** mode where the
     process doesn't restart between cases).
4. **VAE decode — resolved**, see `comfy.sd.VAE(...).decode()` above; no
   node-graph context needed, confirmed by reading `comfy/sd.py:1055`.
5. **Determinism** — `--deterministic` CLI flag exists (`comfy/cli_args.py:161`,
   forces slower deterministic algorithms; upstream comment warns it doesn't
   guarantee bit-identical images). Seeding path is `comfy.sample.prepare_noise(latent_image, seed, batch_inds)`
   (`comfy/sample.py:22`) — use this, not raw `torch.manual_seed`, to match
   what `KSampler` actually does.
6. **`torch.cuda.*` under ROCm** — unverified in this pass, low risk (HIP is
   CUDA-API-compatible in torch); sanity-check once on real hardware before
   trusting VRAM numbers in the sweep, not worth deeper static investigation.
7. **Quant-type override — confirmed not a runtime knob.** `gguf_sd_loader`
   has no override param; the file's tensor `tensor_type` from the GGUF header
   drives dequant dispatch directly. `quant_type_override` in the config
   schema must mean "load a different pre-converted `.gguf` file" — rename
   the field to `model_path` selection in the sweep generator rather than
   pretending it's a load-time flag.
8. **`fused_custom` attention slot** — still correctly left as a stub; given
   item 3's finding, the natural extension point is a dict mapping backend
   name → `(patch_fn, unpatch_fn)` pair, called the same way
   `noflash_attention.enable()/disable()` already is.

## Priority order for implementation

1. Cold mode, single test case, no VAE decode, no correctness checks —
   prove the minimal-import path works at all.
2. Add VAE decode + correctness checks (NaN/Inf, stddev, reference diff).
3. Add warm mode + seed looping + VRAM leak tracking.
4. Add sweep config generator + run the small-vs-large resolution sweep as
   first real validation of the harness itself.
5. Wire in `attention_backend` switching via `NOFLASH_DISABLE`/`noflash_attention.enable()`/`.disable()`.
6. Leave `fused_custom` slot stubbed for later RVII kernel work.

Note: `gpu_profile.py` (repo root, already wired into `main.py`) gives an
opt-in `torch.profiler` trace behind `COMFY_PROFILE=1` — orthogonal to this
harness's own JSONL timings, reuse it standalone for deep single-case
op-level diagnosis rather than reimplementing profiling in `bench.py`.
