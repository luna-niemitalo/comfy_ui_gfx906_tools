# ComfyUI CLI Args — Relevant to My Hardware

Personal reference, not part of the ComfyUI project itself. Untracked (see
`.gitignore`). Source of truth for flag list: `comfy/cli_args.py`.

Hardware this filters for:
- `fox_nest`: Ryzen 9 3950X, 2x Radeon VII (gfx906), 64GB RAM, ROCm 7.2 built
  from source, NixOS/ZFS.
- `mobile_fox`: Framework 13, Ryzen AI 9 HX 370 (XDNA2 NPU), 96GB RAM, NixOS.

No NVIDIA, no Intel, no DirectML, no Windows — those categories/flags are
excluded here as dead weight (see the "explore CLI settings" conversation for
the full unfiltered breakdown if needed).

---

## Device Selection

- `--cuda-device` — ROCm maps to torch's CUDA API, so this selects between the
  two Radeon VIIs on `fox_nest`.
- `--default-device` — same reasoning, sets which of the two is default.

## Precision & Dtype

- `--force-fp32` / `--force-fp16`
- `--fp32-unet` / `--fp64-unet` / `--bf16-unet` / `--fp16-unet`
- `--fp8_e4m3fn-unet` / `--fp8_e5m2-unet` / `--fp8_e8m0fnu-unet`
- `--fp16-vae` / `--fp32-vae` / `--bf16-vae`
- `--cpu-vae`
- `--fp8_e4m3fn-text-enc` / `--fp8_e5m2-text-enc` / `--fp16-text-enc` /
  `--fp32-text-enc` / `--bf16-text-enc`
- `--fp16-intermediates`
- `--force-channels-last`

gfx906 (Vega 20) has no native fp8 hardware support — the fp8-unet/fp8-text-enc
flags run but gain nothing over bf16 on this GPU.

## Attention Backends

- `--use-split-cross-attention`
- `--use-quad-cross-attention`
- `--use-pytorch-cross-attention` — realistic option on gfx906.
- `--use-sage-attention` — CUDA-kernel-only in practice, unlikely to have a
  working ROCm gfx906 path.
- `--use-flash-attention` — same caveat as sage-attention.
- `--use-ck-attention` — Comfy Kitchen, built for gfx906 here; realistic
  option.
- `--disable-xformers` — xformers may not even be installed for this ROCm
  build.
- `--force-upcast-attention` / `--dont-upcast-attention`

## Memory & Caching

- `--cache-ram` (default) / `--cache-classic` / `--cache-lru` / `--cache-none`
- `--high-ram` — worth tuning with 64GB system RAM on `fox_nest`.

## VRAM Management

- `--gpu-only` / `--highvram` / `--lowvram` / `--novram` / `--cpu`
- `--reserve-vram`
- `--vram-headroom`
- `--async-offload` / `--disable-async-offload` — help text says "enabled by
  default on Nvidia," meaning the default may silently be off on ROCm. Worth
  explicitly testing/setting.
- `--disable-dynamic-vram` / `--enable-dynamic-vram`
- `--fast-disk` / `--disable-fast-disk`
- `--disable-cuda-graphs`
- `--disable-comfy-compiler`
- `--assert-graph-breaks`

## Performance Flags

- `--fast [fp16_accumulation|autotune]` — `cublas_ops` and `fp8_matrix_mult`
  excluded here: cuBLAS is Nvidia-only, and fp8_matrix_mult hits the same
  fp8-hardware gap as above.
  - **`fp16_accumulation`: registered as enabled on this hardware — investigate
    what it's actually doing / whether it's helping, at some point.**
- `--force-non-blocking` — help text explicitly calls out potential benefit on
  non-Nvidia systems; worth testing on ROCm.
- `--deterministic`
- `--disable-smart-memory`
- `--disable-pinned-memory`
- `--mmap-torch-files` / `--disable-mmap`
- `--enable-triton-backend` / `--disable-triton-backend`

## Manager & Custom Nodes

- `--enable-manager`
- `--disable-manager-ui` / `--enable-manager-legacy-ui`
- `--disable-all-custom-nodes`
- `--whitelist-custom-nodes`
- `--disable-api-nodes`
- `--disable-metadata`

## Preview

- `--preview-method`
- `--preview-size`

## Logging & Debug

- `--verbose`
- `--log-stdout`
- `--dont-print-server`
- `--debug-hang`
- `--quick-test-for-ci`

## Frontend, Assets & API

- `--front-end-version`
- `--front-end-root`
- `--enable-compress-response-body`
- `--enable-assets`
- `--enable-asset-hashing`
- `--feature-flag`
- `--list-feature-flags`
- `--multi-user`
- `--default-hashing-function`
- `--comfy-api-base`

---

## Instructions for LLM: maintaining this list

When asked to update this file (or proactively, if asked to check whether it's
stale against `comfy/cli_args.py`):

1. Read `comfy/cli_args.py` fresh — it is the only source of truth for what
   flags exist. Do not rely on memory of a previous pass.
2. Diff the current flag set against what's listed here. For each **new**
   flag:
   - Decide relevance using the hardware profile at the top of this file
     (dual Radeon VII / gfx906 / ROCm, no NVIDIA/Intel/DirectML/Windows). A
     flag is dead weight if it only affects a backend, vendor, or OS not in
     that profile (e.g. anything CUDA-graph/cuBLAS/NVML-specific, DirectML,
     oneAPI, Windows-standalone).
   - If relevant, add it to the matching category above (create a new
     category only if none of the existing ones fit within 3 words — see
     `~/nix/claude-rules/CLI.md` §2.1 for the naming rule).
   - If irrelevant, skip it silently — do not maintain a "dead weight" list
     here; that was a one-time discussion artifact, not something to keep
     growing.
3. For each **removed** flag (no longer in `cli_args.py`), delete its line
   here.
4. Do not re-derive or restate the "why" for existing entries unless the
   flag's own help text or behavior changed — only touch what actually
   changed.
5. Never add speculative categories or notes about flags that don't exist yet.
