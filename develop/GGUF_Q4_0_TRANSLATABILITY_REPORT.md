# GGUF → gfx906 int4 translatability report

Scope: every `.gguf` diffusion/text-encoder model under `/media/luna/models`
relevant to the ComfyUI pipeline (excludes `non-image-gen/`, which is
llama.cpp-style LLM/embedding models unrelated to this work).

**Classification rule**: `Gfx906Int4Transcode` (see
`custom_nodes/ComfyUI-Gfx906Int4/`) repacks **Q4_0** tensors losslessly —
same value range (-8..7, symmetric, no zero-point) and block size (32) as
our native format, so it's a pure bit-shuffle, verified bit-exact against
ggml's own dequantizer. Every other ggml quant type (Q4_K, Q5_K, Q6_K,
Q3_K, Q2_K, Q8_0, Q4_1) has a different block structure (two-level scales,
or — for Q4_1 — an added zero-point term our format has no slot for) and
currently falls back to dequantize-then-requantize, which is lossy on top
of whatever loss the original quantization already introduced.

Sorted by file **creation time** (statx birth time, newest first) — this
disk uses ZFS/statx so birth time is real, not a fallback to mtime.

## Krea 2 — flagged first per request, but doesn't apply to this report

`/media/luna/models/unet/krea2/` has **no GGUF files at all**:

| file | format |
|---|---|
| `Krea2_Turbo_fp8mixed.safetensors` | comfy-native fp8 quantized (likely `TensorCoreFP8*` layout) |
| `moodyKrea2Mix_v50.safetensors` | plain safetensors (not yet inspected for dtype) |
| `Krea2_Turbo_convrot_int8mixed.safetensors` | comfy-native int8, almost certainly `TensorCoreConvRotW4A4Layout` (`convrot_w4a4` in `comfy/quant_ops.py`) |

None of these are GGUF, so the Q4_0-lossless-repack question doesn't apply
to them — there's nothing to classify here. More importantly: **the two
"mixed" files are already comfy-native `QuantizedTensor` objects
(comfy_kitchen tensor subclass), a third format `Gfx906Int4Transcode`
has never been tested against.** The node's format detection only checks
for GGUF's `tensor_type` attribute; a `QuantizedTensor` doesn't have that,
so it would fall into the "plain" branch and call `.detach().float()` on
it — whether that transparently dequantizes correctly or silently produces
garbage is **unverified**. Treat Krea 2 as blocked on that check, not as
"ready to convert," until someone tests it directly.

## Diffusion / text-encoder GGUF files, newest first

| date created | file | size (MB) | quant types present | verdict |
|---|---|---|---|---|
| 2026-08-11 | `text_encoders/Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-Q4_K_M.gguf` | 14932 | Q4_K:307, F32:200, Q6_K:44 | lossy (Q4_K, not Q4_0) |
| 2026-08-11 | `unet/h3/MiniMax-H3-FL2VA-Q3_K_M.gguf` | 14241 | Q3_K:266, F32:7, F16:262 | lossy |
| 2026-08-11 | `unet/h3/MiniMax-H3-FL2VA-Q4_K_M.gguf` | 18640 | Q4_K:266, F32:7, F16:262 | lossy |
| 2026-06-09 | `diffusion_models/ideogram/ideogram4-Q4_0.gguf` | 5644 | **Q4_0:204**, BF16:254 | **LOSSLESS — the one real hit** |
| 2026-06-09 | `diffusion_models/ideogram/ideogram4-Q8_0.gguf` | 10136 | Q8_0:204, BF16:254 | lossy (Q8_0, not Q4_0) |
| 2026-06-09 | `diffusion_models/Z Image Turbo/z-image-turbo-Q8_0.gguf` | 7225 | Q8_0:180, F32:245, BF16:28 | lossy |
| 2026-06-09 | `diffusion_models/Z Image Turbo/z-image-turbo-Q4_K_M.gguf` | 5018 | Q4_K:112, Q5_K:36, Q6_K:32, F32:245, BF16:28 | lossy |
| 2026-05-07 | `unet/flux1/flux1-dev-Q4_1.gguf` | 7537 | Q4_1:304, F32:466, BF16:10 | lossy — **closest near-miss**, see below |
| 2026-04-19 | `unet/flux2/flux-2-klein-9b-Q6_K.gguf` | 7865 | Q6_K:112, F32:80, BF16:9 | lossy |
| 2026-04-19 | `unet/flux2/flux-2-klein-9b-Q4_K_M.gguf` | 5910 | Q4_K:76, Q5_K:36, F32:80, BF16:9 | lossy |
| 2026-01-28 | `unet/z-image-turbo-q6_k.gguf` | 5910 | Q6_K:180, F32:243, F16:30 | lossy |
| 2025-12-19 | `unet/flux2/flux2-dev-Q2_K.gguf` | 12858 | Q2_K:136, Q4_K:24, F32:128, BF16:11 | lossy |
| 2025-12-08 | `text_encoders/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` | 14334 | Q4_K:241, Q6_K:41, F32:81 | lossy (text encoder) |
| 2025-09-15 | `unet/Chroma1-HD-Q4_K_M.gguf` | 5167 | Q4_K:201, Q5_K:38, F32:400, F16:1, BF16:3 | lossy |
| 2025-09-15 | `unet/Chroma1-Base-Q4_K_S.gguf` | 5033 | Q4_K:239, F32:400, F16:1, BF16:3 | lossy |
| 2025-08-28 | `unet/flux1/flux1-schnell-Q6_K.gguf` | 9828 | Q6_K:304, F32:464, BF16:8 | lossy |
| 2025-08-17 | `unet/flux1/flux1-kontext-dev-Q6_K.gguf` | 9848 | Q6_K:304, F32:466, BF16:10 | lossy |
| 2025-07-28 | `unet/flux1/flux1-kontext-dev-Q4_K_M.gguf` | 6932 | Q4_K:266, Q5_K:38, F32:466, BF16:10 | lossy |

## Bottom line

- **1 of 18 files (`ideogram4-Q4_0.gguf`) is fully, losslessly translatable today.** Nothing else in the library is Q4_0.
- The library's "UD" / mixed-quant Unsloth-style files (`Qwen3-VL`, would-be `*-UD-Q4_K_XL` style naming elsewhere) deliberately vary quant type per tensor for accuracy — none of that variation happens to land on Q4_0.
- **`flux1-dev-Q4_1.gguf` is the best candidate for a second lossless repack.** Q4_1 is Q4_0 plus a per-block zero-point (asymmetric, 20 bytes/block instead of 18) — not covered by the current repack, but structurally close enough that a dedicated `repack_q4_1_native` is a bounded follow-up, not a redesign, *if* it's worth adding a zero-point term to `Gfx906Int4Ops.Linear`'s forward.
- **Q8_0 (5 files here) is a distinct future opportunity, not covered by this report**: it's 8-bit, so it maps to the `sdot4`-based int8 path explored earlier in this session, not the int4 kernel this repack targets. A lossless Q8_0 repack would be a separate, smaller project (int8 already has no zero-point/asymmetric complication to work around).
- **Krea 2 has no GGUF and needs its own investigation** (comfy-native `QuantizedTensor` format, untested against `Gfx906Int4Transcode`) before it can be classified at all — see above.
- Everything else quantizes via the existing dequant-then-requantize fallback, same accuracy profile already validated for Z-Image Turbo earlier in this project (~10-17% error depending on layer/config) — real, usable, just not free the way Q4_0 is.
