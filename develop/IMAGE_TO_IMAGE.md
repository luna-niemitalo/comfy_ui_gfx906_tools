# Image-to-image and sectional editing with the growth toolchain

Worked out 2026-10-01 on a character portrait and a multi-panel character sheet. Nothing here is a
node yet. The building blocks are the Resolution Growth node's internals
(`nodes/ComfyUI-ResolutionGrowth/growth.py`) and the SAM3 and crop helpers in
`develop/experiments/2026-09-30/zoom_test.py`. The scripts that use them for that project
(part regrow, patch, splice, coalesce) live in the OC content repo, under `experiments/2026-10-01/`.
Model stack: recipe `oc_krea2` (int4 Krea2, fused attention, row split onto cuda:1, int4 CLIP,
int8 VAE).

## The one primitive

Every operation below is the same step, applied to a region:

    encode(region) -> re-noise to sigma -> one tiled denoise step (1024 px tiles, overlap 256,
    low band optionally anchored to a reference) -> decode -> paste back through a soft weight

`growth.Grower.step(k, x, size, sigma, anchor)` is that step. If `size` is larger than `x`, it
upscales first (Lanczos in pixel space), so the same call is a growth step. Everything else is
which pixels go in, which prompt, which sigma, and how the result is pasted.

| sigma | what one step does |
|---|---|
| 0.08-0.2 | polish, texture, blend; content fixed |
| 0.35-0.5 | re-draws seams, re-joins shapes that were cut, fills a cloned patch |
| 0.6 | growth start: same objects, new detail at the new scale |
| 0.8 | adds objects (accessories on a part); needs an isolated input, see below |

## Operations

| operation | how |
|---|---|
| compose | 8 steps from noise at ~1 MP, per-region prompt |
| grow (img2img up) | linear +256 px steps, sigma 0.6 -> 0.1, anchor 1 fading (shape 2) |
| part regrow | crop around a SAM3 detection, grow with the part's own text, paste through its mask |
| add to a part | **prune the seed**: crop x part mask, rest black, sigma 0.8, anchor 0.75, prompt ends "against plain black darkness" |
| layer composite | black = transparent: `out = C + (1 - k) B`, C = grown - black level, k = max channel |
| seam blend | one sigma 0.35 step kept only where a paste weight is partial (`4w(1-w)`, grown, softened) |
| region refine | the region + margin gets low steps with **only that region's prompt**, pasted feathered |
| patch | clone-stamp neighbouring pixels (feathered), then one sigma 0.5 step on a small crop |
| splice | upper part of candidate A + lower part of candidate B, joined at a horizontal seam, seam band re-denoised at sigma 0.45 |

## What decides whether it works

- **Large stretches create artifacts.** A 2.9x Lanczos stretch and then any number of fixed-size
  steps leaves an ~8 px dot lattice (splat path B's mosaic has the same cause). Grow in steps
  instead; linear over geometric (cloth wrinkles read as real wrinkles).
- **Words attach to whatever is in frame.** A region prompt describes only that region. Words
  for a part that is absent get attached to something that is there: creature words for one
  body part reshaped other visible parts, and a part described without its body made the model
  invent a body for it. Name every body part in the frame positively, in the subject's own
  terms.
- **Isolation beats masking for additions.** With the surroundings visible, sigma 0.8 re-imagined
  them. With the surroundings black, the additions can only be additive, and the paste is a plain
  layer composite.
- **Brightness keys break thin dark things.** They cut shadowed ribbons and strings into pieces;
  use the premultiplied layer composite instead. Opacity rising faster than brightness gave a dark
  halo.
- **Some things are seed lottery.** Anatomy placement, splits and framing stopped responding to
  wording. Compose more seeds (~25 s each), or fix the spot with patch or splice.
- **Splice seams go where both sources agree.** A seam across a shape running at different angles
  in the two sources kinks it; a seam where one source has extra content carries a scrap along.
  Put it where both sources run the same way.

## Static cost per refinement sample

A refinement sample = one 1024x1024 tile through one denoise step. Its cost does not depend on
the size of the image, only on the tile. Image size only changes how many tiles there are:

    tiles(W, H) = (1 + ceil((W - 1024) / 768)) * (1 + ceil((H - 1024) / 768))   [W, H >= 1024]
    step_s      ~= 2.25 * tiles + 3.1 (per-step overhead) + VAE round trip (~0.75 s per MP)

Measured 2026-10-01, 2 GPUs (row split), 4 tiles per call, from the run logs of that project:

| size | tiles | denoise s | s / tile | upscale (VAE) s | overhead s |
|---|---|---|---|---|---|
| 1056x1584 | 4 | 9.2 | 2.30 | 0.8 | 0.8 |
| 1280x1920 | 6 | 12.3 | 2.05 | 0.2 | 2.0 |
| 1728x2608 | 8 | 18.2 | 2.27 | 1.5 | 2.9 |
| 1952x2944 | 12 | 28.3 | 2.36 | 3.0 | 2.9 |
| 2400x3616 | 15 | 31.6-39.4 (22 steps) | 2.11-2.63 | 1.4-7.9 | 3.0-3.5 |

So: **~2.0-2.6 s per tile-step, mean ~2.25 s.** 15-tile steps vary by ±10% (thermal; the later
steps of a run are slower).

Whole operations, same setup:

| operation | cost |
|---|---|
| compose ~1 MP, 8 steps (132 runs) | 19.3-29.1 s, mean 25.2 |
| grow a panel 720x1440 -> 1296x2560 (3 steps) | 42-47 s |
| grow 1024x1024 -> 1920x1920 (4 steps) | 67-70 s |
| grow 1184x880 -> 2544x1904 (6 steps) | 110-114 s |
| part regrow, 832x1248 seed crops | 10-134 s each |
| full-canvas step 2400x3616 (finish, seam band) | ~40-45 s incl. VAE |
| patch (one ~480x290 crop, 1 tile, 1 step) | ~5 s + model load |

One card: not measured for these runs. The row split's speedup is 1.75-1.9x (CLAUDE.md
Status), so expect roughly 4-4.5 s per tile-step. Under the GPU-turn rule
(`~/.claude/rules/gpu-lock.md`), these scripts must not run on card1 while Luna uses the desktop.
The `oc_krea2` recipe puts the split's replica on cuda:1, so a single-card run needs the split
removed from the recipe. That change has not been made yet.
