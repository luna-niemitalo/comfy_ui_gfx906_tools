# Prompt/Pipeline Debug Harness — DESIGN

## Context

`develop/PythonLibraryGuide.md` shows how to drive ComfyUI in-process (no server,
no HTTP, models stay resident). `develop/TODO.md` is a *separate* project: a
perf/correctness bench harness for RVII kernel work (timings, VRAM, NaN checks).
This document is a third, related thing: a harness for the actual generation
workflow, aimed at answering "why did this prompt produce the wrong image" —
not "how fast is it."

Concrete driver: `develop/Luna's moodyKrea thumbnails.json` is the current
real workflow and becomes the framework's default recipe. Its nodes:

```
UNETLoaderMultiGPU  (krea2/moodyKrea2Mix_v50.safetensors, cuda:0)
CLIPLoaderMultiGPU  (qwen3vl_4b_bf16.safetensors, type=krea2, cuda:1)
VAELoaderMultiGPU   (qwenImageVAE_qwenImageVAE.safetensors, cuda:0)
CLIPTextEncode      (positive prompt)
ConditioningZeroOut (negative = zeroed positive, not a separate encode)
EmptyLatentImage    (128, 128, batch=1)
SeedNode            (fixed seed, separate from the sampler's own seed widget)
KSamplerAdvanced    (add_noise=enable, steps=8, cfg=0.8, sampler=er_sde,
                     scheduler=simple, start_at=0, end_at=8, return_leftover=enable)
VAEDecodeTiled      (tile=512, overlap=64, temporal 64/8)
SaveImage / PreviewImage
```

This is the multi-GPU / Qwen-VL-encoder path, distinct from the plain
`CheckpointLoaderSimple` path the library guide documents — the framework
must treat "which node sequence builds MODEL/CLIP/VAE" as data, not assume
the simple checkpoint path.

## Goal

One command → one image, from a fixed, inspectable configuration. The same
configuration, minus one deliberately changed field, should be a one-line
diff to produce. This is what makes it possible to tell whether a bad image
is caused by the model, the ComfyUI build, the pipeline wiring, or the
prompt — because everything else is pinned and logged.

Non-goal for this pass: any actual prompt-engineering logic (variant
generation, scoring, "does the model understand front vs foreground").
The Qwen-VL example is only there to justify the shape of the tool — the
tool must make that investigation *possible*, not perform it.

## Architecture

Four small pieces, each independently useful, sharing one module:

```
develop/harness/
  bootstrap.py    # sys.argv/args setup, folder_paths, model-management flags
  pipeline.py     # recipe schema -> MODEL/CLIP/VAE/CONDITIONING/LATENT/IMAGE
  generate.py     # CLI: one recipe (+overrides) -> one image + sidecar JSON
  sweep.py        # CLI: one recipe + one axes file -> N images, reusing loaded
                   #      models across axis values that don't force a reload
  contact_sheet.py# manifest.jsonl -> a labeled grid PNG for eyeballing a sweep
```

`bootstrap.py` is lifted straight from `PythonLibraryGuide.md` §2 (no
`enable_args_parsing()`, set `folder_paths` before `import nodes`) plus the
`--gpu-only`/`--disable-smart-memory` flag-setting from `TODO.md` item 1/2,
since this fork's dynamic-VRAM path is a real confound for reproducibility
here too, not just for the perf harness.

### Recipe = diff against a default, not a transcript

The workflow JSON is a raw web-export: every node's every widget value,
spelled out in full, whether or not it's ever going to change. That's the
wrong shape for an input file — a recipe should only contain what you're
choosing to diverge on. Everything else comes from a **pipeline default**:
one YAML file per pipeline family, under `develop/recipes/defaults/`, named
by its `kind` (`develop/recipes/defaults/moodykrea_qwenvl.yaml` today) —
data, not code, so adding a pipeline family is dropping in a file, matching
the same "recipe = data" principle the recipes themselves follow:

```yaml
# develop/recipes/defaults/moodykrea_qwenvl.yaml
loaders:
  unet: {node: UNETLoaderMultiGPU, unet_name: krea2/moodyKrea2Mix_v50.safetensors,
         weight_dtype: default, device: cuda:0}
  clip: {node: CLIPLoaderMultiGPU, clip_name: qwen3vl_4b_bf16.safetensors,
         type: krea2, device: cuda:1}
  vae:  {node: VAELoaderMultiGPU, vae_name: qwenImageVAE_qwenImageVAE.safetensors,
         device: cuda:0}
conditioning:
  positive: {node: CLIPTextEncode, text: ""}
  negative: {node: ConditioningZeroOut, source: positive}
latent: {node: EmptyLatentImage, width: 128, height: 128, batch_size: 1}
sampler: {node: KSamplerAdvanced, add_noise: enable, noise_seed: 337040858006321,
          steps: 8, cfg: 0.8, sampler_name: er_sde, scheduler: simple,
          start_at_step: 0, end_at_step: 8, return_with_leftover_noise: enable}
vae_decode: {node: VAEDecodeTiled, tile_size: 512, overlap: 64,
             temporal_size: 64, temporal_overlap: 8}
```

A recipe file is just the divergence, plus the one field every run needs:

```yaml
kind: moodykrea_qwenvl
prompt: "simple trivial to glance read icon for a file gallery web server, favicon, monochrome, silouette, tiny icon"
```

That's the whole file for "run the workflow as-is." Diverging on one
setting means writing exactly that one setting:

```yaml
kind: moodykrea_qwenvl
prompt: "character standing, facing the viewer"
sampler: {noise_seed: 42, steps: 12}
```

`pipeline.py`'s `resolve()` loads `develop/recipes/defaults/<kind>.yaml` by
filename stem and deep-merges the recipe onto it (recipe keys win, dicts
merge key-by-key, everything else — including `--set` overrides from
`generate.py`/`sweep.py` — is the same merge applied one more time on top).
`prompt` writes into `conditioning.positive.text` as a named shorthand,
since it's the field that changes on nearly every run and typing
`conditioning.positive.text=` every time would defeat the point.

The *sidecar* written per run (see below) is still the fully-resolved
node graph, not the sparse recipe — provenance needs the complete picture,
input files don't.

Promoting a new pipeline family (a plain SDXL checkpoint, a GGUF Flux
model, whatever comes after Krea) means dropping one more file into
`develop/recipes/defaults/`, transcribed once from its own workflow export
the same way — not writing per-recipe boilerplate for every run of that
family, and not touching `pipeline.py` at all.

### Generic node execution, not per-pipeline Python

`pipeline.py` builds MODEL/CLIP/VAE/CONDITIONING/LATENT by walking the
recipe's `{node: ..., **kwargs}` blocks through
`nodes.NODE_CLASS_MAPPINGS[block["node"]]`, exactly the `run_node` pattern
from `PythonLibraryGuide.md` §5. A new model family (a plain SDXL
checkpoint, a GGUF Flux model, whatever comes after Krea) is a new recipe
file, not new Python — as long as it produces the standard
MODEL/CLIP/VAE/CONDITIONING/LATENT/IMAGE sockets. This is the same
generic-dispatch idea `TODO.md` uses for its `fused_custom` attention slot,
applied one level up (whole loader stages instead of one function pair).

Only write a dedicated Python path for a pipeline when its node graph
doesn't fit the linear loaders→conditioning→latent→sample→decode shape
above (e.g. anything needing more than one sampling pass). Keep that as a
named function in `pipeline.py` registered by `id` prefix, not a general
"arbitrary graph" executor — `execution.py` already owns that job and pulling
it in would defeat the point of skipping the queue/server.

### One-shot: `generate.py`

```
python develop/harness/generate.py --recipe recipes/moodykrea_qwenvl.yaml \
    --set prompt="a photo of the character's front" --set sampler.noise_seed=42
```

- Loads the recipe, applies `--set dotted.path=value` overrides (small,
  no need for a real templating/patch library — a handful of dict-path
  writes).
- Runs bootstrap once, builds the pipeline, samples, decodes, saves
  `out/<timestamp>_<id>.png`.
- Writes a sidecar `out/<timestamp>_<id>.json`: the fully-resolved recipe
  (after overrides), the ComfyUI git commit hash (`git rev-parse HEAD` in
  this checkout), wall-clock timings per phase, and the resolved seed.
  This is the provenance that lets two images be diffed later and tells you
  immediately whether "different build" or "different setting" explains a
  difference — same idea as `TODO.md`'s JSONL log, generalized from
  perf/correctness fields to full config provenance.
- Appends one line to `out/manifest.jsonl` (same record, for later sweep
  tooling to consume without re-reading every sidecar).

### Sweeps: `sweep.py`

The actual tool for the Qwen-VL debugging workflow: many prompt variants,
one fixed recipe, one manifest to compare against afterward.

```yaml
# sweeps/front_vs_foreground.yaml
recipe: recipes/moodykrea_qwenvl.yaml
axes:
  prompt: {file: sweeps/front_vs_foreground.txt}
  sampler.noise_seed: [1, 2, 3]
```

```
# sweeps/front_vs_foreground.txt — one prompt per line
character standing, facing the viewer
character standing, front view
character in the foreground of the image
character close to the camera, background blurred
```

`prompt` is the one axis expected to grow by hand or by script between
runs, so it takes `{file: ...}` instead of an inline YAML list: a plain
newline-delimited text file, one variant per line, blank lines skipped.
Generating more variants is then `echo "..." >> sweeps/foo.txt` — no
YAML/JSON quoting, no parser beyond `splitlines()`, safe to append to
from another process while nothing is reading it mid-sweep. Every other
axis (`sampler.noise_seed`, `sampler.cfg`, whatever) stays an inline YAML list,
since those are short, fixed, and don't benefit from append-only editing.
`sweep.py` reads the `.txt` file once at sweep start, same as it reads the
YAML — no live-reload/watch behavior, that's a different feature if it's
ever needed.

- Cartesian product over declared axes (this is exactly the
  "sweep generation is just config manufacturing" idea already named in
  `TODO.md` — reused, not reinvented, with `prompt` simply being one more
  axis alongside numeric settings).
- Loads models once, reuses them across every combination — `sampler.noise_seed`
  changes never trigger a reload; only a change to `loaders.*` would. This
  matters here for the same reason it mattered in `TODO.md`'s warm mode:
  reloading a Qwen-VL text encoder per prompt variant would make a
  40-variant sweep unusably slow.
- Same per-run sidecar + manifest as `generate.py`, one line per
  combination, `error` populated instead of aborting the whole sweep so one
  bad combination doesn't lose the rest of the sweep's results (breadcrumb
  convention, matches `TODO.md`'s "never let a bad run vanish silently").

### `contact_sheet.py`

Takes a `manifest.jsonl` (or `--dir out/`), lays out the referenced images
in a grid, one cell per run, labeled with whichever axis field actually
varies across the manifest (diff the resolved recipes, label with the
differing key/value only — not the whole config dump). This is the actual
"distill what the model is failing to understand" artifact: a page of
labeled thumbnails you can scan by eye, not a directory of anonymous PNGs.
Deliberately a separate script reading the manifest format, not built into
`sweep.py` — sweeps you want to eyeball and sweeps you want to feed to
something else (a VLM judge, later) both start from the same manifest.

## Relationship to `develop/TODO.md`

Shared: `bootstrap.py`'s argv/flags setup, the "recipe/config = data,
diffable, no implicit inheritance" convention, the JSONL-manifest habit,
the generic node-dispatch idea for extension points.

Not shared: output schema and success criteria. `TODO.md`'s harness cares
about timings/VRAM/NaN-or-not; this harness cares about "is the image what
was asked for," which is a human-judgment field, not a threshold — so its
manifest has no `correctness` block, just the resolved config + where the
PNG landed. Keep the two entrypoints and schemas separate; merging them
would leak perf concerns into a prompt-debugging tool and vice versa.

## Headless gotchas found running this for real

Implemented and run end-to-end against the real weights on `fox_nest`.
Two environment gaps showed up, both fixed in `bootstrap.py`/`pipeline.py`,
neither specific to the moodyKrea recipe — any future `kind` reusing these
loaders or `execution.py`'s conventions will hit them too:

- **`server.PromptServer.instance` doesn't exist outside a running server.**
  It's only ever assigned inside `PromptServer.__init__` (`server.py:217`),
  never given a class-level default — code that checks `is None` still
  crashes with `AttributeError` if the server was never started.
  ComfyUI-MultiGPU's memory-management hooks hit this immediately on
  `VAEDecodeTiled`. `bootstrap.py` sets `server.PromptServer.instance = None`
  once, matching PythonLibraryGuide.md's own warning that "some custom
  nodes assume a running PromptServer."
- **Node calls must happen inside `torch.inference_mode()`.**
  `execution.py`'s `PromptExecutor` wraps a whole prompt's node calls in one
  `with torch.inference_mode():` block; some tensors created under that
  scope (weights loaded via `comfy.ops`, VAE decode output) are only cloned
  back to normal tensors when code checks `torch.is_inference_mode_enabled()`
  and finds it off (see `comfy/utils.py:980`). Calling nodes directly
  without that scope, as this harness does, hits "inplace update to
  inference tensor" on the VAE decode's final clamp. `pipeline.run()` and
  `sweep.py`'s per-combination loop now wrap the loaders+sample+decode call
  in the same block, matching `execution.py`'s scope exactly (not a
  workaround — this is how AGENTS.md's inference-mode rule assumes model
  code gets invoked in the first place).

## Open questions

- Multi-GPU device placement (`cuda:0`/`cuda:1` split across unet/clip/vae)
  is hardcoded per-loader in the recipe today, matching the workflow. Fine
  to leave as literal recipe fields rather than an abstraction — only one
  concrete layout exists so far; revisit if a second layout shows up.
- `ConditioningZeroOut` as "negative" is workflow-specific (Krea2/Qwen-VL
  convention, no real negative prompt). Recipes for pipelines that do use a
  real negative-prompt `CLIPTextEncode` just declare that node instead —
  `pipeline.py` doesn't need to know which convention is in play, it just
  executes whatever the recipe's `conditioning.negative` block says.
- No plan yet for comparing across different *ComfyUI builds/forks*
  checked out in different worktrees — right now that's "run the same
  recipe from each worktree's checkout, diff the sidecar's git-hash field."
  Revisit only if that becomes a frequent enough motion to automate.

## Priority order

1. `bootstrap.py` + `pipeline.py` generic node-dispatch, hardcoded to the
   moodyKrea recipe above — prove one recipe produces the same image the
   real workflow does.
2. `generate.py` with `--set` overrides + sidecar/manifest.
3. `sweep.py` with prompt as one axis, reuse-loaded-model behavior.
4. `contact_sheet.py`.
