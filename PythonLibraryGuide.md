# Using ComfyUI as a Python Library

This guide shows how to drive ComfyUI directly from a Python script — loading models,
building a pipeline out of ComfyUI's own nodes, generating images, and getting the
result back as a tensor/PIL image — **without** running the web server or talking to
its HTTP API.

> There are two very different ways to "use ComfyUI from Python":
>
> 1. **HTTP / prompt-graph API** — you POST a JSON workflow to a running server
>    (`http://127.0.0.1:8188/prompt`). See `script_examples/basic_api_example.py`.
> 2. **In-process library** — you `import` ComfyUI's Python modules and call the node
>    classes yourself. No server, no JSON, models stay resident in your process.
>
> **This guide is about #2.** It's the right foundation for "load models on demand,
> generate, hand the image back." Once that works in-process, wrapping it in your own
> FastAPI/Flask endpoint (your end goal) is trivial — see the last section.

---

## 1. Mental model

A ComfyUI workflow is a graph of **nodes**. Each node is an ordinary Python class in
`nodes.py` (and in custom-node packages). A node advertises:

- `INPUT_TYPES` — a classmethod describing its inputs (used by the UI; you can ignore it).
- `RETURN_TYPES` — a tuple of the output socket types.
- `FUNCTION` — the name of the instance method that does the work.

Calling a node is just:

```python
node = SomeNode()
outputs = node.FUNCTION_METHOD(...)   # returns a tuple, one element per RETURN_TYPES
```

The "wires" between nodes in the UI are just you passing one node's return value into
the next node's method. There is no hidden magic — a text-to-image workflow is 6 method
calls in a row.

### The core data types you'll pass around

| Socket type    | Actual Python object                                                             |
|----------------|----------------------------------------------------------------------------------|
| `MODEL`        | a `ModelPatcher` wrapping the diffusion UNet                                      |
| `CLIP`         | a `comfy.sd.CLIP` text encoder                                                    |
| `VAE`          | a `comfy.sd.VAE`                                                                  |
| `CONDITIONING` | a list of `[tensor, dict]` pairs (output of `CLIPTextEncode`)                     |
| `LATENT`       | a **dict** `{"samples": tensor, ...}` — the tensor is `[B, 4, H/8, W/8]`          |
| `IMAGE`        | a torch tensor, shape `[B, H, W, C]`, float32 in range `0.0–1.0`, channels-last  |
| `MASK`         | a torch tensor, shape `[B, H, W]`, float32 `0.0–1.0`                              |

Note the two easy-to-trip-on conventions:

- **Latents are dicts**, not bare tensors. The tensor lives under `["samples"]`.
- **Images are `[B, H, W, C]`** (batch, height, width, channel) and `0..1` — *not* the
  `[B, C, H, W]`, `0..255` that most other libraries use. Convert at the boundary.

---

## 2. Bootstrapping the environment

Because ComfyUI was written as an app, a few globals must be set up before you import
`nodes`. The order matters.

### 2a. The `sys.argv` gotcha (important)

`comfy/cli_args.py` runs at import time. It only parses the real command line **if**
`comfy.options.enable_args_parsing()` was called first (that's what `main.py` does):

```python
# comfy/cli_args.py
if comfy.options.args_parsing:
    args = parser.parse_args()     # parses YOUR script's sys.argv -> can crash
else:
    args = parser.parse_args([])   # uses defaults -> safe for a library
```

So **as a library, do NOT call `enable_args_parsing()`.** If you skip it, ComfyUI parses
an empty argument list and every setting falls back to its default. You then override
what you need programmatically (see below). This keeps your own script's CLI flags from
colliding with ComfyUI's.

If you *do* want to reuse ComfyUI's flags (`--gpu-only`, `--lowvram`, etc.), you can set
the values on the `args` object manually after import instead:

```python
from comfy.cli_args import args
args.lowvram = True          # example; set before models are loaded
```

### 2b. Where models are found

`folder_paths.py` builds the model search paths at import time, defaulting to
`<ComfyUI dir>/models/checkpoints`, `.../loras`, `.../vae`, `.../text_encoders`, etc.

You have three ways to point it at your models:

```python
import folder_paths

# Option 1: use the default ./models tree (put files there, nothing to do)

# Option 2: add an extra directory for a category
folder_paths.add_model_folder_path("checkpoints", "/data/models/checkpoints")

# Option 3: relocate the whole base dir (do this BEFORE importing nodes if possible)
# via args.base_directory, or load an extra_model_paths.yaml:
import utils.extra_config
utils.extra_config.load_extra_path_config("extra_model_paths.yaml")
```

Set your **output** directory (where `SaveImage` writes) with
`folder_paths.set_output_directory("/data/out")`, and the **input** dir (where
`LoadImage` looks) with `folder_paths.set_input_directory("/data/in")`.

### 2c. Minimal bootstrap block

Put this at the top of your script, run it once per process:

```python
import os, sys

# Make sure the ComfyUI repo root is importable
COMFY_DIR = os.path.dirname(os.path.abspath(__file__))  # adjust as needed
sys.path.insert(0, COMFY_DIR)

# NOTE: we deliberately do NOT call comfy.options.enable_args_parsing()
# so ComfyUI uses default args and ignores our script's argv.

import folder_paths
# folder_paths.add_model_folder_path("checkpoints", "/data/models/checkpoints")
# folder_paths.set_output_directory("/data/out")

# Importing nodes triggers torch import + node registration. Do it once.
import nodes
import comfy.model_management
```

`import nodes` also builds `nodes.NODE_CLASS_MAPPINGS`, the dict of every built-in node
class keyed by its `class_type` string (`"KSampler"`, `"CheckpointLoaderSimple"`, ...).

---

## 3. A complete text-to-image script

This mirrors the default ComfyUI workflow, but as plain Python. It loads a checkpoint,
encodes two prompts, samples, decodes, and saves — reusing the loaded models on every
call.

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image

import folder_paths
import nodes
import comfy.model_management


# ---- load once, reuse many times ----
CheckpointLoaderSimple = nodes.CheckpointLoaderSimple
CLIPTextEncode         = nodes.CLIPTextEncode
EmptyLatentImage       = nodes.EmptyLatentImage
KSampler               = nodes.KSampler
VAEDecode              = nodes.VAEDecode

# load_checkpoint returns a tuple (MODEL, CLIP, VAE)
model, clip, vae = CheckpointLoaderSimple().load_checkpoint("v1-5-pruned-emaonly.safetensors")


def generate(positive_prompt, negative_prompt="", width=512, height=512,
             steps=20, cfg=8.0, seed=0, sampler="euler", scheduler="normal",
             batch_size=1):
    # 1. encode prompts -> CONDITIONING
    (positive,) = CLIPTextEncode().encode(clip, positive_prompt)
    (negative,) = CLIPTextEncode().encode(clip, negative_prompt)

    # 2. empty latent canvas -> LATENT dict
    (latent,) = EmptyLatentImage().generate(width, height, batch_size)

    # 3. sample -> denoised LATENT
    (samples,) = KSampler().sample(
        model=model, seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=latent, denoise=1.0,
    )

    # 4. decode LATENT -> IMAGE tensor [B, H, W, C], 0..1
    (images,) = VAEDecode().decode(vae, samples)
    return images


def tensor_to_pil(images):
    """Convert a ComfyUI IMAGE batch tensor into a list of PIL images."""
    out = []
    for img in images:                       # img: [H, W, C], 0..1
        arr = (255.0 * img.cpu().numpy()).clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


if __name__ == "__main__":
    imgs = generate("a photograph of an astronaut riding a horse",
                    negative_prompt="blurry, low quality",
                    seed=42)
    for i, pil in enumerate(tensor_to_pil(imgs)):
        pil.save(f"out_{i}.png")
    print("done")
```

Valid `sampler` / `scheduler` strings come from
`comfy.samplers.KSampler.SAMPLERS` and `comfy.samplers.KSampler.SCHEDULERS`:

```python
import comfy.samplers
print(comfy.samplers.KSampler.SAMPLERS)     # ['euler', 'euler_ancestral', 'dpmpp_2m', ...]
print(comfy.samplers.KSampler.SCHEDULERS)   # ['normal', 'karras', 'exponential', ...]
```

### Why not just use `SaveImage`?

You *can* (`nodes.SaveImage().save_images(images, "prefix")`) and it writes PNGs with the
workflow embedded to the output dir. But for a library it's usually cleaner to convert
the tensor yourself (as above) so you control where bytes go — e.g. straight into an
HTTP response.

---

## 4. Image-to-image and input images

For img2img, encode a real image into a latent instead of using an empty one, and lower
`denoise`.

```python
def load_image_as_tensor(path):
    """Return an IMAGE tensor [1, H, W, C], float 0..1 (ComfyUI convention)."""
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ]     # add batch dim

def img2img(clip, model, vae, init_image_path, prompt,
            denoise=0.6, steps=20, cfg=8.0, seed=0):
    pixels = load_image_as_tensor(init_image_path)          # IMAGE
    (latent,) = nodes.VAEEncode().encode(vae, pixels)       # -> LATENT
    (positive,) = nodes.CLIPTextEncode().encode(clip, prompt)
    (negative,) = nodes.CLIPTextEncode().encode(clip, "")
    (samples,) = nodes.KSampler().sample(
        model=model, seed=seed, steps=steps, cfg=cfg,
        sampler_name="euler", scheduler="normal",
        positive=positive, negative=negative,
        latent_image=latent, denoise=denoise,
    )
    (images,) = nodes.VAEDecode().decode(vae, samples)
    return images
```

`nodes.LoadImage` also exists, but it only reads from the configured input directory and
does extra alpha/EXIF handling — for arbitrary paths the tiny helper above is simpler.
For inpainting see `nodes.VAEEncodeForInpaint` (takes a `MASK`) and `SetLatentNoiseMask`.

---

## 5. Calling arbitrary nodes generically

You don't need to import each node class by name. Every built-in (and, after custom-node
init, every custom) node is in `nodes.NODE_CLASS_MAPPINGS`:

```python
def run_node(class_type, **inputs):
    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    return fn(**inputs)          # returns the output tuple

(latent,) = run_node("EmptyLatentImage", width=768, height=768, batch_size=1)
```

This is the seam you'll build a data-driven pipeline on: a small JSON/dict description of
`{class_type, inputs}` mapped through `run_node`, wiring outputs to later inputs. That's
essentially a hand-rolled version of what `execution.py` does for the HTTP API — start
simple with linear wiring and only reach for the full graph executor if you need caching,
partial re-execution, or `OUTPUT_NODE` semantics.

To introspect a node's expected inputs at runtime:

```python
cls = nodes.NODE_CLASS_MAPPINGS["KSampler"]
print(cls.INPUT_TYPES())     # dict of required/optional inputs + defaults
print(cls.RETURN_TYPES)      # ('LATENT',)
print(cls.FUNCTION)          # 'sample'
```

---

## 6. Loading custom nodes

Built-in nodes are available as soon as you `import nodes`. Custom nodes in
`custom_nodes/` are **not** — they're registered by an async initializer. If you need
them:

```python
import asyncio
import nodes

asyncio.run(nodes.init_extra_nodes(
    init_custom_nodes=True,
    init_api_nodes=False,     # True also loads the cloud/API nodes
))

# now NODE_CLASS_MAPPINGS includes custom nodes too
```

Some custom nodes assume a running `PromptServer`; those may misbehave headless. Test the
specific packs you depend on.

---

## 7. Memory management (matters for a long-lived service)

ComfyUI manages GPU memory itself. A few useful handles:

```python
import comfy.model_management as mm

mm.unload_all_models()          # evict models from VRAM (they reload on next use)
mm.soft_empty_cache()           # torch.cuda.empty_cache() + housekeeping
mm.interrupt_current_processing(True)   # cooperatively abort an in-flight sample()
```

Guidance for an on-demand server:

- **Keep `model/clip/vae` objects resident** between requests. Loading a checkpoint is the
  slow part; sampling is fast. Load once at startup (or lazily + cache by checkpoint name).
- ComfyUI moves weights on/off the GPU as needed during `sample()`, so holding many
  checkpoints in Python is fine — only the active one occupies VRAM.
- Sampling is **not** re-entrant on a single GPU. Serialize `generate()` calls behind a
  lock/queue; don't run two samples concurrently on the same device.
- Progress previews and the sampler progress bar are enabled by default via
  `latent_preview` / `comfy.utils`. Headless they're harmless, but you can disable preview
  work with `args.preview_method = comfy.cli_args.LatentPreviewMethod.NoPreviews`.

---

## 8. Toward the HTTP pipeline (your end goal)

Once `generate()` works in-process, exposing it is straightforward. Sketch:

```python
# server.py  (pseudo-code)
from fastapi import FastAPI
from fastapi.responses import Response
import io, threading

app = FastAPI()
gpu_lock = threading.Lock()      # one sample at a time per GPU

@app.post("/generate")
def generate_endpoint(req: dict):
    with gpu_lock:
        images = generate(
            positive_prompt=req["prompt"],
            negative_prompt=req.get("negative", ""),
            width=req.get("width", 512),
            height=req.get("height", 512),
            seed=req.get("seed", 0),
        )
    pil = tensor_to_pil(images)[0]
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
```

Design notes for "prebuilt pipeline + prompt + optional input image":

- Represent each named pipeline as a Python function (or a small list of
  `{class_type, inputs}` steps run through `run_node` from §5). The POST body selects the
  pipeline by name and supplies the variable bits (prompt, seed, input image bytes).
- Decode an uploaded image with the `load_image_as_tensor` helper (§4) — accept raw bytes
  and `Image.open(io.BytesIO(data))` instead of a path.
- Cache loaded checkpoints in a dict keyed by name; evict with `unload_all_models()` under
  memory pressure.
- Run the FastAPI app single-worker (or with an external queue) so the GPU lock actually
  serializes work; multiple Uvicorn workers each get their own model copies.

---

## Quick reference — the built-in nodes used above

| Node                     | Method (`FUNCTION`) | Signature (key args)                                                        | Returns          |
|--------------------------|---------------------|-----------------------------------------------------------------------------|------------------|
| `CheckpointLoaderSimple` | `load_checkpoint`   | `ckpt_name`                                                                 | `(MODEL,CLIP,VAE)`|
| `CLIPTextEncode`         | `encode`            | `clip, text`                                                                | `(CONDITIONING,)`|
| `EmptyLatentImage`       | `generate`          | `width, height, batch_size=1`                                               | `(LATENT,)`      |
| `VAEEncode`              | `encode`            | `vae, pixels`                                                               | `(LATENT,)`      |
| `KSampler`               | `sample`            | `model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise` | `(LATENT,)` |
| `VAEDecode`              | `decode`            | `vae, samples`                                                              | `(IMAGE,)`       |
| `SaveImage`              | `save_images`       | `images, filename_prefix="ComfyUI"`                                         | writes PNGs      |
| `LoadImage`              | `load_image`        | `image` (filename in input dir)                                             | `(IMAGE, MASK)`  |

> Every node method returns a **tuple**, even for a single output — hence the
> `(positive,) = ...` unpacking throughout. Match arguments by keyword; the positional
> order of a node's `FUNCTION` method does not always match its `INPUT_TYPES` order.
