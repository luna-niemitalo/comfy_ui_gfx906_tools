"""Resolve a recipe (sparse diff) into a full node graph and run it.

Node dispatch is generic (nodes.NODE_CLASS_MAPPINGS[block["node"]]), so a
new model family is a new file under develop/recipes/defaults/, not new
code — resolve() loads DEFAULTS_DIR/<kind>.yaml by filename stem. The
graph shape itself — loaders -> conditioning -> latent -> sample -> decode
— is fixed, since every recipe so far fits it; see develop/DESIGN.md for
when that would need to change.
"""
import copy
import os

import torch
import yaml

import nodes

DEFAULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "recipes", "defaults")


def load_defaults(kind):
    path = os.path.join(DEFAULTS_DIR, f"{kind}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base, overlay):
    """Merge overlay onto base; dicts merge key-by-key, other values replace."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_nested(items):
    """(dotted.path, value) pairs -> nested dict, for merging axis/--set
    overrides through the same deep_merge() the recipe itself goes through."""
    result = {}
    for dotted_path, value in items:
        node = result
        *parents, leaf = dotted_path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return result


def resolve(recipe, cli_overrides=None):
    """Merge a sparse recipe onto develop/recipes/defaults/<kind>.yaml, then
    onto that the CLI/sweep-axis overrides (same merge, applied one more
    time). The `prompt` shorthand — the field nearly every run changes — is
    resolved last, writing into conditioning.positive.text."""
    kind = recipe["kind"]
    resolved = deep_merge(load_defaults(kind), recipe)
    resolved.pop("kind", None)

    if cli_overrides:
        resolved = deep_merge(resolved, cli_overrides)

    prompt = resolved.pop("prompt", None)
    if prompt is not None:
        resolved["conditioning"]["positive"]["text"] = prompt

    return resolved


def call_node(class_type, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    instance = cls()
    fn = getattr(instance, cls.FUNCTION)
    return fn(**kwargs)


def _kwargs(block):
    return {k: v for k, v in block.items() if k != "node"}


# Socket-type kwarg name a chained step (e.g. Gfx906Int4Transcode) takes the
# previous step's output as, by ComfyUI's own universal naming convention
# for these three socket types (MODEL/CLIP/VAE always the arg named this).
_CHAIN_KWARG = {"unet": "model", "clip": "clip", "vae": "vae"}


def _build_loader_slot(slot_key, spec):
    """A loader slot is either one node block (the common case) or a list
    of blocks (a chain, e.g. CLIPLoaderMultiGPU -> Gfx906Int4TranscodeCLIP)
    -- each step after the first takes the previous step's output as its
    MODEL/CLIP/VAE-typed kwarg, everything else in the block is passed
    through as-is."""
    steps = spec if isinstance(spec, list) else [spec]
    (result,) = call_node(steps[0]["node"], **_kwargs(steps[0]))
    for step in steps[1:]:
        kwargs = _kwargs(step)
        kwargs[_CHAIN_KWARG[slot_key]] = result
        (result,) = call_node(step["node"], **kwargs)
    return result


def build_loaders(resolved):
    loaders = resolved["loaders"]
    model = _build_loader_slot("unet", loaders["unet"])
    clip = _build_loader_slot("clip", loaders["clip"])
    vae = _build_loader_slot("vae", loaders["vae"])
    return model, clip, vae


def run_with_loaders(resolved, model, clip, vae):
    """Everything after the loaders: conditioning, latent, sample, decode.
    Split out from run() so sweeps can reuse (model, clip, vae) across
    combinations that don't touch loaders.*."""
    cond = resolved["conditioning"]
    results = {}
    for name in ("positive", "negative"):
        block = cond[name]
        kwargs = {k: v for k, v in block.items() if k not in ("node", "source")}
        source = block.get("source")
        if source is not None:
            kwargs["conditioning"] = results[source][0]
        else:
            kwargs["clip"] = clip
        results[name] = call_node(block["node"], **kwargs)

    (latent,) = call_node(resolved["latent"]["node"], **_kwargs(resolved["latent"]))

    sampler_kwargs = _kwargs(resolved["sampler"])
    sampler_kwargs.update(model=model, positive=results["positive"][0],
                            negative=results["negative"][0], latent_image=latent)
    (samples,) = call_node(resolved["sampler"]["node"], **sampler_kwargs)

    decode_kwargs = _kwargs(resolved["vae_decode"])
    decode_kwargs.update(samples=samples, vae=vae)
    (images,) = call_node(resolved["vae_decode"]["node"], **decode_kwargs)

    return images


def run(resolved):
    """One-shot: build loaders fresh and run. Sweeps use build_loaders() /
    run_with_loaders() directly instead, to reuse loaded weights.

    execution.py wraps a whole prompt's node calls in one
    torch.inference_mode() block (see execution.py's PromptExecutor); model
    code and comfy.ops rely on that (some tensors are only cloned when NOT
    in inference mode, e.g. comfy/utils.py:980). Calling nodes directly
    outside that block, as this harness does, produces the same "inplace
    update to inference tensor" crash the real app never hits — so
    replicate the same scope here."""
    with torch.inference_mode():
        model, clip, vae = build_loaders(resolved)
        return run_with_loaders(resolved, model, clip, vae)
