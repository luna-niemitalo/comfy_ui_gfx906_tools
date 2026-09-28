"""Bootstrap the in-process ComfyUI environment for headless generation.

Mirrors develop/PythonLibraryGuide.md: no server, no queue, direct node
calls, models stay resident in this process. Import order below is
load-bearing — comfy.model_management snapshots args.* into module globals
at its own import time, so vram/determinism flags must be set on
comfy.cli_args.args before nodes/comfy.sd (which pull it in) are imported.
See develop/DESIGN.md.
"""
import asyncio
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import comfy.cli_args  # noqa: E402  (args_parsing defaults False -> safe to import as a library)

args = comfy.cli_args.args
args.gpu_only = True
args.disable_smart_memory = True
args.disable_dynamic_vram = True
args.deterministic = True

import folder_paths  # noqa: E402

_extra_model_paths = os.path.join(REPO_ROOT, "extra_model_paths.yaml")
if os.path.isfile(_extra_model_paths):
    import utils.extra_config
    utils.extra_config.load_extra_path_config(_extra_model_paths)

import nodes  # noqa: E402  (registers built-in NODE_CLASS_MAPPINGS)

import server  # noqa: E402
if not hasattr(server.PromptServer, "instance"):
    # ComfyUI-MultiGPU's memory-management hooks read server.PromptServer.instance
    # unconditionally (guarding only against it being None, never against the
    # attribute not existing at all) — it's only ever set inside __init__, which
    # only runs for a real running server. PythonLibraryGuide.md's own warning
    # ("some custom nodes assume a running PromptServer") applies here; this is
    # the minimal headless shim, not a PromptServer instance.
    server.PromptServer.instance = None

_custom_nodes_loaded = False


def ensure_custom_nodes():
    """Load custom_nodes/ once. The moodyKrea recipe's loaders (MultiGPU,
    GGUF, Krea2 text encoder type) all live there, unlike the plain
    CheckpointLoaderSimple path — this fork's fidelity to the real workflow
    matters more here than the speed cost the perf harness (TODO.md) avoids
    it for."""
    global _custom_nodes_loaded
    if _custom_nodes_loaded:
        return
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    _custom_nodes_loaded = True
