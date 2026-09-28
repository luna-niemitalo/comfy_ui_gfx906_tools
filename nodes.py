"""
gfx906 int4 quantization test patch -- MODEL -> MODEL.

Wire this between a normal loader (UNETLoader) and KSampler to test int4
weight-only (W4A16) or int4 weight+activation (W4A4) quantization on real
prompts in the real pipeline, instead of synthetic noise in a script.

Quantization: symmetric int4, block=32, naive round-to-nearest, matching
ggml Q4_0's value convention. Not yet using the gfx906 sdot4/sdot8 hardware
path -- this dequantizes to the model's working dtype before the normal
matmul, same as the existing GGUF loader's compute path. It exists to
measure accuracy/quality on real inputs before any kernel work is spent.
"""
import json
import re
import torch

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.ops

from .ops import quantize_pack_int4, repack_q4_0_native, choose_rotation_group, apply_group_rotation, get_group_rotation, Gfx906Int4Ops
from .row_split import Gfx906RowSplitKrea2

try:
    from comfy_kitchen.tensor.convrot_w4a4 import dequantize_convrot_w4a4_weight
except ImportError:
    dequantize_convrot_w4a4_weight = None


BLOCK = 32
DEFAULT_SPARED_SUFFIXES = ("x_embedder", "final_layer", "cap_embedder", "t_embedder")


def quantize_dequantize_int4(t: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    orig_shape = t.shape
    K = orig_shape[-1]
    if K % block != 0:
        return t
    tb = t.reshape(-1, K // block, block).float()
    amax = tb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 7.0
    q = torch.round(tb / scale).clamp(-8, 7)
    return (q * scale).reshape(orig_shape).to(t.dtype)


def find_max_layer_index(diffusion_model, prefix="layers."):
    max_idx = -1
    for name, _ in diffusion_model.named_modules():
        m = re.match(re.escape(prefix) + r"(\d+)\.", name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def make_activation_quant_hook():
    def hook(module, inputs):
        x = inputs[0]
        return (quantize_dequantize_int4(x),) + inputs[1:]
    return hook


class Gfx906Int4Patch:
    """Quantize a loaded diffusion model's Linear weights (and optionally
    activations) to int4 for testing on gfx906-class GPUs. Returns a
    patched MODEL; connect straight into KSampler."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["w4a16", "w4a4_naive"], {"default": "w4a16"}),
                "spare_first_last_layers": ("BOOLEAN", {"default": True}),
                "block_size": ("INT", {"default": 32, "min": 16, "max": 256, "step": 16}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Int4 Quantize (test)"

    def patch(self, model, mode, spare_first_last_layers, block_size):
        model = model.clone()
        diffusion_model = model.model.diffusion_model

        spared_prefixes = tuple()
        if spare_first_last_layers:
            max_idx = find_max_layer_index(diffusion_model)
            spared_prefixes = DEFAULT_SPARED_SUFFIXES
            if max_idx >= 0:
                spared_prefixes = spared_prefixes + (f"layers.0.", f"layers.{max_idx}.")

        quantized, spared, hooked = 0, 0, 0
        named = dict(diffusion_model.named_modules())
        with torch.no_grad():
            for name, param in diffusion_model.named_parameters():
                if not name.endswith(".weight") or param.ndim != 2:
                    continue
                if param.shape[1] % block_size != 0 or param.shape[1] < block_size:
                    continue
                mod_name = name[: -len(".weight")]
                if mod_name.startswith(spared_prefixes):
                    spared += 1
                    continue

                param.data.copy_(quantize_dequantize_int4(param.data.float(), block_size).to(param.dtype))
                quantized += 1

                if mode == "w4a4_naive" and mod_name in named:
                    named[mod_name].register_forward_pre_hook(make_activation_quant_hook())
                    hooked += 1

        print(f"[Gfx906Int4Patch] mode={mode} quantized={quantized} spared={spared} activation_hooks={hooked}")
        return (model,)


def _get_gguf_dequant_module():
    """Lazy, defensive lookup of ComfyUI-GGUF's dequant helpers via
    sys.modules, keyed by ComfyUI's literal custom_nodes folder name
    (dashes included -- not a normal dotted import). Returns None if
    ComfyUI-GGUF isn't loaded; callers must handle that."""
    import sys
    return sys.modules.get("ComfyUI-GGUF.dequant")


def _get_gguf_loader_module():
    """Same lazy lookup as _get_gguf_dequant_module, for ComfyUI-GGUF's
    gguf_sd_loader -- lets Gfx906Int4UnetLoader read a raw .gguf file
    itself (GGML-tensor-type-tagged state dict) instead of only accepting
    plain safetensors, without duplicating GGUF's own file parsing."""
    import sys
    return sys.modules.get("ComfyUI-GGUF.loader")


class _RawTensorModule:
    """Minimal module-shaped wrapper so a raw state-dict tensor (as read by
    Gfx906Int4UnetLoader, before any model is constructed) can flow through
    the same _quantize_module_weight dispatch Gfx906Int4Transcode uses for
    an already-built nn.Linear -- one format-detection path for both nodes
    instead of two that can drift apart."""
    def __init__(self, weight, out_features, in_features):
        self.weight = weight
        self.out_features = out_features
        self.in_features = in_features


def _quantize_module_weight(module, block_size, rotation_group=0):
    """Returns (packed_i8, scale_f32, source_tag) for ANY Linear-like
    module, regardless of source format.

    Q4_0 gets a lossless bit-repack (same value range -8..7, same block
    size 32 as our format -- no dequant/requant round trip, no additional
    quantization error versus the source .gguf file). Every other format
    (plain fp16/bf16, any other GGUF quant type, or a comfy-native
    QuantizedTensor -- fp8/int8/nvfp4/etc. from comfy_kitchen, e.g. the
    on-disk F8_E4M3+U8+weight_scale format some Krea 2 checkpoints use)
    goes through dequantize-to-float32 then our normal RTN quantizer.

    rotation_group != 0 forces the dequant path even for Q4_0 -- folding a
    Hadamard rotation into the weight needs floating point, so it costs
    the lossless bit-repack opportunity. Folding once here means the
    matching activation rotation at inference (Gfx906Int4Ops.Linear) is
    the only per-forward cost; see ops.py's get_group_rotation."""
    w = module.weight
    tensor_type = getattr(w, "tensor_type", None)

    if tensor_type is not None and not rotation_group and block_size == 32:
        try:
            import gguf
            if tensor_type == gguf.GGMLQuantizationType.Q4_0:
                packed, scale = repack_q4_0_native(w.data, module.out_features, module.in_features, block_size)
                return packed, scale, "gguf_q4_0_repack"
        except ImportError:
            pass

    if tensor_type is not None:
        dequant_mod = _get_gguf_dequant_module()
        if dequant_mod is None:
            raise RuntimeError(
                "This model has GGUF-quantized weights but ComfyUI-GGUF is not loaded "
                "(needed to dequantize them before re-quantizing to gfx906 int4)."
            )
        w_float, source_tag = dequant_mod.dequantize_tensor(w, dtype=torch.float32), "gguf_dequant_requant"
    else:
        try:
            from comfy_kitchen.tensor import QuantizedTensor
        except ImportError:
            QuantizedTensor = None
        if QuantizedTensor is not None and isinstance(w, QuantizedTensor):
            # Some QuantizedTensor layouts (ConvRot) do a real rotation-undo
            # matmul inside dequantize() -- if the weight is still
            # CPU-resident here (model not yet moved to its load device),
            # that hits the same broken CPU-matmul segfault apply_group_
            # rotation works around in ops.py, just inside comfy_kitchen's
            # own code where we can't patch the call site directly.
            w_for_dequant = w.cuda() if w.device.type == "cpu" and torch.cuda.is_available() else w
            w_float, source_tag = w_for_dequant.dequantize().float(), "comfy_quantized_dequant_requant"
        else:
            if w.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"weight is already quantized ({w.dtype}), not a plain float weight, GGUF "
                    f"tensor, or recognized QuantizedTensor -- this looks like a third-party "
                    f"quantization format this transcoder doesn't know how to decode."
                )
            w_float, source_tag = w.detach().float(), "plain"

    if rotation_group:
        R = get_group_rotation(module.in_features, rotation_group, w_float.device)
        w_float = apply_group_rotation(w_float, R, rotation_group)
        source_tag += "_rotated"

    packed, scale = quantize_pack_int4(w_float, block_size)
    return packed, scale, source_tag


def _transcode_linears(root_module, block_size, rotate):
    """Walks root_module's Linear layers and replaces them in place with
    the gfx906 native int4 kernel Linear. Shared by the diffusion-model
    and CLIP text-encoder transcode nodes -- same weight formats, same
    kernel, only the root module differs.

    Also catches comfy core's own native mixed-precision Linear
    (comfy.ops.MixedPrecisionOp -- plain fp8/int8/nvfp4/convrot_w4a4/
    asym_w4a8_int8 checkpoints comfy now loads without GGUF or any custom
    node at all). It's a torch.nn.Module, not a torch.nn.Linear subclass,
    so it was previously invisible here -- a real checkpoint would then
    run entirely through comfy's own native dispatch with this node
    silently converting nothing (converted=0), and on gfx906 that native
    path has at least one broken format (asym_w4a8_int8, produced NaN in
    real testing). Its .weight is already a real QuantizedTensor once
    loaded, so it flows through the exact same _quantize_module_weight
    dispatch as GGUF/ConvRot with no extra code -- MoE expert layers
    (weight isn't 2D) are the one shape this can't handle and are skipped,
    same as any other layer whose shape doesn't fit."""
    named = dict(root_module.named_modules())

    converted, skipped = 0, 0
    source_counts = {}
    with torch.no_grad():
        for name, module in list(named.items()):
            is_plain_linear = isinstance(module, torch.nn.Linear)
            is_mixed_precision = isinstance(module, comfy.ops.MixedPrecisionOp) and getattr(module, "weight", None) is not None
            if not (is_plain_linear or is_mixed_precision):
                continue
            if getattr(module.weight, "ndim", 2) != 2:
                skipped += 1  # e.g. MoE expert weights (leading expert dim) -- not a plain 2D Linear
                continue
            in_f, out_f = module.in_features, module.out_features
            if in_f % block_size != 0 or in_f < block_size:
                skipped += 1
                continue

            rotation_group = choose_rotation_group(in_f) if rotate else 0
            device = module.weight.device
            out_dtype = module.weight.dtype
            packed, scale, source_tag = _quantize_module_weight(module, block_size, rotation_group)
            packed, scale = packed.to(device), scale.to(device)
            source_counts[source_tag] = source_counts.get(source_tag, 0) + 1

            new_linear = Gfx906Int4Ops.Linear(in_f, out_f, bias=module.bias is not None, device=device, dtype=out_dtype, rotation_group=rotation_group)
            new_linear.weight_q4.copy_(packed)
            new_linear.weight_scale.copy_(scale)
            if module.bias is not None:
                new_linear.bias.data.copy_(module.bias.detach().float().to(out_dtype))

            parent_name, _, attr = name.rpartition(".")
            parent = named[parent_name] if parent_name else root_module
            setattr(parent, attr, new_linear)
            converted += 1

    # The replaced full-precision weights stay reserved in torch's caching
    # allocator (Qwen3-VL-4B: 11 GB reserved vs 3.1 GB live) -- hand them back.
    # `named` and the loop variable still reference the old modules; drop them
    # first or nothing is free yet.
    del named, module
    torch.cuda.empty_cache()
    return converted, skipped, source_counts


class Gfx906Int4Transcode:
    """MODEL -> MODEL. Walks any already-loaded model's Linear layers and
    replaces them in place with the gfx906 native int4 (v_dot8_i32_i4)
    kernel Linear -- regardless of what format they arrived in: plain
    fp16/bf16, or GGUF-quantized (Q4_K, Q8_0, etc., dequantized here first
    then re-quantized to our native two's-complement int4 format).

    Q4_0-quantized weights get a lossless bit-repack (no dequant/requant
    round trip, no additional error versus the source .gguf file) since
    Q4_0 already shares our value range and block size. Every other
    format (plain fp16/bf16, or any other GGUF quant type such as Q4_K or
    Q8_0) is dequantized then re-quantized with our normal RTN scheme.

    Works after ANY upstream loader, including MultiGPU-wrapped ones --
    device is inherited from wherever the model already lives; this node
    has no device logic of its own, so there's nothing to duplicate per
    source format or per multi-GPU wrapper the way a dedicated per-format
    loader would need."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "activation_rotation": ("BOOLEAN", {"default": False, "tooltip": "Randomized Hadamard rotation of weight+activation before quantizing, per SpinQuant/QuIP#. Spreads outlier energy without needing calibration data; costs the lossless Q4_0 repack path where it would otherwise apply."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "transcode"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Int4 Transcode (native kernel, any source)"

    def transcode(self, model, activation_rotation):
        model = model.clone()
        converted, skipped, source_counts = _transcode_linears(model.model.diffusion_model, BLOCK, activation_rotation)
        print(f"[Gfx906Int4Transcode] converted={converted} skipped={skipped} sources={source_counts}")
        return (model,)


class Gfx906Int4TranscodeCLIP:
    """CLIP -> CLIP. Same native int4 kernel and format handling as
    Gfx906Int4Transcode, applied to a loaded text encoder's Linear layers
    instead of a diffusion model's. Wire it between CLIPLoader and the
    conditioning nodes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "activation_rotation": ("BOOLEAN", {"default": False, "tooltip": "Randomized Hadamard rotation of weight+activation before quantizing, per SpinQuant/QuIP#. Spreads outlier energy without needing calibration data; costs the lossless Q4_0 repack path where it would otherwise apply."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "transcode"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Int4 Transcode CLIP (native kernel, any source)"

    def transcode(self, clip, activation_rotation):
        clip = clip.clone()
        converted, skipped, source_counts = _transcode_linears(clip.cond_stage_model, BLOCK, activation_rotation)
        print(f"[Gfx906Int4TranscodeCLIP] converted={converted} skipped={skipped} sources={source_counts}")
        return (clip,)


def _device_list():
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return devices


def _default_device(devices):
    # Same convention ComfyUI-MultiGPU's override_class uses: default away
    # from device 0 when a second GPU is present, since that's usually the
    # one already carrying the desktop/other load.
    non_cpu = [d for d in devices if d != "cpu"]
    return non_cpu[1] if len(non_cpu) > 1 else (non_cpu[0] if non_cpu else "cpu")


def _parse_convrot_layers(metadata, prefix):
    """Some Krea2 checkpoints ship pre-quantized with comfy_kitchen's own
    ConvRot W4A4 format (int8-packed nibbles + per-output-channel scale
    under the plain ".weight"/".weight_scale" key names) and embed a
    `_quantization_metadata` JSON blob documenting exactly that, per layer
    (format/convrot_groupsize/quant_group_size). Returns {stripped_key:
    cfg} so those layers can be dequantized with their own real decoder
    (comfy_kitchen.dequantize_convrot_w4a4_weight) before being folded into
    our native int4 -- lets a file with no plain bf16 source (e.g. the
    Muse checkpoint) be used at all, and skips the 26GB bf16 load for ones
    that do have one."""
    if not metadata:
        return {}
    raw = metadata.get("_quantization_metadata")
    if not raw:
        return {}
    layers = json.loads(raw).get("layers", {})
    out = {}
    for key, cfg in layers.items():
        if cfg.get("format") != "convrot_w4a4":
            continue
        stripped = key[len(prefix):] if prefix and key.startswith(prefix) else key
        out[stripped] = cfg
    return out


def _unet_filename_list():
    names = list(folder_paths.get_filename_list("diffusion_models"))
    if "unet_gguf" in folder_paths.folder_names_and_paths:
        names += [n for n in folder_paths.get_filename_list("unet_gguf") if n not in names]
    return names


class Gfx906Int4UnetLoader:
    """Loads a diffusion model directly into gfx906 native int4 (v_dot8_i32_i4)
    Linear layers -- weights are quantized at load time and never
    materialized as full-precision tensors, unlike Gfx906Int4Patch. This is
    the real VRAM-saving, kernel-backed path; Gfx906Int4Patch remains useful
    for quick accuracy experiments (dequant-then-fp16-matmul, no kernel).

    Accepts plain fp16/bf16 safetensors, .gguf (read directly via
    ComfyUI-GGUF's own gguf_sd_loader -- never materialized as a full model
    first), and checkpoints pre-quantized with comfy_kitchen's ConvRot W4A4
    format (detected via embedded checkpoint metadata). All three funnel
    through the same _quantize_module_weight dispatch Gfx906Int4Transcode
    uses, via the _RawTensorModule adapter, so format support only needs
    to be taught once.

    Limitation: quantizes every eligible Linear weight uniformly. Comfy's
    custom_operations mechanism builds the whole model with one Ops class,
    so there's no clean per-layer opt-out the way Gfx906Int4Patch's
    spare_first_last_layers has -- sparing specific layers here would need
    a per-tensor quant_config, which this node doesn't implement."""

    @classmethod
    def INPUT_TYPES(s):
        devices = _device_list()
        return {
            "required": {
                "unet_name": (_unet_filename_list(),),
                "device": (devices, {"default": _default_device(devices)}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "gfx906"
    TITLE = "Gfx906 Int4 UNet Loader (native kernel)"

    def load_unet(self, unet_name, device):
        block_size = BLOCK
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        torch_device = torch.device(device)

        if unet_path.endswith(".gguf"):
            gguf_loader = _get_gguf_loader_module()
            if gguf_loader is None:
                raise RuntimeError("Loading a .gguf model requires ComfyUI-GGUF to be loaded (for gguf_sd_loader).")
            sd, _extra = gguf_loader.gguf_sd_loader(unet_path)  # already strips its own prefix internally
            metadata = None
            diffusion_model_prefix = ""
        else:
            sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
            # Strip any "diffusion_model." (or similar) prefix BEFORE quantizing,
            # so the quantized state dict's keys stay aligned with the
            # unquantized one used for architecture detection below.
            diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
            temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
            if len(temp_sd) > 0:
                sd = temp_sd

        convrot_layers = _parse_convrot_layers(metadata, diffusion_model_prefix) if metadata else {}
        # The metadata's own per-layer key list can drift from the actual
        # tensor names (seen in the wild: metadata names some txtfusion
        # layers "layerwise_blocks", the real tensors are "refiner_blocks")
        # -- every layer in a given file uses the same convrot_groupsize/
        # quant_group_size in practice, so fall back to that shared config
        # for any int8 2D weight with a matching per-channel scale sidecar
        # that the name-based lookup above missed, rather than refusing a
        # tensor that's clearly the same format under a renamed key.
        default_convrot_cfg = next(iter(convrot_layers.values()), None)

        # Resolve every convrot layer in one upfront pass, before touching
        # quantized_sd. A layer's ".weight_scale" is a real, separate key in
        # sd (the file's own sidecar) -- if it were only skipped once its
        # ".weight" sibling had already been processed, an unlucky key
        # order (sidecar visited first, or as an update to a dict already
        # populated by ".weight") could let it fall through to plain
        # passthrough and silently overwrite the real computed scale with
        # the original one afterward.
        convrot_cfg_for_base = {}
        for key in sd:
            if not key.endswith(".weight"):
                continue
            base = key[: -len(".weight")]
            cfg = convrot_layers.get(base)
            if cfg is None and default_convrot_cfg is not None:
                t = sd[key]
                scale_key = base + ".weight_scale"
                if (t.dtype == torch.int8 and t.ndim == 2 and scale_key in sd
                        and sd[scale_key].dtype == torch.float32 and tuple(sd[scale_key].shape) == (t.shape[0],)):
                    cfg = default_convrot_cfg
            if cfg is not None:
                convrot_cfg_for_base[base] = cfg
        convrot_scale_keys = {base + ".weight_scale" for base in convrot_cfg_for_base}

        quantized_sd = {}
        quantized, passthrough, convrot_repacked = 0, 0, 0
        for key in list(sd.keys()):
            if key in convrot_scale_keys:
                continue  # consumed alongside its sibling ".weight" below
            t = sd[key]
            base = key[: -len(".weight")] if key.endswith(".weight") else None
            cfg = convrot_cfg_for_base.get(base) if base is not None else None

            if cfg is not None:
                if dequantize_convrot_w4a4_weight is None:
                    raise RuntimeError(
                        f"{key} is ConvRot-quantized but comfy_kitchen "
                        f"is not available to decode it."
                    )
                scale = sd[base + ".weight_scale"].to(torch_device)
                w_float = dequantize_convrot_w4a4_weight(
                    t.to(torch_device), scale,
                    convrot_groupsize=cfg["convrot_groupsize"], quant_group_size=cfg["quant_group_size"],
                    output_dtype=torch.float32,
                )
                out_f, in_f = w_float.shape
                fake = _RawTensorModule(w_float, out_f, in_f)
                packed, scale, source_tag = _quantize_module_weight(fake, block_size)
                quantized_sd[base + ".weight_q4"] = packed.to(torch_device).cpu()
                quantized_sd[base + ".weight_scale"] = scale.to(torch_device).cpu()
                convrot_repacked += 1
            elif base is not None and t.ndim == 2 and t.shape[1] % block_size == 0 and t.shape[1] >= block_size:
                fake = _RawTensorModule(t, t.shape[0], t.shape[1])
                packed, scale, source_tag = _quantize_module_weight(fake, block_size)
                quantized_sd[base + ".weight_q4"] = packed.to(torch_device).cpu()
                quantized_sd[base + ".weight_scale"] = scale.to(torch_device).cpu()
                quantized += 1
            else:
                quantized_sd[key] = t
                passthrough += 1
        print(f"[Gfx906Int4UnetLoader] quantized={quantized} convrot_repacked={convrot_repacked} passthrough_tensors={passthrough} device={device}")

        # Architecture detection needs the ORIGINAL (unquantized) state dict --
        # model_config_from_unet inspects specific "<name>.weight" keys by
        # shape directly, which quantized_sd no longer has under that name.
        model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        if model_config is None:
            raise RuntimeError(f"Could not detect model type of: {unet_path}")

        parameters = comfy.utils.calculate_parameters(sd)
        weight_dtype = comfy.utils.weight_dtype(sd)
        offload_device = comfy.model_management.unet_offload_device()
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, torch_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=torch_device)
        model_config.custom_operations = Gfx906Int4Ops

        model = model_config.get_model(quantized_sd, "")
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch_device, offload_device=offload_device, fast_disk=comfy.storage.state_dict_fast_disk(quantized_sd))
        if not comfy.model_management.is_device_cpu(offload_device):
            model.to(offload_device)
        model.load_model_weights(quantized_sd, "")
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "Gfx906Int4Patch": Gfx906Int4Patch,
    "Gfx906Int4UnetLoader": Gfx906Int4UnetLoader,
    "Gfx906Int4Transcode": Gfx906Int4Transcode,
    "Gfx906Int4TranscodeCLIP": Gfx906Int4TranscodeCLIP,
    "Gfx906RowSplitKrea2": Gfx906RowSplitKrea2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gfx906Int4Transcode": "Gfx906 Int4 Transcode (native kernel, any source)",
    "Gfx906Int4TranscodeCLIP": "Gfx906 Int4 Transcode CLIP (native kernel, any source)",
    "Gfx906Int4Patch": "Gfx906 Int4 Quantize (test)",
    "Gfx906Int4UnetLoader": "Gfx906 Int4 UNet Loader (native kernel)",
    "Gfx906RowSplitKrea2": "Gfx906 Split Krea2 (2 GPUs)",
}
