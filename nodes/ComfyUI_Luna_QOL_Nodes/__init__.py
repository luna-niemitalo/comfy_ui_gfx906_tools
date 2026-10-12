from .DualGPUTileSampler import DualGPUTileSampler
from .ComfyUI_MaterialSaver import LunaMaterialSaver
from .resolution_selector import LunaResolutionSelector

NODE_CLASS_MAPPINGS = {
    "LunaMaterialSaver": LunaMaterialSaver,
    "LunaResolutionSelector": LunaResolutionSelector,
	"DualGPUTileSampler": DualGPUTileSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaMaterialSaver": "Luna Material Saver",
    "LunaResolutionSelector": "Luna Resolution Selector",
	"DualGPUTileSampler": "Dual GPU Tile Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
