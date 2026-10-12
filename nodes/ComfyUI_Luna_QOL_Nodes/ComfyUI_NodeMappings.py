from .ComfyUI_MaterialSaver import MaterialSaver

NODE_CLASS_MAPPINGS = {
    "MaterialSaver": MaterialSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaterialSaver": "Material Saver",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
