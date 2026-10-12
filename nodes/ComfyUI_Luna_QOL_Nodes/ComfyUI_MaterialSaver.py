import os
import uuid
from PIL import Image
import cv2
import numpy as np
import comfy.utils


class LunaMaterialSaver:
    OUTPUT_NODE = True

    """
    Material texture exporter with channel packing.

    scalar_map  RGB 8-bit:   R = organic,  G = glow,  B = height
    surface_map RGB 16-bit:  R = metallic, G = hard,  B = fabric
    """

    # -------------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "material_name": ("STRING", {"default": "material"}),

                "diffuse_image": ("IMAGE",),
                "diffuse_prefix": ("STRING", {"default": "diffuse"}),
            },
            "optional": {
                "output_base_dir": ("STRING", {"default": "materials"}),

                "diffuse_high_image": ("IMAGE",),
                "diffuse_high_prefix": ("STRING", {"default": "diffuse_high"}),

                "normal_image": ("IMAGE",),
                "normal_prefix": ("STRING", {"default": "normal"}),

                "height_high_image": ("IMAGE",),
                "height_high_prefix": ("STRING", {"default": "height_high"}),

                # scalar packed (8-bit RGB)
                "organic_image": ("IMAGE",),
                "glow_image": ("IMAGE",),
                "height_image": ("IMAGE",),
                "scalar_prefix": ("STRING", {"default": "scalar"}),

                # surface packed (16-bit RGB)
                "metallic_image": ("IMAGE",),
                "hard_image": ("IMAGE",),
                "fabric_image": ("IMAGE",),
                "surface_prefix": ("STRING", {"default": "surface"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "utils"

    # -------------------------------------------------------------

    def save(self,
             material_name,
             diffuse_image,
             diffuse_prefix,
             output_base_dir="materials",

             diffuse_high_image=None,
             diffuse_high_prefix="diffuse_high",

             normal_image=None,
             normal_prefix="normal",

             height_high_image=None,
             height_high_prefix="height_high",

             organic_image=None,
             glow_image=None,
             height_image=None,
             scalar_prefix="scalar",

             metallic_image=None,
             hard_image=None,
             fabric_image=None,
             surface_prefix="surface"
             ):

        material_id = material_name or f"material_{uuid.uuid4().hex[:6]}"
        print(f"Saving material '{material_name}' with ID '{material_id}'...")

        output_dir = os.path.join(output_base_dir, material_id)
        os.makedirs(output_dir, exist_ok=True)

        base_h, base_w = self._get_base_resolution(diffuse_image)

        # ---------- RGB textures (8-bit) ----------
        self._save_rgb(diffuse_image, output_dir,
                       f"{diffuse_prefix}_{material_id}.png", base_h, base_w)

        if diffuse_high_image is not None:
            self._save_rgb(diffuse_high_image, output_dir,
                           f"{diffuse_high_prefix}_{material_id}.png", base_h, base_w)

        if normal_image is not None:
            self._save_rgb(normal_image, output_dir,
                           f"{normal_prefix}_{material_id}.png", base_h, base_w)

        # ---------- height high (8-bit gray) ----------
        if height_high_image is not None:
            self._save_gray(height_high_image, output_dir,
                            f"{height_high_prefix}_{material_id}.png")

        # ---------- scalar pack (8-bit RGB: R=organic, G=glow, B=height) ----------
        scalar_inputs = (organic_image, glow_image, height_image)
        if any(t is not None for t in scalar_inputs):
            self._save_channel_pack(
                organic_image, glow_image, height_image,
                output_dir, f"{scalar_prefix}_{material_id}.png",
                base_h, base_w, bits=16
            )

        # ---------- surface pack (16-bit RGB: R=metallic, G=hard, B=fabric) ----------
        surface_inputs = (metallic_image, hard_image, fabric_image)
        if any(t is not None for t in surface_inputs):
            self._save_channel_pack(
                metallic_image, hard_image, fabric_image,
                output_dir, f"{surface_prefix}_{material_id}.png",
                base_h, base_w, bits=16
            )

        return ()

    # -------------------------------------------------------------

    def _get_base_resolution(self, diffuse_tensor):
        if diffuse_tensor is None:
            return 1, 1
        img = diffuse_tensor[0]
        return img.shape[0], img.shape[1]

    @staticmethod
    def _tensor_to_float(tensor):
        """Return (H, W, C) float32 array in [0, 1]."""
        return tensor[0].cpu().numpy().astype(np.float32)

    # -------------------------------------------------------------

    def _save_rgb(self, tensor, folder, filename, h, w):
        arr = np.clip(self._tensor_to_float(tensor) * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(arr, "RGB").save(
            os.path.join(folder, filename),
            compress_level=9, optimize=True
        )

    # -------------------------------------------------------------

    def _save_gray(self, tensor, folder, filename):
        arr = np.clip(self._tensor_to_float(tensor) * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]

        Image.fromarray(arr, "L").save(
            os.path.join(folder, filename),
            compress_level=9, optimize=True
        )

    # -------------------------------------------------------------

    def _gray_channel(self, tensor, h, w):
        """Extract first channel as float32 in [0, 1]."""
        if tensor is None:
            return np.zeros((h, w), dtype=np.float32)

        arr = self._tensor_to_float(tensor)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.clip(arr, 0.0, 1.0)

    # -------------------------------------------------------------

    def _save_channel_pack(self, r_tensor, g_tensor, b_tensor,
                           folder, filename, h, w, bits=8):
        """
        Pack three grayscale inputs into an RGB PNG.
        Missing channels are filled with black (zero).

        bits=8  -> standard 24-bit PNG  (uint8,  0-255)
        bits=16 -> 48-bit PNG           (uint16, 0-65535)
        """
        r = self._gray_channel(r_tensor, h, w)
        g = self._gray_channel(g_tensor, h, w)
        b = self._gray_channel(b_tensor, h, w)

        if bits == 16:
            max_val = 65535.0
            dtype = np.uint16
        else:
            max_val = 255.0
            dtype = np.uint8

        packed = np.stack([r, g, b], axis=-1)
        packed = np.clip(packed * max_val, 0, max_val).astype(dtype)

        path = os.path.join(folder, filename)

        if bits == 16:
            # cv2 handles 16-bit PNG natively; it expects BGR
            bgr = packed[..., ::-1].copy()
            cv2.imwrite(path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            Image.fromarray(packed, "RGB").save(
                path, compress_level=9, optimize=True
            )

        print(f"Saved {bits}-bit channel-packed map: {filename}")