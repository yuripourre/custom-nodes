import torch
import numpy as np
from PIL import Image
import os
import folder_paths

class LoadImageAlpha:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        image_files = []
        
        if os.path.exists(input_dir):
            image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'))]
        
        return {
            "required": {
                "image": (sorted(image_files) if image_files else [""], {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image):
        # Get the full path to the image file
        try:
            # Try ComfyUI's standard method first
            image_path = folder_paths.get_annotated_filepath(image)
        except (AttributeError, TypeError):
            # Fallback: construct path manually
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image)
        
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        # Load image with PIL, preserving alpha if present
        img = Image.open(image_path)

        # Convert to RGBA to ensure alpha channel is preserved
        # If image already has alpha, it's preserved
        # If image is RGB, we add a fully opaque alpha channel
        if img.mode != "RGBA":
            if img.mode in ("P", "LA", "PA"):
                # Palette or grayscale with alpha - convert to RGBA
                img = img.convert("RGBA")
            elif img.mode == "L":
                # Grayscale - convert to RGBA
                img = img.convert("RGBA")
            else:
                # RGB or other modes - convert to RGBA
                img = img.convert("RGBA")

        # Convert PIL image to numpy array
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr, dtype=np.float32)

        # Convert to tensor [H, W, 4] (RGBA)
        tensor = torch.from_numpy(arr)

        # Add batch dimension [1, H, W, 4]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        return (tensor,)

NODE_CLASS_MAPPINGS = {
    "LoadImageAlpha": LoadImageAlpha
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageAlpha": "Load Image (Alpha)"
}

