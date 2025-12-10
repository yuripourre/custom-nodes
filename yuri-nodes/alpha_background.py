import torch
import numpy as np
from scipy import ndimage
from PIL import Image

class AlphaBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "tolerance": ("INT", {"default": 10, "min": 0, "max": 127, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "background")
    FUNCTION = "create_alpha"
    CATEGORY = "mask"

    def tensor_to_pil(self, tensor_img: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor [B,H,W,C] or [H,W,C] to PIL RGBA."""
        if tensor_img.ndim == 4:
            tensor_img = tensor_img[0]

        arr = tensor_img.detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        # Handle RGB input by adding alpha channel
        if arr.shape[-1] == 3:
            alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)

        arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr, mode="RGBA")

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL RGBA to ComfyUI tensor [1,H,W,C] float32."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        tensor = torch.from_numpy(arr)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [1,H,W,4]

        return tensor

    def create_alpha(self, image, red, green, blue, tolerance):
        # Convert to PIL for processing
        if isinstance(image, torch.Tensor):
            img = self.tensor_to_pil(image)
        elif isinstance(image, list) and len(image) > 0 and isinstance(image[0], torch.Tensor):
            img = self.tensor_to_pil(image[0])
        elif hasattr(image, "convert"):
            img = image.convert("RGBA")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

        w, h = img.size
        arr = np.array(img)
        rgb = arr[..., :3].astype(np.float32) / 255.0

        # Convert RGB values from 0-255 to 0.0-1.0 range
        target_color = np.array([red / 255.0, green / 255.0, blue / 255.0])

        # Convert tolerance from 0-127 to 0.0-0.498 range
        tolerance_normalized = tolerance / 255.0

        # Set connectivity structure (8-way)
        structure = ndimage.generate_binary_structure(2, 2)

        # Find all pixels similar to the target color
        diff = np.abs(rgb - target_color)
        similar_mask = np.all(diff <= tolerance_normalized, axis=-1)

        # Label connected components of similar pixels
        labeled, num_features = ndimage.label(similar_mask, structure=structure)

        # Flood fill starting from (0,0)
        # Get the label at position (0,0)
        seed_label = labeled[0, 0]

        # If (0,0) is part of a region matching the target color, make that region transparent
        if seed_label > 0:
            # This region contains (0,0), make it transparent
            result_mask = (labeled == seed_label)
        else:
            # (0,0) doesn't match the target color, nothing to fill
            result_mask = np.zeros((h, w), dtype=bool)

        # Create alpha mask: flood-filled region becomes transparent (0), others stay opaque (1)
        alpha_mask = (~result_mask).astype(np.float32)

        # Get original alpha channel
        orig_alpha = arr[..., 3].astype(np.float32) / 255.0

        # Combine with original alpha channel
        new_alpha = orig_alpha * alpha_mask

        # Update alpha channel
        arr[..., 3] = np.clip(new_alpha * 255.0, 0, 255).astype(np.uint8)

        out_img = Image.fromarray(arr, mode="RGBA")
        
        # Create background color image for preview
        # Convert RGB values from 0-255 to 0.0-1.0 range
        bg_r = red / 255.0
        bg_g = green / 255.0
        bg_b = blue / 255.0
        
        # Create solid color background matching image dimensions
        background_arr = np.full((h, w, 3), [bg_r, bg_g, bg_b], dtype=np.float32)
        background_tensor = torch.from_numpy(background_arr)
        
        # Add batch dimension [1, H, W, 3]
        if background_tensor.ndim == 3:
            background_tensor = background_tensor.unsqueeze(0)
        
        return (self.pil_to_tensor(out_img), background_tensor)

NODE_CLASS_MAPPINGS = {
    "AlphaBackground": AlphaBackground
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaBackground": "Alpha Background"
}

