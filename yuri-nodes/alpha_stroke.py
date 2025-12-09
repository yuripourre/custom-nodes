from typing import Dict, Tuple
import numpy as np
import torch
from scipy import ndimage

class AlphaStrokeNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "stroke_size": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "apply_alpha_stroke"
    CATEGORY = "image/filter"

    def apply_alpha_stroke(self, image, stroke_size, opacity, red=0, green=0, blue=0):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape

        # Handle RGB input by adding alpha channel
        if channels == 3:
            alpha = np.ones((batch_size, height, width, 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=-1)
            channels = 4

        results = []

        for b in range(batch_size):
            # Get single image
            img_single = img_np[b]  # [H, W, C]

            # Extract alpha channel to determine foreground/background
            alpha_channel = img_single[:, :, 3]
            ALPHA_THRESHOLD = 0.5

            # Create binary mask: pixels that are NOT transparent
            binary_mask = (alpha_channel >= ALPHA_THRESHOLD).astype(np.float32)

            # Detect contour/edge pixels: only pixels that are TRANSPARENT and have at least one OPAQUE neighbor
            # This ensures we only get the contour OUTSIDE the shape, not on image edges
            transparent_mask = (alpha_channel < ALPHA_THRESHOLD).astype(np.float32)
            contour_mask = np.zeros_like(binary_mask, dtype=np.float32)

            # Check 8-connected neighbors to find edge pixels
            # A contour pixel is transparent AND has at least one opaque neighbor
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # Shift the binary mask (opaque pixels)
                    shifted_opaque = np.roll(np.roll(binary_mask, dy, axis=0), dx, axis=1)
                    # Contour pixel if current pixel is transparent AND neighbor is opaque
                    contour_mask = np.maximum(contour_mask, transparent_mask * shifted_opaque)

            # Apply stroke_size to dilate the contour if needed
            if stroke_size > 0 and np.any(contour_mask > 0):
                # Use scipy dilation with 8-connected structure
                structure = ndimage.generate_binary_structure(2, 2)
                # Dilate by stroke_size iterations (size 1 = 1 pixel thick, size 2 = 2 pixels thick)
                contour_mask = ndimage.binary_dilation(
                    contour_mask.astype(bool),
                    structure=structure,
                    iterations=stroke_size
                ).astype(np.float32)

            # Create new transparent image (all zeros)
            result_arr = np.zeros((height, width, 4), dtype=np.float32)

            # Apply contour color with opacity
            stroke_color = np.array([red, green, blue], dtype=np.float32) / 255.0

            # Set RGB channels to stroke color where contour exists
            for c in range(3):
                result_arr[:, :, c] = stroke_color[c] * contour_mask * opacity

            # Set alpha channel: contour area becomes opaque based on opacity
            result_arr[:, :, 3] = contour_mask * opacity

            # Convert back to tensor [H,W,C]
            result_tensor = torch.from_numpy(result_arr)

            results.append(result_tensor)

        # Stack results into batch [B,H,W,C]
        output = torch.stack(results, dim=0)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaStrokeNode": AlphaStrokeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaStrokeNode": "Alpha Stroke"
}

