from typing import Dict, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch

class AlphaStrokeNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "stroke_size": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "antialias": ("BOOLEAN", {"default": True}),
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

    def apply_alpha_stroke(self, image, stroke_size, opacity, antialias, red=0, green=0, blue=0):
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
            # Transparent pixels (alpha < threshold) = background, opaque = foreground
            alpha_channel = img_single[:, :, 3]
            ALPHA_THRESHOLD = 0.5

            # Create mask: pixels that are NOT transparent (keep foreground pixels)
            # Transparent pixels = 0, opaque pixels = 255
            foreground_mask = (alpha_channel >= ALPHA_THRESHOLD).astype(np.uint8) * 255
            mask_img = Image.fromarray(foreground_mask, mode="L")

            # Store original mask
            original_mask_arr = np.array(mask_img, dtype=np.float32) / 255.0

            # Apply stroke expansion if needed
            if stroke_size > 0:
                # Use MaxFilter with correct kernel size for stroke
                # MaxFilter kernel size should be (stroke_size * 2 + 1) for stroke pixels
                kernel_size = stroke_size * 2 + 1
                expanded_mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel_size))

                # Calculate stroke area (expanded minus original)
                expanded_mask_arr = np.array(expanded_mask_img, dtype=np.float32) / 255.0
                stroke_area = expanded_mask_arr - original_mask_arr
                stroke_area = np.clip(stroke_area, 0, 1)

                # Apply antialiasing to stroke if requested
                if antialias:
                    stroke_img = Image.fromarray((stroke_area * 255).astype(np.uint8), mode="L")
                    antialiased_stroke = stroke_img.filter(ImageFilter.GaussianBlur(radius=0.5))
                    stroke_area = np.array(antialiased_stroke, dtype=np.float32) / 255.0

                # Apply stroke color with opacity
                stroke_color = np.array([red, green, blue], dtype=np.float32) / 255.0

                # Get original image array
                result_arr = img_single.copy()

                # Apply stroke color to RGB channels where stroke exists
                for c in range(3):
                    original_channel = img_single[:, :, c]
                    stroke_channel = stroke_color[c]

                    # Blend: original * (1 - stroke_area * opacity) + stroke_color * (stroke_area * opacity)
                    blended = (original_channel * (1.0 - stroke_area * opacity) +
                              stroke_channel * (stroke_area * opacity))

                    result_arr[:, :, c] = np.clip(blended, 0.0, 1.0)

                # Update alpha channel: make stroke area opaque, keep background transparent
                # The expanded mask defines what should be opaque (foreground + stroke)
                result_arr[:, :, 3] = np.clip(expanded_mask_arr, 0.0, 1.0)

                # Convert back to tensor [H,W,C]
                result_tensor = torch.from_numpy(result_arr)
            else:
                # No stroke: return original
                result_tensor = torch.from_numpy(img_single)

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

