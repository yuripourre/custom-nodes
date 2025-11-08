from typing import Dict, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch

class OutlineNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "background_red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "background_green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "background_blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "tolerance": ("INT", {"default": 0, "min": 0, "max": 127, "step": 1}),
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
    FUNCTION = "apply_stroke"
    CATEGORY = "image/filter"

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

    def apply_stroke(self, image, background_red, background_green, background_blue, tolerance,
                     stroke_size, opacity, antialias, red=0, green=0, blue=0):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape

        # Handle RGB input by adding alpha channel
        if channels == 3:
            alpha = np.ones((batch_size, height, width, 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=-1)
            channels = 4

        # Convert background color to float32
        background_color = np.array([background_red, background_green, background_blue], dtype=np.float32)

        results = []

        for b in range(batch_size):
            # Get single image
            img_single = img_np[b]  # [H, W, C]

            # Extract RGB channels
            rgb_channels = img_single[:, :, :3]
            # Convert to 0-255 range for comparison
            rgb_255 = (rgb_channels * 255.0).astype(np.float32)

            # Calculate Euclidean distance from background color
            diff = rgb_255 - background_color
            euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=-1))

            # Create mask: pixels that are NOT background (keep foreground pixels)
            # Pixels matching background color (within tolerance) = 0, others = 255
            foreground_mask = (euclidean_dist > tolerance).astype(np.uint8) * 255
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
    "OutlineNode": OutlineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OutlineNode": "Outline"
}

