from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch

class ResizeAndPadAlphaNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 1, "max": 10000, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
                "pad_top": ("BOOLEAN", {"default": True}),
                "pad_bottom": ("BOOLEAN", {"default": True}),
                "pad_left": ("BOOLEAN", {"default": True}),
                "pad_right": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_and_pad"
    CATEGORY = "image/transform"

    def tensor_to_pil(self, tensor_img: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor [H,W,C] to PIL RGBA."""
        arr = tensor_img.detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        # Handle RGB input by adding alpha channel
        if arr.shape[-1] == 3:
            alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)

        arr = np.ascontiguousarray(arr)
        return Image.fromarray(arr, mode="RGBA")

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL RGBA to ComfyUI tensor [H,W,C] float32."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        tensor = torch.from_numpy(arr)

        return tensor

    def resize_and_pad(self, image, target_width, target_height, pad_top=True, pad_bottom=True, pad_left=True, pad_right=True):
        """
        Resize and pad image to target dimensions with alpha background.
        Padding is distributed based on pad_top, pad_bottom, pad_left, pad_right flags.
        When all flags are True, the image is centered.
        If target dimensions are smaller, rows/columns are removed based on padding flags.
        """
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, original_height, original_width, channels = img_np.shape

        # Handle RGB input by adding alpha channel
        if channels == 3:
            # Add alpha channel
            alpha = np.ones((batch_size, original_height, original_width, 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=-1)
            channels = 4

        # Calculate differences
        width_diff = target_width - original_width
        height_diff = target_height - original_height

        # Determine crop offsets (for when target is smaller)
        crop_left = 0
        crop_top = 0
        crop_width = original_width
        crop_height = original_height

        # Handle width cropping/padding
        if width_diff < 0:
            # Need to crop width
            horizontal_sides = sum([pad_left, pad_right])
            if horizontal_sides == 0:
                # Center crop
                crop_left = (-width_diff) // 2
            elif pad_left and not pad_right:
                # Crop from right (keep left)
                crop_left = 0
            elif pad_right and not pad_left:
                # Crop from left (keep right)
                crop_left = -width_diff
            else:
                # Both enabled, center crop
                crop_left = (-width_diff) // 2
            crop_width = target_width
        else:
            # Need to pad width
            horizontal_sides = sum([pad_left, pad_right])
            if horizontal_sides == 0:
                # Center by default
                left_pad = width_diff // 2
                right_pad = width_diff - left_pad
            elif horizontal_sides == 1:
                # Only one side enabled
                if pad_left:
                    left_pad = width_diff
                    right_pad = 0
                else:
                    left_pad = 0
                    right_pad = width_diff
            else:
                # Both sides enabled, distribute evenly
                left_pad = width_diff // 2
                right_pad = width_diff - left_pad

        # Handle height cropping/padding
        if height_diff < 0:
            # Need to crop height
            vertical_sides = sum([pad_top, pad_bottom])
            if vertical_sides == 0:
                # Center crop
                crop_top = (-height_diff) // 2
            elif pad_top and not pad_bottom:
                # Crop from top (keep bottom)
                crop_top = -height_diff
            elif pad_bottom and not pad_top:
                # Crop from bottom (keep top)
                crop_top = 0
            else:
                # Both enabled, center crop
                crop_top = (-height_diff) // 2
            crop_height = target_height
        else:
            # Need to pad height
            vertical_sides = sum([pad_top, pad_bottom])
            if vertical_sides == 0:
                # Center by default
                top_pad = height_diff // 2
                bottom_pad = height_diff - top_pad
            elif vertical_sides == 1:
                # Only one side enabled
                if pad_top:
                    top_pad = height_diff
                    bottom_pad = 0
                else:
                    top_pad = 0
                    bottom_pad = height_diff
            else:
                # Both sides enabled, distribute evenly
                top_pad = height_diff // 2
                bottom_pad = height_diff - top_pad

        results = []

        for b in range(batch_size):
            # Convert single image tensor to PIL
            img_single = torch.from_numpy(img_np[b])
            img_pil = self.tensor_to_pil(img_single)

            # Crop if needed
            if width_diff < 0 or height_diff < 0:
                img_pil = img_pil.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

            # Create a new transparent image with target dimensions
            padded_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

            # Paste the (possibly cropped) image at the calculated offset
            if width_diff >= 0 and height_diff >= 0:
                # Only padding needed
                padded_img.paste(img_pil, (left_pad, top_pad), img_pil)
            elif width_diff < 0 and height_diff >= 0:
                # Width was cropped, height needs padding
                padded_img.paste(img_pil, (0, top_pad), img_pil)
            elif width_diff >= 0 and height_diff < 0:
                # Height was cropped, width needs padding
                padded_img.paste(img_pil, (left_pad, 0), img_pil)
            else:
                # Both were cropped, just paste at origin
                padded_img.paste(img_pil, (0, 0), img_pil)

            # Convert back to tensor [H,W,C]
            padded_tensor = self.pil_to_tensor(padded_img)
            results.append(padded_tensor)

        # Stack results into batch [B,H,W,C]
        output = torch.stack(results, dim=0)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ResizeAndPadAlphaNode": ResizeAndPadAlphaNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeAndPadAlphaNode": "Resize And Pad Alpha"
}

