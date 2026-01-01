from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch

class AlphaCentralizeNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "horizontal": ("BOOLEAN", {"default": True}),
                "vertical": ("BOOLEAN", {"default": True}),
                "alpha_tolerance": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "centralize"
    CATEGORY = "alpha"

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

    def find_bounding_box(self, img: Image.Image, alpha_tolerance: float = 0.01) -> Tuple[int, int, int, int]:
        """Find the bounding box of the alpha blob (pixels with alpha > tolerance)."""
        arr = np.array(img)
        alpha = arr[..., 3]

        # Convert tolerance from 0.0-1.0 range to 0-255 range for uint8
        tolerance_uint8 = int(alpha_tolerance * 255)

        # Find rows and columns with any pixels above tolerance
        rows = np.any(alpha > tolerance_uint8, axis=1)
        cols = np.any(alpha > tolerance_uint8, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Image is fully transparent, return full image bounds
            return (0, 0, img.width, img.height)

        # Find first and last row/column with content
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        top = row_indices[0]
        bottom = row_indices[-1] + 1
        left = col_indices[0]
        right = col_indices[-1] + 1

        return (left, top, right, bottom)

    def centralize(self, image, horizontal=True, vertical=True, alpha_tolerance=0.01):
        """
        Centralize a transparent image by finding its bounding box and centering it.
        The horizontal and vertical toggles control whether to center in each axis.
        alpha_tolerance: Minimum alpha value (0.0-1.0) to consider as part of the blob.
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

        results = []

        for b in range(batch_size):
            # Convert single image tensor to PIL
            img_single = torch.from_numpy(img_np[b])
            img_pil = self.tensor_to_pil(img_single)

            # Find bounding box of the alpha blob
            left, top, right, bottom = self.find_bounding_box(img_pil, alpha_tolerance)
            content_width = right - left
            content_height = bottom - top

            # Extract the blob region
            blob_region = img_pil.crop((left, top, right, bottom))

            # Create a new transparent image with same dimensions
            centralized_img = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))

            # Calculate center position for the blob
            if horizontal:
                paste_x = (original_width - content_width) // 2
            else:
                paste_x = left

            if vertical:
                paste_y = (original_height - content_height) // 2
            else:
                paste_y = top

            # Paste the extracted blob at the calculated position
            centralized_img.paste(blob_region, (paste_x, paste_y), blob_region)

            # Convert back to tensor [H,W,C]
            centralized_tensor = self.pil_to_tensor(centralized_img)
            results.append(centralized_tensor)

        # Stack results into batch [B,H,W,C]
        output = torch.stack(results, dim=0)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaCentralizeNode": AlphaCentralizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaCentralizeNode": "🫟 Alpha Centralize"
}

