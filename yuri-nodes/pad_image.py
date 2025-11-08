from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch

class PadImageNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "pad_image"
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

    def pad_image(self, image, left, top, right, bottom):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, original_height, original_width, channels = img_np.shape

        # Calculate new dimensions
        new_width = original_width + left + right
        new_height = original_height + top + bottom

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

            # Create a new transparent image
            padded_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

            # Paste the original image at the specified offset
            padded_img.paste(img_pil, (left, top), img_pil)

            # Convert back to tensor [H,W,C]
            padded_tensor = self.pil_to_tensor(padded_img)
            results.append(padded_tensor)

        # Stack results into batch [B,H,W,C]
        output = torch.stack(results, dim=0)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "PadImageNode": PadImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadImageNode": "Pad Image"
}

