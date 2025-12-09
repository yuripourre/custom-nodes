import torch
import numpy as np

class AddBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "add_background"
    CATEGORY = "image"

    def add_background(self, image, red, green, blue):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape

        # Convert RGB background color to normalized float
        bg_r = red / 255.0
        bg_g = green / 255.0
        bg_b = blue / 255.0

        # Handle RGB input by adding alpha channel
        if channels == 3:
            # Add alpha channel (fully opaque)
            alpha = np.ones((batch_size, height, width, 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=-1)
            channels = 4

        results = []

        for b in range(batch_size):
            # Extract RGB and alpha channels for this batch item
            rgb = img_np[b, ..., :3]  # [H, W, 3]
            alpha = img_np[b, ..., 3:4]  # [H, W, 1]

            # Create background color array matching image dimensions
            background = np.array([bg_r, bg_g, bg_b], dtype=np.float32)
            background = np.broadcast_to(background, (height, width, 3))

            # Alpha compositing: result = foreground * alpha + background * (1 - alpha)
            # Expand alpha to match RGB dimensions
            alpha_expanded = np.broadcast_to(alpha, (height, width, 3))
            result_rgb = rgb * alpha_expanded + background * (1.0 - alpha_expanded)

            # Clip to valid range
            result_rgb = np.clip(result_rgb, 0.0, 1.0)

            # Convert to tensor [H, W, 3]
            result_tensor = torch.from_numpy(result_rgb.astype(np.float32))
            results.append(result_tensor)

        # Stack results into batch [B, H, W, 3]
        output = torch.stack(results, dim=0)

        return (output,)

NODE_CLASS_MAPPINGS = {
    "AddBackground": AddBackground
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddBackground": "Add Background"
}

