import torch
import numpy as np
from scipy import ndimage

class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "kernel_size": ("INT", {"default": 3, "min": 0, "max": 101, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "blur_mask"
    CATEGORY = "mask"

    def blur_mask(self, mask, sigma, kernel_size):
        # Convert to numpy
        mask_np = mask.cpu().numpy()

        # Handle different tensor shapes
        if mask_np.ndim == 2:
            # Single mask (H, W) - add batch dimension
            mask_np = mask_np[np.newaxis, ...]
            squeeze_output = True
        else:
            # Batch of masks (B, H, W)
            squeeze_output = False

        batch_size = mask_np.shape[0]
        results = []

        for b in range(batch_size):
            mask_single = mask_np[b]

            # Apply Gaussian blur
            if sigma > 0.0:
                if kernel_size > 0:
                    # Calculate truncate based on kernel_size and sigma
                    # kernel_size = 2 * truncate * sigma + 1
                    # truncate = (kernel_size - 1) / (2 * sigma)
                    truncate = (kernel_size - 1) / (2.0 * sigma) if sigma > 0 else 4.0
                    blurred = ndimage.gaussian_filter(mask_single, sigma=sigma, truncate=truncate)
                else:
                    # Use default truncate (4.0) if kernel_size is 0
                    blurred = ndimage.gaussian_filter(mask_single, sigma=sigma)
            else:
                # No blur if sigma is 0
                blurred = mask_single

            # Ensure values stay in valid range [0, 1]
            blurred = np.clip(blurred, 0.0, 1.0).astype(np.float32)
            results.append(blurred)

        # Stack results
        output = np.stack(results, axis=0)

        # Remove batch dimension if input was 2D
        if squeeze_output:
            output = output[0]

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output)

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "MaskBlur": MaskBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBlur": "🫟 Mask Blur"
}

