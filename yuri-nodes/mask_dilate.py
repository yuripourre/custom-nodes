import torch
import numpy as np
from scipy import ndimage

class MaskDilate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "dilate_mask"
    CATEGORY = "mask"

    def dilate_mask(self, mask, iterations, kernel_size):
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
        
        # Create structure element based on kernel size
        # Start with 8-connected base structure and iterate to desired size
        base_structure = ndimage.generate_binary_structure(2, 2)
        if kernel_size == 1:
            structure = np.array([[1]], dtype=bool)
        elif kernel_size == 3:
            structure = base_structure
        else:
            # Iterate the structure to expand it to the desired kernel size
            iterations_needed = (kernel_size - 1) // 2
            structure = ndimage.iterate_structure(base_structure, iterations_needed)
        
        for b in range(batch_size):
            mask_single = mask_np[b]
            
            # Convert to binary (threshold at 0.5)
            binary_mask = (mask_single > 0.5).astype(np.float32)
            
            # Apply dilation iteratively
            dilated = binary_mask
            for _ in range(iterations):
                dilated = ndimage.binary_dilation(dilated, structure=structure).astype(np.float32)
            
            # Convert back to float32 mask format
            result_mask = dilated.astype(np.float32)
            results.append(result_mask)
        
        # Stack results
        output = np.stack(results, axis=0)
        
        # Remove batch dimension if input was 2D
        if squeeze_output:
            output = output[0]
        
        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output)
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "MaskDilate": MaskDilate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskDilate": "🫟 Mask Dilate"
}

