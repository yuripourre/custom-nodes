from typing import Dict, Tuple
import numpy as np
import torch
from scipy import ndimage
from PIL import Image, ImageFilter

class AlphaEdgeDilate:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "antialias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dilate_alpha_edges"
    CATEGORY = "alpha"

    def dilate_alpha_edges(self, image, iterations, kernel_size, antialias=True):
        # Convert tensor to numpy
        img_np = image.cpu().numpy()
        
        # Handle different tensor shapes
        if img_np.ndim == 3:
            # Single image (H, W, C) - add batch dimension
            img_np = img_np[np.newaxis, ...]
            squeeze_output = True
        else:
            # Batch of images (B, H, W, C)
            squeeze_output = False
        
        batch_size, height, width, channels = img_np.shape
        
        # Ensure we have alpha channel
        if channels < 4:
            # Add alpha channel if missing
            alpha_channel = np.ones((batch_size, height, width, 1), dtype=img_np.dtype)
            img_np = np.concatenate([img_np, alpha_channel], axis=-1)
            channels = 4
        
        results = []
        
        # Create structure element for dilation
        base_structure = ndimage.generate_binary_structure(2, 2)
        if kernel_size == 1:
            structure = np.array([[1]], dtype=bool)
        elif kernel_size == 3:
            structure = base_structure
        else:
            iterations_needed = (kernel_size - 1) // 2
            structure = ndimage.iterate_structure(base_structure, iterations_needed)
        
        for b in range(batch_size):
            img = img_np[b].copy()
            original_alpha = img[..., 3].copy()
            
            # Convert alpha to binary mask (threshold at 0.5)
            binary_alpha = (original_alpha > 0.5).astype(np.float32)
            
            # Detect edges: pixels that are on the boundary between transparent and opaque
            # An edge pixel is one that has at least one neighbor with different alpha value
            edge_mask = np.zeros_like(binary_alpha, dtype=bool)
            
            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # Shift the binary alpha
                    shifted = np.roll(np.roll(binary_alpha, dy, axis=0), dx, axis=1)
                    # Edge pixel if current pixel differs from neighbor
                    edge_mask |= (binary_alpha != shifted)
            
            # Only process if edges exist
            if not np.any(edge_mask):
                # No edges found, return original
                results.append(img)
                continue
            
            # Create edge region mask: dilate edge pixels to get a band around edges
            # The width should account for iterations and kernel size to capture all affected pixels
            edge_region_structure = ndimage.generate_binary_structure(2, 2)
            edge_region_iterations = max(1, iterations + (kernel_size // 2))
            edge_region_mask = ndimage.binary_dilation(edge_mask.astype(bool), structure=edge_region_structure, iterations=edge_region_iterations).astype(np.float32)
            
            # Apply dilation to the entire alpha channel
            working_alpha = binary_alpha.copy()
            for _ in range(iterations):
                working_alpha = ndimage.binary_dilation(working_alpha, structure=structure).astype(np.float32)
            
            # Blend: use dilated alpha in edge regions, keep original elsewhere
            final_alpha = np.where(edge_region_mask > 0.5, working_alpha, binary_alpha)
            
            # Apply antialiasing if requested
            if antialias:
                # Convert alpha to PIL Image for blur
                alpha_img = Image.fromarray((final_alpha * 255).astype(np.uint8), mode="L")
                antialiased_alpha = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.5))
                final_alpha = np.array(antialiased_alpha, dtype=np.float32) / 255.0
            
            # Convert back to original alpha range and apply to alpha channel
            img[..., 3] = np.clip(final_alpha, 0.0, 1.0)
            
            results.append(img)
        
        # Stack results
        output = np.stack(results, axis=0)
        
        # Remove batch dimension if input was 3D
        if squeeze_output:
            output = output[0]
        
        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output)
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "AlphaEdgeDilate": AlphaEdgeDilate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaEdgeDilate": "🫟 Alpha Edge Dilate"
}

