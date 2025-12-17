from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch
from scipy import ndimage

class AlphaRemoveThinLinesNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "preserve_alpha": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def tensor_to_pil(self, tensor_img: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor [1,H,W,C] or [H,W,C] to PIL RGBA."""
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
    
    def process(self, image, line_thickness=2, iterations=1, preserve_alpha=True):
        """
        Remove thin lines from alpha channel using morphological opening.
        This works even when lines are connected to larger blobs.
        
        Morphological opening (erosion + dilation) removes thin protrusions
        and thin lines while preserving larger regions.
        """
        # Handle different possible image formats
        if isinstance(image, torch.Tensor):
            img = self.tensor_to_pil(image)
        elif isinstance(image, list) and len(image) > 0 and isinstance(image[0], torch.Tensor):
            img = self.tensor_to_pil(image[0])
        elif hasattr(image, "convert"):
            img = image.convert("RGBA")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
        
        w, h = img.size
        arr = np.array(img)
        
        # Extract alpha channel
        original_alpha = arr[..., 3].astype(np.float32) / 255.0
        
        # Convert to binary mask (threshold at 0.5)
        # Pixels with alpha > 0.5 are considered opaque, <= 0.5 are transparent
        binary_mask = (original_alpha > 0.5).astype(bool)
        
        # Create structure element based on line_thickness
        # This determines how thick a line needs to be to survive
        base_structure = ndimage.generate_binary_structure(2, 2)
        
        # Calculate structure size: line_thickness determines the kernel radius
        # For a line_thickness of N, we need a structure that's roughly 2*N+1 in diameter
        # to remove lines thinner than that
        if line_thickness == 1:
            structure = np.array([[1]], dtype=bool)
        elif line_thickness <= 3:
            # Use base 3x3 structure
            structure = base_structure
        else:
            # Expand structure to handle thicker lines
            # For line_thickness N, we want to remove lines thinner than N pixels
            iterations_needed = (line_thickness - 1) // 2
            structure = ndimage.iterate_structure(base_structure, iterations_needed)
        
        # Apply morphological opening: erosion followed by dilation
        # This removes thin lines and protrusions while preserving larger blobs
        processed_mask = binary_mask.copy()
        
        for _ in range(iterations):
            # Step 1: Erosion - shrinks all regions, breaks thin connections
            eroded = ndimage.binary_erosion(processed_mask, structure=structure)
            
            # Step 2: Dilation - restores size of larger blobs (but not removed thin lines)
            processed_mask = ndimage.binary_dilation(eroded, structure=structure)
        
        # Convert back to float alpha
        new_alpha = processed_mask.astype(np.float32)
        
        # If preserve_alpha is True, blend with original alpha to maintain smooth transitions
        # This helps preserve anti-aliased edges
        if preserve_alpha:
            # For pixels that remain opaque, use original alpha value
            # For pixels that were removed, set to 0
            new_alpha = np.where(processed_mask, original_alpha, 0.0)
        else:
            # Hard binary: either fully opaque or fully transparent
            new_alpha = new_alpha
        
        # Update alpha channel in the image array
        arr[..., 3] = np.clip(new_alpha * 255.0, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(arr, mode="RGBA")
        return (self.pil_to_tensor(out_img),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaRemoveThinLinesNode": AlphaRemoveThinLinesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaRemoveThinLinesNode": "Alpha Remove Thin Lines"
}


