from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch
from scipy import ndimage

class AlphaRemoveSmallRegionsNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "min_region_size": ("INT", {"default": 100, "min": 1, "max": 1000000, "step": 1}),
            },
            "optional": {
                "preserve_opaque": ("BOOLEAN", {"default": False}),
                "preserve_transparent": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "alpha"
    
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
    
    def process(self, image, min_region_size=100, preserve_opaque=True, preserve_transparent=True):
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
        alpha = arr[..., 3].astype(np.float32) / 255.0
        
        # Convert to binary mask (threshold at 0.5)
        # Pixels with alpha > 0.5 are considered opaque, <= 0.5 are transparent
        binary_mask = (alpha > 0.5).astype(bool)
        
        # Use 8-connected structure for flood fill
        structure = ndimage.generate_binary_structure(2, 2)
        
        # Label all connected components (regions)
        # This performs flood fill to identify all separate regions
        labeled, num_features = ndimage.label(binary_mask, structure=structure)
        
        # Calculate size of each region
        region_sizes = {}
        for label_id in range(1, num_features + 1):
            region_mask = (labeled == label_id)
            region_sizes[label_id] = np.sum(region_mask)
        
        # Determine which regions to keep
        # Keep regions that are large enough OR should be preserved based on settings
        regions_to_remove = set()
        
        for label_id, size in region_sizes.items():
            if size < min_region_size:
                # Check if this region should be preserved
                region_mask = (labeled == label_id)
                is_opaque_region = np.any(binary_mask[region_mask])
                
                should_preserve = False
                if is_opaque_region and preserve_opaque:
                    should_preserve = True
                elif not is_opaque_region and preserve_transparent:
                    should_preserve = True
                
                if not should_preserve:
                    regions_to_remove.add(label_id)
        
        # Create mask of regions to keep
        keep_mask = np.ones((h, w), dtype=bool)
        for label_id in regions_to_remove:
            region_mask = (labeled == label_id)
            keep_mask[region_mask] = False
        
        # Apply the mask to alpha channel
        # Remove small regions by setting their alpha to 0
        new_alpha = alpha.copy()
        new_alpha[~keep_mask] = 0.0
        
        # Update alpha channel in the image array
        arr[..., 3] = np.clip(new_alpha * 255.0, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(arr, mode="RGBA")
        return (self.pil_to_tensor(out_img),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaRemoveSmallRegionsNode": AlphaRemoveSmallRegionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaRemoveSmallRegionsNode": "🫟 Alpha Remove Small Regions"
}

