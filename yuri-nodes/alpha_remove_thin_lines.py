from typing import Dict, Tuple
from PIL import Image
import numpy as np
import torch

class AlphaRemoveThinLinesNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "horizontal": ("BOOLEAN", {"default": True}),
                "vertical": ("BOOLEAN", {"default": True}),
                "alpha_tolerance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "preserve_alpha": ("BOOLEAN", {"default": True}),
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
    
    def process_scanline_horizontal(self, alpha, line_thickness, alpha_tolerance):
        """
        Process horizontal scanlines: scan row by row, find segments,
        and remove segments thinner than line_thickness.
        """
        h, w = alpha.shape
        removal_mask = np.zeros_like(alpha, dtype=bool)
        
        for y in range(h):
            row = alpha[y, :]
            # Find segments where alpha >= alpha_tolerance
            above_threshold = row >= alpha_tolerance
            
            # Find segment boundaries
            # Start of segment: transition from False to True
            segment_starts = np.where(np.diff(np.concatenate(([False], above_threshold, [False]))))[0]
            
            # Process pairs of start/end indices
            for i in range(0, len(segment_starts), 2):
                if i + 1 < len(segment_starts):
                    start_idx = segment_starts[i]
                    end_idx = segment_starts[i + 1]
                    segment_length = end_idx - start_idx
                    
                    # If segment is thinner than line_thickness, mark for removal
                    if segment_length < line_thickness:
                        removal_mask[y, start_idx:end_idx] = True
        
        return removal_mask
    
    def process_scanline_vertical(self, alpha, line_thickness, alpha_tolerance):
        """
        Process vertical scanlines: scan column by column, find segments,
        and remove segments thinner than line_thickness.
        """
        h, w = alpha.shape
        removal_mask = np.zeros_like(alpha, dtype=bool)
        
        for x in range(w):
            col = alpha[:, x]
            # Find segments where alpha >= alpha_tolerance
            above_threshold = col >= alpha_tolerance
            
            # Find segment boundaries
            # Start of segment: transition from False to True
            segment_starts = np.where(np.diff(np.concatenate(([False], above_threshold, [False]))))[0]
            
            # Process pairs of start/end indices
            for i in range(0, len(segment_starts), 2):
                if i + 1 < len(segment_starts):
                    start_idx = segment_starts[i]
                    end_idx = segment_starts[i + 1]
                    segment_length = end_idx - start_idx
                    
                    # If segment is thinner than line_thickness, mark for removal
                    if segment_length < line_thickness:
                        removal_mask[start_idx:end_idx, x] = True
        
        return removal_mask
    
    def process(self, image, line_thickness=2, horizontal=True, vertical=True, alpha_tolerance=0.5, preserve_alpha=True):
        """
        Remove thin lines from alpha channel using scanline-based processing.
        
        Scans horizontally and/or vertically, finding segments of consecutive pixels
        that meet the alpha tolerance threshold, and removes segments thinner than
        the specified line_thickness.
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
        
        # Start with all pixels marked as keep
        keep_mask = np.ones_like(original_alpha, dtype=bool)
        
        # Process horizontal scanlines first (if enabled)
        if horizontal:
            horizontal_removal = self.process_scanline_horizontal(original_alpha, line_thickness, alpha_tolerance)
            keep_mask = keep_mask & ~horizontal_removal
        
        # Process vertical scanlines (if enabled)
        if vertical:
            # Use the current state (after horizontal processing if it was done)
            current_alpha = np.where(keep_mask, original_alpha, 0.0)
            vertical_removal = self.process_scanline_vertical(current_alpha, line_thickness, alpha_tolerance)
            keep_mask = keep_mask & ~vertical_removal
        
        # Generate final alpha channel
        if preserve_alpha:
            # For pixels that remain, use original alpha value
            # For pixels that were removed, set to 0
            new_alpha = np.where(keep_mask, original_alpha, 0.0)
        else:
            # Hard binary: either fully opaque or fully transparent
            new_alpha = keep_mask.astype(np.float32)
        
        # Update alpha channel in the image array
        arr[..., 3] = np.clip(new_alpha * 255.0, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(arr, mode="RGBA")
        return (self.pil_to_tensor(out_img),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaRemoveThinLinesNode": AlphaRemoveThinLinesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaRemoveThinLinesNode": "🫟 Alpha Remove Thin Lines"
}






