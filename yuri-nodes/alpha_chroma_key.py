from typing import Dict, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch

class AlphaChromaKeyNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "red": ("INT", {"default": 0, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
                "variance": ("INT", {"default": 0, "min": 0, "max": 255}),
                "inner_outline": ("INT", {"default": 0, "min": 0, "max": 200}),
                "outline": ("INT", {"default": 0, "min": 0, "max": 200}),
                "antialias": ("BOOLEAN", {"default": True}),
                "invert_output": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "mask"
    
    # ---------- Conversion helpers ----------
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
    
    # ---------- Main processing ----------
    def process(self, image, red=0, green=0, blue=0, variance=0, inner_outline=0, outline=0, antialias=True, invert_output=False):
        
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
        rgb = arr[..., :3].astype(np.int16)
        
        # Store original alpha channel for comparison
        orig_alpha = arr[..., 3].astype(np.float32) / 255.0
        
        # Target color & distance calculation
        target = np.array([red, green, blue], dtype=np.float32)
        rgb_float = rgb.astype(np.float32)
        
        # Calculate Euclidean distance from target color
        diff = rgb_float - target
        euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Create mask based on variance threshold
        if not invert_output:
            # Normal chroma key: pixels matching target color (within variance) become transparent
            # So we KEEP pixels that are FARTHER than variance from target
            keep_pixels = euclidean_dist > variance
        else:
            # Inverted: keep pixels that match the target color
            keep_pixels = euclidean_dist <= variance
        
        # Create initial mask (255 = keep/opaque, 0 = remove/transparent)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[keep_pixels] = 255
        mask_img = Image.fromarray(mask, mode="L")
        
        # Apply inner outline erosion if needed (before outline expansion)
        if inner_outline > 0:
            # Use MinFilter with correct kernel size for exact pixel erosion
            # MinFilter kernel size should be (inner_outline * 2 + 1) for inner_outline pixels
            kernel_size = inner_outline * 2 + 1
            mask_img = mask_img.filter(ImageFilter.MinFilter(kernel_size))
        
        # Store the mask after inner outline for outline processing
        original_mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        
        # Track which pixels will be modified
        # A pixel is modified if its alpha or RGB will change from the original
        modified_pixels = np.zeros((h, w), dtype=bool)
        
        # Apply outline expansion if needed
        if outline > 0:
            # Use MaxFilter with correct kernel size for exact pixel outline
            # MaxFilter kernel size should be (outline * 2 + 1) for outline pixels
            kernel_size = outline * 2 + 1
            expanded_mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel_size))
            
            # Calculate outline area (expanded minus original)
            expanded_mask_arr = np.array(expanded_mask_img, dtype=np.float32) / 255.0
            outline_area = expanded_mask_arr - original_mask_arr
            outline_area = np.clip(outline_area, 0, 1)
            
            # Mark outline area pixels as modified (RGB will change)
            modified_pixels |= (outline_area > 0)
            
            # Fill outline area with the chroma key color
            background_color = np.array([red, green, blue], dtype=np.uint8)
            
            # Apply outline color to RGB channels where outline exists
            for c in range(3):
                arr[..., c] = np.where(outline_area > 0, background_color[c], arr[..., c])
            
            # Final mask includes both original areas and outline areas (both fully opaque)
            final_mask_arr = expanded_mask_arr
        else:
            # No outline: just use the original chroma key mask
            final_mask_arr = original_mask_arr
            expanded_mask_img = mask_img
        
        # Calculate which pixels will have their alpha modified
        # Alpha is modified if the mask changes the effective alpha value
        original_effective_alpha = orig_alpha  # What alpha would be without mask
        new_effective_alpha = orig_alpha * final_mask_arr  # What alpha will be with mask
        alpha_changed = np.abs(original_effective_alpha - new_effective_alpha) > 0.001
        modified_pixels |= alpha_changed
        
        # Apply antialiasing only if requested and only to modified pixels
        if antialias:
            # Convert back to PIL for filtering
            final_mask_img = Image.fromarray((final_mask_arr * 255).astype(np.uint8), mode="L")
            # Apply light blur for antialiasing
            antialiased_mask = final_mask_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            antialiased_mask_arr = np.array(antialiased_mask, dtype=np.float32) / 255.0
            
            # Only use antialiased mask where pixels were modified
            # For unmodified pixels, use the original mask (no antialiasing)
            final_mask_arr = np.where(modified_pixels, antialiased_mask_arr, final_mask_arr)
        
        # Combine with original alpha channel
        new_alpha = orig_alpha * final_mask_arr
        
        # Update alpha channel
        arr[..., 3] = np.clip(new_alpha * 255.0, 0, 255).astype(np.uint8)
        
        out_img = Image.fromarray(arr, mode="RGBA")
        return (self.pil_to_tensor(out_img),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaChromaKeyNode": AlphaChromaKeyNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaChromaKeyNode": "Alpha Chroma Key"
}