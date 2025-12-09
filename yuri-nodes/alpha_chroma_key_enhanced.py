from typing import Dict, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch
from scipy import ndimage

class AlphaChromaKeyEnhancedNode:
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
                "invert_output": ("BOOLEAN", {"default": False}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
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
    def process(self, image, red=0, green=0, blue=0, variance=0, inner_outline=0, outline=0, 
                antialias=True, invert_output=False, process_left=True, process_top=True, 
                process_right=True, process_bottom=True):
        
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
        
        # Identify chroma key pixels (pixels that match the target color within variance)
        if not invert_output:
            # Normal chroma key: pixels matching target color (within variance)
            chroma_key_pixels = euclidean_dist <= variance
        else:
            # Inverted: pixels NOT matching the target color
            chroma_key_pixels = euclidean_dist > variance
        
        # CRITICAL: Start with ALL pixels as opaque (keep everything)
        # Only the floodfill pass will determine what becomes transparent
        # This prevents holes from being created - isolated chroma key regions
        # that are not connected to borders will remain opaque
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        # Apply border region removal (floodfill from borders)
        # This is the ONLY pass that removes pixels - no holes will be created
        if process_left or process_top or process_right or process_bottom:
            # Find border-connected regions of chroma key pixels
            structure = ndimage.generate_binary_structure(2, 2)
            labeled, num_features = ndimage.label(chroma_key_pixels, structure=structure)
            
            # Get border coordinates based on enabled borders
            border_coords = []
            
            # Top border
            if process_top:
                for x in range(w):
                    border_coords.append((0, x))
            
            # Bottom border
            if process_bottom:
                for x in range(w):
                    border_coords.append((h - 1, x))
            
            # Left border (excluding corners already added)
            if process_left:
                start_y = 1 if process_top else 0
                end_y = h - 1 if process_bottom else h
                for y in range(start_y, end_y):
                    border_coords.append((y, 0))
            
            # Right border (excluding corners already added)
            if process_right:
                start_y = 1 if process_top else 0
                end_y = h - 1 if process_bottom else h
                for y in range(start_y, end_y):
                    border_coords.append((y, w - 1))
            
            # Find regions that touch the border and remove them from the mask
            # Only border-connected chroma key regions are removed
            # Isolated chroma key regions (holes) remain opaque
            processed_labels = set()
            for seed_y, seed_x in border_coords:
                # Check if this border pixel is a chroma key pixel
                if chroma_key_pixels[seed_y, seed_x]:
                    seed_label = labeled[seed_y, seed_x]
                    
                    if seed_label > 0 and seed_label not in processed_labels:
                        # This chroma key region touches the border - remove it from mask
                        region = (labeled == seed_label)
                        mask[region] = 0
                        processed_labels.add(seed_label)
        
        mask_img = Image.fromarray(mask, mode="L")
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        
        # Convert to binary mask
        binary_mask = (mask_arr > 0.5).astype(bool)
        
        # Detect ALL edge pixels: pixels that have at least one neighbor with different value
        # Edge pixels are those that are NOT completely surrounded by same-type neighbors
        all_edge_mask = np.zeros_like(binary_mask, dtype=bool)
        
        # Check 8-connected neighbors to find edge pixels
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                # Shift the binary mask
                shifted = np.roll(np.roll(binary_mask, dy, axis=0), dx, axis=1)
                # Edge pixel: pixel that differs from at least one neighbor
                all_edge_mask |= (binary_mask != shifted)
        
        # CRITICAL: Identify holes (transparent regions completely surrounded by opaque pixels)
        # Holes should NEVER be processed - only external edges should be processed
        transparent_mask = ~binary_mask
        structure = ndimage.generate_binary_structure(2, 2)
        
        # Label all transparent regions
        labeled_transparent, num_transparent = ndimage.label(transparent_mask, structure=structure)
        
        # Identify which transparent regions are holes (not connected to border)
        # A hole is a transparent region that doesn't touch any image border
        # We check ALL borders regardless of processing settings, since a hole is
        # defined as a region that doesn't touch ANY border of the image
        hole_labels = set()
        border_coords_for_holes = []
        
        # Collect ALL border coordinates (all edges of the image)
        # Top border
        for x in range(w):
            border_coords_for_holes.append((0, x))
        # Bottom border
        for x in range(w):
            border_coords_for_holes.append((h - 1, x))
        # Left border (excluding corners already added)
        for y in range(1, h - 1):
            border_coords_for_holes.append((y, 0))
        # Right border (excluding corners already added)
        for y in range(1, h - 1):
            border_coords_for_holes.append((y, w - 1))
        
        # Find transparent regions that touch the border (these are NOT holes)
        border_connected_transparent_labels = set()
        for seed_y, seed_x in border_coords_for_holes:
            label = labeled_transparent[seed_y, seed_x]
            if label > 0:
                border_connected_transparent_labels.add(label)
        
        # All transparent regions that don't touch the border are holes
        for label in range(1, num_transparent + 1):
            if label not in border_connected_transparent_labels:
                hole_labels.add(label)
        
        # Create mask of hole regions
        hole_mask = np.zeros_like(binary_mask, dtype=bool)
        for hole_label in hole_labels:
            hole_mask |= (labeled_transparent == hole_label)
        
        # Identify hole edge pixels: opaque edge pixels that border holes
        # These should NOT be processed - only external edges should be processed
        hole_edge_mask = np.zeros_like(binary_mask, dtype=bool)
        if np.any(hole_mask):
            # Check if any neighbor of an opaque edge pixel is a hole
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # Shift the hole mask to check neighbors
                    shifted_hole = np.roll(np.roll(hole_mask, dy, axis=0), dx, axis=1)
                    # Hole edge: opaque pixel that is on an edge AND adjacent to a hole
                    hole_edge_mask |= (binary_mask & all_edge_mask & shifted_hole)
        
        # External edges are all edges EXCEPT hole edges
        # Only external edges should be processed (edges of the main region, not holes)
        edge_mask = all_edge_mask & ~hole_edge_mask
        
        # Detect interior pixels: pixels where ALL neighbors are the same type
        # These should NEVER be processed
        interior_mask = ~all_edge_mask
        
        # Store the original mask before any outline operations
        original_mask_before_outline = mask_arr.copy()
        
        # Create edge region mask: dilate edge pixels to include area for outline operations
        max_outline_distance = max(inner_outline, outline)
        edge_region_mask = edge_mask.copy()
        
        if max_outline_distance > 0 and np.any(edge_mask):
            # Dilate edge pixels to create a region around edges for outline operations
            structure = ndimage.generate_binary_structure(2, 2)
            edge_region_mask = ndimage.binary_dilation(edge_mask, structure=structure, 
                                                      iterations=max_outline_distance).astype(bool)
            # CRITICAL: Exclude interior pixels and hole regions from edge region
            # Interior pixels and holes should never be processed
            edge_region_mask = edge_region_mask & ~interior_mask & ~hole_mask
        
        # Apply inner outline erosion if needed (before outline expansion)
        if inner_outline > 0 and np.any(edge_mask):
            kernel_size = inner_outline * 2 + 1
            eroded_mask_img = mask_img.filter(ImageFilter.MinFilter(kernel_size))
            eroded_mask_arr = np.array(eroded_mask_img, dtype=np.float32) / 255.0
            
            # Only apply erosion to edge pixels (not interior pixels)
            # Edge pixels that are part of opaque regions can be eroded
            erosion_applicable = edge_mask & (mask_arr > 0.5)
            mask_arr = np.where(erosion_applicable, eroded_mask_arr, mask_arr)
            
            # CRITICAL: Restore interior pixels - they should NEVER change
            mask_arr = np.where(interior_mask, original_mask_before_outline, mask_arr)
            
            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        
        # Store the mask after inner outline for outline processing
        original_mask_arr = mask_arr.copy()
        
        # Track which pixels will be modified
        modified_pixels = np.zeros((h, w), dtype=bool)
        
        # Apply outline expansion if needed
        if outline > 0 and np.any(edge_mask):
            kernel_size = outline * 2 + 1
            expanded_mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel_size))
            
            # Calculate outline area (expanded minus original)
            expanded_mask_arr = np.array(expanded_mask_img, dtype=np.float32) / 255.0
            outline_area = expanded_mask_arr - original_mask_arr
            outline_area = np.clip(outline_area, 0, 1)
            
            # Only apply outline in edge region (not interior)
            outline_area = np.where(edge_region_mask, outline_area, 0)
            
            # Blend: use expanded mask in edge regions, original mask elsewhere
            final_mask_arr = np.where(edge_region_mask, expanded_mask_arr, original_mask_arr)
            
            # CRITICAL: Restore interior pixels - they should NEVER change
            final_mask_arr = np.where(interior_mask, original_mask_before_outline, final_mask_arr)
            outline_area = np.where(interior_mask, 0, outline_area)
            
            # Mark outline area pixels as modified (RGB will change)
            modified_pixels |= (outline_area > 0)
            
            # Fill outline area with the chroma key color
            background_color = np.array([red, green, blue], dtype=np.uint8)
            
            # Apply outline color to RGB channels where outline exists
            for c in range(3):
                arr[..., c] = np.where(outline_area > 0, background_color[c], arr[..., c])
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
    "AlphaChromaKeyEnhancedNode": AlphaChromaKeyEnhancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaChromaKeyEnhancedNode": "Alpha Chroma Key Enhanced"
}

