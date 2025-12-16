from typing import Dict, Tuple
import numpy as np
import torch
from scipy import ndimage

class AlphaStrokeNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "stroke_size": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "offset": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "antialias_radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_antialias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "apply_alpha_stroke"
    CATEGORY = "image/filter"

    def apply_alpha_stroke(self, image, stroke_size, opacity, red=0, green=0, blue=0, 
                          offset=0, antialias_radius=0.5, alpha_threshold=0.5, enable_antialias=True):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape

        # Handle RGB input by adding alpha channel
        if channels == 3:
            alpha = np.ones((batch_size, height, width, 1), dtype=np.float32)
            img_np = np.concatenate([img_np, alpha], axis=-1)
            channels = 4

        results = []

        for b in range(batch_size):
            # Get single image
            img_single = img_np[b]  # [H, W, C]

            # Extract alpha channel to determine foreground/background
            alpha_channel = img_single[:, :, 3]

            # Create binary mask: pixels that are NOT transparent
            binary_mask = (alpha_channel >= alpha_threshold).astype(np.float32)

            # Apply offset: positive = expand outward (dilate), negative = shrink inward (erode)
            if offset != 0:
                structure = ndimage.generate_binary_structure(2, 2)
                if offset > 0:
                    # Positive offset: dilate (expand outward)
                    binary_mask = ndimage.binary_dilation(
                        binary_mask.astype(bool),
                        structure=structure,
                        iterations=offset
                    ).astype(np.float32)
                else:
                    # Negative offset: erode (shrink inward)
                    binary_mask = ndimage.binary_erosion(
                        binary_mask.astype(bool),
                        structure=structure,
                        iterations=-offset
                    ).astype(np.float32)

            # Better edge detection: find pixels where the mask differs from at least one neighbor
            # This ensures we only get actual edge pixels, not interior pixels
            edge_mask = np.zeros_like(binary_mask, dtype=np.float32)
            
            # Check 8-connected neighbors to find edge pixels
            # An edge pixel is one where the current pixel differs from at least one neighbor
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # Shift the binary mask
                    shifted_mask = np.roll(np.roll(binary_mask, dy, axis=0), dx, axis=1)
                    # Edge pixel if current pixel differs from neighbor
                    edge_mask = np.maximum(edge_mask, np.abs(binary_mask - shifted_mask))

            # Only keep edge pixels that are OUTSIDE the shape (transparent side of the edge)
            # Edge pixels on the transparent side (where binary_mask is 0)
            contour_mask = edge_mask * (1.0 - binary_mask)

            # Exclude edges where the opaque shape touches the image boundaries
            # Check if any opaque pixels (binary_mask) are at the image edges
            shape_touches_edge = (
                np.any(binary_mask[0, :] > 0.5) or   # Top edge
                np.any(binary_mask[-1, :] > 0.5) or  # Bottom edge
                np.any(binary_mask[:, 0] > 0.5) or   # Left edge
                np.any(binary_mask[:, -1] > 0.5)     # Right edge
            )
            
            # If shape touches any edge, find connected regions and exclude those edges
            if shape_touches_edge:
                # Dilate binary_mask to find regions connected to edges
                structure = ndimage.generate_binary_structure(2, 2)
                # Start with border pixels
                border_mask = np.zeros_like(binary_mask, dtype=bool)
                border_mask[0, :] = binary_mask[0, :] > 0.5    # Top edge
                border_mask[-1, :] = binary_mask[-1, :] > 0.5  # Bottom edge
                border_mask[:, 0] = binary_mask[:, 0] > 0.5    # Left edge
                border_mask[:, -1] = binary_mask[:, -1] > 0.5  # Right edge
                
                # Expand border region to cover areas connected to the edge
                MAX_EXPANSION = max(stroke_size + 10, 20)
                for _ in range(MAX_EXPANSION):
                    expanded = ndimage.binary_dilation(border_mask, structure=structure)
                    # Only expand within the binary_mask
                    expanded = expanded & (binary_mask > 0.5)
                    if np.array_equal(expanded, border_mask):
                        break
                    border_mask = expanded
                
                # Dilate once more to get the edge region
                edge_exclusion = ndimage.binary_dilation(border_mask, structure=structure, iterations=2)
                
                # Exclude contour pixels near edges that touch the boundary
                contour_mask = contour_mask * (1.0 - edge_exclusion.astype(np.float32))

            # Apply stroke_size to dilate the contour if needed
            # contour_mask is already 1 pixel thick, so we dilate by (stroke_size - 1) to get the desired thickness
            if stroke_size > 1 and np.any(contour_mask > 0):
                # Use scipy dilation with 8-connected structure
                structure = ndimage.generate_binary_structure(2, 2)
                # Dilate by (stroke_size - 1) iterations to achieve the desired thickness
                # stroke_size=1 → no dilation (1 pixel thick)
                # stroke_size=2 → dilate by 1 (2 pixels thick)
                # stroke_size=3 → dilate by 2 (3 pixels thick)
                contour_mask = ndimage.binary_dilation(
                    contour_mask.astype(bool),
                    structure=structure,
                    iterations=stroke_size - 1
                ).astype(np.float32)
            elif stroke_size == 0:
                # No stroke requested
                contour_mask = np.zeros_like(binary_mask, dtype=np.float32)

            # Store the sharp contour mask for the core stroke
            sharp_mask = contour_mask.copy()

            # Create professional vector-style antialiasing:
            # 1. Sharp core stroke (controlled by opacity)
            # 2. Feathered/antialiased edges extending beyond the sharp stroke
            
            if enable_antialias and antialias_radius > 0.0 and np.any(contour_mask > 0):
                # Create expanded antialiased edge zone by blurring the sharp mask
                # This creates smooth feathered edges that extend beyond the sharp stroke
                antialiased_edge = ndimage.gaussian_filter(
                    contour_mask,
                    sigma=antialias_radius,
                    truncate=4.0
                )
                antialiased_edge = np.clip(antialiased_edge, 0.0, 1.0)
                
                # Apply smooth curve for better falloff
                antialiased_edge = antialiased_edge * antialiased_edge * (3.0 - 2.0 * antialiased_edge)
            else:
                # No antialiasing: use sharp mask
                antialiased_edge = contour_mask

            # Start with the original image
            result_arr = img_single.copy()

            # Apply stroke color
            stroke_color = np.array([red, green, blue], dtype=np.float32) / 255.0

            # Professional vector-style compositing:
            # Layer 1: Antialiased feathered edges (soft, extends beyond core)
            # Layer 2: Sharp core stroke on top (full opacity, controlled by opacity parameter)
            
            # First, apply the antialiased edge layer
            antialias_alpha = antialiased_edge * opacity
            for c in range(3):
                # Blend antialiased edges with original
                result_arr[:, :, c] = (
                    result_arr[:, :, c] * (1.0 - antialias_alpha) +
                    stroke_color[c] * antialiased_edge * antialias_alpha
                )
            
            # Update alpha for antialiased edges
            result_arr[:, :, 3] = np.maximum(result_arr[:, :, 3], antialias_alpha)
            
            # Then, draw sharp core stroke on top at full opacity
            # This ensures the center is crisp and only controlled by opacity
            sharp_alpha = sharp_mask * opacity
            for c in range(3):
                # Overwrite with sharp stroke color where sharp mask exists
                result_arr[:, :, c] = np.where(
                    sharp_mask > 0.5,
                    result_arr[:, :, c] * (1.0 - sharp_alpha) + stroke_color[c] * sharp_alpha,
                    result_arr[:, :, c]
                )
            
            # Update alpha for sharp core
            result_arr[:, :, 3] = np.maximum(result_arr[:, :, 3], sharp_alpha)

            # Convert back to tensor [H,W,C]
            result_tensor = torch.from_numpy(result_arr)

            results.append(result_tensor)

        # Stack results into batch [B,H,W,C]
        output = torch.stack(results, dim=0)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaStrokeNode": AlphaStrokeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaStrokeNode": "Alpha Stroke"
}

