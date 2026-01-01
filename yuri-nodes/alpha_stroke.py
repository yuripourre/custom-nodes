import numpy as np
import torch
from scipy import ndimage

class AlphaStrokeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stroke_size": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "offset": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "inner_softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "outer_softness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "falloff_curve": (["linear", "smoothstep", "smootherstep", "cosine"], {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "apply_alpha_stroke"
    CATEGORY = "alpha"

    def _apply_falloff_curve(self, mask, curve_type):
        """Apply different falloff curves to the stroke mask for artistic control."""
        if curve_type == "linear":
            return mask
        elif curve_type == "smoothstep":
            # Hermite interpolation: 3t^2 - 2t^3
            return mask * mask * (3.0 - 2.0 * mask)
        elif curve_type == "smootherstep":
            # Ken Perlin's improved version: 6t^5 - 15t^4 + 10t^3
            return mask * mask * mask * (mask * (mask * 6.0 - 15.0) + 10.0)
        elif curve_type == "cosine":
            # Cosine interpolation
            return (1.0 - np.cos(mask * np.pi)) * 0.5
        else:
            return mask

    def apply_alpha_stroke(self, image, stroke_size, opacity, red=0, green=0, blue=0,
                          offset=0.0, inner_softness=0.0, outer_softness=1.0,
                          alpha_threshold=0.5, falloff_curve="smoothstep"):
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
            binary_mask = (alpha_channel >= alpha_threshold).astype(bool)

            if not np.any(binary_mask):
                # No opaque pixels, return original image
                results.append(torch.from_numpy(img_single))
                continue

            # Exclude edges where the opaque shape touches the image boundaries
            shape_touches_edge = (
                np.any(binary_mask[0, :]) or   # Top edge
                np.any(binary_mask[-1, :]) or  # Bottom edge
                np.any(binary_mask[:, 0]) or   # Left edge
                np.any(binary_mask[:, -1])     # Right edge
            )

            # Create exclusion mask for border-touching regions
            border_exclusion = np.zeros_like(binary_mask, dtype=bool)
            if shape_touches_edge:
                structure = ndimage.generate_binary_structure(2, 2)
                border_mask = np.zeros_like(binary_mask, dtype=bool)
                border_mask[0, :] = binary_mask[0, :]    # Top edge
                border_mask[-1, :] = binary_mask[-1, :]  # Bottom edge
                border_mask[:, 0] = binary_mask[:, 0]    # Left edge
                border_mask[:, -1] = binary_mask[:, -1]  # Right edge

                # Expand border region to cover areas connected to the edge
                MAX_EXPANSION = int(max(stroke_size + outer_softness + 10, 20))
                for _ in range(MAX_EXPANSION):
                    expanded = ndimage.binary_dilation(border_mask, structure=structure)
                    expanded = expanded & binary_mask
                    if np.array_equal(expanded, border_mask):
                        break
                    border_mask = expanded

                # Create exclusion zone around border regions
                border_exclusion = ndimage.binary_dilation(
                    border_mask,
                    structure=structure,
                    iterations=int(stroke_size + outer_softness + 2)
                )

            # ===== SDF-based stroke generation =====

            # Compute distance from the edge using SDF
            # Distance inside the shape (positive distances)
            # Use distance_transform_edt for Euclidean distance transform
            try:
                dist_inside = ndimage.distance_transform_edt(binary_mask)
                dist_outside = ndimage.distance_transform_edt(~binary_mask)
            except AttributeError:
                # Fallback for environments where distance_transform_edt is not available
                # Use standard distance_transform with euclidean metric
                dist_inside = ndimage.distance_transform(binary_mask.astype(np.uint8))
                dist_outside = ndimage.distance_transform((~binary_mask).astype(np.uint8))

            # Create signed distance field: negative inside, positive outside
            sdf = dist_outside - dist_inside

            # Apply offset to the SDF
            # Positive offset pushes the stroke outward
            # Negative offset pulls it inward
            sdf_offset = sdf - offset

            # Define stroke boundaries
            # Inner edge: where stroke meets the shape interior
            # Outer edge: where stroke fades to transparent
            inner_edge = -inner_softness
            outer_edge = stroke_size + outer_softness

            # Calculate stroke mask using SDF with smooth falloff
            stroke_mask = np.zeros_like(sdf, dtype=np.float32)

            if stroke_size > 0:
                # Inner transition: from inner_edge to 0
                # Outer transition: from stroke_size to outer_edge
                if inner_softness > 0:
                    # Smooth inner edge
                    inner_ramp = np.clip((sdf_offset - inner_edge) / (0 - inner_edge + 1e-6), 0.0, 1.0)
                else:
                    # Sharp inner edge
                    inner_ramp = (sdf_offset >= 0).astype(np.float32)

                if outer_softness > 0:
                    # Smooth outer edge
                    outer_ramp = np.clip((outer_edge - sdf_offset) / (outer_softness + 1e-6), 0.0, 1.0)
                else:
                    # Sharp outer edge
                    outer_ramp = (sdf_offset <= stroke_size).astype(np.float32)

                # Combine inner and outer ramps
                stroke_mask = inner_ramp * outer_ramp

                # Apply falloff curve for more artistic control
                stroke_mask = self._apply_falloff_curve(stroke_mask, falloff_curve)

            # Apply border exclusion
            stroke_mask = stroke_mask * (~border_exclusion).astype(np.float32)

            # Apply opacity
            stroke_mask = stroke_mask * opacity

            # Start with the original image
            result_arr = img_single.copy()

            # Apply stroke color
            stroke_color = np.array([red, green, blue], dtype=np.float32) / 255.0

            # Composite stroke onto image
            for c in range(3):
                result_arr[:, :, c] = (
                    result_arr[:, :, c] * (1.0 - stroke_mask) +
                    stroke_color[c] * stroke_mask
                )

            # Update alpha channel
            result_arr[:, :, 3] = np.maximum(result_arr[:, :, 3], stroke_mask)

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
    "AlphaStrokeNode": "🫟 Alpha Stroke"
}

