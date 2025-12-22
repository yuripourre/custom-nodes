"""
Alpha Chroma Key Pro - Professional automatic chroma keying with color decontamination

This node provides Photoshop/Nuke-level quality chroma keying with:
- Automatic background color detection
- Color decontamination (removes background color from edges)
- Multi-scale matte refinement for ultra-smooth gradients
- Independent control over keying aggressiveness and edge quality

PARAMETER GUIDE:
================

Key Parameters (adjust these first):
- key_threshold (0.10-0.20): Lower = more aggressive keying
- edge_feather (0.10-0.25): Controls edge softness (independent of threshold)
- color_match_power (1.0-2.0): Color sensitivity (1.0 = balanced, higher = more conservative)

Refinement:
- edge_softness (2.0-3.0): Additional edge smoothing via distance transform
- spill_removal_strength (0.7-1.0): How much to remove background color from edges
- matte_refinement (1.5-2.5): Multi-scale smoothing strength

Advanced:
- edge_erosion (0.0-1.0): Pull edges inward slightly
- edge_gamma (0.8-1.2): Control alpha gradient curve
- aggressive_despill: Enable for stubborn color spill
- preserve_highlights: Keep bright areas opaque

RECOMMENDED WORKFLOW:
====================
1. Start with defaults
2. Adjust key_threshold until background is removed
3. Adjust edge_feather to get desired edge softness
4. Fine-tune color_match_power if foreground is being keyed
5. Increase spill_removal_strength if you see color halos
"""

from typing import Dict, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter

class AlphaChromaKeyAutoNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "detection_method": (["corners", "edges", "mode"], {"default": "corners"}),
                "sample_size": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "color_match_power": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "key_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_erosion": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "edge_feather": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_gamma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "edge_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "matte_refinement": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "spill_removal_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aggressive_despill": ("BOOLEAN", {"default": True}),
                "preserve_highlights": ("BOOLEAN", {"default": True}),
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

    # ---------- Background color detection ----------
    def detect_background_color(self, rgb, method="corners", sample_size=20):
        """
        Automatically detect the background color from the image.

        Args:
            rgb: RGB array (H, W, 3)
            method: Detection method - 'corners', 'edges', or 'mode'
            sample_size: Size of sample area to use

        Returns:
            Detected background color as [R, G, B] array
        """
        h, w = rgb.shape[:2]

        if method == "corners":
            # Sample from all four corners
            samples = []

            # Top-left
            tl_sample = rgb[0:sample_size, 0:sample_size]
            samples.append(tl_sample.reshape(-1, 3))

            # Top-right
            tr_sample = rgb[0:sample_size, w-sample_size:w]
            samples.append(tr_sample.reshape(-1, 3))

            # Bottom-left
            bl_sample = rgb[h-sample_size:h, 0:sample_size]
            samples.append(bl_sample.reshape(-1, 3))

            # Bottom-right
            br_sample = rgb[h-sample_size:h, w-sample_size:w]
            samples.append(br_sample.reshape(-1, 3))

            # Combine all samples
            all_samples = np.vstack(samples)

            # Use median to be robust to outliers
            bg_color = np.median(all_samples, axis=0)

        elif method == "edges":
            # Sample from all edges
            samples = []

            # Top edge
            top_sample = rgb[0:sample_size, :]
            samples.append(top_sample.reshape(-1, 3))

            # Bottom edge
            bottom_sample = rgb[h-sample_size:h, :]
            samples.append(bottom_sample.reshape(-1, 3))

            # Left edge (excluding corners)
            left_sample = rgb[sample_size:h-sample_size, 0:sample_size]
            samples.append(left_sample.reshape(-1, 3))

            # Right edge (excluding corners)
            right_sample = rgb[sample_size:h-sample_size, w-sample_size:w]
            samples.append(right_sample.reshape(-1, 3))

            all_samples = np.vstack(samples)
            bg_color = np.median(all_samples, axis=0)

        elif method == "mode":
            # Find the most common color in the image
            # Quantize to reduce color space
            QUANTIZE_LEVELS = 32
            rgb_quantized = (rgb // (256 // QUANTIZE_LEVELS)).astype(np.int32)

            # Flatten and find mode
            pixels = rgb_quantized.reshape(-1, 3)

            # Create unique color keys
            color_keys = pixels[:, 0] * (QUANTIZE_LEVELS**2) + pixels[:, 1] * QUANTIZE_LEVELS + pixels[:, 2]

            # Find most common color
            unique_keys, counts = np.unique(color_keys, return_counts=True)
            most_common_idx = np.argmax(counts)
            most_common_key = unique_keys[most_common_idx]

            # Decode back to RGB
            b = most_common_key % QUANTIZE_LEVELS
            g = (most_common_key // QUANTIZE_LEVELS) % QUANTIZE_LEVELS
            r = most_common_key // (QUANTIZE_LEVELS**2)

            bg_color = np.array([r, g, b]) * (256 // QUANTIZE_LEVELS) + (128 // QUANTIZE_LEVELS)

        else:
            bg_color = np.array([0, 0, 0])

        return bg_color.astype(np.float32)

    # ---------- Distance calculation ----------
    def calculate_color_distance(self, rgb, target_color, color_match_power=1.5):
        """
        Calculate perceptual distance from target color.

        Uses a weighted Euclidean distance with optional power transformation
        to emphasize color differences.

        Args:
            rgb: RGB array (H, W, 3)
            target_color: Background color [R, G, B]
            color_match_power: Controls color sensitivity
                             - 1.0 = linear (balanced)
                             - < 1.0 = more tolerant (easier to key, but may key foreground)
                             - > 1.0 = less tolerant (preserves more, but harder to key background)
                             Recommended: 1.0-2.0
        """
        # Convert to float
        rgb_float = rgb.astype(np.float32)
        target_float = target_color.astype(np.float32)

        # Calculate difference per channel
        diff = rgb_float - target_float

        # Weighted Euclidean distance (slight green bias for common green screens)
        # This is more perceptually accurate than simple Euclidean
        weights = np.array([1.0, 1.1, 1.0])  # Slightly boost green sensitivity
        weighted_diff = diff * weights.reshape(1, 1, 3)

        # Calculate distance
        euclidean_dist = np.sqrt(np.sum(weighted_diff ** 2, axis=-1))

        # Normalize to 0-1 range
        # Max weighted distance is sqrt((255^2 * 1.0) + (255^2 * 1.1) + (255^2 * 1.0))
        MAX_DISTANCE = np.sqrt(255.0**2 * (1.0**2 + 1.1**2 + 1.0**2))
        normalized_dist = euclidean_dist / MAX_DISTANCE

        # Apply power transformation to control sensitivity
        # Power < 1.0: expands the distance (more aggressive keying)
        # Power > 1.0: compresses the distance (more conservative keying)
        # Note: We use 1.0 / power for intuitive parameter behavior
        if color_match_power != 1.0:
            distance_map = np.power(normalized_dist, 1.0 / color_match_power)
        else:
            distance_map = normalized_dist

        return distance_map

    # ---------- Matte generation ----------
    def generate_initial_matte(self, distance_map, threshold, feather_range):
        """
        Generate initial alpha matte from distance map.

        Creates a soft gradient based on distance from background color.

        Args:
            distance_map: Normalized color distance (0-1)
            threshold: Key threshold - pixels below this are transparent
            feather_range: Width of the transition zone (0 = hard edge, higher = softer)
        """
        # Create soft matte with gradient falloff
        # Pixels close to background color (low distance) = transparent
        # Pixels far from background (high distance) = opaque

        if feather_range < 0.001:
            # Hard threshold (binary mask) - no feathering
            matte = (distance_map > threshold).astype(np.float32)
        else:
            # Soft threshold with smooth transition
            # Transition zone: from threshold to (threshold + feather_range)
            # This creates a gradient independent of the threshold value

            # Linear ramp
            matte = np.clip((distance_map - threshold) / feather_range, 0.0, 1.0)

            # Apply smoothstep for smoother gradient (optional, could be parameter)
            # This removes the linear look and makes it more natural
            matte = matte * matte * (3.0 - 2.0 * matte)

        return matte

    # ---------- Edge refinement ----------
    def refine_edges(self, matte, edge_softness, edge_erosion, edge_gamma=1.0):
        """
        Refine matte edges using distance transforms and morphological operations.

        This creates smooth, natural-looking edges without artifacts.
        edge_gamma controls the alpha gradient curve (< 1.0 = more transparency, > 1.0 = more opacity)
        """
        if edge_softness <= 0 and edge_erosion <= 0:
            return matte

        # Create binary mask
        BINARY_THRESHOLD = 0.5
        binary_mask = (matte > BINARY_THRESHOLD).astype(bool)

        if not np.any(binary_mask):
            return matte

        # Apply erosion if needed
        if edge_erosion > 0:
            structure = ndimage.generate_binary_structure(2, 2)
            iterations = int(np.ceil(edge_erosion))
            binary_mask = ndimage.binary_erosion(binary_mask, structure=structure, iterations=iterations)

        # Calculate distance from edge for smooth falloff
        if edge_softness > 0:
            # Distance transform inside foreground
            dist_inside = ndimage.distance_transform_edt(binary_mask)
            # Distance transform outside foreground
            dist_outside = ndimage.distance_transform_edt(~binary_mask)

            # Create smooth gradient at edges
            # Inside: gradually increase from edge
            # Outside: gradually decrease from edge
            edge_distance = dist_inside - dist_outside

            # Apply smooth transition
            refined_matte = np.clip((edge_distance + edge_softness) / (2 * edge_softness), 0.0, 1.0)

            # Apply smoothstep for even smoother falloff
            refined_matte = refined_matte * refined_matte * (3.0 - 2.0 * refined_matte)

            # Apply gamma correction to control the gradient curve
            # This gives fine control over the alpha ramp
            if edge_gamma != 1.0:
                refined_matte = np.power(refined_matte, edge_gamma)
        else:
            refined_matte = binary_mask.astype(np.float32)

        return refined_matte

    # ---------- Matte refinement ----------
    def refine_matte_advanced(self, matte, refinement_strength):
        """
        Apply advanced matte refinement to remove noise and smooth transitions.
        Uses multi-scale smoothing for ultra-smooth alpha gradients.
        """
        if refinement_strength <= 0:
            return matte

        # Multi-scale smoothing: process at different scales and combine
        # This creates ultra-smooth gradients without banding
        scales = [0.5, 1.0, 2.0]
        refined_layers = []

        for scale in scales:
            sigma = refinement_strength * scale
            smoothed = gaussian_filter(matte, sigma=sigma)
            refined_layers.append(smoothed)

        # Combine scales using weighted average
        # More weight to medium scale, less to extremes
        weights = [0.2, 0.6, 0.2]
        result = np.zeros_like(matte)

        for layer, weight in zip(refined_layers, weights):
            result += layer * weight

        # Edge-aware blending: preserve sharp edges where needed
        # Calculate local variance to detect edges
        WINDOW_SIZE = 5
        mean_filter = gaussian_filter(matte, sigma=WINDOW_SIZE / 2.0)
        sq_mean_filter = gaussian_filter(matte * matte, sigma=WINDOW_SIZE / 2.0)
        variance = np.maximum(sq_mean_filter - mean_filter * mean_filter, 0.0)

        # Edges have high variance
        edge_strength = np.sqrt(variance)

        # Blend between original and refined based on edge strength
        EDGE_PRESERVATION = 0.3
        blend_factor = 1.0 - (edge_strength * EDGE_PRESERVATION)
        result = matte * (1.0 - blend_factor) + result * blend_factor

        # Final subtle smoothing to eliminate any banding
        ANTI_BANDING_SIGMA = 0.5
        result = gaussian_filter(result, sigma=ANTI_BANDING_SIGMA)

        return result

    # ---------- Color decontamination ----------
    def decontaminate_colors(self, rgb, matte, bg_color, strength, aggressive=True):
        """
        Remove background color contamination from semi-transparent edges.

        This solves the equation: observed = foreground * alpha + background * (1 - alpha)
        to extract the true foreground color without background color bleeding.
        """
        if strength <= 0:
            return rgb

        # Work in float32
        rgb_float = rgb.astype(np.float32) / 255.0
        bg_float = bg_color / 255.0

        # Identify edge/transition regions where color contamination occurs
        # These are semi-transparent pixels that contain a mix of fg/bg
        MIN_ALPHA_THRESHOLD = 0.05
        MAX_ALPHA_THRESHOLD = 0.95
        edge_region = (matte > MIN_ALPHA_THRESHOLD) & (matte < MAX_ALPHA_THRESHOLD)

        # Also process nearby opaque pixels in aggressive mode
        if aggressive:
            structure = ndimage.generate_binary_structure(2, 2)
            DILATION_ITERATIONS = 3
            edge_region = ndimage.binary_dilation(edge_region, structure=structure, iterations=DILATION_ITERATIONS)
            edge_region = edge_region & (matte > MIN_ALPHA_THRESHOLD)

        if not np.any(edge_region):
            return rgb

        result = rgb_float.copy()

        # For each semi-transparent pixel, solve for the true foreground color
        # Formula: foreground = (observed - background * (1 - alpha)) / alpha
        # This removes the background color contribution

        edge_pixels = edge_region
        edge_alpha = matte[edge_pixels]

        # Prevent division by zero
        EPSILON = 0.001
        edge_alpha_safe = np.maximum(edge_alpha, EPSILON)

        # Calculate background contribution
        bg_contribution = 1.0 - edge_alpha

        # Solve for foreground color for each channel
        for c in range(3):
            observed = rgb_float[:, :, c][edge_pixels]
            bg_value = bg_float[c]

            # Remove background contamination
            # foreground = (observed - bg * (1 - alpha)) / alpha
            decontaminated = (observed - bg_value * bg_contribution) / edge_alpha_safe

            # Apply strength parameter (blend between original and decontaminated)
            blended = observed * (1.0 - strength) + decontaminated * strength

            # Clamp to valid range
            result[:, :, c][edge_pixels] = np.clip(blended, 0.0, 1.0)

        # Additional spill suppression for remaining color cast
        if aggressive:
            # Calculate how close each pixel is to background color
            diff = result - bg_float.reshape(1, 1, 3)
            distance_to_bg = np.sqrt(np.sum(diff ** 2, axis=-1))

            # Identify pixels still too close to background color
            SPILL_THRESHOLD = 0.2
            has_spill = edge_region & (distance_to_bg < SPILL_THRESHOLD)

            if np.any(has_spill):
                # For pixels with remaining spill, desaturate and shift color
                spill_factor = np.clip(1.0 - (distance_to_bg / SPILL_THRESHOLD), 0.0, 1.0)
                spill_factor = spill_factor * strength

                # Calculate desaturated version (shift toward average)
                luminance = (0.299 * result[:, :, 0] +
                           0.587 * result[:, :, 1] +
                           0.114 * result[:, :, 2])

                # Shift the most similar channel to bg toward luminance
                for c in range(3):
                    channel_similarity = 1.0 - np.abs(result[:, :, c] - bg_float[c])
                    despill_amount = spill_factor * channel_similarity
                    result[:, :, c] = np.where(
                        has_spill,
                        result[:, :, c] * (1.0 - despill_amount) + luminance * despill_amount,
                        result[:, :, c]
                    )

        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result

    # ---------- Highlight preservation ----------
    def preserve_bright_highlights(self, matte, rgb, threshold=220):
        """
        Preserve bright highlights that might be incorrectly keyed out.

        Bright areas in the foreground should remain opaque.
        """
        # Calculate luminance
        luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Identify bright highlights
        highlights = luminance > threshold

        # Boost matte in highlight areas
        highlight_boost = highlights.astype(np.float32) * 0.5

        result = np.clip(matte + highlight_boost, 0.0, 1.0)

        return result

    # ---------- Main processing ----------
    def process(self, image, detection_method="corners", sample_size=20,
                color_match_power=1.0, key_threshold=0.15, edge_erosion=0.5,
                edge_feather=0.15, edge_gamma=1.0, edge_softness=2.0,
                matte_refinement=1.5, spill_removal_strength=0.8,
                aggressive_despill=True, preserve_highlights=True):

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
        rgb = arr[..., :3]

        # Store original alpha channel
        orig_alpha = arr[..., 3].astype(np.float32) / 255.0

        # Step 1: Detect background color automatically
        bg_color = self.detect_background_color(rgb, method=detection_method, sample_size=sample_size)

        print(f"Detected background color: RGB({int(bg_color[0])}, {int(bg_color[1])}, {int(bg_color[2])})")

        # Step 2: Calculate color distance map
        distance_map = self.calculate_color_distance(rgb, bg_color, color_match_power)

        # Step 3: Generate initial matte with proper feathering
        matte = self.generate_initial_matte(distance_map, key_threshold, edge_feather)

        # Step 4: Refine edges for smooth transitions
        matte = self.refine_edges(matte, edge_softness, edge_erosion, edge_gamma)

        # Step 5: Advanced matte refinement
        matte = self.refine_matte_advanced(matte, matte_refinement)

        # Step 6: Preserve bright highlights if enabled
        if preserve_highlights:
            matte = self.preserve_bright_highlights(matte, rgb)

        # Step 7: Decontaminate colors (remove background color from semi-transparent edges)
        rgb_despilled = self.decontaminate_colors(rgb, matte, bg_color,
                                                   spill_removal_strength, aggressive_despill)

        # Step 8: Combine with original alpha channel
        final_alpha = orig_alpha * matte

        # Update array
        arr[..., :3] = rgb_despilled
        arr[..., 3] = np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)

        out_img = Image.fromarray(arr, mode="RGBA")
        return (self.pil_to_tensor(out_img),)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaChromaKeyAutoNode": AlphaChromaKeyAutoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaChromaKeyAutoNode": "🎯 Alpha Chroma Key Pro"
}

