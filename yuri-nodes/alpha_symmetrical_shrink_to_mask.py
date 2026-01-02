import torch
import torch.nn.functional as F
import numpy as np

class AlphaSymmetricalShrinkToMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "interpolation": (["nearest-neighbor", "bilinear", "bicubic"], {"default": "bilinear"}),
                "horizontal": ("BOOLEAN", {"default": True}),
                "vertical": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "shrink"
    CATEGORY = "alpha"

    def shrink(self, image, mask, interpolation, horizontal, vertical):
        # Map human-readable names to torch interpolation modes
        interp_mode = {
            "nearest-neighbor": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic"
        }[interpolation]

        # Ensure image is [B, H, W, C] and mask is [B, H, W]
        # If mask is [B, C, H, W], we adjust
        if len(mask.shape) == 4:
            mask = mask.mean(dim=-1)

        batch_size, height, width, channels = image.shape

        # If neither horizontal nor vertical is enabled, return original image
        if not horizontal and not vertical:
            return (image,)

        # Initialize output - if only vertical is enabled, start with original image
        # Otherwise start with zeros (horizontal will fill it)
        output = image.clone() if not horizontal else torch.zeros_like(image)

        # Process horizontal direction (rows)
        if horizontal:
            center_x = width // 2

            for b in range(batch_size):
                for y in range(height):
                    row_pixels = image[b, y]  # [W, C]
                    row_mask = mask[b, y]  # [W]

                    # --- Process Right Side ---
                    # Find indices of non-transparent pixels (mask > 0.5) to the right of center
                    right_indices = torch.where(row_mask[center_x:] > 0.5)[0]
                    if len(right_indices) > 0:
                        first_right = right_indices[0] + center_x
                        # Segment from first mask pixel to the end
                        segment_right = row_pixels[first_right:]  # [SegmentWidth, C]

                        # Prepare for interpolation if needed
                        if segment_right.shape[0] > 0:
                            # Calculate target width (from center to end of image)
                            target_width = width - center_x

                            # If segment needs resizing, use interpolation
                            if segment_right.shape[0] != target_width and target_width > 0:
                                # Prepare tensor for interpolation: [1, C, SegmentWidth, 1]
                                segment_tensor = segment_right.permute(1, 0).unsqueeze(0).unsqueeze(2)

                                # Perform interpolation
                                scaled_segment = F.interpolate(
                                    segment_tensor,
                                    size=(target_width, 1),
                                    mode=interp_mode,
                                    align_corners=False if interp_mode != "nearest" else None
                                )

                                # Reshape back to [TargetWidth, C]
                                segment_right = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                            # Place segment starting at the center
                            seg_len = min(segment_right.shape[0], target_width)
                            output[b, y, center_x : center_x + seg_len] = segment_right[:seg_len]

                    # --- Process Left Side ---
                    # Find indices of non-transparent pixels to the left of center
                    left_indices = torch.where(row_mask[:center_x] > 0.5)[0]
                    if len(left_indices) > 0:
                        last_left = left_indices[-1]
                        # Segment from the start of the line to the last mask pixel
                        segment_left = row_pixels[:last_left + 1]  # [SegmentWidth, C]

                        # Prepare for interpolation if needed
                        if segment_left.shape[0] > 0:
                            # Calculate target width (from start to center)
                            target_width = center_x

                            # If segment needs resizing, use interpolation
                            if segment_left.shape[0] != target_width and target_width > 0:
                                # Prepare tensor for interpolation: [1, C, SegmentWidth, 1]
                                segment_tensor = segment_left.permute(1, 0).unsqueeze(0).unsqueeze(2)

                                # Perform interpolation
                                scaled_segment = F.interpolate(
                                    segment_tensor,
                                    size=(target_width, 1),
                                    mode=interp_mode,
                                    align_corners=False if interp_mode != "nearest" else None
                                )

                                # Reshape back to [TargetWidth, C]
                                segment_left = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                            # Place segment ending at the center
                            seg_len = min(segment_left.shape[0], target_width)
                            output[b, y, center_x - seg_len : center_x] = segment_left[:seg_len]

        # Process vertical direction (columns)
        if vertical:
            # If horizontal was also processed, use its output as input for vertical
            working_image = output if horizontal else image
            center_y = height // 2

            for b in range(batch_size):
                for x in range(width):
                    col_pixels = working_image[b, :, x]  # [H, C]
                    col_mask = mask[b, :, x]  # [H]

                    # --- Process Bottom Side ---
                    # Find indices of non-transparent pixels (mask > 0.5) below center
                    bottom_indices = torch.where(col_mask[center_y:] > 0.5)[0]
                    if len(bottom_indices) > 0:
                        first_bottom = bottom_indices[0] + center_y
                        # Segment from first mask pixel to the end
                        segment_bottom = col_pixels[first_bottom:]  # [SegmentHeight, C]

                        # Prepare for interpolation if needed
                        if segment_bottom.shape[0] > 0:
                            # Calculate target height (from center to end of image)
                            target_height = height - center_y

                            # If segment needs resizing, use interpolation
                            if segment_bottom.shape[0] != target_height and target_height > 0:
                                # Prepare tensor for interpolation: [1, C, SegmentHeight, 1]
                                segment_tensor = segment_bottom.permute(1, 0).unsqueeze(0).unsqueeze(-1)

                                # Perform interpolation
                                scaled_segment = F.interpolate(
                                    segment_tensor,
                                    size=(target_height, 1),
                                    mode=interp_mode,
                                    align_corners=False if interp_mode != "nearest" else None
                                )

                                # Reshape back to [TargetHeight, C]
                                segment_bottom = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                            # Place segment starting at the center
                            seg_len = min(segment_bottom.shape[0], target_height)
                            output[b, center_y : center_y + seg_len, x] = segment_bottom[:seg_len]

                    # --- Process Top Side ---
                    # Find indices of non-transparent pixels above center
                    top_indices = torch.where(col_mask[:center_y] > 0.5)[0]
                    if len(top_indices) > 0:
                        last_top = top_indices[-1]
                        # Segment from the start of the column to the last mask pixel
                        segment_top = col_pixels[:last_top + 1]  # [SegmentHeight, C]

                        # Prepare for interpolation if needed
                        if segment_top.shape[0] > 0:
                            # Calculate target height (from start to center)
                            target_height = center_y

                            # If segment needs resizing, use interpolation
                            if segment_top.shape[0] != target_height and target_height > 0:
                                # Prepare tensor for interpolation: [1, C, SegmentHeight, 1]
                                segment_tensor = segment_top.permute(1, 0).unsqueeze(0).unsqueeze(-1)

                                # Perform interpolation
                                scaled_segment = F.interpolate(
                                    segment_tensor,
                                    size=(target_height, 1),
                                    mode=interp_mode,
                                    align_corners=False if interp_mode != "nearest" else None
                                )

                                # Reshape back to [TargetHeight, C]
                                segment_top = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                            # Place segment ending at the center
                            seg_len = min(segment_top.shape[0], target_height)
                            output[b, center_y - seg_len : center_y, x] = segment_top[:seg_len]

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaSymmetricalShrinkToMaskNode": AlphaSymmetricalShrinkToMaskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaSymmetricalShrinkToMaskNode": "🫟 Alpha Symmetrical Shrink to Mask"
}

