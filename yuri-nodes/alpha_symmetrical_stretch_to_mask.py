import torch
import torch.nn.functional as F

class AlphaSymmetricalStretchToMaskNode:
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
    FUNCTION = "stretch"
    CATEGORY = "alpha"

    def stretch(self, image, mask, interpolation, horizontal, vertical):
        # Map human-readable names to torch interpolation modes
        interp_mode = {
            "nearest-neighbor": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic"
        }[interpolation]

        # ComfyUI Image: [B, H, W, C]
        # ComfyUI Mask: [B, H, W]
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
            for b in range(batch_size):
                for y in range(height):
                    row_img = image[b, y]  # [W, C]
                    row_mask = mask[b, y]  # [W]

                    # Find the absolute outer bounds of image content (ignoring internal gaps)
                    # .any(dim=-1) checks if a pixel has any color/alpha content
                    img_indices = torch.where(row_img.any(dim=-1))[0]
                    # Find mask bounds (where mask is active)
                    mask_indices = torch.where(row_mask > 0.5)[0]

                    # Skip if no image content or no mask area in this row
                    if len(img_indices) == 0 or len(mask_indices) == 0:
                        continue

                    img_start, img_end = img_indices[0], img_indices[-1]
                    mask_start, mask_end = mask_indices[0], mask_indices[-1]

                    # 1. Extract the segment (including internal alpha gaps)
                    segment = row_img[img_start : img_end + 1] # [SegmentWidth, C]

                    # 2. Prepare for torch.nn.functional.interpolate
                    # Needs to be [Batch, Channels, Height, Width]
                    # We treat our segment as Width, with Height = 1
                    segment_tensor = segment.permute(1, 0).unsqueeze(0).unsqueeze(2) # [1, C, SegmentWidth, 1]

                    target_width = int((mask_end - mask_start + 1).item())

                    # 3. Perform the stretch
                    # Note: 'bicubic' and 'bilinear' require 4D input
                    scaled_segment = F.interpolate(
                        segment_tensor,
                        size=(target_width, 1),
                        mode=interp_mode,
                        align_corners=False if interp_mode != "nearest" else None
                    )

                    # Reshape back to [TargetWidth, C]
                    scaled_row_part = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                    # 4. Place scaled segment into output at mask coordinates
                    # Boundary check to prevent crashes if mask spans edge of screen
                    place_end = min(mask_start + target_width, width)
                    actual_segment_slice = scaled_row_part[:place_end - mask_start]

                    output[b, y, mask_start:place_end] = actual_segment_slice

        # Process vertical direction (columns)
        if vertical:
            # If horizontal was also processed, use its output as input for vertical
            working_image = output if horizontal else image

            for b in range(batch_size):
                for x in range(width):
                    col_img = working_image[b, :, x]  # [H, C]
                    col_mask = mask[b, :, x]  # [H]

                    # Find the absolute outer bounds of image content (ignoring internal gaps)
                    img_indices = torch.where(col_img.any(dim=-1))[0]
                    # Find mask bounds (where mask is active)
                    mask_indices = torch.where(col_mask > 0.5)[0]

                    # Skip if no image content or no mask area in this column
                    if len(img_indices) == 0 or len(mask_indices) == 0:
                        continue

                    img_start, img_end = img_indices[0], img_indices[-1]
                    mask_start, mask_end = mask_indices[0], mask_indices[-1]

                    # 1. Extract the segment (including internal alpha gaps)
                    segment = col_img[img_start : img_end + 1] # [SegmentHeight, C]

                    # 2. Prepare for torch.nn.functional.interpolate
                    # Needs to be [Batch, Channels, Height, Width]
                    # We treat our segment as Height, with Width = 1
                    segment_tensor = segment.permute(1, 0).unsqueeze(0).unsqueeze(-1) # [1, C, SegmentHeight, 1]

                    target_height = int((mask_end - mask_start + 1).item())

                    # 3. Perform the stretch
                    scaled_segment = F.interpolate(
                        segment_tensor,
                        size=(target_height, 1),
                        mode=interp_mode,
                        align_corners=False if interp_mode != "nearest" else None
                    )

                    # Reshape back to [TargetHeight, C]
                    scaled_col_part = scaled_segment.squeeze(0).squeeze(-1).permute(1, 0)

                    # 4. Place scaled segment into output at mask coordinates
                    # Boundary check to prevent crashes if mask spans edge of screen
                    place_end = min(mask_start + target_height, height)
                    actual_segment_slice = scaled_col_part[:place_end - mask_start]

                    output[b, mask_start:place_end, x] = actual_segment_slice

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "AlphaSymmetricalStretchToMaskNode": AlphaSymmetricalStretchToMaskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaSymmetricalStretchToMaskNode": "🫟 Alpha Symmetrical Stretch to Mask"
}

