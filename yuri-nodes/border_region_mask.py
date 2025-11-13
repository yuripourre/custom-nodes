import torch
import numpy as np
from scipy import ndimage

class BorderMaskFromColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "tolerance": ("INT", {"default": 10, "min": 0, "max": 127, "step": 1}),
                "process_left": ("BOOLEAN", {"default": True}),
                "process_top": ("BOOLEAN", {"default": True}),
                "process_right": ("BOOLEAN", {"default": True}),
                "process_bottom": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "mask"

    def create_mask(self, image, red, green, blue, tolerance, process_left, process_top, process_right, process_bottom):
        # Convert to numpy (B, H, W, C)
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape

        # Convert RGB values from 0-255 to 0.0-1.0 range
        target_color = np.array([red / 255.0, green / 255.0, blue / 255.0])

        # Convert tolerance from 0-127 to 0.0-0.498 range
        tolerance_normalized = tolerance / 255.0

        # Set connectivity structure (8-way)
        structure = ndimage.generate_binary_structure(2, 2)

        results = []

        for b in range(batch_size):
            img = img_np[b]

            # Find all pixels similar to the target color
            diff = np.abs(img - target_color)
            similar_mask = np.all(diff <= tolerance_normalized, axis=-1)

            # Label connected components of similar pixels
            labeled, num_features = ndimage.label(similar_mask, structure=structure)

            # Start with no pixels selected
            result_mask = np.zeros((height, width), dtype=bool)

            # Get all border pixels based on enabled borders
            border_coords = []

            # Top border
            if process_top:
                for x in range(width):
                    border_coords.append((0, x))

            # Bottom border
            if process_bottom:
                for x in range(width):
                    border_coords.append((height - 1, x))

            # Left border (excluding corners already added)
            if process_left:
                start_y = 1 if process_top else 0
                end_y = height - 1 if process_bottom else height
                for y in range(start_y, end_y):
                    border_coords.append((y, 0))

            # Right border (excluding corners already added)
            if process_right:
                start_y = 1 if process_top else 0
                end_y = height - 1 if process_bottom else height
                for y in range(start_y, end_y):
                    border_coords.append((y, width - 1))

            # Find regions that touch the border
            processed_labels = set()
            for seed_y, seed_x in border_coords:
                seed_label = labeled[seed_y, seed_x]

                if seed_label > 0 and seed_label not in processed_labels:
                    # This region touches the border
                    region = (labeled == seed_label)
                    result_mask |= region
                    processed_labels.add(seed_label)

            # Convert to binary mask
            # Found pixels (border-connected) = 0, not included = 1
            binary = (~result_mask).astype(np.float32)
            results.append(binary)

        # Convert back to torch tensor
        output = torch.from_numpy(np.stack(results, axis=0))

        return (output,)

NODE_CLASS_MAPPINGS = {
    "BorderMaskFromColor": BorderMaskFromColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BorderMaskFromColor": "Border Mask from Color"
}