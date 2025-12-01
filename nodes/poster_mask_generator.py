import json
import numpy as np
import torch


class PosterMaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1080, "min": 16, "max": 8192}),
                "height": ("INT", {"default": 1920, "min": 16, "max": 8192}),
                "visual_blocks": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("visual_mask", "text_protect_mask")
    FUNCTION = "generate"
    CATEGORY = "6yuan/poster"

    def generate(self, width, height, visual_blocks):
        try:
            blocks = json.loads(visual_blocks)
        except Exception:
            blocks = []

        mask = np.zeros((int(height), int(width)), dtype=np.uint8)

        for b in blocks if isinstance(blocks, list) else []:
            try:
                xr = float(b.get("x", 0.0))
                yr = float(b.get("y", 0.0))
                wr = float(b.get("w", 0.0))
                hr = float(b.get("h", 0.0))
            except Exception:
                continue
            x0 = int(max(0, min(width, xr * width)))
            y0 = int(max(0, min(height, yr * height)))
            x1 = int(max(0, min(width, (xr + wr) * width)))
            y1 = int(max(0, min(height, (yr + hr) * height)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255

        visual_mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        text_protect = 1.0 - (visual_mask)
        return (visual_mask, text_protect)


NODE_CLASS_MAPPINGS = {
    "PosterMaskGenerator": PosterMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PosterMaskGenerator": "Poster Mask Generator",
}

