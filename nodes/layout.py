import torch
import numpy as np
import cv2
import random

class SimpleTextLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 768, "min": 64, "max": 8192}),
                "large_text_count": ("INT", {"default": 1, "min": 0, "max": 10}),
                "medium_text_count": ("INT", {"default": 1, "min": 0, "max": 10}),
                "small_text_count": ("INT", {"default": 2, "min": 0, "max": 20}),
                "alignment": (["center", "left", "right", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "padding": ("INT", {"default": 32, "min": 0, "max": 512}),
                "gap": ("INT", {"default": 16, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_layout"
    CATEGORY = "6yuan/text"

    def generate_layout(self, width, height, large_text_count, medium_text_count, small_text_count, alignment, seed, padding=32, gap=16):
        random.seed(seed)
        
        # Canvas (Black background)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define heights based on resolution (approximate)
        # Large: ~15% of min dimension
        # Medium: ~8%
        # Small: ~4%
        base_dim = min(width, height)
        h_large = int(base_dim * 0.15)
        h_medium = int(base_dim * 0.08)
        h_small = int(base_dim * 0.04)
        
        # Limit widths
        max_w = width - 2 * padding
        
        # Create list of blocks to place
        # Order: Large -> Medium -> Small (Classic Hierarchy)
        blocks = []
        for _ in range(large_text_count): blocks.append(h_large)
        for _ in range(medium_text_count): blocks.append(h_medium)
        for _ in range(small_text_count): blocks.append(h_small)
        
        current_y = padding
        
        for bh in blocks:
            if current_y + bh > height - padding:
                break # No more space
            
            # Determine width for this block
            # Randomize width between 50% and 100% of max_w to add variety
            # But for Title (Large), we usually want it wider/prominent
            if bh == h_large:
                block_w = random.randint(int(max_w * 0.7), max_w)
            else:
                block_w = random.randint(int(max_w * 0.4), max_w)
            
            # Determine X based on alignment
            if alignment == "center":
                x = padding + (max_w - block_w) // 2
            elif alignment == "left":
                x = padding
            elif alignment == "right":
                x = width - padding - block_w
            else: # random
                eff_align = random.choice(["center", "left", "right"])
                if eff_align == "center":
                    x = padding + (max_w - block_w) // 2
                elif eff_align == "left":
                    x = padding
                else:
                    x = width - padding - block_w
            
            # Draw rectangle (White)
            cv2.rectangle(mask, (x, current_y), (x + block_w, current_y + bh), 255, -1)
            
            current_y += bh + gap

        # Convert to tensor [B, H, W, C]
        # Expand to 3 channels (RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        t = torch.from_numpy(mask_rgb.astype(np.float32) / 255.0)[None, ...]
        return (t,)

NODE_CLASS_MAPPINGS = {
    "SimpleTextLayout": SimpleTextLayout
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTextLayout": "Simple Text Layout Generator"
}
