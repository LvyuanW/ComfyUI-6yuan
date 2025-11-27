import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class StringPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 256, "min": 64, "max": 4096}),
                "font_size": ("INT", {"default": 16, "min": 8, "max": 256}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 256}),
                "line_spacing": ("INT", {"default": 4, "min": 0, "max": 128}),
                "auto_fit": ("BOOLEAN", {"default": True}),
                "min_font_size": ("INT", {"default": 10, "min": 6, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "6yuan/utils"

    def render(self, text, width=512, height=256, font_size=16, padding=8, line_spacing=4, auto_fit=True, min_font_size=10):
        W = int(width)
        H = int(height)
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        fs = int(font_size)
        x = int(padding)
        y = int(padding)
        cw = max(1, W - 2 * int(padding))
        ch = max(1, H - 2 * int(padding))

        def load_font(sz):
            try:
                return ImageFont.truetype("arial.ttf", int(sz))
            except:
                return ImageFont.load_default()

        def wrap_lines(s, font_obj):
            lines = []
            for para in str(s).split("\n"):
                current = ""
                for ch in para:
                    candidate = current + ch
                    w = draw.textbbox((0, 0), candidate, font=font_obj)[2]
                    if w <= cw:
                        current = candidate
                    else:
                        if current:
                            lines.append(current)
                        current = ch
                if current:
                    lines.append(current)
            return lines

        while True:
            font = load_font(fs)
            lines = wrap_lines(text, font)
            total_h = 0
            for ln in lines:
                bbox = draw.textbbox((0, 0), ln, font=font)
                total_h += (bbox[3] - bbox[1])
            if len(lines) > 0:
                total_h += int(line_spacing) * (len(lines) - 1)
            if not auto_fit or total_h <= ch or fs <= int(min_font_size):
                break
            fs -= 1

        for ln in lines:
            draw.text((x, y), ln, fill=(0, 0, 0), font=font)
            bbox = draw.textbbox((x, y), ln, font=font)
            y = bbox[3] + int(line_spacing)
            if y >= H - int(padding):
                break

        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]
        return (t,)


NODE_CLASS_MAPPINGS = {
    "StringPreview": StringPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringPreview": "String Preview",
}
