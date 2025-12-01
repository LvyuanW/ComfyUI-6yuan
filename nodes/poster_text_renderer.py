import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class PosterTextRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "template_layout": ("STRING", {"multiline": True}),
                "font_path": ("STRING", {"default": ""}),
                "text_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "title_text": ("STRING", {"multiline": True, "default": ""}),
                "subtitle_text": ("STRING", {"multiline": True, "default": ""}),
                "time_text": ("STRING", {"multiline": False, "default": ""}),
                "booth_text": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_text",)
    FUNCTION = "render"
    CATEGORY = "6yuan/poster"

    def _tensor_to_pil(self, image):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr)

    def _pil_to_tensor(self, pil):
        arr = np.asarray(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return torch.from_numpy(arr)[None, ...]

    def _parse_color(self, s):
        s = str(s).strip()
        if s.startswith("#") and len(s) in (7, 9):
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return (r, g, b)
        return (0, 0, 0)

    def _load_font(self, path, size):
        try:
            if path:
                return ImageFont.truetype(path, size)
        except Exception:
            pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, draw, text, font, max_w):
        lines = []
        for para in str(text).split("\n"):
            current = ""
            for ch in para:
                candidate = current + ch
                w = draw.textbbox((0, 0), candidate, font=font)[2]
                if w <= max_w:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        return lines

    def _draw_block(self, draw, rect, text, align, color, font_path, base_h, kind):
        x_px, y_px, w_px, h_px = rect
        if kind == "title":
            size = max(12, int(base_h * 0.06))
        elif kind == "subtitle":
            size = max(12, int(base_h * 0.035))
        else:
            size = max(12, int(base_h * 0.028))

        font = self._load_font(font_path, size)
        lines = self._wrap_text(draw, text, font, w_px)
        total_h = 0
        bboxes = []
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            bboxes.append(bbox)
            total_h += (bbox[3] - bbox[1])
        spacing = int(size * 0.25)
        total_h += spacing * max(0, len(lines) - 1)

        while total_h > h_px and size > 8:
            size -= 1
            font = self._load_font(font_path, size)
            lines = self._wrap_text(draw, text, font, w_px)
            total_h = 0
            bboxes = []
            for ln in lines:
                bbox = draw.textbbox((0, 0), ln, font=font)
                bboxes.append(bbox)
                total_h += (bbox[3] - bbox[1])
            spacing = int(size * 0.25)
            total_h += spacing * max(0, len(lines) - 1)

        if align == "center":
            x_start = x_px + (w_px // 2)
        elif align == "right":
            x_start = x_px + w_px
        else:
            x_start = x_px

        y = y_px
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if align == "center":
                x = x_start - w // 2
            elif align == "right":
                x = x_start - w
            else:
                x = x_start
            draw.text((x, y), ln, fill=color, font=font)
            y += h + spacing

    def render(self, image, template_layout, font_path, text_color, title_text="", subtitle_text="", time_text="", booth_text=""):
        pil = self._tensor_to_pil(image)
        draw = ImageDraw.Draw(pil)
        W, H = pil.size

        try:
            layout = json.loads(template_layout)
        except Exception:
            layout = {}

        slots = {}
        if isinstance(layout, dict):
            slots = layout

        color = self._parse_color(text_color)

        def rect_from_ratio(slot):
            xr = float(slot.get("x", 0.0))
            yr = float(slot.get("y", 0.0))
            wr = float(slot.get("w", 0.0))
            hr = float(slot.get("h", 0.0))
            x0 = int(xr * W)
            y0 = int(yr * H)
            w0 = int(wr * W)
            h0 = int(hr * H)
            return (x0, y0, w0, h0)

        title_slot = slots.get("title", None)
        if isinstance(title_slot, dict) and title_text:
            align = str(title_slot.get("align", "left"))
            self._draw_block(draw, rect_from_ratio(title_slot), title_text, align, color, font_path, H, "title")

        subtitle_slot = slots.get("subtitle", None)
        if isinstance(subtitle_slot, dict) and subtitle_text:
            align = str(subtitle_slot.get("align", "left"))
            self._draw_block(draw, rect_from_ratio(subtitle_slot), subtitle_text, align, color, font_path, H, "subtitle")

        time_slot = slots.get("time", None)
        if isinstance(time_slot, dict) and time_text:
            align = str(time_slot.get("align", "left"))
            self._draw_block(draw, rect_from_ratio(time_slot), time_text, align, color, font_path, H, "detail")

        booth_slot = slots.get("booth", None)
        if isinstance(booth_slot, dict) and booth_text:
            align = str(booth_slot.get("align", "left"))
            self._draw_block(draw, rect_from_ratio(booth_slot), booth_text, align, color, font_path, H, "detail")

        return (self._pil_to_tensor(pil),)


NODE_CLASS_MAPPINGS = {
    "PosterTextRenderer": PosterTextRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PosterTextRenderer": "Poster Text Renderer",
}

