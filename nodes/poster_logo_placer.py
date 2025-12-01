import json
import numpy as np
import torch
from PIL import Image


class PosterLogoPlacer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "logo_image": ("IMAGE",),
                "template_layout": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_logo",)
    FUNCTION = "place"
    CATEGORY = "6yuan/poster"

    def _tensor_to_pil(self, image):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _pil_to_tensor(self, pil):
        arr = np.asarray(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        return torch.from_numpy(arr)[None, ...]

    def place(self, image, logo_image, template_layout):
        base = self._tensor_to_pil(image).convert("RGBA")
        logo = self._tensor_to_pil(logo_image).convert("RGBA")
        W, H = base.size

        try:
            layout = json.loads(template_layout)
        except Exception:
            layout = {}

        slot = layout.get("logo", None) if isinstance(layout, dict) else None
        if not isinstance(slot, dict):
            return (self._pil_to_tensor(base.convert("RGB")),)

        xr = float(slot.get("x", 0.0))
        yr = float(slot.get("y", 0.0))
        wr = float(slot.get("w", 0.0))
        hr = float(slot.get("h", 0.0))
        align = str(slot.get("align", "center"))

        x0 = int(xr * W)
        y0 = int(yr * H)
        w0 = int(max(1, wr * W))
        h0 = int(max(1, hr * H))

        lw, lh = logo.size
        scale = min(w0 / lw, h0 / lh)
        new_size = (max(1, int(lw * scale)), max(1, int(lh * scale)))
        logo_resized = logo.resize(new_size, Image.LANCZOS)

        if align == "left":
            px = x0
        elif align == "right":
            px = x0 + w0 - logo_resized.size[0]
        else:
            px = x0 + (w0 - logo_resized.size[0]) // 2
        py = y0 + (h0 - logo_resized.size[1]) // 2

        base.alpha_composite(logo_resized, (px, py))
        out = base.convert("RGB")
        return (self._pil_to_tensor(out),)


NODE_CLASS_MAPPINGS = {
    "PosterLogoPlacer": PosterLogoPlacer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PosterLogoPlacer": "Poster Logo Placer",
}

