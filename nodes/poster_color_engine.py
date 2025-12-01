import numpy as np
import torch
from PIL import Image


class PosterColorEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "logo_image": ("IMAGE",),
            },
            "optional": {
                "clusters": ("INT", {"default": 3, "min": 1, "max": 8}),
                "min_saturation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("bg_color", "text_main", "text_inverse")
    FUNCTION = "analyze"
    CATEGORY = "6yuan/poster"

    def _tensor_to_np(self, image):
        if image.dim() == 4:
            img = image[0]
        else:
            img = image
        arr = img.cpu().numpy()
        return arr

    def _rgb_to_hex(self, rgb):
        r, g, b = [int(max(0, min(255, v))) for v in rgb]
        return f"#{r:02X}{g:02X}{b:02X}"

    def _relative_luminance(self, rgb):
        srgb = np.array(rgb, dtype=np.float32) / 255.0
        def lin(c):
            return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = lin(srgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _contrast_ratio(self, c1, c2):
        L1 = self._relative_luminance(c1)
        L2 = self._relative_luminance(c2)
        L1, L2 = float(max(L1, L2)), float(min(L1, L2))
        return (L1 + 0.05) / (L2 + 0.05)

    def _kmeans(self, data, k, iters=8):
        N = data.shape[0]
        idx = np.random.choice(N, k, replace=False)
        centers = data[idx].copy()
        for _ in range(iters):
            d = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            for i in range(k):
                m = labels == i
                if np.any(m):
                    centers[i] = data[m].mean(axis=0)
        return centers

    def analyze(self, logo_image, clusters=3, min_saturation=0.2):
        arr = self._tensor_to_np(logo_image)
        h, w, c = arr.shape
        if c == 4:
            rgb = arr[..., :3]
            alpha = arr[..., 3]
            mask = alpha > 0.01
            rgb = rgb[mask]
        else:
            rgb = arr[..., :3].reshape(-1, 3)
        if rgb.size == 0:
            bg = (255, 255, 255)
            black = (0, 0, 0)
            white = (255, 255, 255)
            cr_black = self._contrast_ratio(bg, black)
            cr_white = self._contrast_ratio(bg, white)
            text_main = black if cr_black >= cr_white else white
            text_inverse = white if text_main == black else black
            return (self._rgb_to_hex(bg), self._rgb_to_hex(text_main), self._rgb_to_hex(text_inverse))

        rgb = rgb.astype(np.float32)
        rgb01 = rgb / 255.0
        cmax = np.max(rgb01, axis=1)
        cmin = np.min(rgb01, axis=1)
        delta = cmax - cmin
        s = np.where(cmax == 0, 0.0, delta / cmax)
        mask_sat = s >= float(min_saturation)
        rgb = rgb[mask_sat]
        if rgb.size == 0:
            rgb = (arr[..., :3].reshape(-1, 3)).astype(np.float32)

        sample_n = min(5000, rgb.shape[0])
        idx = np.random.choice(rgb.shape[0], sample_n, replace=False)
        samples = rgb[idx]
        k = max(1, min(int(clusters), samples.shape[0]))
        centers = self._kmeans(samples, k)

        ratios = []
        d = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        for i in range(len(centers)):
            ratios.append(float(np.sum(labels == i)) / float(sample_n))
        palette = [(centers[i], ratios[i]) for i in range(len(centers))]
        palette.sort(key=lambda x: x[1], reverse=True)
        bg_rgb = palette[0][0]

        black = np.array([0, 0, 0], dtype=np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)
        cr_black = self._contrast_ratio(bg_rgb, black)
        cr_white = self._contrast_ratio(bg_rgb, white)
        text_main = black if cr_black >= cr_white else white
        text_inverse = white if (text_main == black).all() else black

        return (self._rgb_to_hex(bg_rgb), self._rgb_to_hex(text_main), self._rgb_to_hex(text_inverse))


NODE_CLASS_MAPPINGS = {
    "PosterColorEngine": PosterColorEngine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PosterColorEngine": "Poster Color Engine",
}

