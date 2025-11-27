import json
import math

import numpy as np
from PIL import Image, ImageDraw

import torch

class ColorKMeans:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "k": ("INT", {"default": 3, "min": 1, "max": 8, "forceInput": False}),
                "resize_to": ("INT", {"default": 64, "min": 16, "max": 256, "forceInput": False}),
                "min_ratio": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": False}),
                "min_saturation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("color_json", "hist_image",)
    FUNCTION = "analyze"
    CATEGORY = "6yuan/analyze"

    def analyze(self, image, k, resize_to, min_ratio, min_saturation):
        """
        image: torch.Tensor, shape [B,H,W,C] or [H,W,C]
        """

        # 1. 取单张 / 去 batch
        if image.dim() == 4:
            assert image.shape[0] == 1
            img_t = image[0]
        else:
            img_t = image

        # img_t: [H,W,C], C=3 or 4, 0~1 float
        img_np = img_t.cpu().numpy()
        h, w, c = img_np.shape
        if max(h, w) > resize_to:
            arr = (img_np * 255).astype(np.uint8)
            mode = "RGBA" if c == 4 else "RGB"
            pil = Image.fromarray(arr, mode=mode)
            if w >= h:
                new_w = resize_to
                new_h = int(h * resize_to / w)
            else:
                new_h = resize_to
                new_w = int(w * resize_to / h)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            img_np = np.asarray(pil).astype(np.float32) / 255.0
            h, w, c = img_np.shape

        # 2. 处理 alpha：只保留非透明像素
        if c == 4:
            rgb = img_np[..., :3]
            alpha = img_np[..., 3]
            mask = alpha > 0.01
            rgb_flat = rgb[mask]
        else:
            rgb_flat = img_np[..., :3].reshape(-1, 3)

        if rgb_flat.size == 0:
            empty_img = Image.new("RGB", (256, 128), (0, 0, 0))
            hist_image_t = self.pil_to_tensor(empty_img)
            color_json = json.dumps({"palette": []})
            return (color_json, hist_image_t)

        cmax = np.max(rgb_flat, axis=1)
        cmin = np.min(rgb_flat, axis=1)
        delta = cmax - cmin
        s = np.where(cmax == 0, 0.0, delta / cmax)
        sat_mask = s >= float(min_saturation)
        rgb_flat = rgb_flat[sat_mask]

        if rgb_flat.size == 0:
            empty_img = Image.new("RGB", (256, 128), (0, 0, 0))
            hist_image_t = self.pil_to_tensor(empty_img)
            color_json = json.dumps({"palette": []})
            return (color_json, hist_image_t)

        # 3. 下采样：随机采样 N 个像素
        sample_n = min(5000, rgb_flat.shape[0])
        idx = np.random.choice(rgb_flat.shape[0], sample_n, replace=False)
        samples = rgb_flat[idx]

        k = max(1, min(k, samples.shape[0]))
        centers = self.kmeans(samples, k)

        # 5. 把每个像素分配到最近中心，统计占比
        labels = self.assign_clusters(samples, centers)
        ratios = []
        for i in range(len(centers)):
            count = np.sum(labels == i)
            ratios.append(count / float(sample_n))

        # 6. 转 0~255，按占比排序
        palette = []
        for center, ratio in zip(centers, ratios):
            r, g, b = (center * 255).astype(int).tolist()
            palette.append({"r": r, "g": g, "b": b, "ratio": ratio})

        palette.sort(key=lambda x: x["ratio"], reverse=True)
        palette = [p for p in palette if p["ratio"] >= float(min_ratio)]

        # 7. 生成 JSON：primary / secondary / accent
        primary = palette[0] if len(palette) > 0 else None
        secondary = palette[1] if len(palette) > 1 else None
        accent = palette[2] if len(palette) > 2 else None

        color_info = {
            "primary": primary,
            "secondary": secondary,
            "accent": accent,
            "palette": palette,
        }
        color_json = json.dumps(color_info, ensure_ascii=False)

        # 8. 生成直方图可视化（简单 RGB 三通道直方图）
        hist_img = self.make_hist_image(rgb_flat)
        hist_image_t = self.pil_to_tensor(hist_img)

        print(color_json)

        return (color_json, hist_image_t)

    def kmeans(self, data, k, iters=10):
        # data: [N,3] 0~1
        # 初始化：随机选 k 个样本
        N = data.shape[0]
        idx = np.random.choice(N, k, replace=False)
        centers = data[idx].copy()

        for _ in range(iters):
            labels = self.assign_clusters(data, centers)
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    centers[i] = data[mask].mean(axis=0)
        return centers

    def assign_clusters(self, data, centers):
        # data: [N,3], centers: [k,3]
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return labels

    def make_hist_image(self, rgb_flat):
        # rgb_flat: [N,3], 0~1
        rgb_255 = (rgb_flat * 255).astype(np.uint8)
        # 每个通道 256 bins
        hist_r, _ = np.histogram(rgb_255[:, 0], bins=256, range=(0, 255))
        hist_g, _ = np.histogram(rgb_255[:, 1], bins=256, range=(0, 255))
        hist_b, _ = np.histogram(rgb_255[:, 2], bins=256, range=(0, 255))

        # 归一化到高度
        h = 128
        max_v = max(hist_r.max(), hist_g.max(), hist_b.max(), 1)
        hr = (hist_r / max_v * h).astype(int)
        hg = (hist_g / max_v * h).astype(int)
        hb = (hist_b / max_v * h).astype(int)

        img = Image.new("RGB", (256, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for x in range(256):
            if hr[x] > 0:
                draw.line((x, h, x, h - hr[x]), fill=(255, 0, 0))
            if hg[x] > 0:
                draw.line((x, h, x, h - hg[x]), fill=(0, 255, 0))
            if hb[x] > 0:
                draw.line((x, h, x, h - hb[x]), fill=(0, 0, 255))

        return img

    def pil_to_tensor(self, img):
        # PIL -> torch tensor [1,H,W,3], 0~1
        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]
        return t

NODE_CLASS_MAPPINGS = {
    "ColorKMeans": ColorKMeans,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorKMeans": "Color K-Means",
}
