import torch
import comfy
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class CropAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 500}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "crop"
    CATEGORY = "6yuan/image"

    def crop(self, image, padding: int = 0):
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        batch, height, width, channels = image.shape
        cropped_images = []
        cropped_masks = []

        for b in range(batch):
            img = image[b]
            alpha = img[..., 3]

            mask = (alpha > 0.01)

            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)

            ymin, ymax = self._find_boundary(rows)
            xmin, xmax = self._find_boundary(cols)

            if ymin is None or xmin is None:
                cropped_images.append(img)
                cropped_masks.append(torch.zeros_like(alpha))
                continue

            ymin = max(0, ymin - padding)
            ymax = min(height, ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(width, xmax + padding)

            cropped = img[ymin:ymax, xmin:xmax, :4]
            cropped_mask = alpha[ymin:ymax, xmin:xmax]

            cropped_images.append(cropped)
            cropped_masks.append(cropped_mask)

        cropped_images = torch.stack(cropped_images, dim=0)
        cropped_masks = torch.stack(cropped_masks, dim=0)

        return (cropped_images, cropped_masks)

    def _find_boundary(self, arr: torch.Tensor):
        nz = torch.nonzero(arr, as_tuple=False)
        if nz.numel() == 0:
            return (None, None)
        return (nz[0].item(), nz[-1].item() + 1)


class ShrinkImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["scale", "pixels"], {"default": "scale"}),
                "resize_algorithm": (list(resize_algorithms.keys()), {"default": "LANCZOS"})
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1}),
                "height": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shrink_image"
    CATEGORY = "6yuan/image"

    def calculate_scale(self, img: Image.Image, mode, scale=None, width=None, height=None):
        if mode == "scale":
            return scale
        else:
            img_width, img_height = img.size
            width = min(width, img_width)
            height = min(height, img_height)
            scale_x = width / img_width
            scale_y = height / img_height
            return min(scale_x, scale_y)

    def shrink_image_with_scale(self, img: Image.Image, scale, algorithm):
        width, height = img.size
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return img.resize((new_width, new_height), algorithm)

    def shrink_image(self, image, mode, resize_algorithm, scale=None, width=None, height=None):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        algorithm = resize_algorithms[resize_algorithm]

        if isinstance(image, list):
            image = torch.stack(image, dim=0)

        batch = image.shape[0]
        output_images = []

        for b in range(batch):
            img_t = image[b]
            img_pil = to_pil_image(img_t.permute(2, 0, 1))

            s = self.calculate_scale(img_pil, mode, scale, width, height)
            resized_img = self.shrink_image_with_scale(img_pil, s, algorithm)

            resized_img_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_img_t = torch.from_numpy(resized_img_np)
            output_images.append(resized_img_t)

        output_images = torch.stack(output_images, dim=0)
        return (output_images,)


NODE_CLASS_MAPPINGS = {
    "CropAlpha": CropAlpha,
    "ShrinkImage": ShrinkImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAlpha": "Crop Alpha",
    "ShrinkImage": "Shrink Image"
}
