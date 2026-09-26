"""Create a smaller image used only by the Qwen Image 2.1 Prompt Enhancer.

The original source image remains untouched for Qwen Image 2.1 conditioning.
Reducing the PE vision input lowers multimodal prefill/context cost.
"""

import torch
import torch.nn.functional as F


class PEPreviewResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_dimension": (
                    "INT",
                    {"default": 768, "min": 256, "max": 2048, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize"
    CATEGORY = "image/qwen-image-2.1"

    def resize(self, image, max_dimension):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        longest = max(height, width)

        if longest <= int(max_dimension):
            return (image,)

        scale = float(max_dimension) / float(longest)
        target_h = max(1, round(height * scale))
        target_w = max(1, round(width * scale))

        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            nchw,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (resized.permute(0, 2, 3, 1).contiguous(),)


NODE_CLASS_MAPPINGS = {"PEPreviewResize": PEPreviewResize}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PEPreviewResize": "Qwen Image 2.1 · PE Preview Resize"
}
