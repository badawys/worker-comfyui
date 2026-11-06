"""Keep Qwen Image 2.1 edits at the exact size and position of the source image.

QwenImage21TextEncodeList (resolution=0) rounds each side of the reference
image to a multiple of 32 and *stretches* the image to fit. The output canvas
uses that rounded size, so a 1216x868 source comes back as 1216x864 with the
content squeezed by a few pixels, which looks like a shift.

Fix: pad the source (right/bottom, edge pixels repeated) up to the next
multiple of 32 so no stretching happens, then crop the decoded output back to
the original width and height.
"""

import torch.nn.functional as F

MULTIPLE = 32


class QwenPadToMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple": ("INT", {"default": MULTIPLE, "min": 8, "max": 128, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad"
    CATEGORY = "image/qwen-image-2.1"

    def pad(self, image, multiple):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")
        height, width = int(image.shape[1]), int(image.shape[2])
        pad_h = (-height) % int(multiple)
        pad_w = (-width) % int(multiple)
        if pad_h == 0 and pad_w == 0:
            return (image,)
        nchw = image.permute(0, 3, 1, 2)
        padded = F.pad(nchw, (0, pad_w, 0, pad_h), mode="replicate")
        return (padded.permute(0, 2, 3, 1).contiguous(),)


class QwenCropToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop"
    CATEGORY = "image/qwen-image-2.1"

    def crop(self, image, reference):
        ref_h, ref_w = int(reference.shape[1]), int(reference.shape[2])
        height, width = int(image.shape[1]), int(image.shape[2])
        if (height, width) == (ref_h, ref_w):
            return (image,)
        # Expected path: output is the padded canvas, so cut the padding off.
        if height >= ref_h and width >= ref_w and height - ref_h < MULTIPLE * 4 and width - ref_w < MULTIPLE * 4:
            return (image[:, :ref_h, :ref_w, :].contiguous(),)
        # Canvas was a different size entirely (e.g. resolution > 0): resize.
        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(nchw, size=(ref_h, ref_w), mode="bicubic", align_corners=False, antialias=True)
        return (resized.clamp(0, 1).permute(0, 2, 3, 1).contiguous(),)


NODE_CLASS_MAPPINGS = {
    "QwenPadToMultiple": QwenPadToMultiple,
    "QwenCropToReference": QwenCropToReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPadToMultiple": "Qwen Image 2.1 · Pad To Multiple",
    "QwenCropToReference": "Qwen Image 2.1 · Crop To Source Size",
}
