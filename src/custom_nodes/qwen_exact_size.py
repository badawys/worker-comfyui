"""Return Qwen Image 2.1 edits at the exact size of the source image.

QwenImage21TextEncodeList (resolution=0) rounds each side of the reference
image to a multiple of 32 and stretches it slightly; the output canvas uses
that rounded size. The model maps the whole reference canvas onto the whole
output canvas, so resizing the decoded image back to the source size puts
every pixel back where it was (measured < 1px drift).

Padding the source instead does NOT work: the model squeezes the picture into
the padded canvas, which shows up as a growing vertical shift.
"""

import torch.nn.functional as F


class QwenResizeToReference:
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
    FUNCTION = "resize"
    CATEGORY = "image/qwen-image-2.1"

    def resize(self, image, reference):
        ref_h, ref_w = int(reference.shape[1]), int(reference.shape[2])
        if (int(image.shape[1]), int(image.shape[2])) == (ref_h, ref_w):
            return (image,)
        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            nchw, size=(ref_h, ref_w), mode="bicubic", align_corners=False, antialias=True
        )
        return (resized.clamp(0, 1).permute(0, 2, 3, 1).contiguous(),)


NODE_CLASS_MAPPINGS = {"QwenResizeToReference": QwenResizeToReference}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenResizeToReference": "Qwen Image 2.1 · Resize To Source Size",
}
