"""Exact Viggle v0.2.1 sigma schedule for Qwen Image 2.1 Fast mode."""

import math
import torch


class ViggleTurboSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "nodes": (
                    "STRING",
                    {
                        "default": "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, latent, nodes):
        samples = latent["samples"]
        ratio = latent.get("downscale_ratio_spacial", 16) / 16
        tokens = round(samples.shape[-2] * ratio) * round(samples.shape[-1] * ratio)
        mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)

        raw = [float(value.strip()) for value in nodes.split(",") if value.strip()]
        if not raw or any(value <= 0 for value in raw):
            raise ValueError("Viggle sigma nodes must contain positive values")

        t = torch.tensor(raw, dtype=torch.float64)
        shifted = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        return (torch.cat([shifted, shifted.new_zeros(1)]).float(),)


NODE_CLASS_MAPPINGS = {"ViggleTurboSigmas": ViggleTurboSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ViggleTurboSigmas": "Qwen Image 2.1 · Viggle Turbo Sigmas"
}
