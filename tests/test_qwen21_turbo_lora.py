import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts.prepare_qwen21_turbo_lora import prepare_lora, split_fused_mlp_lora


class TestQwen21TurboLora(unittest.TestCase):
    prefix = "diffusion_model.transformer_blocks.0.img_mlp."

    def adapter(self):
        return {
            self.prefix + "gate_up.lora_up.weight": torch.arange(-12, 12, dtype=torch.bfloat16).reshape(8, 3),
            self.prefix + "gate_up.lora_down.weight": torch.arange(-6, 6, dtype=torch.bfloat16).reshape(3, 4),
            self.prefix + "gate_up.alpha": torch.tensor(3, dtype=torch.bfloat16),
            "diffusion_model.img_in.alpha": torch.tensor(7, dtype=torch.bfloat16),
        }

    def test_split_preserves_adapter_matrix_product_and_scaling(self):
        original = self.adapter()
        converted, count = split_fused_mlp_lora(original)
        self.assertEqual(count, 1)
        before = original[self.prefix + "gate_up.lora_up.weight"].float() @ original[self.prefix + "gate_up.lora_down.weight"].float()
        halves = []
        for part in ("gate_layer", "proj"):
            up = converted[self.prefix + part + ".lora_up.weight"]
            down = converted[self.prefix + part + ".lora_down.weight"]
            self.assertEqual(up.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(converted[self.prefix + part + ".alpha"], original[self.prefix + "gate_up.alpha"]))
            halves.append(up.float() @ down.float())
        self.assertTrue(torch.equal(before, torch.cat(halves)))
        self.assertTrue(torch.equal(converted["diffusion_model.img_in.alpha"], original["diffusion_model.img_in.alpha"]))
        self.assertFalse(any("gate_up" in key for key in converted))
        self.assertEqual(len(original), 4)  # Caller data is not changed.

    def test_safetensors_round_trip_retains_metadata_and_bf16_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "official.safetensors"
            destination = Path(directory) / "unfused.safetensors"
            save_file(self.adapter(), str(source), metadata={"ss_network_alpha": "Dynamic"})
            prepare_lora(source, destination, expected_blocks=1)
            converted = load_file(str(destination))
            self.assertEqual(converted[self.prefix + "gate_layer.lora_up.weight"].shape, (4, 3))
            self.assertTrue(torch.equal(converted[self.prefix + "proj.lora_up.weight"], self.adapter()[self.prefix + "gate_up.lora_up.weight"][4:]))
            with safe_open(str(destination), framework="pt") as result:
                self.assertEqual(result.metadata()["ss_network_alpha"], "Dynamic")
                self.assertEqual(result.metadata()["qwen21_source_lora"], source.name)
            with self.assertRaisesRegex(ValueError, "Expected 32"):
                prepare_lora(source, destination)
            with self.assertRaisesRegex(ValueError, "different files"):
                prepare_lora(source, source)

    def test_invalid_layouts_fail_instead_of_silently_skipping_weights(self):
        for kind in ("missing_down", "odd_rows", "rank_mismatch", "collision", "no_fused"):
            with self.subTest(kind=kind):
                tensors = self.adapter()
                if kind == "missing_down":
                    del tensors[self.prefix + "gate_up.lora_down.weight"]
                elif kind == "odd_rows":
                    tensors[self.prefix + "gate_up.lora_up.weight"] = torch.zeros(7, 3)
                elif kind == "rank_mismatch":
                    tensors[self.prefix + "gate_up.lora_down.weight"] = torch.zeros(2, 4)
                elif kind == "collision":
                    tensors[self.prefix + "proj.alpha"] = torch.tensor(3)
                else:
                    tensors = {"unrelated.alpha": torch.tensor(1)}
                with self.assertRaises(ValueError):
                    split_fused_mlp_lora(tensors)


if __name__ == "__main__":
    unittest.main()
