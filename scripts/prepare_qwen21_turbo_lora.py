"""Split the official Turbo LoRA's fused MLP for the unfused Quality GGUF.

This changes tensor layout only: [gate; proj] @ down becomes the same two
matrix products, with the original alpha and rank retained for each half.
See https://github.com/Comfy-Org/ComfyUI/issues/16944.
"""

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def split_fused_mlp_lora(tensors):
    suffix = ".img_mlp.gate_up.lora_up.weight"
    prefixes = [key.removesuffix(".lora_up.weight") for key in tensors if key.endswith(suffix)]
    if not prefixes:
        raise ValueError("No fused Qwen Image 2.1 MLP LoRA weights found")

    converted = dict(tensors)
    for prefix in prefixes:
        keys = [prefix + ending for ending in (".lora_up.weight", ".lora_down.weight", ".alpha")]
        if any(key not in tensors for key in keys):
            raise ValueError(f"Incomplete LoRA weights for {prefix}")
        up, down, alpha = (tensors[key] for key in keys)
        if up.ndim != 2 or up.shape[0] % 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"Invalid fused LoRA matrix dimensions for {prefix}")

        for name, half in zip(("gate_layer", "proj"), up.chunk(2, dim=0)):
            target = prefix.removesuffix("gate_up") + name
            for ending, tensor in ((".lora_up.weight", half), (".lora_down.weight", down), (".alpha", alpha)):
                key = target + ending
                if key in converted:
                    raise ValueError(f"Refusing to overwrite existing adapter weight {key}")
                # safetensors requires separate contiguous storage for duplicated tensors.
                converted[key] = tensor.clone().contiguous()
        for key in keys:
            del converted[key]
    if any(".img_mlp.gate_up." in key for key in converted):
        raise ValueError("Unconverted fused MLP adapter keys remain")
    return converted, len(prefixes)


def prepare_lora(source: Path, destination: Path, expected_blocks: int = 32):
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files")
    tensors, blocks = split_fused_mlp_lora(load_file(str(source), device="cpu"))
    if blocks != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} fused MLP blocks, found {blocks}")
    with safe_open(str(source), framework="pt", device="cpu") as original:
        metadata = original.metadata() or {}
    metadata["qwen21_mlp_layout"] = "unfused_gate_layer_proj"
    metadata["qwen21_source_lora"] = source.name
    save_file(tensors, str(destination), metadata=metadata)
    print(f"Prepared {destination.name}: losslessly split {blocks} fused MLP adapters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_lora(args.source, args.destination)
