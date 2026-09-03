#!/usr/bin/env bash
set -euo pipefail

# Use tcmalloc when it is available in the image. Do not set LD_PRELOAD to an
# empty/non-existent library because that adds noisy loader warnings.
TCMALLOC="$(ldconfig -p 2>/dev/null | grep -Po 'libtcmalloc.so.\d+' | head -n 1 || true)"
if [[ -n "${TCMALLOC}" ]]; then
    export LD_PRELOAD="${TCMALLOC}"
fi

echo "worker-comfyui: Checking PyTorch / CUDA compatibility"
python - <<'PY'
import sys
import torch

print(f"worker-comfyui: PyTorch {torch.__version__}")
print(f"worker-comfyui: PyTorch CUDA runtime {torch.version.cuda}")

if torch.version.cuda != "12.8":
    print(
        f"worker-comfyui: ERROR: expected PyTorch cu128, got CUDA runtime {torch.version.cuda!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print(
        "worker-comfyui: ERROR: torch.cuda.is_available() is false. "
        "The RunPod host NVIDIA driver is incompatible with this image.",
        file=sys.stderr,
    )
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(f"worker-comfyui: GPU {props.name}")
print(f"worker-comfyui: GPU VRAM {props.total_memory / (1024**3):.1f} GiB")
PY

echo "worker-comfyui: Starting ComfyUI"

# Production defaults: no sampler previews, quiet logs, and high-RAM caching.
# These change runtime overhead only; they do not change Qwen sampling math.
: "${COMFY_LOG_LEVEL:=WARNING}"
: "${COMFY_HIGH_RAM:=true}"
: "${COMFY_HIGH_VRAM:=auto}"
: "${COMFY_HIGH_VRAM_MIN_MB:=45000}"

COMFY_ARGS=(
    --disable-auto-launch
    --disable-metadata
    --preview-method none
    --verbose "${COMFY_LOG_LEVEL}"
    --log-stdout
)

if [[ "${COMFY_HIGH_RAM}" == "true" ]]; then
    COMFY_ARGS+=(--high-ram)
fi

# Qwen Image Edit FP8 + Qwen-VL + VAE + two LoRAs is intentionally left on
# DynamicVRAM for 24/32 GB GPUs. On >=45 GB GPUs, keeping models resident can
# remove repeated model shuffling between jobs. Override with true/false.
ENABLE_HIGH_VRAM="false"
case "${COMFY_HIGH_VRAM}" in
    true)
        ENABLE_HIGH_VRAM="true"
        ;;
    auto)
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
            if [[ "${GPU_VRAM_MB}" =~ ^[0-9]+$ ]] && (( GPU_VRAM_MB >= COMFY_HIGH_VRAM_MIN_MB )); then
                ENABLE_HIGH_VRAM="true"
            fi
        fi
        ;;
    false)
        ;;
    *)
        echo "worker-comfyui: Invalid COMFY_HIGH_VRAM=${COMFY_HIGH_VRAM}; expected auto|true|false" >&2
        ;;
esac

if [[ "${ENABLE_HIGH_VRAM}" == "true" ]]; then
    echo "worker-comfyui: Enabling --highvram"
    COMFY_ARGS+=(--highvram)
else
    echo "worker-comfyui: Using DynamicVRAM"
fi

if [[ -n "${COMFY_OUTPUT_DIR:-}" ]]; then
    COMFY_ARGS+=(--output-directory "${COMFY_OUTPUT_DIR}")
fi

if [[ "${SERVE_API_LOCALLY:-false}" == "true" ]]; then
    python -u /comfyui/main.py "${COMFY_ARGS[@]}" --listen &

    echo "worker-comfyui: Starting RunPod Handler"
    exec python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py "${COMFY_ARGS[@]}" &

    echo "worker-comfyui: Starting RunPod Handler"
    exec python -u /handler.py
fi
