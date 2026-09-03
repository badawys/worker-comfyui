#!/usr/bin/env bash
set -euo pipefail

COMFY_PYTHON="${COMFY_PYTHON:-/comfyui/.venv/bin/python}"
if [[ ! -x "${COMFY_PYTHON}" ]]; then
    echo "worker-comfyui: ERROR: ComfyUI Python not found at ${COMFY_PYTHON}" >&2
    exit 1
fi

TCMALLOC="$(ldconfig -p 2>/dev/null | grep -Po 'libtcmalloc.so.\d+' | head -n 1 || true)"
if [[ -n "${TCMALLOC}" ]]; then
    export LD_PRELOAD="${TCMALLOC}"
fi

echo "worker-comfyui: Checking PyTorch / CUDA"
"${COMFY_PYTHON}" - <<'PY'
import sys
import torch

print(f"worker-comfyui: PyTorch {torch.__version__}")
print(f"worker-comfyui: PyTorch CUDA runtime {torch.version.cuda}")

if not torch.cuda.is_available():
    print(
        "worker-comfyui: ERROR: CUDA is not available to PyTorch.",
        file=sys.stderr,
    )
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(f"worker-comfyui: GPU {props.name}")
print(f"worker-comfyui: GPU VRAM {props.total_memory / (1024**3):.1f} GiB")
PY

echo "worker-comfyui: Starting ComfyUI"

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
    "${COMFY_PYTHON}" -u /comfyui/main.py "${COMFY_ARGS[@]}" --listen &

    echo "worker-comfyui: Starting RunPod Handler"
    exec "${COMFY_PYTHON}" -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    "${COMFY_PYTHON}" -u /comfyui/main.py "${COMFY_ARGS[@]}" &

    echo "worker-comfyui: Starting RunPod Handler"
    exec "${COMFY_PYTHON}" -u /handler.py
fi
