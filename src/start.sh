#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

COMFY_ARGS="--disable-auto-launch --disable-metadata --verbose ${COMFY_LOG_LEVEL} --log-stdout"

if [ -n "${COMFY_OUTPUT_DIR}" ]; then
    COMFY_ARGS="${COMFY_ARGS} --output-directory ${COMFY_OUTPUT_DIR}"
fi

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py ${COMFY_ARGS} --listen &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py ${COMFY_ARGS} &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi