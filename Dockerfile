# syntax=docker/dockerfile:1.7

# =============================================================================
# Base Image
# =============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS comfy-base


# =============================================================================
# Build Arguments
# =============================================================================

ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL


# =============================================================================
# Environment
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PIP_NO_INPUT=1


# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        git \
        wget \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# =============================================================================
# UV / Python Environment
# =============================================================================

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"


# =============================================================================
# Comfy CLI
# =============================================================================

RUN uv pip install \
    comfy-cli \
    pip \
    setuptools \
    wheel


# =============================================================================
# Install ComfyUI
# =============================================================================

RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
        /usr/bin/yes | comfy \
            --workspace /comfyui \
            install \
            --version "${COMFYUI_VERSION}" \
            --cuda-version "${CUDA_VERSION_FOR_COMFY}" \
            --nvidia; \
    else \
        /usr/bin/yes | comfy \
            --workspace /comfyui \
            install \
            --version "${COMFYUI_VERSION}" \
            --nvidia; \
    fi


# =============================================================================
# Install ComfyUI Runtime Requirements
#
# Explicitly install ComfyUI requirements after comfy-cli installation.
#
# This ensures runtime dependencies such as:
#
# - SQLAlchemy
# - Alembic
# - aiohttp
# - safetensors
# - transformers
# - other ComfyUI dependencies
#
# remain available even when ComfyUI adds new requirements.
# =============================================================================

RUN python -m pip install \
        --no-cache-dir \
        -r /comfyui/requirements.txt \
    && python -c \
        "import sqlalchemy, alembic; print('ComfyUI database dependencies OK')"


# =============================================================================
# Optional PyTorch Upgrade
# =============================================================================

RUN if [ "${ENABLE_PYTORCH_UPGRADE}" = "true" ]; then \
        uv pip install \
            --force-reinstall \
            torch \
            torchvision \
            torchaudio \
            --index-url "${PYTORCH_INDEX_URL}"; \
    fi


# =============================================================================
# Custom Nodes
# =============================================================================

WORKDIR /comfyui/custom_nodes

RUN git clone \
        --depth 1 \
        https://github.com/cubiq/ComfyUI_essentials.git \
    && git clone \
        --depth 1 \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone \
        --depth 1 \
        https://github.com/kijai/ComfyUI-WanVideoWrapper.git


# =============================================================================
# Custom Node Requirements
# =============================================================================

RUN if [ -f ComfyUI_essentials/requirements.txt ]; then \
        uv pip install \
            --no-cache-dir \
            -r ComfyUI_essentials/requirements.txt; \
    fi \
    && if [ -f ComfyUI-VideoHelperSuite/requirements.txt ]; then \
        uv pip install \
            --no-cache-dir \
            -r ComfyUI-VideoHelperSuite/requirements.txt; \
    fi \
    && if [ -f ComfyUI-WanVideoWrapper/requirements.txt ]; then \
        uv pip install \
            --no-cache-dir \
            -r ComfyUI-WanVideoWrapper/requirements.txt; \
    fi


# =============================================================================
# RunPod Worker Runtime Dependencies
# =============================================================================

WORKDIR /

RUN uv pip install \
    runpod \
    requests \
    websocket-client


# =============================================================================
# =============================================================================
# Model Downloader Stage
#
# Current production image:
#
#   Qwen Image Edit 2511
#
# WAN is intentionally disabled for now.
#
# Keeping this stage separate means changes to:
#
# - handler.py
# - start.sh
# - helper scripts
# - runtime dependencies
#
# do not need to re-download the Qwen models when Docker cache is available.
# =============================================================================
# =============================================================================

FROM comfy-base AS model-downloader


# =============================================================================
# Hugging Face Downloader
# =============================================================================

ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ETAG_TIMEOUT=60
ENV HF_HUB_DISABLE_UPDATE_CHECK=1

RUN uv pip install \
    huggingface_hub \
    hf_xet


# =============================================================================
# Model Output / Temporary Directories
# =============================================================================

RUN mkdir -p \
    /model-output/unet \
    /model-output/clip \
    /model-output/vae \
    /model-output/loras \
    /tmp/qwen-edit \
    /tmp/qwen-base \
    /tmp/qwen-lora


# =============================================================================
# Download Qwen Image Edit 2511
#
# Three independent Hugging Face jobs are started in parallel:
#
# 1. Qwen Image Edit diffusion model
# 2. Qwen text encoder + VAE
# 3. Qwen Image Edit 2511 Lightning LoRA
#
# All repositories are public, so no Hugging Face token is required.
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/huggingface \
    set -eu; \
    \
    echo "============================================================"; \
    echo "Downloading Qwen Image Edit 2511"; \
    echo "============================================================"; \
    \
    echo "[1/3] Starting Qwen Image Edit 2511 diffusion model..."; \
    hf download \
        Comfy-Org/Qwen-Image-Edit_ComfyUI \
        split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        --local-dir /tmp/qwen-edit \
        --max-workers 2 \
        > /tmp/qwen-edit.log 2>&1 \
        & PID_QWEN_EDIT=$!; \
    \
    echo "[2/3] Starting Qwen text encoder + VAE..."; \
    hf download \
        Comfy-Org/Qwen-Image_ComfyUI \
        split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        split_files/vae/qwen_image_vae.safetensors \
        --local-dir /tmp/qwen-base \
        --max-workers 2 \
        > /tmp/qwen-base.log 2>&1 \
        & PID_QWEN_BASE=$!; \
    \
    echo "[3/3] Starting Qwen 2511 Lightning LoRA..."; \
    hf download \
        lightx2v/Qwen-Image-Edit-2511-Lightning \
        Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        --local-dir /tmp/qwen-lora \
        --max-workers 2 \
        > /tmp/qwen-lora.log 2>&1 \
        & PID_QWEN_LORA=$!; \
    \
    echo ""; \
    echo "All Qwen downloads started."; \
    echo "Waiting for completion..."; \
    echo ""; \
    \
    FAILED=0; \
    \
    if wait "${PID_QWEN_EDIT}"; then \
        echo "[OK] Qwen Image Edit 2511 diffusion model"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] Qwen Image Edit 2511 failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/qwen-edit.log || true; \
        FAILED=1; \
    fi; \
    \
    if wait "${PID_QWEN_BASE}"; then \
        echo "[OK] Qwen text encoder + VAE"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] Qwen text encoder / VAE failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/qwen-base.log || true; \
        FAILED=1; \
    fi; \
    \
    if wait "${PID_QWEN_LORA}"; then \
        echo "[OK] Qwen Image Edit 2511 Lightning LoRA"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] Qwen Lightning LoRA failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/qwen-lora.log || true; \
        FAILED=1; \
    fi; \
    \
    if [ "${FAILED}" -ne 0 ]; then \
        echo ""; \
        echo "============================================================"; \
        echo "ONE OR MORE MODEL DOWNLOADS FAILED"; \
        echo "============================================================"; \
        exit 1; \
    fi; \
    \
    echo ""; \
    echo "============================================================"; \
    echo "All Qwen downloads completed."; \
    echo "Moving models into stable output directories..."; \
    echo "============================================================"; \
    \
    mv \
        /tmp/qwen-edit/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        /model-output/unet/qwen_image_edit_2511_fp8mixed.safetensors; \
    \
    mv \
        /tmp/qwen-base/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        /model-output/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors; \
    \
    mv \
        /tmp/qwen-base/split_files/vae/qwen_image_vae.safetensors \
        /model-output/vae/qwen_image_vae.safetensors; \
    \
    mv \
        /tmp/qwen-lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        /model-output/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors; \
    \
    rm -rf \
        /tmp/qwen-edit \
        /tmp/qwen-base \
        /tmp/qwen-lora \
        /tmp/qwen-edit.log \
        /tmp/qwen-base.log \
        /tmp/qwen-lora.log; \
    \
    echo ""; \
    echo "============================================================"; \
    echo "QWEN MODELS READY"; \
    echo "============================================================"; \
    echo "qwen_image_edit_2511_fp8mixed.safetensors"; \
    echo "qwen_2.5_vl_7b_fp8_scaled.safetensors"; \
    echo "qwen_image_vae.safetensors"; \
    echo "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"; \
    echo "============================================================"


# =============================================================================
# =============================================================================
# Final Production Image
#
# Large model COPY commands intentionally come before:
#
# - C compiler/runtime dependencies
# - handler.py
# - start.sh
# - helper scripts
#
# This preserves Docker layer caching for the large model weights.
# =============================================================================
# =============================================================================

FROM comfy-base AS final


# =============================================================================
# Standard ComfyUI Model Directories
# =============================================================================

RUN mkdir -p \
    /comfyui/models/unet \
    /comfyui/models/clip \
    /comfyui/models/vae \
    /comfyui/models/loras


# =============================================================================
# Qwen Image Edit 2511
# =============================================================================

COPY --from=model-downloader \
    /model-output/unet/qwen_image_edit_2511_fp8mixed.safetensors \
    /comfyui/models/unet/qwen_image_edit_2511_fp8mixed.safetensors

COPY --from=model-downloader \
    /model-output/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    /comfyui/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors

COPY --from=model-downloader \
    /model-output/vae/qwen_image_vae.safetensors \
    /comfyui/models/vae/qwen_image_vae.safetensors

COPY --from=model-downloader \
    /model-output/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors


# =============================================================================
# Runtime Build Tools
#
# Triton JIT compilation requires a C compiler at inference time.
#
# The NVIDIA CUDA runtime image intentionally does not contain GCC/G++.
#
# These tools are intentionally installed AFTER the model layers so this
# runtime layer can change without invalidating the huge Qwen layers.
# =============================================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++


# =============================================================================
# Custom Node Runtime Dependencies
#
# Fixes:
#
# ComfyUI-VideoHelperSuite:
#     ModuleNotFoundError: No module named 'cv2'
#
# ComfyUI-WanVideoWrapper:
#     ModuleNotFoundError: No module named 'accelerate'
#
# OpenCV headless is used because this is a serverless environment.
# =============================================================================

RUN uv pip install \
    --no-cache-dir \
    opencv-python-headless \
    imageio-ffmpeg \
    accelerate


# =============================================================================
# Runtime Verification
#
# Fail the Docker build immediately if these important runtime dependencies
# are not available.
# =============================================================================

RUN echo "============================================================" \
    && echo "Verifying runtime dependencies..." \
    && echo "============================================================" \
    && gcc --version \
    && g++ --version \
    && python -c "import sqlalchemy; print('SQLAlchemy OK:', sqlalchemy.__version__)" \
    && python -c "import alembic; print('Alembic OK:', alembic.__version__)" \
    && python -c "import cv2; print('OpenCV OK:', cv2.__version__)" \
    && python -c "import accelerate; print('Accelerate OK:', accelerate.__version__)" \
    && python -c "import torch; print('PyTorch OK:', torch.__version__)" \
    && python -c "import triton; print('Triton OK:', triton.__version__)" \
    && echo "============================================================" \
    && echo "Runtime dependency verification passed." \
    && echo "============================================================"


# =============================================================================
# Worker Files
#
# Keep frequently changing application files at the end.
#
# Updating handler.py or start.sh will therefore only invalidate these small
# final layers.
# =============================================================================

WORKDIR /

COPY src/start.sh /start.sh
COPY handler.py /handler.py
COPY test_input.json /test_input.json

RUN chmod +x /start.sh


# =============================================================================
# Helper Scripts
# =============================================================================

COPY scripts/comfy-node-install.sh \
    /usr/local/bin/comfy-node-install

RUN chmod +x /usr/local/bin/comfy-node-install


COPY scripts/comfy-manager-set-mode.sh \
    /usr/local/bin/comfy-manager-set-mode

RUN chmod +x /usr/local/bin/comfy-manager-set-mode


# =============================================================================
# Final Image Information
# =============================================================================

RUN echo "============================================================" \
    && echo "worker-comfyui production image ready" \
    && echo "Qwen Image Edit 2511: enabled" \
    && echo "Triton runtime compiler: enabled" \
    && echo "OpenCV: enabled" \
    && echo "Accelerate: enabled" \
    && echo "============================================================"


# =============================================================================
# Start
# =============================================================================

WORKDIR /

CMD ["/start.sh"]
