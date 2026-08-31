# syntax=docker/dockerfile:1.7

# =============================================================================
# Base Image
# =============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS base


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

# Hugging Face
#
# Default download timeout is much lower.
# Large 10-20 GB model files benefit from a much larger timeout.
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ETAG_TIMEOUT=60

# Do not waste time checking CLI versions during Docker builds.
ENV HF_HUB_DISABLE_UPDATE_CHECK=1


# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update \
    && apt-get install -y \
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
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
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
# Python Packages
#
# hf_xet is automatically used by huggingface_hub for Xet-backed repositories.
# =============================================================================

RUN uv pip install \
    comfy-cli \
    pip \
    setuptools \
    wheel \
    huggingface_hub \
    hf_xet


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
# ComfyUI Configuration
# =============================================================================

WORKDIR /comfyui

ADD src/extra_model_paths.yaml ./


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
        pip install \
            --no-cache-dir \
            -r ComfyUI_essentials/requirements.txt; \
    fi \
    && if [ -f ComfyUI-VideoHelperSuite/requirements.txt ]; then \
        pip install \
            --no-cache-dir \
            -r ComfyUI-VideoHelperSuite/requirements.txt; \
    fi \
    && if [ -f ComfyUI-WanVideoWrapper/requirements.txt ]; then \
        pip install \
            --no-cache-dir \
            -r ComfyUI-WanVideoWrapper/requirements.txt; \
    fi


# =============================================================================
# Worker Runtime
# =============================================================================

WORKDIR /

RUN uv pip install \
    runpod \
    requests \
    websocket-client


# =============================================================================
# Worker Files
# =============================================================================

ADD src/start.sh handler.py test_input.json ./

RUN chmod +x /start.sh


# =============================================================================
# Helper Scripts
# =============================================================================

COPY scripts/comfy-node-install.sh \
    /usr/local/bin/comfy-node-install

RUN chmod +x /usr/local/bin/comfy-node-install


ENV PIP_NO_INPUT=1


COPY scripts/comfy-manager-set-mode.sh \
    /usr/local/bin/comfy-manager-set-mode

RUN chmod +x /usr/local/bin/comfy-manager-set-mode


# =============================================================================
# Default Command
# =============================================================================

CMD ["/start.sh"]


# =============================================================================
# =============================================================================
# Model Downloader
#
# The worker always includes:
#
#   1. Qwen Image Edit 2511
#   2. Qwen 2511 Lightning 4-step LoRA
#   3. WAN 2.1 VAE / text encoder
#   4. WAN 2.2 I2V high + low noise models
#   5. WAN 2.2 Lightning LoRAs
#
# SDXL, SD3 and Flux are intentionally not included.
# =============================================================================
# =============================================================================

FROM base AS downloader

WORKDIR /comfyui


# =============================================================================
# Model Directories
# =============================================================================

RUN mkdir -p \
    models/checkpoints \
    models/vae \
    models/unet \
    models/clip \
    models/loras


# =============================================================================
# Download Qwen + WAN
#
# We use five parallel jobs rather than ten separate downloads.
#
# Individual repositories containing multiple files use --max-workers.
#
# This provides:
#
# - parallel downloads
# - Xet acceleration
# - fewer processes
# - better error reporting
# - no Hugging Face token dependency
# - less temporary disk usage
# =============================================================================

RUN set -eu; \
    \
    echo "============================================================"; \
    echo "Downloading ComfyUI models"; \
    echo "Qwen Image Edit 2511 + WAN 2.2"; \
    echo "============================================================"; \
    \
    \
    mkdir -p \
        /tmp/qwen-edit \
        /tmp/qwen-base \
        /tmp/qwen-lora \
        /tmp/wan21 \
        /tmp/wan22; \
    \
    \
    echo ""; \
    echo "[1/5] Starting Qwen Image Edit 2511 diffusion model..."; \
    \
    hf download \
        Comfy-Org/Qwen-Image-Edit_ComfyUI \
        split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        --local-dir /tmp/qwen-edit \
        --max-workers 2 \
        > /tmp/qwen-edit.log 2>&1 \
        & PID_QWEN_EDIT=$!; \
    \
    \
    echo "[2/5] Starting Qwen text encoder + VAE..."; \
    \
    hf download \
        Comfy-Org/Qwen-Image_ComfyUI \
        split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        split_files/vae/qwen_image_vae.safetensors \
        --local-dir /tmp/qwen-base \
        --max-workers 2 \
        > /tmp/qwen-base.log 2>&1 \
        & PID_QWEN_BASE=$!; \
    \
    \
    echo "[3/5] Starting Qwen 2511 Lightning LoRA..."; \
    \
    hf download \
        lightx2v/Qwen-Image-Edit-2511-Lightning \
        Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        --local-dir /tmp/qwen-lora \
        --max-workers 2 \
        > /tmp/qwen-lora.log 2>&1 \
        & PID_QWEN_LORA=$!; \
    \
    \
    echo "[4/5] Starting WAN 2.1 VAE + text encoder..."; \
    \
    hf download \
        Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        split_files/vae/wan_2.1_vae.safetensors \
        split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        --local-dir /tmp/wan21 \
        --max-workers 2 \
        > /tmp/wan21.log 2>&1 \
        & PID_WAN21=$!; \
    \
    \
    echo "[5/5] Starting WAN 2.2 models + LoRAs..."; \
    \
    hf download \
        Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
        split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
        split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
        split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
        --local-dir /tmp/wan22 \
        --max-workers 2 \
        > /tmp/wan22.log 2>&1 \
        & PID_WAN22=$!; \
    \
    \
    echo ""; \
    echo "All download jobs started."; \
    echo "Waiting for completion..."; \
    echo ""; \
    \
    \
    FAILED=0; \
    \
    \
    if wait "${PID_QWEN_EDIT}"; then \
        echo "[OK] Qwen Image Edit 2511"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] Qwen Image Edit 2511 failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/qwen-edit.log || true; \
        FAILED=1; \
    fi; \
    \
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
    \
    if wait "${PID_QWEN_LORA}"; then \
        echo "[OK] Qwen Lightning LoRA"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] Qwen Lightning LoRA failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/qwen-lora.log || true; \
        FAILED=1; \
    fi; \
    \
    \
    if wait "${PID_WAN21}"; then \
        echo "[OK] WAN 2.1 VAE + text encoder"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] WAN 2.1 files failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/wan21.log || true; \
        FAILED=1; \
    fi; \
    \
    \
    if wait "${PID_WAN22}"; then \
        echo "[OK] WAN 2.2 models + LoRAs"; \
    else \
        STATUS=$?; \
        echo "============================================================"; \
        echo "[ERROR] WAN 2.2 files failed: ${STATUS}"; \
        echo "============================================================"; \
        cat /tmp/wan22.log || true; \
        FAILED=1; \
    fi; \
    \
    \
    if [ "${FAILED}" -ne 0 ]; then \
        echo ""; \
        echo "============================================================"; \
        echo "ONE OR MORE MODEL DOWNLOADS FAILED"; \
        echo "See the error output above."; \
        echo "============================================================"; \
        exit 1; \
    fi; \
    \
    \
    echo ""; \
    echo "============================================================"; \
    echo "All downloads completed successfully."; \
    echo "Moving models into ComfyUI..."; \
    echo "============================================================"; \
    \
    \
    mv \
        /tmp/qwen-edit/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        models/unet/qwen_image_edit_2511_fp8mixed.safetensors; \
    \
    \
    mv \
        /tmp/qwen-base/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors; \
    \
    \
    mv \
        /tmp/qwen-base/split_files/vae/qwen_image_vae.safetensors \
        models/vae/qwen_image_vae.safetensors; \
    \
    \
    mv \
        /tmp/qwen-lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors; \
    \
    \
    mv \
        /tmp/wan21/split_files/vae/wan_2.1_vae.safetensors \
        models/vae/wan_2.1_vae.safetensors; \
    \
    \
    mv \
        /tmp/wan21/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors; \
    \
    \
    mv \
        /tmp/wan22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
        models/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; \
    \
    \
    mv \
        /tmp/wan22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
        models/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors; \
    \
    \
    mv \
        /tmp/wan22/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
        models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors; \
    \
    \
    mv \
        /tmp/wan22/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
        models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors; \
    \
    \
    rm -rf \
        /tmp/qwen-edit \
        /tmp/qwen-base \
        /tmp/qwen-lora \
        /tmp/wan21 \
        /tmp/wan22 \
        /tmp/*.log; \
    \
    \
    echo ""; \
    echo "============================================================"; \
    echo "MODEL INSTALLATION COMPLETED"; \
    echo "============================================================"; \
    echo ""; \
    echo "Qwen:"; \
    echo "  qwen_image_edit_2511_fp8mixed.safetensors"; \
    echo "  qwen_2.5_vl_7b_fp8_scaled.safetensors"; \
    echo "  qwen_image_vae.safetensors"; \
    echo "  Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"; \
    echo ""; \
    echo "WAN:"; \
    echo "  wan_2.1_vae.safetensors"; \
    echo "  umt5_xxl_fp8_e4m3fn_scaled.safetensors"; \
    echo "  wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"; \
    echo "  wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"; \
    echo "  wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"; \
    echo "  wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"; \
    echo "============================================================"


# =============================================================================
# Final Image
# =============================================================================

FROM base AS final

COPY --from=downloader \
    /comfyui/models \
    /comfyui/models
