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
# =============================================================================

RUN python -m pip install \
        --no-cache-dir \
        -r /comfyui/requirements.txt \
    && python -c "import sqlalchemy, alembic; print('ComfyUI database dependencies OK')"


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
# IMPORTANT:
#
# This stage is kept separate from the final image.
#
# Changing handler.py / start.sh / other worker code will NOT trigger the
# expensive model downloads again.
#
# Models:
#
#   Qwen Image Edit 2511
#   WAN 2.2 I2V
#
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
# Output Directories
# =============================================================================

RUN mkdir -p \
    /model-output/unet \
    /model-output/clip \
    /model-output/vae \
    /model-output/loras \
    /tmp/qwen-edit \
    /tmp/qwen-base \
    /tmp/qwen-lora \
    /tmp/wan21 \
    /tmp/wan22


# =============================================================================
# Download Qwen 2511 + WAN
#
# Downloads are performed in parallel.
#
# No Hugging Face token is required because all model repositories used here
# are public.
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/huggingface \
    set -eu; \
    \
    echo "============================================================"; \
    echo "Downloading Qwen Image Edit 2511 + WAN 2.2"; \
    echo "============================================================"; \
    \
    echo "[1/3] Qwen Image Edit 2511 diffusion model"; \
    hf download \
        Comfy-Org/Qwen-Image-Edit_ComfyUI \
        split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        --local-dir /tmp/qwen-edit \
        --max-workers 2 \
        > /tmp/qwen-edit.log 2>&1 \
        & PID_QWEN_EDIT=$!; \
    \
    echo "[2/3] Qwen text encoder + VAE"; \
    hf download \
        Comfy-Org/Qwen-Image_ComfyUI \
        split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        split_files/vae/qwen_image_vae.safetensors \
        --local-dir /tmp/qwen-base \
        --max-workers 2 \
        > /tmp/qwen-base.log 2>&1 \
        & PID_QWEN_BASE=$!; \
    \
    echo "[3/3] Qwen 2511 Lightning LoRA"; \
    hf download \
        lightx2v/Qwen-Image-Edit-2511-Lightning \
        Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        --local-dir /tmp/qwen-lora \
        --max-workers 2 \
        > /tmp/qwen-lora.log 2>&1 \
        & PID_QWEN_LORA=$!; \
    \
    # echo "[4/5] WAN 2.1 VAE + text encoder"; \
    # hf download \
    #     Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    #     split_files/vae/wan_2.1_vae.safetensors \
    #     split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    #     --local-dir /tmp/wan21 \
    #     --max-workers 2 \
    #     > /tmp/wan21.log 2>&1 \
    #     & PID_WAN21=$!; \
    # \
    # echo "[5/5] WAN 2.2 diffusion models + Lightning LoRAs"; \
    # hf download \
    #     Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    #     split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
    #     split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
    #     split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
    #     split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
    #     --local-dir /tmp/wan22 \
    #     --max-workers 2 \
    #     > /tmp/wan22.log 2>&1 \
    #     & PID_WAN22=$!; \
    # \
    echo "Waiting for model downloads..."; \
    \
    FAILED=0; \
    \
    if wait "${PID_QWEN_EDIT}"; then \
        echo "[OK] Qwen Image Edit 2511"; \
    else \
        STATUS=$?; \
        echo "[ERROR] Qwen Image Edit 2511 failed: ${STATUS}"; \
        cat /tmp/qwen-edit.log || true; \
        FAILED=1; \
    fi; \
    \
    if wait "${PID_QWEN_BASE}"; then \
        echo "[OK] Qwen text encoder + VAE"; \
    else \
        STATUS=$?; \
        echo "[ERROR] Qwen text encoder / VAE failed: ${STATUS}"; \
        cat /tmp/qwen-base.log || true; \
        FAILED=1; \
    fi; \
    \
    if wait "${PID_QWEN_LORA}"; then \
        echo "[OK] Qwen Lightning LoRA"; \
    else \
        STATUS=$?; \
        echo "[ERROR] Qwen Lightning LoRA failed: ${STATUS}"; \
        cat /tmp/qwen-lora.log || true; \
        FAILED=1; \
    fi; \
    \
    # if wait "${PID_WAN21}"; then \
    #     echo "[OK] WAN 2.1 files"; \
    # else \
    #     STATUS=$?; \
    #     echo "[ERROR] WAN 2.1 files failed: ${STATUS}"; \
    #     cat /tmp/wan21.log || true; \
    #     FAILED=1; \
    # fi; \
    # \
    # if wait "${PID_WAN22}"; then \
    #     echo "[OK] WAN 2.2 files"; \
    # else \
    #     STATUS=$?; \
    #     echo "[ERROR] WAN 2.2 files failed: ${STATUS}"; \
    #     cat /tmp/wan22.log || true; \
    #     FAILED=1; \
    # fi; \
    # \
    if [ "${FAILED}" -ne 0 ]; then \
        echo "============================================================"; \
        echo "MODEL DOWNLOAD FAILED"; \
        echo "============================================================"; \
        exit 1; \
    fi; \
    \
    echo "Moving models into stable output directories..."; \
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
    # mv \
    #     /tmp/wan21/split_files/vae/wan_2.1_vae.safetensors \
    #     /model-output/vae/wan_2.1_vae.safetensors; \
    # \
    # mv \
    #     /tmp/wan21/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    #     /model-output/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors; \
    # \
    # mv \
    #     /tmp/wan22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
    #     /model-output/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; \
    # \
    # mv \
    #     /tmp/wan22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
    #     /model-output/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors; \
    # \
    # mv \
    #     /tmp/wan22/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
    #     /model-output/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors; \
    # \
    # mv \
    #     /tmp/wan22/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
    #     /model-output/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors; \
    # \
    rm -rf \
        /tmp/qwen-edit \
        /tmp/qwen-base \
        /tmp/qwen-lora \
        # /tmp/wan21 \
        # /tmp/wan22 \
        # /tmp/*.log; \
    \
    echo "============================================================"; \
    echo "ALL MODELS READY"; \
    echo "============================================================"


# =============================================================================
# =============================================================================
# Final Production Image
#
# Important:
#
# Each major model is copied as a SEPARATE Docker layer.
#
# Benefits:
#
# - Docker Hub can cache/deduplicate individual model layers.
# - Future code-only releases do not rebuild these layers.
# - RunPod can pull multiple layers concurrently.
# - Changing one model doesn't invalidate every other model.
#
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
# WAN
# =============================================================================

# COPY --from=model-downloader \
#     /model-output/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
#     /comfyui/models/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors

# COPY --from=model-downloader \
#     /model-output/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
#     /comfyui/models/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors

# COPY --from=model-downloader \
#     /model-output/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
#     /comfyui/models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# COPY --from=model-downloader \
#     /model-output/vae/wan_2.1_vae.safetensors \
#     /comfyui/models/vae/wan_2.1_vae.safetensors

# COPY --from=model-downloader \
#     /model-output/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
#     /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

# COPY --from=model-downloader \
#     /model-output/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
#     /comfyui/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors


# =============================================================================
# Worker Files
#
# KEEP THESE AFTER THE MODEL LAYERS.
#
# This is intentional.
#
# Changing handler.py or start.sh should only invalidate these tiny final
# layers, NOT the huge model layers above.
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
# Start
# =============================================================================

WORKDIR /

CMD ["/start.sh"]
