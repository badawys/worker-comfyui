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

# Hugging Face download configuration
ENV HF_HUB_DOWNLOAD_TIMEOUT=300
ENV HF_HUB_ETAG_TIMEOUT=30


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

RUN git clone --depth 1 \
        https://github.com/cubiq/ComfyUI_essentials.git \
    && git clone --depth 1 \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone --depth 1 \
        https://github.com/kijai/ComfyUI-WanVideoWrapper.git


# =============================================================================
# Custom Node Requirements
# =============================================================================

RUN pip install \
        --no-cache-dir \
        -r ComfyUI_essentials/requirements.txt \
        || echo "No ComfyUI Essentials requirements" \
    && pip install \
        --no-cache-dir \
        -r ComfyUI-VideoHelperSuite/requirements.txt \
        || echo "No VideoHelperSuite requirements" \
    && pip install \
        --no-cache-dir \
        -r ComfyUI-WanVideoWrapper/requirements.txt \
        || echo "No WanVideoWrapper requirements"


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
# =============================================================================
# =============================================================================

FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN

# Supported:
#
# default
# qwen-edit-2511
# wan
#
# default = Qwen Image Edit 2511 + WAN 2.2
#
ARG MODEL_TYPE=default


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
# DEFAULT
#
# Qwen Image Edit 2511 + WAN
#
# All independent model files are downloaded in parallel.
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/huggingface \
    if [ "${MODEL_TYPE}" = "default" ]; then \
        \
        set -eu; \
        \
        echo "============================================================"; \
        echo "Downloading DEFAULT model set"; \
        echo "Qwen Image Edit 2511 + WAN"; \
        echo "Downloads running in parallel"; \
        echo "============================================================"; \
        \
        mkdir -p \
            /tmp/qwen-unet \
            /tmp/qwen-clip \
            /tmp/qwen-vae \
            /tmp/qwen-lora \
            /tmp/wan-vae \
            /tmp/wan-clip \
            /tmp/wan-low \
            /tmp/wan-high \
            /tmp/wan-lora-low \
            /tmp/wan-lora-high; \
        \
        \
        echo "Starting Qwen Image Edit 2511 model..."; \
        hf download \
            Comfy-Org/Qwen-Image-Edit_ComfyUI \
            split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
            --local-dir /tmp/qwen-unet \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_UNET=$!; \
        \
        \
        echo "Starting Qwen text encoder..."; \
        hf download \
            Comfy-Org/Qwen-Image_ComfyUI \
            split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
            --local-dir /tmp/qwen-clip \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_CLIP=$!; \
        \
        \
        echo "Starting Qwen VAE..."; \
        hf download \
            Comfy-Org/Qwen-Image_ComfyUI \
            split_files/vae/qwen_image_vae.safetensors \
            --local-dir /tmp/qwen-vae \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_VAE=$!; \
        \
        \
        echo "Starting Qwen 2511 Lightning LoRA..."; \
        hf download \
            lightx2v/Qwen-Image-Edit-2511-Lightning \
            Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
            --local-dir /tmp/qwen-lora \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_LORA=$!; \
        \
        \
        echo "Starting WAN VAE..."; \
        hf download \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/vae/wan_2.1_vae.safetensors \
            --local-dir /tmp/wan-vae \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_VAE=$!; \
        \
        \
        echo "Starting WAN text encoder..."; \
        hf download \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
            --local-dir /tmp/wan-clip \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_CLIP=$!; \
        \
        \
        echo "Starting WAN low-noise model..."; \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
            --local-dir /tmp/wan-low \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LOW=$!; \
        \
        \
        echo "Starting WAN high-noise model..."; \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
            --local-dir /tmp/wan-high \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_HIGH=$!; \
        \
        \
        echo "Starting WAN low-noise Lightning LoRA..."; \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
            --local-dir /tmp/wan-lora-low \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LORA_LOW=$!; \
        \
        \
        echo "Starting WAN high-noise Lightning LoRA..."; \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
            --local-dir /tmp/wan-lora-high \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LORA_HIGH=$!; \
        \
        \
        echo "Waiting for all model downloads..."; \
        \
        wait "${PID_QWEN_UNET}"; \
        wait "${PID_QWEN_CLIP}"; \
        wait "${PID_QWEN_VAE}"; \
        wait "${PID_QWEN_LORA}"; \
        wait "${PID_WAN_VAE}"; \
        wait "${PID_WAN_CLIP}"; \
        wait "${PID_WAN_LOW}"; \
        wait "${PID_WAN_HIGH}"; \
        wait "${PID_WAN_LORA_LOW}"; \
        wait "${PID_WAN_LORA_HIGH}"; \
        \
        \
        echo "Moving Qwen models..."; \
        \
        cp \
            /tmp/qwen-unet/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
            models/unet/qwen_image_edit_2511_fp8mixed.safetensors; \
        \
        cp \
            /tmp/qwen-clip/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
            models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/qwen-vae/split_files/vae/qwen_image_vae.safetensors \
            models/vae/qwen_image_vae.safetensors; \
        \
        cp \
            /tmp/qwen-lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
            models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors; \
        \
        \
        echo "Moving WAN models..."; \
        \
        cp \
            /tmp/wan-vae/split_files/vae/wan_2.1_vae.safetensors \
            models/vae/wan_2.1_vae.safetensors; \
        \
        cp \
            /tmp/wan-clip/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
            models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors; \
        \
        cp \
            /tmp/wan-low/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
            models/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/wan-high/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
            models/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/wan-lora-low/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
            models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors; \
        \
        cp \
            /tmp/wan-lora-high/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
            models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors; \
        \
        \
        rm -rf \
            /tmp/qwen-unet \
            /tmp/qwen-clip \
            /tmp/qwen-vae \
            /tmp/qwen-lora \
            /tmp/wan-vae \
            /tmp/wan-clip \
            /tmp/wan-low \
            /tmp/wan-high \
            /tmp/wan-lora-low \
            /tmp/wan-lora-high; \
        \
        echo "============================================================"; \
        echo "DEFAULT model set downloaded successfully"; \
        echo "Qwen Image Edit 2511 + WAN ready"; \
        echo "============================================================"; \
    fi


# =============================================================================
# QWEN IMAGE EDIT 2511 ONLY
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/huggingface \
    if [ "${MODEL_TYPE}" = "qwen-edit-2511" ]; then \
        \
        set -eu; \
        \
        echo "Downloading Qwen Image Edit 2511..."; \
        \
        mkdir -p \
            /tmp/qwen-unet \
            /tmp/qwen-clip \
            /tmp/qwen-vae \
            /tmp/qwen-lora; \
        \
        \
        hf download \
            Comfy-Org/Qwen-Image-Edit_ComfyUI \
            split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
            --local-dir /tmp/qwen-unet \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_UNET=$!; \
        \
        hf download \
            Comfy-Org/Qwen-Image_ComfyUI \
            split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
            --local-dir /tmp/qwen-clip \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_CLIP=$!; \
        \
        hf download \
            Comfy-Org/Qwen-Image_ComfyUI \
            split_files/vae/qwen_image_vae.safetensors \
            --local-dir /tmp/qwen-vae \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_VAE=$!; \
        \
        hf download \
            lightx2v/Qwen-Image-Edit-2511-Lightning \
            Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
            --local-dir /tmp/qwen-lora \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_QWEN_LORA=$!; \
        \
        \
        wait "${PID_QWEN_UNET}"; \
        wait "${PID_QWEN_CLIP}"; \
        wait "${PID_QWEN_VAE}"; \
        wait "${PID_QWEN_LORA}"; \
        \
        \
        cp \
            /tmp/qwen-unet/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
            models/unet/qwen_image_edit_2511_fp8mixed.safetensors; \
        \
        cp \
            /tmp/qwen-clip/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
            models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/qwen-vae/split_files/vae/qwen_image_vae.safetensors \
            models/vae/qwen_image_vae.safetensors; \
        \
        cp \
            /tmp/qwen-lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
            models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors; \
        \
        \
        rm -rf \
            /tmp/qwen-unet \
            /tmp/qwen-clip \
            /tmp/qwen-vae \
            /tmp/qwen-lora; \
        \
        echo "Qwen Image Edit 2511 ready."; \
    fi


# =============================================================================
# WAN ONLY
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/huggingface \
    if [ "${MODEL_TYPE}" = "wan" ]; then \
        \
        set -eu; \
        \
        echo "Downloading WAN models..."; \
        \
        mkdir -p \
            /tmp/wan-vae \
            /tmp/wan-clip \
            /tmp/wan-low \
            /tmp/wan-high \
            /tmp/wan-lora-low \
            /tmp/wan-lora-high; \
        \
        \
        hf download \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/vae/wan_2.1_vae.safetensors \
            --local-dir /tmp/wan-vae \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_VAE=$!; \
        \
        hf download \
            Comfy-Org/Wan_2.1_ComfyUI_repackaged \
            split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
            --local-dir /tmp/wan-clip \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_CLIP=$!; \
        \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
            --local-dir /tmp/wan-low \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LOW=$!; \
        \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
            --local-dir /tmp/wan-high \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_HIGH=$!; \
        \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
            --local-dir /tmp/wan-lora-low \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LORA_LOW=$!; \
        \
        hf download \
            Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
            split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
            --local-dir /tmp/wan-lora-high \
            --token "${HUGGINGFACE_ACCESS_TOKEN}" \
            & PID_WAN_LORA_HIGH=$!; \
        \
        \
        wait "${PID_WAN_VAE}"; \
        wait "${PID_WAN_CLIP}"; \
        wait "${PID_WAN_LOW}"; \
        wait "${PID_WAN_HIGH}"; \
        wait "${PID_WAN_LORA_LOW}"; \
        wait "${PID_WAN_LORA_HIGH}"; \
        \
        \
        cp \
            /tmp/wan-vae/split_files/vae/wan_2.1_vae.safetensors \
            models/vae/wan_2.1_vae.safetensors; \
        \
        cp \
            /tmp/wan-clip/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
            models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors; \
        \
        cp \
            /tmp/wan-low/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
            models/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/wan-high/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
            models/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors; \
        \
        cp \
            /tmp/wan-lora-low/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
            models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors; \
        \
        cp \
            /tmp/wan-lora-high/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
            models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors; \
        \
        \
        rm -rf \
            /tmp/wan-vae \
            /tmp/wan-clip \
            /tmp/wan-low \
            /tmp/wan-high \
            /tmp/wan-lora-low \
            /tmp/wan-lora-high; \
        \
        echo "WAN models ready."; \
    fi


# =============================================================================
# Final Image
# =============================================================================

FROM base AS final

COPY --from=downloader \
    /comfyui/models \
    /comfyui/models
