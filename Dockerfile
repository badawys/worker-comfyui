# syntax=docker/dockerfile:1.7

# =============================================================================
# Qwen Image 2.1 GGUF RunPod worker
# =============================================================================

# CUDA 12.8 is the minimum for Blackwell (sm_120), e.g. RTX PRO 6000 / MIG slices.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
# =============================================================================
# ComfyUI base
# =============================================================================

FROM ${BASE_IMAGE} AS comfy-base

ARG COMFYUI_VERSION=0.37.0
ARG CUDA_VERSION_FOR_COMFY=12.8

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PIP_NO_INPUT=1
# CUDA modules are loaded on first use instead of eagerly at process startup.
# This reduces cold-start work and GPU memory overhead on serverless workers.
ENV CUDA_MODULE_LOADING=LAZY

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
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install \
    comfy-cli \
    pip \
    setuptools \
    wheel

# ComfyUI 0.37.0 contains the native Qwen Image 2.1 nodes.
# Everything in this image targets CUDA 12.8 (base image, PyTorch cu128 and the
# llama.cpp cu128 wheel), so it runs on any host driver that supports CUDA 12.8+
# (driver 570+), including Blackwell GPUs such as RTX PRO 6000 and its MIG slices.
# Hosts that are all CUDA 13 can build with CUDA_VERSION_FOR_COMFY=13.0 for
# ComfyUI's cu130-only optimized kernels; that build then needs driver 580+.
RUN /usr/bin/yes | comfy \
        --workspace /comfyui \
        install \
        --version "${COMFYUI_VERSION}" \
        --cuda-version "${CUDA_VERSION_FOR_COMFY}" \
        --nvidia

# comfy-cli owns /comfyui/.venv. Make that the one runtime environment for
# ComfyUI, custom nodes and the RunPod handler so dependencies cannot diverge.
ENV VIRTUAL_ENV=/comfyui/.venv
ENV PATH="/comfyui/.venv/bin:/opt/venv/bin:${PATH}"

# =============================================================================
# Qwen Image 2.1 custom nodes
# =============================================================================

WORKDIR /comfyui/custom_nodes

RUN git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF.git \
    && git clone --depth 1 https://github.com/xiaowuapple-pixel/ComfyUI-Prompt-Enhancer.git

RUN if [ -f ComfyUI-GGUF/requirements.txt ]; then \
        uv pip install --python /comfyui/.venv/bin/python \
            --no-cache-dir -r ComfyUI-GGUF/requirements.txt; \
    fi \
    && if [ -f ComfyUI-Prompt-Enhancer/requirements.txt ]; then \
        uv pip install --python /comfyui/.venv/bin/python \
            --no-cache-dir -r ComfyUI-Prompt-Enhancer/requirements.txt; \
    fi

# Use JamePeng's prebuilt CUDA 12.8 / Python 3.12 Linux wheel. Same 0.3.49
# release as before; the cu128 build includes sm_120 (Blackwell) kernels.
# This avoids compiling llama.cpp during the RunPod image build (the previous
# source build could consume a large part of RunPod's 30-minute build limit).
ARG LLAMA_CPP_WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.49-cu128-linux-20260831/llama_cpp_python-0.3.49%2Bcu128-cp312-cp312-linux_x86_64.whl

RUN uv pip install --python /comfyui/.venv/bin/python \
        "${LLAMA_CPP_WHEEL_URL}" \
    && uv pip install --python /comfyui/.venv/bin/python \
        sentencepiece \
        protobuf

# RunPod worker dependencies live in the same environment as ComfyUI.
WORKDIR /

RUN uv pip install --python /comfyui/.venv/bin/python \
    runpod \
    requests \
    websocket-client


# =============================================================================
# Model downloader
# =============================================================================

FROM comfy-base AS model-downloader

ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ETAG_TIMEOUT=60
ENV HF_HUB_DISABLE_UPDATE_CHECK=1

RUN uv pip install --python /comfyui/.venv/bin/python \
    huggingface_hub \
    hf_xet

COPY scripts/prepare_qwen21_turbo_lora.py /prepare_qwen21_turbo_lora.py

RUN mkdir -p \
    /model-output/diffusion_models \
    /model-output/loras \
    /model-output/text_encoders \
    /model-output/vae \
    /model-output/LLM \
    /tmp/qwen21-diffusion \
    /tmp/qwen21-encoder \
    /tmp/qwen21-vae \
    /tmp/qwen21-pe \
    /tmp/qwen21-turbo-lora

# The original abenzerps/Qwen-Image-2.1-Uncensored-GGUF repo was renamed and its
# files replaced (2026-11-03), so the uncensored Q5_K_M file now comes from
# mirrors pinned to a commit. Every mirror checked has the same SHA-256.
ARG QWEN21_UC_Q5_SHA256=af0bf278cf16d204fb31c384dc82fd41dca82d976b15fe9305a60c726fd6f821

# Download the Quality profile's Qwen Image 2.1 components in parallel.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    set -eu; \
    \
    echo "[1/5] Qwen Image 2.1 uncensored Q5_K_M diffusion"; \
    ( \
        for SRC in \
            "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF@240ef1d68f778115f8ff69965eca9ad6d3ccdcf1" \
            "basileleuwa/Qwen-Image-2.1-Uncensored-GGUF@dcbc8aaae12500aab2a00052ad6b5bbff1dea2d6"; do \
            echo "Trying ${SRC}"; \
            if hf download "${SRC%@*}" qwen-image-2.1-UC-Q5_K_M.gguf \
                    --revision "${SRC#*@}" \
                    --local-dir /tmp/qwen21-diffusion \
                    --max-workers 2 \
                && echo "${QWEN21_UC_Q5_SHA256}  /tmp/qwen21-diffusion/qwen-image-2.1-UC-Q5_K_M.gguf" | sha256sum -c -; then \
                exit 0; \
            fi; \
            rm -f /tmp/qwen21-diffusion/qwen-image-2.1-UC-Q5_K_M.gguf; \
        done; \
        exit 1 \
    ) > /tmp/qwen21-diffusion.log 2>&1 \
        & PID_DIFFUSION=$!; \
    \
    echo "[2/5] Qwen3-VL 8B Q4_K_M encoder + vision projector"; \
    hf download \
        gguf-org/qwen-image-2.1-gguf \
        qwen3vl-8b-it-q4_k_m.gguf \
        mmproj-qwen3vl-8b-it-q8_0.gguf \
        --local-dir /tmp/qwen21-encoder \
        --max-workers 2 \
        > /tmp/qwen21-encoder.log 2>&1 \
        & PID_ENCODER=$!; \
    \
    echo "[3/5] Official Qwen Image 2.1 VAE"; \
    hf download \
        Comfy-Org/Qwen-Image-2.1 \
        vae/qwen_image_2.1_vae_bf16.safetensors \
        --local-dir /tmp/qwen21-vae \
        --max-workers 2 \
        > /tmp/qwen21-vae.log 2>&1 \
        & PID_VAE=$!; \
    \
    echo "[4/5] Qwen Image 2.1 I2I prompt enhancer Q4_K_M + projector"; \
    hf download \
        prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF \
        Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf \
        Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf \
        --local-dir /tmp/qwen21-pe \
        --max-workers 2 \
        > /tmp/qwen21-pe.log 2>&1 \
        & PID_PE=$!; \
    \
    echo "[5/5] Quality · official Qwen Image 2.1 Turbo LoRA"; \
    hf download \
        Comfy-Org/Qwen-Image-2.1 \
        loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors \
        --local-dir /tmp/qwen21-turbo-lora \
        --max-workers 2 \
        > /tmp/qwen21-turbo-lora.log 2>&1 \
        & PID_TURBO_LORA=$!; \
    \
    FAILED=0; \
    for ITEM in \
        "DIFFUSION:${PID_DIFFUSION}:/tmp/qwen21-diffusion.log" \
        "ENCODER:${PID_ENCODER}:/tmp/qwen21-encoder.log" \
        "VAE:${PID_VAE}:/tmp/qwen21-vae.log" \
        "PROMPT_ENHANCER:${PID_PE}:/tmp/qwen21-pe.log" \
        "TURBO_LORA:${PID_TURBO_LORA}:/tmp/qwen21-turbo-lora.log"; do \
        NAME="${ITEM%%:*}"; \
        REST="${ITEM#*:}"; \
        PID="${REST%%:*}"; \
        LOG="${REST#*:}"; \
        if wait "${PID}"; then \
            echo "[OK] ${NAME}"; \
        else \
            echo "[ERROR] ${NAME}"; \
            cat "${LOG}" || true; \
            FAILED=1; \
        fi; \
    done; \
    test "${FAILED}" -eq 0; \
    \
    python /prepare_qwen21_turbo_lora.py \
        /tmp/qwen21-turbo-lora/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors \
        /model-output/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16_unfused.safetensors; \
    \
    mv /tmp/qwen21-diffusion/qwen-image-2.1-UC-Q5_K_M.gguf \
        /model-output/diffusion_models/qwen-image-2.1-UC-Q5_K_M.gguf; \
    mv /tmp/qwen21-encoder/qwen3vl-8b-it-q4_k_m.gguf \
        /model-output/text_encoders/qwen3vl-8b-it-q4_k_m.gguf; \
    mv /tmp/qwen21-encoder/mmproj-qwen3vl-8b-it-q8_0.gguf \
        /model-output/text_encoders/mmproj-qwen3vl-8b-it-q8_0.gguf; \
    mv /tmp/qwen21-vae/vae/qwen_image_2.1_vae_bf16.safetensors \
        /model-output/vae/qwen_image_2.1_vae_bf16.safetensors; \
    mv /tmp/qwen21-pe/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf \
        /model-output/LLM/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf; \
    mv /tmp/qwen21-pe/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf \
        /model-output/LLM/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf; \
    \
    rm -rf \
        /tmp/qwen21-diffusion \
        /tmp/qwen21-encoder \
        /tmp/qwen21-vae \
        /tmp/qwen21-pe \
        /tmp/qwen21-turbo-lora \
        /tmp/qwen21-diffusion.log \
        /tmp/qwen21-encoder.log \
        /tmp/qwen21-vae.log \
        /tmp/qwen21-pe.log \
        /tmp/qwen21-turbo-lora.log


# =============================================================================
# Final production image
# =============================================================================

FROM comfy-base AS final

# Small local nodes. Copied here (not in comfy-base) so editing them does not
# force the model-downloader stage to download every model again.
COPY src/custom_nodes/pe_preview_resize.py /comfyui/custom_nodes/pe_preview_resize.py
COPY src/custom_nodes/qwen_exact_size.py /comfyui/custom_nodes/qwen_exact_size.py

RUN mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/loras \
    /comfyui/models/text_encoders \
    /comfyui/models/vae \
    /comfyui/models/LLM

COPY --from=model-downloader \
    /model-output/diffusion_models/qwen-image-2.1-UC-Q5_K_M.gguf \
    /comfyui/models/diffusion_models/qwen-image-2.1-UC-Q5_K_M.gguf

COPY --from=model-downloader \
    /model-output/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16_unfused.safetensors \
    /comfyui/models/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16_unfused.safetensors

COPY --from=model-downloader \
    /model-output/text_encoders/qwen3vl-8b-it-q4_k_m.gguf \
    /comfyui/models/text_encoders/qwen3vl-8b-it-q4_k_m.gguf

COPY --from=model-downloader \
    /model-output/text_encoders/mmproj-qwen3vl-8b-it-q8_0.gguf \
    /comfyui/models/text_encoders/mmproj-qwen3vl-8b-it-q8_0.gguf

COPY --from=model-downloader \
    /model-output/vae/qwen_image_2.1_vae_bf16.safetensors \
    /comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors

COPY --from=model-downloader \
    /model-output/LLM/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf \
    /comfyui/models/LLM/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf

COPY --from=model-downloader \
    /model-output/LLM/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf \
    /comfyui/models/LLM/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf

# Keep a compiler available for PyTorch/Triton JIT kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.version.cuda and tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8), 'PyTorch cu128+ is required (Blackwell / sm_120)'" \
    && python -c "import runpod; print('RunPod SDK OK')" \
    && python -c "import gguf; print('ComfyUI-GGUF dependency OK')" \
    && python -c "import llama_cpp; print('llama-cpp-python:', llama_cpp.__version__)"

WORKDIR /

COPY src/start.sh /start.sh
COPY handler.py /handler.py
COPY test_input.json /test_input.json

RUN chmod +x /start.sh

COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

RUN echo "============================================================" \
    && echo "worker-comfyui Qwen Image 2.1 GGUF image ready" \
    && echo "Quality: uncensored Q5_K_M GGUF + Turbo LoRA · 8 steps" \
    && echo "Encoder: Qwen3-VL 8B Q4_K_M GGUF" \
    && echo "Prompt enhancer: I2I Q4_K_M GGUF" \
    && echo "ComfyUI: ${COMFYUI_VERSION:-0.37.0}" \
    && echo "============================================================"

CMD ["/start.sh"]
