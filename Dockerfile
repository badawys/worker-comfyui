# syntax=docker/dockerfile:1.7

# =============================================================================
# Qwen Image 2.1 GGUF RunPod worker
# =============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# =============================================================================
# ComfyUI base
# =============================================================================

FROM ${BASE_IMAGE} AS comfy-base

ARG COMFYUI_VERSION=0.37.0
ARG CUDA_VERSION_FOR_COMFY=12.6

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PIP_NO_INPUT=1

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

# Small local scheduler node used by the premerged Viggle v0.2.1 Fast checkpoint.
COPY src/custom_nodes/viggle_turbo_sigmas.py /comfyui/custom_nodes/viggle_turbo_sigmas.py

# Use JamePeng's prebuilt CUDA 12.6 / Python 3.12 Linux wheel.
# This avoids compiling llama.cpp during the RunPod image build (the previous
# source build could consume a large part of RunPod's 30-minute build limit).
ARG LLAMA_CPP_WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.49-cu126-linux-20260831/llama_cpp_python-0.3.49%2Bcu126-cp312-cp312-linux_x86_64.whl

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

RUN mkdir -p \
    /model-output/diffusion_models \
    /model-output/text_encoders \
    /model-output/vae \
    /model-output/LLM \
    /tmp/qwen21-diffusion \
    /tmp/qwen21-encoder \
    /tmp/qwen21-vae \
    /tmp/qwen21-pe \
    /tmp/qwen21-fast \
    /tmp/qwen21-turbo

# Download shared Qwen Image 2.1 components plus Fast/Turbo diffusion profiles in parallel.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    set -eu; \
    \
    echo "[1/4] Qwen Image 2.1 uncensored Q5_K_M diffusion"; \
    hf download \
        abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
        qwen-image-2.1-UC-Q5_K_M.gguf \
        --local-dir /tmp/qwen21-diffusion \
        --max-workers 2 \
        > /tmp/qwen21-diffusion.log 2>&1 \
        & PID_DIFFUSION=$!; \
    \
    echo "[2/4] Qwen3-VL 8B Q4_K_M encoder + vision projector"; \
    hf download \
        gguf-org/qwen-image-2.1-gguf \
        qwen3vl-8b-it-q4_k_m.gguf \
        mmproj-qwen3vl-8b-it-q8_0.gguf \
        --local-dir /tmp/qwen21-encoder \
        --max-workers 2 \
        > /tmp/qwen21-encoder.log 2>&1 \
        & PID_ENCODER=$!; \
    \
    echo "[3/4] Official Qwen Image 2.1 VAE"; \
    hf download \
        Comfy-Org/Qwen-Image-2.1 \
        vae/qwen_image_2.1_vae_bf16.safetensors \
        --local-dir /tmp/qwen21-vae \
        --max-workers 2 \
        > /tmp/qwen21-vae.log 2>&1 \
        & PID_VAE=$!; \
    \
    echo "[4/4] Qwen Image 2.1 I2I prompt enhancer Q4_K_M + projector"; \
    hf download \
        prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF \
        Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf \
        Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf \
        --local-dir /tmp/qwen21-pe \
        --max-workers 2 \
        > /tmp/qwen21-pe.log 2>&1 \
        & PID_PE=$!; \
    \
    echo "[5/6] Fast · Viggle v0.2.1 premerged 6-step Q5_K_M GGUF"; \
    hf download \
        Abiray/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-GGUF \
        qwen_image_2.1_turbo_Q5_K_M.gguf \
        --local-dir /tmp/qwen21-fast \
        --max-workers 2 \
        > /tmp/qwen21-fast.log 2>&1 \
        & PID_FAST=$!; \
    \
    echo "[6/6] Turbo · Viggle v0.1 premerged 4-step Q5_K_M GGUF"; \
    hf download \
        Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF \
        qwen_image_2.1_turbo_Q5_K_M.gguf \
        --local-dir /tmp/qwen21-turbo \
        --max-workers 2 \
        > /tmp/qwen21-turbo.log 2>&1 \
        & PID_TURBO=$!; \
    \
    FAILED=0; \
    for ITEM in \
        "DIFFUSION:${PID_DIFFUSION}:/tmp/qwen21-diffusion.log" \
        "ENCODER:${PID_ENCODER}:/tmp/qwen21-encoder.log" \
        "VAE:${PID_VAE}:/tmp/qwen21-vae.log" \
        "PROMPT_ENHANCER:${PID_PE}:/tmp/qwen21-pe.log" \
        "FAST:${PID_FAST}:/tmp/qwen21-fast.log" \
        "TURBO:${PID_TURBO}:/tmp/qwen21-turbo.log"; do \
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
    mv /tmp/qwen21-fast/qwen_image_2.1_turbo_Q5_K_M.gguf \
        /model-output/diffusion_models/qwen_image_2.1_fast_v0.2.1_Q5_K_M.gguf; \
    mv /tmp/qwen21-turbo/qwen_image_2.1_turbo_Q5_K_M.gguf \
        /model-output/diffusion_models/qwen_image_2.1_turbo_v0.1_Q5_K_M.gguf; \
    \
    rm -rf \
        /tmp/qwen21-diffusion \
        /tmp/qwen21-encoder \
        /tmp/qwen21-vae \
        /tmp/qwen21-pe \
        /tmp/qwen21-fast \
        /tmp/qwen21-turbo \
        /tmp/qwen21-diffusion.log \
        /tmp/qwen21-encoder.log \
        /tmp/qwen21-vae.log \
        /tmp/qwen21-pe.log \
        /tmp/qwen21-fast.log \
        /tmp/qwen21-turbo.log


# =============================================================================
# Final production image
# =============================================================================

FROM comfy-base AS final

RUN mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/text_encoders \
    /comfyui/models/vae \
    /comfyui/models/LLM

COPY --from=model-downloader \
    /model-output/diffusion_models/qwen-image-2.1-UC-Q5_K_M.gguf \
    /comfyui/models/diffusion_models/qwen-image-2.1-UC-Q5_K_M.gguf

COPY --from=model-downloader \
    /model-output/diffusion_models/qwen_image_2.1_fast_v0.2.1_Q5_K_M.gguf \
    /comfyui/models/diffusion_models/qwen_image_2.1_fast_v0.2.1_Q5_K_M.gguf

COPY --from=model-downloader \
    /model-output/diffusion_models/qwen_image_2.1_turbo_v0.1_Q5_K_M.gguf \
    /comfyui/models/diffusion_models/qwen_image_2.1_turbo_v0.1_Q5_K_M.gguf

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

RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)" \
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
    && echo "Quality: uncensored Q5_K_M GGUF · 25 steps" \
    && echo "Fast: Viggle v0.2.1 Q5_K_M GGUF · 6 steps" \
    && echo "Turbo: Viggle v0.1 Q5_K_M GGUF · 4 steps" \
    && echo "Encoder: Qwen3-VL 8B Q4_K_M GGUF" \
    && echo "Prompt enhancer: I2I Q4_K_M GGUF" \
    && echo "ComfyUI: ${COMFYUI_VERSION:-0.37.0}" \
    && echo "============================================================"

CMD ["/start.sh"]
