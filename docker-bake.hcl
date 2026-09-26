variable "DOCKERHUB_REPO" {
  default = "runpod"
}

variable "DOCKERHUB_IMG" {
  default = "worker-comfyui"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "COMFYUI_VERSION" {
  default = "0.37.0"
}

variable "BASE_IMAGE" {
  default = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
}

variable "CUDA_VERSION_FOR_COMFY" {
  default = "12.6"
}


# =============================================================================
# Default Build
#
# Dedicated Qwen Image 2.1 image-editing worker:
# - Q5_K_M uncensored GGUF diffusion
# - Q4_K_M Qwen3-VL encoder
# - Q4_K_M local I2I prompt enhancer
# =============================================================================

group "default" {
  targets = ["worker"]
}


# =============================================================================
# Production Worker
# =============================================================================

target "worker" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "final"

  platforms = [
    "linux/amd64"
  ]

  args = {
    BASE_IMAGE             = "${BASE_IMAGE}"
    COMFYUI_VERSION        = "${COMFYUI_VERSION}"
    CUDA_VERSION_FOR_COMFY = "${CUDA_VERSION_FOR_COMFY}"
  }

  tags = [
    "${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}",
    "${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:latest"
  ]
}
