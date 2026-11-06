# worker-comfyui

> [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as a serverless API on [RunPod](https://www.runpod.io/)

<p align="center">
  <img src="assets/worker_sitting_in_comfy_chair.jpg" title="Worker sitting in comfy chair" />
</p>

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-comfyui)](https://www.runpod.io/console/hub/runpod-workers/worker-comfyui)

---

This project allows you to run ComfyUI workflows as a serverless API endpoint on the RunPod platform. Submit workflows via API calls and receive generated images as base64 strings or S3 URLs.



## Qwen Image 2.1 GGUF Worker

The `qwen-image-2.1-gguf` branch provides one Qwen Image 2.1 image-editing profile, **Quality (8 steps)**, for RunPod Serverless, with an optional local I2I prompt enhancer.

### Quality profile

| Diffusion | Sampling |
| --- | --- |
| `qwen-image-2.1-UC-Q5_K_M.gguf` + official Turbo LoRA (strength 1) | Official 8-step sigma schedule · Euler · CFG 1 |

Quality applies the [official extracted Turbo LoRA](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors) through `LoraLoaderModelOnly`, between `UnetLoaderGGUF` and `QwenImage21Cache`, then samples with `CFGGuider` (CFG 1), `KSamplerSelect` (Euler), `ManualSigmas` and `SamplerCustomAdvanced`. The sigma schedule comes from the [official image-edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_turbo_image_edit.json): `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`.

The Fast (Viggle) and Turbo profiles and their models have been removed.

**Model source:** the original `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` repo was renamed and its files replaced on 2026-11-03. The build now downloads `qwen-image-2.1-UC-Q5_K_M.gguf` from a mirror pinned to a commit, falls back to a second mirror, and checks the file's SHA-256 (`QWEN21_UC_Q5_SHA256` build arg).

### Exact output size

`QwenImage21TextEncodeList` rounds the reference image to multiples of 32 and stretches it, which made outputs a few pixels off in size and slightly shifted. The workflow now pads the source to a multiple of 32 (`QwenPadToMultiple`, edge pixels repeated) and crops the decoded image back to the source size (`QwenCropToReference`). Both nodes live in `src/custom_nodes/qwen_exact_size.py`.

### Quality LoRA compatibility

The Quality GGUF stores separate `img_mlp.gate_layer` and `img_mlp.proj` tensors. The official LoRA stores fused `img_mlp.gate_up` adapters, which can be silently skipped by the current ComfyUI mapping ([upstream issue #16944](https://github.com/Comfy-Org/ComfyUI/issues/16944)). The model-downloader stage runs `scripts/prepare_qwen21_turbo_lora.py` once to split all 32 MLP adapters: split `lora_up` row-wise and duplicate `lora_down` and `alpha`. This preserves the adapter matrix products, BF16 precision, rank and scale. The prepared file is stored as `models/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16_unfused.safetensors`. The script fails the build on unexpected or incomplete layouts.

### Shared models

| Component | Model |
| --- | --- |
| Qwen3-VL image/text encoder | `qwen3vl-8b-it-q4_k_m.gguf` |
| Qwen3-VL vision projector | `mmproj-qwen3vl-8b-it-q8_0.gguf` |
| I2I prompt enhancer | `Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf` |
| Prompt-enhancer vision projector | `Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |

The prompt enhancer is optional per request and off by default in RadiantLunar, which then wires the raw prompt directly to the text encoder.

### Workflows

- `workflows/qwen_image_2_1_quality.json`
- `workflows/qwen_image_2_1_edit_gguf.json` remains as the original unaccelerated 25-step workflow for backwards compatibility.

RadiantLunar normalizes every uploaded image to a maximum 768px longest side while preserving aspect ratio. The worker enforces the same cap as a fallback for direct API callers.

### Prompt enhancer settings

Thinking mode with 400 plan tokens, a 1536-token answer cap, 12288-token context, and a 1024px vision preview. The preview is a separate tensor used only by the prompt enhancer. The explicit answer cap stops a runaway completion from filling the context and failing with `decode: failed to find a memory slot for batch of size 1`.

ComfyUI starts with `--fast fp16_accumulation` by default. Set `COMFY_PERFORMANCE_ARGS=""` to disable it without rebuilding the image, or override the variable with another supported ComfyUI performance flag set.

The handler already uses direct writes to `/comfyui/input`, websocket image output, in-memory result handling and direct S3 upload; the slower localhost multipart/history/view/temp-file path is retained only as a compatibility fallback.

Worker input normalization defaults to `COMFY_INPUT_MAX_DIMENSION=768` and `COMFY_INPUT_JPEG_QUALITY=92`. Set `COMFY_INPUT_MAX_DIMENSION=0` to disable the worker-side fallback.

### CUDA 12.8+ runtime

Everything in the image targets CUDA 12.8:

- Base image `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
- PyTorch cu128 (`comfy-cli --cuda-version 12.8`)
- JamePeng `llama-cpp-python 0.3.49+cu128` wheel, built with `sm_120` (Blackwell) kernels

It runs on any host driver that supports CUDA 12.8 or newer (driver 570+), including Blackwell GPUs such as RTX PRO 6000 and its MIG slices. RunPod endpoint setting: allow CUDA **12.8, 12.9 and 13.0**.

If every host you use supports CUDA 13 (driver 580+), you can build with `CUDA_VERSION_FOR_COMFY=13.0` to get ComfyUI's cu130-only optimized `comfy_kitchen` kernels. That build no longer runs on CUDA 12.x hosts.

`CUDA_MODULE_LOADING=LAZY` is enabled to reduce serverless startup work, and the default ComfyUI log level is `INFO`.

### Runtime

- ComfyUI 0.37.0
- `leejet/ComfyUI-GGUF`
- `xiaowuapple-pixel/ComfyUI-Prompt-Enhancer`
- Local nodes: `PEPreviewResize`, `QwenPadToMultiple`, `QwenCropToReference`
- JamePeng prebuilt CUDA 12.8 / Python 3.12 `llama-cpp-python` wheel; no llama.cpp source compilation during the RunPod build

Rebuild and deploy the worker image before sending the updated RadiantLunar workflow: older images lack the pad/crop nodes. The RunPod input/output contract is unchanged.

### GPU target

NVIDIA Ampere, Ada and Blackwell GPUs with 24 GB or more, e.g. RTX 4090 and RTX PRO 6000 (including 24 GB MIG slices).

## Table of Contents

- [Quickstart](#quickstart)
- [Available Docker Images](#available-docker-images)
- [API Specification](#api-specification)
- [Usage](#usage)
- [Getting the Workflow JSON](#getting-the-workflow-json)
- [Further Documentation](#further-documentation)

---

## Quickstart

1.  🐳 Choose one of the [available Docker images](#available-docker-images) for your serverless endpoint (e.g., `runpod/worker-comfyui:<version>-sd3`).
2.  📄 Follow the [Deployment Guide](docs/deployment.md) to set up your RunPod template and endpoint.
3.  ⚙️ Optionally configure the worker (e.g., for S3 upload) using environment variables - see the full [Configuration Guide](docs/configuration.md).
4.  🧪 Pick an example workflow from [`test_resources/workflows/`](./test_resources/workflows/) or [get your own](#getting-the-workflow-json).
5.  🚀 Follow the [Usage](#usage) steps below to interact with your deployed endpoint.

## Available Docker Images

These images are available on Docker Hub under `runpod/worker-comfyui`:

- **`runpod/worker-comfyui:<version>-base`**: Clean ComfyUI install with no models.
- **`runpod/worker-comfyui:<version>-flux1-schnell`**: Includes checkpoint, text encoders, and VAE for [FLUX.1 schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- **`runpod/worker-comfyui:<version>-flux1-dev`**: Includes checkpoint, text encoders, and VAE for [FLUX.1 dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- **`runpod/worker-comfyui:<version>-sdxl`**: Includes checkpoint and VAEs for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- **`runpod/worker-comfyui:<version>-sd3`**: Includes checkpoint for [Stable Diffusion 3 medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium).

Replace `<version>` with the current release tag, check the [releases page](https://github.com/runpod-workers/worker-comfyui/releases) for the latest version.

## API Specification

The worker exposes standard RunPod serverless endpoints (`/run`, `/runsync`, `/health`). By default, images are returned as base64 strings. You can configure the worker to upload images to an S3 bucket instead by setting specific environment variables (see [Configuration Guide](docs/configuration.md)).

Use the `/runsync` endpoint for synchronous requests that wait for the job to complete and return the result directly. Use the `/run` endpoint for asynchronous requests that return immediately with a job ID; you'll need to poll the `/status` endpoint separately to get the result.

### Input

```json
{
  "input": {
    "workflow": {
      "6": {
        "inputs": {
          "text": "a ball on the table",
          "clip": ["30", 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
          "title": "CLIP Text Encode (Positive Prompt)"
        }
      }
    },
    "images": [
      {
        "name": "input_image_1.png",
        "image": "data:image/png;base64,iVBOR..."
      }
    ]
  }
}
```

The following tables describe the fields within the `input` object:

| Field Path                | Type   | Required | Description                                                                                                                                |
| ------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `input`                   | Object | Yes      | Top-level object containing request data.                                                                                                  |
| `input.workflow`          | Object | Yes      | The ComfyUI workflow exported in the [required format](#getting-the-workflow-json).                                                        |
| `input.images`            | Array  | No       | Optional array of input images. Each image is uploaded to ComfyUI's `input` directory and can be referenced by its `name` in the workflow. |
| `input.comfy_org_api_key` | String | No       | Optional per-request Comfy.org API key for API Nodes. Overrides the `COMFY_ORG_API_KEY` environment variable if both are set.              |

#### `input.images` Object

Each object within the `input.images` array must contain:

| Field Name | Type   | Required | Description                                                                                                                       |
| ---------- | ------ | -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `name`     | String | Yes      | Filename used to reference the image in the workflow (e.g., via a "Load Image" node). Must be unique within the array.            |
| `image`    | String | Yes      | Base64 encoded string of the image. A data URI prefix (e.g., `data:image/png;base64,`) is optional and will be handled correctly. |

> [!NOTE]
>
> **Size Limits:** RunPod endpoints have request size limits (e.g., 10MB for `/run`, 20MB for `/runsync`). Large base64 input images can exceed these limits. See [RunPod Docs](https://docs.runpod.io/docs/serverless-endpoint-urls).

### Output

> [!WARNING]
>
> **Breaking Change in Output Format (5.0.0+)**
>
> Versions `< 5.0.0` returned the primary image data (S3 URL or base64 string) directly within an `output.message` field.
> Starting with `5.0.0`, the output format has changed significantly, see below

```json
{
  "id": "sync-uuid-string",
  "status": "COMPLETED",
  "output": {
    "images": [
      {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "base64",
        "data": "iVBORw0KGgoAAAANSUhEUg..."
      }
    ]
  },
  "delayTime": 123,
  "executionTime": 4567
}
```

### Output Scenarios

#### 1. Default (Base64 Encoded)
When S3 is NOT configured and `COMFY_SKIP_BASE64` is `false` (default).

```json
{
  "output": {
    "images": [
      {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "base64",
        "data": "iVBORw0KGgoAAAANSUhEUg..."
      }
    ]
  }
}
```

#### 2. S3 Upload
When S3 is configured (via `BUCKET_ENDPOINT_URL` etc.).

```json
{
  "output": {
    "images": [
      {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "s3_url",
        "data": "https://my-bucket.s3.amazonaws.com/job-id/ComfyUI_00001_.png"
      }
    ]
  }
}
```

#### 3. Local File Only (Skip Base64)
When `COMFY_SKIP_BASE64=true` and S3 is NOT configured. Useful for network volumes.

```json
{
  "output": {
    "images": [
      {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "type": "local_file"
      }
    ]
  }
}
```

| Field Path      | Type             | Required | Description                                                                                                 |
| --------------- | ---------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `output`        | Object           | Yes      | Top-level object containing the results of the job execution.                                               |
| `output.images` | Array of Objects | No       | Present if the workflow generated images. Contains a list of objects, each representing one output image.   |
| `output.errors` | Array of Strings | No       | Present if non-fatal errors or warnings occurred during processing (e.g., S3 upload failure, missing data). |

#### `output.images`

Each object in the `output.images` array has the following structure:

| Field Name  | Type   | Description                                                                                              |
| ----------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `filename`  | String | The original filename assigned by ComfyUI during generation.                                             |
| `subfolder` | String | The subfolder where the image is stored.                                                                 |
| `type`      | String | Indicates the format of the data. `base64`, `s3_url`, or `local_file` (if `COMFY_SKIP_BASE64` is set).   |
| `data`      | String | Contains either the base64 encoded image string or the S3 URL. Omitted if `type` is `local_file`.        |

> [!NOTE]
> The `output.images` field provides a list of all generated images (excluding temporary ones).
>
> - If S3 upload is **not** configured (default), `type` will be `"base64"` and `data` will contain the base64 encoded image string.
> - If S3 upload **is** configured, `type` will be `"s3_url"` and `data` will contain the S3 URL. See the [Configuration Guide](docs/configuration.md#example-s3-response) for an S3 example response.
> - Clients interacting with the API need to handle this list-based structure under `output.images`.

## Usage

To interact with your deployed RunPod endpoint:

1.  **Get API Key:** Generate a key in RunPod [User Settings](https://www.runpod.io/console/serverless/user/settings) (`API Keys` section).
2.  **Get Endpoint ID:** Find your endpoint ID on the [Serverless Endpoints](https://www.runpod.io/console/serverless/user/endpoints) page or on the `Overview` page of your endpoint.

### Generate Image (Sync Example)

Send a workflow to the `/runsync` endpoint (waits for completion). Replace `<api_key>` and `<endpoint_id>`. The `-d` value should contain the [JSON input described above](#input).

```bash
curl -X POST \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"input":{"workflow":{... your workflow JSON ...}}}' \
  https://api.runpod.ai/v2/<endpoint_id>/runsync
```

You can also use the `/run` endpoint for asynchronous jobs and then poll the `/status` to see when the job is done. Or you [add a `webhook` into your request](https://docs.runpod.io/serverless/endpoints/send-requests#webhook-notifications) to be notified when the job is done.

Refer to [`test_input.json`](./test_input.json) for a complete input example.

## Getting the Workflow JSON

To get the correct `workflow` JSON for the API:

1.  Open ComfyUI in your browser.
2.  In the top navigation, select `Workflow > Export (API)`
3.  A `workflow.json` file will be downloaded. Use the content of this file as the value for the `input.workflow` field in your API requests.

## Further Documentation

- **[Deployment Guide](docs/deployment.md):** Detailed steps for deploying on RunPod.
- **[Configuration Guide](docs/configuration.md):** Full list of environment variables (including S3 setup).
- **[Customization Guide](docs/customization.md):** Adding custom models and nodes (Network Volumes, Docker builds).
- **[Development Guide](docs/development.md):** Setting up a local environment for development & testing
- **[CI/CD Guide](docs/ci-cd.md):** Information about the automated Docker build and publish workflows.
- **[Acknowledgments](docs/acknowledgments.md):** Credits and thanks
