import base64
import copy
import json
import os
from pathlib import Path
import socket
import struct
import time
import traceback
import uuid
from io import BytesIO

import requests
import runpod
from runpod.serverless.utils import rp_upload
import websocket


COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input"))
COMFY_API_AVAILABLE_INTERVAL_MS = int(
    os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", "50")
)
COMFY_API_AVAILABLE_MAX_RETRIES = int(
    os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", "500")
)
WEBSOCKET_RECONNECT_ATTEMPTS = int(
    os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", "5")
)
WEBSOCKET_RECONNECT_DELAY_S = float(
    os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", "3")
)
COMFY_DIRECT_INPUT_WRITE = (
    os.environ.get("COMFY_DIRECT_INPUT_WRITE", "true").lower() == "true"
)
COMFY_WEBSOCKET_OUTPUT = (
    os.environ.get("COMFY_WEBSOCKET_OUTPUT", "true").lower() == "true"
)
COMFY_SKIP_BASE64 = os.environ.get("COMFY_SKIP_BASE64", "false").lower() == "true"
WORKER_VERBOSE = os.environ.get("WORKER_VERBOSE", "false").lower() == "true"

# ComfyUI binary websocket event type for an encoded preview image.
# SaveImageWebsocket uses this exact event path and sends an 8-byte header:
#   4 bytes event type + 4 bytes image format + encoded image bytes.
COMFY_BINARY_PREVIEW_IMAGE = 1

if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)


def _debug(message):
    if WORKER_VERBOSE:
        print(f"worker-comfyui - {message}")


def _bucket_configured():
    return bool(
        os.environ.get("BUCKET_NAME")
        and os.environ.get("BUCKET_ENDPOINT_URL")
        and os.environ.get("BUCKET_ACCESS_KEY_ID")
        and os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    )


def _comfy_server_status():
    try:
        response = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": response.status_code == 200,
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    print(
        "worker-comfyui - Websocket disconnected unexpectedly; "
        f"attempting reconnect: {initial_error}"
    )
    last_error = initial_error

    for attempt in range(max_attempts):
        status = _comfy_server_status()
        if not status["reachable"]:
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP endpoint became unavailable during websocket reconnect"
            )

        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print("worker-comfyui - Websocket reconnected")
            return new_ws
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(delay_s)

    raise websocket.WebSocketConnectionClosedException(
        f"Failed to reconnect websocket: {last_error}"
    )


def validate_input(job_input):
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    if not isinstance(job_input, dict):
        return None, "Input must be an object"

    workflow = job_input.get("workflow")
    if not isinstance(workflow, dict):
        return None, "Missing or invalid 'workflow' parameter"

    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            isinstance(image, dict) and "name" in image and "image" in image
            for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    return {
        "workflow": workflow,
        "images": images,
        "comfy_org_api_key": job_input.get("comfy_org_api_key"),
    }, None


def check_server(url, retries=None, delay=None):
    retries = retries or COMFY_API_AVAILABLE_MAX_RETRIES
    delay = delay or COMFY_API_AVAILABLE_INTERVAL_MS

    for _ in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay / 1000)

    return False


def _decode_image_data_uri(image_data_uri):
    if not isinstance(image_data_uri, str):
        raise ValueError("Image data must be a base64 string")

    base64_data = image_data_uri.split(",", 1)[1] if "," in image_data_uri else image_data_uri
    return base64.b64decode(base64_data)


def _safe_input_name(name):
    safe_name = Path(str(name)).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("Invalid input image filename")
    return safe_name


def write_input_images(images):
    """Write request images directly into ComfyUI's input directory.

    The handler and ComfyUI run in the same container, so this avoids an
    unnecessary localhost multipart HTTP upload and its extra copy/parse work.
    """
    if not images:
        return {"status": "success", "details": []}

    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    errors = []

    for image in images:
        try:
            name = _safe_input_name(image["name"])
            blob = _decode_image_data_uri(image["image"])
            destination = COMFY_INPUT_DIR / name
            temp_path = COMFY_INPUT_DIR / f".{name}.{uuid.uuid4().hex}.tmp"
            temp_path.write_bytes(blob)
            os.replace(temp_path, destination)
            written.append(name)
            _debug(f"Wrote input image directly: {destination}")
        except Exception as exc:
            errors.append(f"{image.get('name', 'unknown')}: {exc}")

    if errors:
        return {"status": "error", "details": errors}
    return {"status": "success", "details": written}


def upload_images_http(images):
    """Compatibility fallback for environments that disable direct input writes."""
    if not images:
        return {"status": "success", "details": []}

    uploaded = []
    errors = []
    for image in images:
        try:
            name = _safe_input_name(image["name"])
            blob = _decode_image_data_uri(image["image"])
            files = {
                "image": (name, BytesIO(blob), "application/octet-stream"),
                "overwrite": (None, "true"),
            }
            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()
            uploaded.append(name)
        except Exception as exc:
            errors.append(f"{image.get('name', 'unknown')}: {exc}")

    if errors:
        return {"status": "error", "details": errors}
    return {"status": "success", "details": uploaded}


def _sanitize_output_prefix(prefix):
    prefix = str(prefix or "ComfyUI").replace("\\", "/").split("/")[-1]
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in prefix)
    return cleaned or "ComfyUI"


def prepare_workflow(workflow):
    """Convert SaveImage output nodes to zero-disk SaveImageWebsocket nodes.

    This is enabled by default for API jobs. Set COMFY_WEBSOCKET_OUTPUT=false to
    preserve the workflow exactly as submitted.
    """
    prepared = copy.deepcopy(workflow)
    output_prefix = "ComfyUI"
    converted = 0

    if not COMFY_WEBSOCKET_OUTPUT:
        return prepared, output_prefix, converted

    # If the caller explicitly requested local-file-only results and there is no
    # bucket, retain SaveImage so the legacy history path can still reference it.
    if COMFY_SKIP_BASE64 and not _bucket_configured():
        return prepared, output_prefix, converted

    for node in prepared.values():
        if not isinstance(node, dict):
            continue

        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "SaveImage":
            if converted == 0:
                output_prefix = _sanitize_output_prefix(
                    inputs.get("filename_prefix", "ComfyUI")
                )
            image_input = inputs.get("images")
            if image_input is None:
                continue
            node["class_type"] = "SaveImageWebsocket"
            node["inputs"] = {"images": image_input}
            node.setdefault("_meta", {})["title"] = "Save Image (Websocket)"
            converted += 1
        elif class_type == "SaveImageWebsocket" and converted == 0:
            converted += 1

    return prepared, output_prefix, converted


def get_available_models():
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            options = (
                object_info["CheckpointLoaderSimple"]
                .get("input", {})
                .get("required", {})
                .get("ckpt_name")
            )
            if options:
                available_models["checkpoints"] = (
                    options[0] if isinstance(options[0], list) else []
                )
        return available_models
    except Exception:
        return {}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    payload = {"prompt": workflow, "client_id": client_id}
    effective_key = comfy_org_api_key or os.environ.get("COMFY_ORG_API_KEY")
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}

    response = requests.post(
        f"http://{COMFY_HOST}/prompt",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if response.status_code == 400:
        try:
            error_data = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Workflow validation failed: {response.text}")

        error_message = "Workflow validation failed"
        error_details = []

        error_info = error_data.get("error")
        if isinstance(error_info, dict):
            error_message = error_info.get("message", error_message)
        elif error_info:
            error_message = str(error_info)

        for node_id, node_error in error_data.get("node_errors", {}).items():
            error_details.append(f"Node {node_id}: {node_error}")

        if error_details:
            raise ValueError(error_message + "\n" + "\n".join(error_details))

        models = get_available_models()
        if models.get("checkpoints"):
            error_message += (
                "\nAvailable checkpoints: " + ", ".join(models["checkpoints"])
            )
        raise ValueError(error_message)

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_output_bytes(filename, subfolder, output_type):
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": output_type,
    }
    response = requests.get(f"http://{COMFY_HOST}/view", params=params, timeout=60)
    response.raise_for_status()
    return response.content


def _detect_image_extension(image_bytes):
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp", "image/webp"
    return ".png", "image/png"


def parse_websocket_image_frame(frame):
    if not isinstance(frame, (bytes, bytearray)) or len(frame) <= 8:
        return None

    event_type = struct.unpack(">I", frame[:4])[0]
    if event_type != COMFY_BINARY_PREVIEW_IMAGE:
        return None

    # Bytes 4:8 contain ComfyUI's image-format enum. Detecting from magic bytes
    # keeps the handler forward-compatible with format-enum changes.
    image_bytes = bytes(frame[8:])
    extension, content_type = _detect_image_extension(image_bytes)
    return image_bytes, extension, content_type


def upload_bytes_to_s3(job_id, filename, data):
    bucket_name = os.environ.get("BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("BUCKET_NAME is required for S3 upload")

    bucket_creds = {
        "endpointUrl": os.environ.get("BUCKET_ENDPOINT_URL"),
        "accessId": os.environ.get("BUCKET_ACCESS_KEY_ID"),
        "accessSecret": os.environ.get("BUCKET_SECRET_ACCESS_KEY"),
    }

    return rp_upload.upload_in_memory_object(
        filename,
        data,
        bucket_creds,
        bucket_name=bucket_name,
        prefix=job_id,
    )


def _append_output_from_bytes(output_data, job_id, filename, data, subfolder=""):
    if _bucket_configured():
        url = upload_bytes_to_s3(job_id, filename, data)
        output_data.append(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": "s3_url",
                "data": url,
            }
        )
        return

    if COMFY_SKIP_BASE64:
        output_data.append(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": "local_file_unavailable",
            }
        )
        return

    output_data.append(
        {
            "filename": filename,
            "subfolder": subfolder,
            "type": "base64",
            "data": base64.b64encode(data).decode("utf-8"),
        }
    )


def _collect_history_outputs(prompt_id, job_id, output_data, errors):
    """Legacy fallback for workflows not using SaveImageWebsocket."""
    history = get_history(prompt_id)
    prompt_history = history.get(prompt_id)
    if not prompt_history:
        errors.append(f"Prompt ID {prompt_id} not found in history")
        return

    outputs = prompt_history.get("outputs", {})
    for node_id, node_output in outputs.items():
        for output_key in ("images", "gifs"):
            for item in node_output.get(output_key, []):
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                output_type = item.get("type")

                if not filename or output_type == "temp":
                    continue

                try:
                    data = get_output_bytes(filename, subfolder, output_type)
                    _append_output_from_bytes(
                        output_data, job_id, filename, data, subfolder=subfolder
                    )
                except Exception as exc:
                    errors.append(
                        f"Failed to process output from node {node_id} ({filename}): {exc}"
                    )


def handler(job):
    job_input = job.get("input")
    job_id = str(job.get("id") or uuid.uuid4())

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    if not check_server(f"http://{COMFY_HOST}/"):
        return {"error": f"ComfyUI server ({COMFY_HOST}) is not reachable"}

    input_images = validated_data.get("images")
    if input_images:
        if COMFY_DIRECT_INPUT_WRITE:
            input_result = write_input_images(input_images)
        else:
            input_result = upload_images_http(input_images)
        if input_result["status"] == "error":
            return {
                "error": "Failed to prepare one or more input images",
                "details": input_result["details"],
            }

    workflow, output_prefix, websocket_output_nodes = prepare_workflow(
        validated_data["workflow"]
    )
    _debug(
        f"Prepared workflow; websocket output nodes={websocket_output_nodes}, "
        f"prefix={output_prefix}"
    )

    client_id = str(uuid.uuid4())
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    ws = None
    prompt_id = None
    websocket_images = []
    output_data = []
    errors = []

    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)

        queued = queue_workflow(
            workflow,
            client_id,
            comfy_org_api_key=validated_data.get("comfy_org_api_key"),
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise ValueError(f"Missing prompt_id in queue response: {queued}")

        print(f"worker-comfyui - Executing prompt {prompt_id}")
        execution_done = False

        while True:
            try:
                message = ws.recv()

                if isinstance(message, (bytes, bytearray)):
                    parsed = parse_websocket_image_frame(message)
                    if parsed is not None:
                        websocket_images.append(parsed)
                    continue

                payload = json.loads(message)
                message_type = payload.get("type")

                if message_type == "executing":
                    data = payload.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        execution_done = True
                        break
                elif message_type == "execution_error":
                    data = payload.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        errors.append(
                            "Workflow execution error: "
                            f"node={data.get('node_id')} "
                            f"type={data.get('node_type')} "
                            f"message={data.get('exception_message')}"
                        )
                        break
                elif message_type == "status":
                    _debug(
                        "Queue remaining: "
                        + str(
                            payload.get("data", {})
                            .get("status", {})
                            .get("exec_info", {})
                            .get("queue_remaining", "N/A")
                        )
                    )

            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException as exc:
                ws = _attempt_websocket_reconnect(
                    ws_url,
                    WEBSOCKET_RECONNECT_ATTEMPTS,
                    WEBSOCKET_RECONNECT_DELAY_S,
                    exc,
                )
            except json.JSONDecodeError:
                _debug("Ignored non-JSON websocket text message")

        if not execution_done and not errors:
            raise ValueError("Execution ended without completion or error event")

        if websocket_images:
            for index, (data, extension, _content_type) in enumerate(
                websocket_images, start=1
            ):
                filename = f"{output_prefix}_{index:05d}{extension}"
                _append_output_from_bytes(output_data, job_id, filename, data)
        elif prompt_id:
            # Compatibility path for workflows that did not emit websocket images.
            _collect_history_outputs(prompt_id, job_id, output_data, errors)

    except websocket.WebSocketException as exc:
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {exc}"}
    except requests.RequestException as exc:
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {exc}"}
    except ValueError as exc:
        print(traceback.format_exc())
        return {"error": str(exc)}
    except Exception as exc:
        print(traceback.format_exc())
        return {"error": f"Unexpected handler error: {exc}"}
    finally:
        if ws and ws.connected:
            ws.close()

    result = {}
    if output_data:
        result["images"] = output_data
    if errors:
        result["errors"] = errors

    if not output_data and errors:
        return {"error": "Job processing failed", "details": errors}
    if not output_data:
        result["status"] = "success_no_images"
        result["images"] = []

    print(f"worker-comfyui - Job complete; returned {len(output_data)} output(s)")
    return result


if __name__ == "__main__":
    print("worker-comfyui - Starting optimized handler")
    runpod.serverless.start({"handler": handler})
