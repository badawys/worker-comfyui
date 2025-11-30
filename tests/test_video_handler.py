import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import base64

# Add parent directory to path to import handler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all dependencies before importing handler
sys.modules['runpod'] = MagicMock()
sys.modules['runpod.serverless'] = MagicMock()
sys.modules['runpod.serverless.utils'] = MagicMock()
sys.modules['runpod.serverless.utils.rp_upload'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['websocket'] = MagicMock()

import handler

class TestVideoHandler(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        sys.modules['requests'].reset_mock()
        sys.modules['websocket'].reset_mock()

    def test_handler_video_output(self):
        # Setup mocks
        mock_uuid = MagicMock()
        handler.uuid = mock_uuid
        mock_uuid.uuid4.return_value = "test-uuid"

        mock_requests = sys.modules['requests']
        mock_websocket = sys.modules['websocket']

        # Mock ComfyUI server check
        mock_requests.get.return_value.status_code = 200

        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_websocket.WebSocket.return_value = mock_ws
        
        # Mock WebSocket messages
        # 1. Status
        # 2. Executing (finished)
        mock_ws.recv.side_effect = [
            json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "test-prompt-id"}})
        ]

        # Mock queue_workflow response
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"prompt_id": "test-prompt-id"}

        # Mock history response with 'gifs'
        mock_history_response = {
            "test-prompt-id": {
                "outputs": {
                    "node-1": {
                        "gifs": [
                            {
                                "filename": "test_video.mp4",
                                "subfolder": "",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        
        # Mock get_history
        def side_effect_get(url, timeout=None):
            if "/history/" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_history_response
                return mock_resp
            elif "/view" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b"fake-video-content"
                return mock_resp
            else:
                # Default for server check
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                return mock_resp

        mock_requests.get.side_effect = side_effect_get

        # Input job
        job = {
            "id": "test-job-id",
            "input": {
                "workflow": {"node": "data"}
            }
        }

        # Run handler
        result = handler.handler(job)

        # Verify results
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["filename"], "test_video.mp4")
        self.assertEqual(result["images"][0]["type"], "base64")
        self.assertEqual(result["images"][0]["data"], base64.b64encode(b"fake-video-content").decode("utf-8"))

    @patch.dict(os.environ, {
        "BUCKET_ENDPOINT_URL": "https://test-bucket-url",
        "BUCKET_ACCESS_KEY_ID": "test-key",
        "BUCKET_SECRET_ACCESS_KEY": "test-secret",
        "BUCKET_NAME": "test-bucket"
    })
    @patch('handler.rp_upload')
    def test_handler_video_output_with_bucket(self, mock_rp_upload):
        # Setup mocks
        mock_uuid = MagicMock()
        handler.uuid = mock_uuid
        mock_uuid.uuid4.return_value = "test-uuid"

        mock_requests = sys.modules['requests']
        mock_websocket = sys.modules['websocket']
        
        # Mock ComfyUI server check
        mock_requests.get.return_value.status_code = 200

        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_websocket.WebSocket.return_value = mock_ws
        
        # Mock WebSocket messages
        mock_ws.recv.side_effect = [
            json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "test-prompt-id"}})
        ]

        # Mock queue_workflow response
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"prompt_id": "test-prompt-id"}

        # Mock history response with 'gifs'
        mock_history_response = {
            "test-prompt-id": {
                "outputs": {
                    "node-1": {
                        "gifs": [
                            {
                                "filename": "test_video.mp4",
                                "subfolder": "",
                                "type": "output"
                            }
                        ]
                    }
                }
            }
        }
        
        # Mock get_history
        def side_effect_get(url, timeout=None):
            if "/history/" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_history_response
                return mock_resp
            elif "/view" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b"fake-video-content"
                return mock_resp
            else:
                # Default for server check
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                return mock_resp

        mock_requests.get.side_effect = side_effect_get

        # Mock upload_file_to_bucket
        mock_rp_upload.upload_file_to_bucket.return_value = "https://test-bucket-url/test-bucket/test-job-id/test_video.mp4"

        # Input job
        job = {
            "id": "test-job-id",
            "input": {
                "workflow": {"node": "data"}
            }
        }

        # Run handler
        result = handler.handler(job)

        # Verify results
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["filename"], "test_video.mp4")
        self.assertEqual(result["images"][0]["type"], "s3_url")
        self.assertEqual(result["images"][0]["data"], "https://test-bucket-url/test-bucket/test-job-id/test_video.mp4")

        # Verify upload_file_to_bucket was called
        mock_rp_upload.upload_file_to_bucket.assert_called_once()
        args, kwargs = mock_rp_upload.upload_file_to_bucket.call_args
        self.assertEqual(args[0], "test_video.mp4") # filename
        # args[1] is temp file path, skip
        self.assertEqual(args[2], {
            "endpointUrl": "https://test-bucket-url",
            "accessId": "test-key",
            "accessSecret": "test-secret",
        }) # bucket_creds
        self.assertEqual(kwargs["bucket_name"], "test-bucket")
        self.assertEqual(kwargs["prefix"], "test-job-id")

if __name__ == '__main__':
    unittest.main()
