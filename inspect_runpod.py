import inspect
try:
    from runpod.serverless.utils import rp_upload
    print("Found rp_upload")
    
    if hasattr(rp_upload, 'upload_image'):
        print("upload_image signature:", inspect.signature(rp_upload.upload_image))
        print("upload_image docstring:", rp_upload.upload_image.__doc__)
    else:
        print("upload_image not found in rp_upload")

    if hasattr(rp_upload, 'upload_file_to_bucket'):
        print("upload_file_to_bucket signature:", inspect.signature(rp_upload.upload_file_to_bucket))
    else:
        print("upload_file_to_bucket not found in rp_upload")

except ImportError as e:
    print(f"Could not import runpod: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
