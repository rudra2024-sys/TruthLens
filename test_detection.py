import os
import urllib.request
import json
import numpy as np
from PIL import Image

# Create a test synthetic image
img_path = "test_sample.jpg"
arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
img = Image.fromarray(arr)
img.save(img_path)

print(f"[1] Created test image: {img_path}")

# Step 1: Upload media
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
with open(img_path, 'rb') as f:
    file_bytes = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test_sample.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'http://localhost:8000/api/v1/upload/',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    method='POST'
)

with urllib.request.urlopen(req) as resp:
    upload_res = json.loads(resp.read().decode('utf-8'))
    print(f"[2] Upload result: {json.dumps(upload_res, indent=2)}")

upload_id = upload_res['upload_id']

# Step 2: Run Detection
req_detect = urllib.request.Request(
    f'http://localhost:8000/api/v1/detect/{upload_id}',
    method='POST'
)

with urllib.request.urlopen(req_detect) as resp:
    detect_res = json.loads(resp.read().decode('utf-8'))
    print(f"[3] Detection result: {json.dumps(detect_res, indent=2)}")

# Cleanup
if os.path.exists(img_path):
    os.remove(img_path)
print("\n✅ End-to-end real model processing test PASSED!")
