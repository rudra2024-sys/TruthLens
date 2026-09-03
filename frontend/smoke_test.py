import json
import os
import urllib.request
import urllib.parse
from pathlib import Path
from io import BytesIO

base = 'http://localhost:8000/api/v1'

# quick health check
with urllib.request.urlopen(f'{base}/health') as resp:
    print('health_status', resp.status)
    print(resp.read().decode())

img_path = Path('frontend/public/test-image.png')
if not img_path.exists():
    from PIL import Image
    Image.new('RGB', (200, 200), color=(10, 20, 30)).save(img_path)

video_path = Path('frontend/public/test-video.mp4')
if not video_path.exists():
    video_path.write_bytes(b'fake mp4 bytes for smoke test')

audio_path = Path('frontend/public/test-audio.wav')
if not audio_path.exists():
    audio_path.write_bytes(b'fake wav bytes for smoke test')

for name, path in [('image', img_path), ('video', video_path), ('audio', audio_path)]:
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    data = []
    with open(path, 'rb') as f:
        content = f.read()
    body = []
    body.append(f'--{boundary}\r\n'.encode())
    body.append(f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode())
    body.append(b'Content-Type: application/octet-stream\r\n\r\n')
    body.append(content)
    body.append(f'\r\n--{boundary}--\r\n'.encode())
    payload = b''.join(body)
    req = urllib.request.Request(f'{base}/upload/', method='POST', data=payload)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(f'[{name}] upload_status={resp.status}')
        print(resp.read().decode())
    # detect
    upload_data = json.loads(resp.read().decode())
