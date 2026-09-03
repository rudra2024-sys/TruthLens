import requests
from pathlib import Path

base = 'http://localhost:8000/api/v1'

payloads = [
    ('image', Path('frontend/public/test-image.png'), 'image/png'),
    ('video', Path('frontend/public/test-video.mp4'), 'video/mp4'),
    ('audio', Path('frontend/public/test-audio.wav'), 'audio/wav'),
]

for name, path, mime in payloads:
    path.write_bytes(path.read_bytes() if path.exists() else b'placeholder')
    with open(path, 'rb') as f:
        upload_resp = requests.post(
            f'{base}/upload/',
            files={'file': (path.name, f, mime)},
            timeout=60,
        )
    print(f'[{name}] upload_status={upload_resp.status_code}')
    print(upload_resp.text)
    if upload_resp.ok:
        upload_id = upload_resp.json().get('upload_id')
        detect_resp = requests.post(f'{base}/detect/{upload_id}', timeout=60)
        print(f'[{name}] detect_status={detect_resp.status_code}')
        print(detect_resp.text)
    print('---')
