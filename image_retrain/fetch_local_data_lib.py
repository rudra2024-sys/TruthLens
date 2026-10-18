import io, os, time, threading, urllib.request, zipfile
from PIL import Image

URLS = {}

class R(io.RawIOBase):
    def __init__(s, url):
        s.url = url; h = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})).headers
        s.size = int(h["Content-Range"].split("/")[1]); s.pos = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o; return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        for a in range(4):
            try:
                d = urllib.request.urlopen(urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{s.pos + n - 1}"}), timeout=60).read(); break
            except Exception:
                time.sleep(1 + a)
        else:
            raise IOError("range read failed")
        b[:len(d)] = d; s.pos += len(d); return len(d)


_tls = threading.local()


def zf(ds):
    z = getattr(_tls, "z", None) or {}
    if ds not in z:
        z[ds] = zipfile.ZipFile(io.BufferedReader(R(URLS[ds]), 128 * 1024)); _tls.z = z
    return z[ds]


def save(ds, member, dest, force_jpeg=False):
    data = zf(ds).read(member)
    ext = member.rsplit(".", 1)[-1].lower()
    im = Image.open(io.BytesIO(data)); im.load()
    big = max(im.size) > 1024
    if ext in ("jpg", "jpeg") and not big and not force_jpeg:
        open(dest, "wb").write(data); return
    im = im.convert("RGB")
    if big:
        s = 1024 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dest, "JPEG", quality=92)


