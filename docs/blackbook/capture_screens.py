"""Capture real screenshots of the running TruthLens app for the project report.

Assumes the backend is on http://127.0.0.1:8000 and the frontend on http://localhost:3000.
Creates a throwaway local test account (credentials saved to work/screenshot_account.json,
git-ignored), runs real scans through the API, then screenshots the live pages with Playwright.
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

import pymupdf as fitz
import requests
from playwright.sync_api import sync_playwright

API = "http://127.0.0.1:8000/api/v1"
WEB = "http://localhost:3000"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "screens"
WORK = HERE / "work"
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

SAMPLES = {
    "photo": ROOT / "test_sample.jpg",
    "chatgpt": ROOT / "backend/test_data/blind_spot_images/chatgpt_everyday/image1.png",
    "portrait": ROOT / "backend/test_data/blind_spot_images/portrait_app/part-000002__1f61c200-aa30-4c96-b249-b0db419dd8b7.png",
    "video": Path(r"D:\eval_data\kagglehub\datasets\stplusplus\rtfs-10k\versions\2\inswapper\inswapper_qZoWizbMI5w_00030_00055_3-568_man.mp4"),
    "audio": Path(r"C:\Users\admin\Downloads\amberaudio.mp3"),
}


def account() -> dict:
    f = WORK / "screenshot_account.json"
    if f.exists():
        creds = json.loads(f.read_text())
        r = requests.post(f"{API}/auth/login", json={"email": creds["email"], "password": creds["password"]})
        r.raise_for_status()
        return r.json()
    creds = {"name": "Report Demo", "email": f"report.demo.{secrets.token_hex(3)}@example.com",
             "password": secrets.token_urlsafe(12)}
    r = requests.post(f"{API}/auth/signup", json={**creds, "account_type": "student"})
    r.raise_for_status()
    f.write_text(json.dumps(creds))
    return r.json()


def scan(token: str, key: str) -> dict:
    cache = WORK / f"scan_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    h = {"Authorization": f"Bearer {token}"}
    path = SAMPLES[key]
    with open(path, "rb") as fh:
        mime = {".jpg": "image/jpeg", ".png": "image/png", ".mp4": "video/mp4", ".mp3": "audio/mpeg"}[path.suffix.lower()]
        r = requests.post(f"{API}/upload/", headers=h, files={"file": (path.name, fh, mime)})
    r.raise_for_status()
    up = r.json()
    r = requests.post(f"{API}/detect/{up['upload_id']}", headers=h, timeout=900)
    r.raise_for_status()
    res = {"upload": up, "result": r.json()}
    cache.write_text(json.dumps(res, indent=1))
    print(key, res["result"].get("verdict"), res["result"].get("confidence_score"))
    return res


def pdf_pages(token: str, upload_id: str, name: str, pages: int = 2) -> None:
    r = requests.get(f"{API}/report/{upload_id}", headers={"Authorization": f"Bearer {token}"}, timeout=600)
    r.raise_for_status()
    pdf = WORK / f"{name}.pdf"
    pdf.write_bytes(r.content)
    doc = fitz.open(pdf)
    for i in range(min(pages, doc.page_count)):
        doc[i].get_pixmap(dpi=110).save(OUT / f"{name}_p{i + 1}.png")


def main() -> None:
    auth = account()
    token = auth["access_token"]
    scans = {k: scan(token, k) for k in (sys.argv[1:] or SAMPLES)}
    user = {k: auth[k] for k in ("user_id", "name", "email", "role")}

    for k in ("chatgpt", "video"):
        if k in scans:
            pdf_pages(token, scans[k]["upload"]["upload_id"], f"pdf_{k}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()

        def shot(url: str, name: str, full: bool = False, wait: float = 2.5, setup=None):
            page.goto(WEB + url, wait_until="networkidle")
            time.sleep(wait)
            if setup:
                setup()
                time.sleep(wait)
            if full:  # fixed nav/background would otherwise repeat mid-page in a full-page capture
                page.evaluate("""() => { for (const el of document.querySelectorAll('body *')) {
                    if (getComputedStyle(el).position === 'fixed') el.style.position = 'absolute'; } }""")
                time.sleep(0.5)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
            print("shot", name)

        shot("/", "home_logged_out")
        shot("/login", "login")
        page.evaluate("([t, u]) => { localStorage.setItem('truthlens_token', t); localStorage.setItem('truthlens_user', u); }",
                      [token, json.dumps(user)])
        shot("/", "home")
        shot("/verify", "verify")
        shot("/history", "history")
        shot("/dashboard", "dashboard")
        shot("/proctorlens", "proctorlens")
        shot("/about", "about")

        def open_explain():
            btn = page.get_by_text("explanation", exact=False).first
            try:
                btn.click(timeout=3000)
            except Exception:
                return
            try:  # video/audio explanations are computed on demand and take a while on CPU
                page.wait_for_function("() => !document.body.innerText.includes('Analysing the specimen')", timeout=240000)
            except Exception:
                pass

        for k, s in scans.items():
            rid = s["upload"]["upload_id"]
            shot(f"/report/{rid}", f"report_{k}", full=True, wait=4)
            shot(f"/report/{rid}", f"report_{k}_explain", full=True, wait=6, setup=open_explain)
        browser.close()


if __name__ == "__main__":
    main()
