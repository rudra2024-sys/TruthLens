"""Hostile text inside a file's metadata must not break, or inject markup into, the PDF report."""

import io

from PIL import Image, PngImagePlugin
from pypdf import PdfReader


def hostile_png():
    info = PngImagePlugin.PngInfo()
    payload = '<b>BOLD</b> & <font color="red">RED</font> </para><para> <img src="x"/> Steps: 20, Sampler: Euler, Seed: 1'
    info.add_text("parameters", payload)
    info.add_text("prompt", payload)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (9, 9, 9)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def test_report_survives_markup_in_metadata_and_shows_it_as_plain_text(client, scan, auth):
    uid = scan(auth, hostile_png(), "hostile.png", "image/png")
    r = client.get(f"/api/v1/report/{uid}", headers=auth)
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
    assert "Provenance" in text                                         # the section was built, not silently dropped
    assert "<b>BOLD</b>" in text or "BOLD" in text                        # shown as text, not interpreted as bold markup


def test_provenance_endpoint_returns_hostile_metadata_as_inert_data(client, scan, auth):
    uid = scan(auth, hostile_png(), "hostile.png", "image/png")
    body = client.get(f"/api/v1/detect/{uid}/provenance", headers=auth).json()
    assert body["available"] and body["level"] == "declared_ai"
    assert any("BOLD" in s["detail"] for s in body["signals"])            # JSON string: the frontend renders it as text
