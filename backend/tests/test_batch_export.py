"""Batch export: CSV / PDF summary of scanned uploads (routes/batch.py, services/batch_export.py)."""

import csv
import io

from pypdf import PdfReader

from app.services.batch_export import BatchRow, _safe_cell, to_csv
from app.services.report.batch_report import counts, overview_sentence
from tests.conftest import png_bytes, wav_bytes

URL = "/api/v1/batch/export"


def _csv(resp):
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/csv")
    return list(csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))))


def test_csv_has_one_row_per_file_read_from_the_stored_results(client, scan, auth):
    a = scan(auth, png_bytes(), "one.png", "image/png")
    b = scan(auth, wav_bytes(), "two.wav", "audio/wav")
    rows = _csv(client.post(URL, headers=auth, json={"format": "csv", "items": [
        {"upload_id": a, "file_name": "folder/one.png"}, {"upload_id": b, "file_name": "two.wav"}]}))
    assert [r["File"] for r in rows] == ["folder/one.png", "two.wav"]
    assert rows[0]["Type"] == "image" and rows[1]["Type"] == "audio"
    assert rows[0]["Status"] == "Scanned" and rows[0]["Verdict code"] == "FAKE" and rows[0]["Verdict"] == "Manipulated"
    assert rows[0]["Case ID"] == a[:8].upper() and float(rows[0]["Tool confidence (%)"]) > 0


def test_a_file_that_could_not_be_scanned_still_gets_a_row_with_the_reason(client, scan, auth):
    a = scan(auth)
    rows = _csv(client.post(URL, headers=auth, json={"format": "csv", "items": [
        {"upload_id": a, "file_name": "ok.png"}, {"file_name": "broken.mov", "error": "Unsupported file type"}]}))
    assert rows[1]["Status"] == "Not scanned" and rows[1]["Verdict"] == "Not scanned"
    assert rows[1]["Problem"] == "Unsupported file type" and rows[1]["Verdict code"] == ""


def test_pdf_summary_has_the_counts_and_plain_words(client, scan, auth):
    a, b = scan(auth), scan(auth)
    r = client.post(URL, headers=auth, json={"format": "pdf", "items": [
        {"upload_id": a, "file_name": "a.png"}, {"upload_id": b, "file_name": "b.png"},
        {"file_name": "c.avi", "error": "too big"}]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    text = " ".join(" ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages).split())
    assert "Batch Summary Report" in text and "TruthLens looked at 3 files" in text
    assert "2 were flagged as probably made or edited by AI" in text and "1 could not be scanned" in text
    assert "a.png" in text and "c.avi" in text and "too big" in text
    assert "What the terms in this report mean" in text and "CLAUDE.md" not in text


def test_someone_elses_upload_is_a_404_and_nothing_leaks(client, scan, auth, other_auth):
    a = scan(auth)
    r = client.post(URL, headers=other_auth, json={"format": "csv", "items": [{"upload_id": a, "file_name": "x.png"}]})
    assert r.status_code == 404
    assert client.post(URL, headers=auth, json={"format": "csv", "items": [
        {"upload_id": "00000000-0000-0000-0000-000000000000", "file_name": "x"}]}).status_code == 404


def test_limits_and_auth_are_enforced(client, scan, auth):
    a = scan(auth)
    item = {"upload_id": a, "file_name": "a.png"}
    assert client.post(URL, json={"format": "csv", "items": [item]}).status_code in (401, 403)      # no token
    assert client.post(URL, headers=auth, json={"format": "csv", "items": [item] * 101}).status_code == 422
    assert client.post(URL, headers=auth, json={"format": "csv", "items": []}).status_code == 422
    assert client.post(URL, headers=auth, json={"format": "xml", "items": [item]}).status_code == 422
    assert client.post(URL, headers=auth, json={"format": "csv", "items": [item] * 100}).status_code == 200


def test_spreadsheet_formulas_in_file_names_are_neutralised():
    for bad in ("=HYPERLINK(\"http://x\")", "+cmd", "-1+1", "@SUM(A1)"):
        assert _safe_cell(bad).startswith("'")
    assert _safe_cell("holiday.png") == "holiday.png"
    out = to_csv([BatchRow(file_name="=evil()", problem="@x")]).decode("utf-8-sig")
    assert "'=evil()" in out and "'@x" in out


def test_overview_sentence_counts():
    rows = [BatchRow("a", scanned=True, verdict_code="REAL"), BatchRow("b", scanned=True, verdict_code="FAKE"),
            BatchRow("c", scanned=True, verdict_code="UNCERTAIN"), BatchRow("d")]
    c = counts(rows)
    assert c == {"total": 4, "authentic": 1, "manipulated": 1, "uncertain": 1, "not_scanned": 1}
    s = overview_sentence(c)
    assert "4 files" in s and "1 look authentic" in s and "and 1 could not be scanned" in s
