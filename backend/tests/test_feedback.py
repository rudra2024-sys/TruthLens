"""User feedback ("was this correct?"): validation, ownership, idempotency and the consent-respecting export."""

import asyncio
import csv
import os

import pytest

from tests.conftest import png_bytes


def fb_url(uid):
    return f"/api/v1/detect/{uid}/feedback"


def test_agreeing_is_stored_and_needs_no_label(client, scan, auth):
    uid = scan(auth)
    r = client.put(fb_url(uid), json={"agrees": True}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["agrees"] is True and body["true_label"] is None and body["allow_reuse"] is False   # consent is opt-in
    assert body["verdict"] == "FAKE" and body["upload_id"] == uid


def test_disagreeing_records_what_the_file_really_is(client, scan, auth):
    uid = scan(auth)
    body = client.put(fb_url(uid), json={"agrees": False, "true_label": "real", "comment": "  my own photo  "},
                      headers=auth).json()
    assert body["agrees"] is False and body["true_label"] == "real" and body["comment"] == "my own photo"


def test_disagreeing_without_a_label_counts_as_unsure(client, scan, auth):
    uid = scan(auth)
    assert client.put(fb_url(uid), json={"agrees": False}, headers=auth).json()["true_label"] == "unsure"


def test_a_label_is_ignored_when_the_user_agrees(client, scan, auth):
    uid = scan(auth)
    assert client.put(fb_url(uid), json={"agrees": True, "true_label": "real"}, headers=auth).json()["true_label"] is None


def test_resubmitting_replaces_instead_of_duplicating(client, scan, auth):
    uid = scan(auth)
    first = client.put(fb_url(uid), json={"agrees": True}, headers=auth).json()
    second = client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth).json()
    assert second["agrees"] is False and second["allow_reuse"] is True and second["created_at"] == first["created_at"]
    assert client.get(fb_url(uid), headers=auth).json()["true_label"] == "ai"


def test_get_returns_null_until_feedback_exists_then_the_feedback(client, scan, auth):
    uid = scan(auth)
    assert client.get(fb_url(uid), headers=auth).json() is None
    client.put(fb_url(uid), json={"agrees": True}, headers=auth)
    assert client.get(fb_url(uid), headers=auth).json()["agrees"] is True


def test_withdrawing_removes_it_and_is_idempotent(client, scan, auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": True, "allow_reuse": True}, headers=auth)
    assert client.delete(fb_url(uid), headers=auth).status_code == 204
    assert client.get(fb_url(uid), headers=auth).json() is None
    assert client.delete(fb_url(uid), headers=auth).status_code == 204             # nothing left to delete: still fine


@pytest.mark.parametrize("payload", [
    {},                                                    # agrees is required
    {"agrees": "maybe"},
    {"agrees": False, "true_label": "fake"},               # only real | ai | unsure
    {"agrees": False, "comment": "x" * 501},               # comment cap
])
def test_invalid_feedback_is_rejected(client, scan, auth, payload):
    assert client.put(fb_url(scan(auth)), json=payload, headers=auth).status_code == 422


def test_a_blank_comment_is_stored_as_none(client, scan, auth):
    assert client.put(fb_url(scan(auth)), json={"agrees": True, "comment": "   "}, headers=auth).json()["comment"] is None


def test_feedback_needs_a_detection_result(client, auth, upload):
    uid = upload(auth, png_bytes())                                            # uploaded, never scanned
    assert client.put(fb_url(uid), json={"agrees": True}, headers=auth).status_code == 404


def test_feedback_is_private_to_the_owner(client, scan, auth, other_auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": True}, headers=auth)
    for method, kw in (("get", {}), ("put", {"json": {"agrees": False}}), ("delete", {})):
        r = getattr(client, method)(fb_url(uid), headers=other_auth, **kw)
        assert r.status_code == 404, method
    assert client.get(fb_url(uid), headers=auth).json()["agrees"] is True          # untouched by the other user
    for method, kw in (("get", {}), ("put", {"json": {"agrees": True}}), ("delete", {})):
        assert getattr(client, method)(fb_url(uid), **kw).status_code in (401, 403)


# ------------------------------------------------------------------ export (consent is the whole point)

def _export(tmp_path):
    from app.core.database import AsyncSessionLocal
    from app.services.feedback_export import export_feedback

    async def run():
        async with AsyncSessionLocal() as db:
            return await export_feedback(db, tmp_path)
    return asyncio.run(run())


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_export_only_copies_files_and_comments_of_users_who_opted_in(client, scan, auth, other_auth, tmp_path):
    consented = scan(auth)
    private = scan(other_auth)
    client.put(fb_url(consented), json={"agrees": False, "true_label": "real", "comment": "family photo", "allow_reuse": True},
               headers=auth)
    client.put(fb_url(private), json={"agrees": False, "true_label": "real", "comment": "secret note"}, headers=other_auth)

    stats = _export(tmp_path)
    manifest = _read(tmp_path / "manifest_feedback.csv")
    files = os.listdir(tmp_path / "files")
    summary = _read(tmp_path / "feedback_summary.csv")

    assert [os.path.basename(r["path"]).split(".")[0] for r in manifest].count(consented) == 1
    assert not any(private in name for name in files) and not any(private in r["path"] for r in manifest)
    text = (tmp_path / "feedback_summary.csv").read_text(encoding="utf-8")
    assert "family photo" in text and "secret note" not in text                # a comment leaves only with consent
    assert stats["feedback_total"] >= 2 and stats["exported_files"] >= 1
    assert "path" not in summary[0] and "storage_url" not in summary[0]


def test_export_derives_labels_only_when_they_are_unambiguous(client, scan, auth, stub_detectors, tmp_path):
    cases = [
        ("FAKE", {"agrees": True, "allow_reuse": True}, 1),                              # agreed FAKE -> ai
        ("REAL", {"agrees": True, "allow_reuse": True}, 0),                              # agreed REAL -> real
        ("UNCERTAIN", {"agrees": True, "allow_reuse": True}, None),                     # agreed UNCERTAIN carries no label
        ("FAKE", {"agrees": False, "true_label": "real", "allow_reuse": True}, 0),       # disagreed -> the user's label
        ("REAL", {"agrees": False, "true_label": "ai", "allow_reuse": True}, 1),
        ("FAKE", {"agrees": False, "true_label": "unsure", "allow_reuse": True}, None),  # unsure carries no label
    ]
    expected = {}
    for verdict, body, label in cases:
        stub_detectors("image", verdict=verdict, confidence=0.9 if verdict == "FAKE" else 0.1)
        uid = scan(auth, png_bytes((len(expected) * 20 % 255, 10, 10)))
        client.put(fb_url(uid), json=body, headers=auth)
        expected[uid] = label

    _export(tmp_path)
    got = {os.path.basename(r["path"]).split(".")[0]: int(r["label"]) for r in _read(tmp_path / "manifest_feedback.csv")}
    for uid, label in expected.items():
        assert (got.get(uid) == label) if label is not None else (uid not in got), (uid, label, got.get(uid))


def test_export_reports_how_often_confident_verdicts_were_judged_wrong(client, scan, auth, stub_detectors, tmp_path):
    stub_detectors("image", verdict="FAKE", confidence=0.95)
    a, b = scan(auth, png_bytes((1, 2, 3))), scan(auth, png_bytes((4, 5, 6)))
    client.put(fb_url(a), json={"agrees": True}, headers=auth)
    client.put(fb_url(b), json={"agrees": False, "true_label": "real"}, headers=auth)
    stats = _export(tmp_path)
    assert stats["disagree"] >= 1 and stats["confident_verdicts_judged_wrong"] >= 1
    assert 0 <= stats["agreement_rate"] <= 1 and "FAKE->real" in stats["confusion"]
