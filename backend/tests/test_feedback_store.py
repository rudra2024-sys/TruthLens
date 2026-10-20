"""Files users flagged as wrongly judged are kept for retraining - only with consent and a clear label."""

import json
from pathlib import Path

from app.core.config import settings


def fb_url(uid):
    return f"/api/v1/detect/{uid}/feedback"


def kept(uid):
    return sorted(p for p in Path(settings.FEEDBACK_WRONG_DIR).rglob(f"{uid}.*"))


def test_wrong_with_label_and_consent_is_kept_with_its_details(client, scan, auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": False, "true_label": "real", "comment": "mine", "allow_reuse": True}, headers=auth)
    files = kept(uid)
    assert [f.suffix for f in files] == [".json", ".png"]
    assert files[0].parent == Path(settings.FEEDBACK_WRONG_DIR) / "real" / "image"
    meta = json.loads(files[0].read_text())
    assert meta["verdict_shown"] == "FAKE" and meta["user_says_it_is"] == "real" and meta["comment"] == "mine"


def test_nothing_is_kept_without_consent_agreement_or_a_label(client, scan, auth):
    for body in ({"agrees": False, "true_label": "ai"},                                  # no consent
                 {"agrees": True, "allow_reuse": True},                                  # agreed
                 {"agrees": False, "true_label": "unsure", "allow_reuse": True},         # no clear label
                 {"agrees": False, "allow_reuse": True}):                                # defaults to unsure
        uid = scan(auth)
        client.put(fb_url(uid), json=body, headers=auth)
        assert kept(uid) == [], body


def test_changing_feedback_moves_or_removes_the_copy(client, scan, auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": False, "true_label": "real", "allow_reuse": True}, headers=auth)
    client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth)
    assert {f.parent.parent.name for f in kept(uid)} == {"ai"}                           # moved, not duplicated
    client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": False}, headers=auth)
    assert kept(uid) == []                                                               # consent withdrawn
    client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth)
    client.put(fb_url(uid), json={"agrees": True}, headers=auth)
    assert kept(uid) == []                                                               # changed mind


def test_withdrawing_feedback_removes_the_copy(client, scan, auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth)
    assert kept(uid)
    assert client.delete(fb_url(uid), headers=auth).status_code == 204
    assert kept(uid) == []


def test_a_storage_problem_does_not_stop_feedback_saving(client, scan, auth, monkeypatch):
    uid = scan(auth)
    monkeypatch.setattr(settings, "FEEDBACK_WRONG_DIR", "/proc/nope/cannot/write")
    r = client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth)
    assert r.status_code == 200 and r.json()["agrees"] is False


def test_deleting_the_account_removes_the_copy(client, scan, auth):
    uid = scan(auth)
    client.put(fb_url(uid), json={"agrees": False, "true_label": "ai", "allow_reuse": True}, headers=auth)
    assert kept(uid)
    assert client.delete("/api/v1/auth/me", headers=auth).status_code in (200, 204)
    assert kept(uid) == []
