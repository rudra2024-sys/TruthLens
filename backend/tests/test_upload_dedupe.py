"""Batch dedupe: GET /upload/by-hash/{sha256} finds the caller's earlier scan of identical bytes."""
import hashlib

from conftest import png_bytes


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_finds_earlier_scan_of_identical_bytes(client, auth, scan):
    data = png_bytes((11, 22, 33))
    uid = scan(auth, data)
    r = client.get(f"/api/v1/upload/by-hash/{_sha(data)}", headers=auth)
    assert r.status_code == 200
    assert r.json()["upload_id"] == uid
    assert r.json()["verdict"] == "FAKE"


def test_different_bytes_not_found(client, auth, scan):
    scan(auth, png_bytes((1, 2, 3)))
    assert client.get(f"/api/v1/upload/by-hash/{_sha(png_bytes((9, 9, 9)))}", headers=auth).status_code == 404


def test_uploaded_but_never_scanned_is_not_a_match(client, auth, upload):
    data = png_bytes((44, 55, 66))
    upload(auth, data)
    assert client.get(f"/api/v1/upload/by-hash/{_sha(data)}", headers=auth).status_code == 404


def test_other_users_scan_is_not_visible(client, auth, other_auth, scan):
    data = png_bytes((77, 88, 99))
    scan(auth, data)
    assert client.get(f"/api/v1/upload/by-hash/{_sha(data)}", headers=other_auth).status_code == 404


def test_renamed_copy_still_matches(client, auth, scan):
    data = png_bytes((5, 6, 7))
    uid = scan(auth, data, name="a.png")
    assert client.get(f"/api/v1/upload/by-hash/{_sha(data)}", headers=auth).json()["upload_id"] == uid


def test_bad_hash_rejected_and_auth_required(client, auth):
    assert client.get("/api/v1/upload/by-hash/not-a-hash", headers=auth).status_code == 422
    assert client.get(f"/api/v1/upload/by-hash/{'0' * 64}").status_code in (401, 403)
