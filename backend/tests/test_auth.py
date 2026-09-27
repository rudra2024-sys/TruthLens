"""Authentication: signup / login / token validation. (Auth is considered done; these tests only guard it.)"""

import uuid

import pytest
from jose import jwt

from app.core import security
from tests.conftest import PASSWORD


def _email():
    return f"auth-{uuid.uuid4().hex[:8]}@example.com"


def test_signup_returns_token_and_user(client):
    email = _email()
    r = client.post("/api/v1/auth/signup", json={"name": "  Ada  ", "email": email.upper(), "password": PASSWORD})
    assert r.status_code == 201
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["email"] == email                      # normalised to lower case
    assert body["name"] == "Ada"                       # trimmed
    assert "password" not in body and "password_hash" not in body


def test_signup_rejects_short_password(client):
    r = client.post("/api/v1/auth/signup", json={"name": "A", "email": _email(), "password": "short"})
    assert r.status_code == 400
    assert "8 characters" in r.json()["detail"]


def test_signup_rejects_duplicate_email_case_insensitively(client):
    email = _email()
    assert client.post("/api/v1/auth/signup", json={"name": "A", "email": email, "password": PASSWORD}).status_code == 201
    r = client.post("/api/v1/auth/signup", json={"name": "B", "email": email.upper(), "password": PASSWORD})
    assert r.status_code == 409


def test_signup_rejects_malformed_email(client):
    r = client.post("/api/v1/auth/signup", json={"name": "A", "email": "not-an-email", "password": PASSWORD})
    assert r.status_code == 422


def test_login_success_and_me(client, user):
    _, email = user
    r = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["email"] == email


@pytest.mark.parametrize("password", ["wrong-password-123", ""])
def test_login_wrong_password_is_401(client, user, password):
    _, email = user
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 401


def test_login_unknown_user_gives_same_error_as_wrong_password(client, user):
    """No user-enumeration: unknown email and wrong password must be indistinguishable."""
    _, email = user
    a = client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password-123"})
    b = client.post("/api/v1/auth/login", json={"email": _email(), "password": "wrong-password-123"})
    assert (a.status_code, a.json()) == (b.status_code, b.json())


def test_passwords_are_stored_hashed(client, user):
    import asyncio

    from sqlalchemy import select

    from app.core.database import AsyncSessionLocal
    from app.models.models import User

    _, email = user

    async def fetch():
        async with AsyncSessionLocal() as s:
            return (await s.execute(select(User).where(User.email == email))).scalar_one()

    row = asyncio.run(fetch())
    assert row.password_hash != PASSWORD and row.password_hash.startswith("$2")     # bcrypt


def test_me_rejects_garbage_and_tampered_tokens(client, auth):
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    good = auth["Authorization"].split()[1]
    tampered = good[:-3] + ("AAA" if not good.endswith("AAA") else "BBB")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401


def test_token_signed_with_wrong_key_is_rejected(client, user):
    forged = jwt.encode({"sub": "someone", "exp": 9999999999}, "another-secret", algorithm=security.ALGORITHM)
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_expired_token_is_rejected(client):
    expired = jwt.encode({"sub": "someone", "exp": 1}, security.SECRET_KEY, algorithm=security.ALGORITHM)
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
