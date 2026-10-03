"""Request correlation ID (app/core/request_context.py, added 2026-10-03) - closes a real production-
debugging gap: before this, a user-reported error had no identifier at all connecting it to server logs."""
import re

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def test_every_response_carries_an_x_request_id_header(client):
    r = client.get("/health")
    assert "X-Request-ID" in r.headers
    assert UUID_RE.match(r.headers["X-Request-ID"])


def test_a_fresh_request_id_is_generated_per_request(client):
    a = client.get("/health").headers["X-Request-ID"]
    b = client.get("/health").headers["X-Request-ID"]
    assert a != b


def test_an_inbound_x_request_id_is_echoed_back_not_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "my-own-trace-id-123"})
    assert r.headers["X-Request-ID"] == "my-own-trace-id-123"


def test_request_id_is_present_even_on_an_unauthenticated_401(client):
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401
    assert "X-Request-ID" in r.headers


def test_unhandled_exception_handler_includes_the_current_request_id():
    """Unit-level, not through a live route: FastAPI resolves a route's handler function to a fixed reference
    at decoration time, so monkeypatching a route module's function after import never affects dispatch -
    calling the exception handler directly is the reliable way to test it."""
    import asyncio

    from starlette.requests import Request

    from app.core.request_context import _request_id
    from app.main import unhandled_exception_handler

    token = _request_id.set("test-fixed-request-id")
    try:
        scope = {"type": "http", "method": "GET", "path": "/whatever", "headers": []}
        request = Request(scope)
        response = asyncio.run(unhandled_exception_handler(request, RuntimeError("boom")))
    finally:
        _request_id.reset(token)

    assert response.status_code == 500
    import json
    body = json.loads(response.body)
    assert body["request_id"] == "test-fixed-request-id"
    assert "boom" in body["detail"]
