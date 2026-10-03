"""Per-request correlation ID (added 2026-10-03) -- closes a real production-debugging gap: before this,
nothing tied one request's log lines together, or let a user-reported error be matched back to server logs at
all (the 500 response body had no identifier in it whatsoever).

`RequestIdMiddleware` reads an inbound `X-Request-ID` (trusting an upstream proxy/load balancer's own
correlation ID when present, generating a fresh uuid4 otherwise), stores it in a contextvar for the duration
of the request, and echoes it back as a response header. `get_request_id()` reads that contextvar so any log
line can embed the current request's ID in its message text directly (see main.py's exception handler) --
this deliberately does NOT reconfigure the root logger's formatters/handlers to inject it automatically: this
app is run under different setups (uvicorn CLI, pytest's TestClient, a future gunicorn+uvicorn-workers
deployment), each already configuring logging its own way, and globally overriding that risks silently
breaking whichever setup is actually in use just to save callers one f-string. Putting the ID in the message
text works identically everywhere.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

HEADER_NAME = "X-Request-ID"


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(HEADER_NAME)
        request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
        token = _request_id.set(request_id)
        try:
            request.state.request_id = request_id
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers[HEADER_NAME] = request_id
        return response
