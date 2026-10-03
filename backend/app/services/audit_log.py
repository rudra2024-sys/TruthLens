"""Records sensitive actions (added 2026-10-03) - closes the gap docs/ETHICS_AND_LIMITATIONS.md section 5
has flagged since it was written ("no audit log"). Read-side (listing/exporting the log) is deliberately not
built yet - this only writes entries; see CLAUDE.md section 25 for the scope line on what's intentionally
left for later.

log_event() uses the CALLER's request-scoped DB session, but commits immediately instead of just flushing.
Two things were tried and rejected before landing here, both found by actually testing, not assumed safe:
  * flush-only on the caller's session: app/core/database.py's get_db() rolls back the WHOLE session if the
    route ends up raising (e.g. login's 401 on a bad password), which silently discarded a "login_failed"
    entry logged just before the exception, in exactly the case where keeping the record matters most.
  * a separate, independently-committed session: SQLite only allows one writer at a time, and the caller's
    own session is usually still mid-transaction (e.g. signup's `db.add(user)` hasn't committed yet) when
    log_event runs - a second connection trying to write at the same time deadlocks against it ("database is
    locked"), since the first transaction can't release the write lock until the route function (which is
    waiting on log_event) returns.
Committing the caller's own session immediately is the one approach that avoids both: no second connection
(no lock contention), and the audit entry - plus anything already pending in that session, which is fine in
every caller here (see the call sites) - becomes durable right away, surviving any later exception in the
same request. SQLAlchemy sessions support multiple commits in their lifetime; anything added to the session
after this point still gets its own commit (or rollback) when the request ends normally.

Never raises on its own account either way: a logging failure must not break the action being logged (a
failed login must still return 401, not 500 because the audit write hit a DB error).
"""

from __future__ import annotations

import logging

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import AuditLogEntry

logger = logging.getLogger(__name__)


def client_ip(request: Request | None) -> str | None:
    if request is None or request.client is None:
        return None
    return request.client.host


async def log_event(
    db: AsyncSession,
    action: str,
    user_id: str | None = None,
    detail: str | None = None,
    request: Request | None = None,
) -> None:
    try:
        db.add(AuditLogEntry(user_id=user_id, action=action, detail=detail, ip_address=client_ip(request)))
        await db.commit()
    except Exception:
        logger.exception("Audit log write failed (action=%s, user_id=%s) - continuing anyway", action, user_id)
