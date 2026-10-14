"""Promote a user account to organisation (added 2026-10-11, repurposed 2026-10-12 now that
account_type is chosen at signup -- see auth.py::SignupRequest). Kept as a secondary override
path, e.g. to fix an account that picked the wrong type at signup; no longer the only way to get
organisation access.

  cd backend && ../.venv/Scripts/python.exe scripts/promote_admin.py --email you@example.com

Reads/writes the database configured by DATABASE_URL (the same one the app uses).
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select                       # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession      # noqa: E402

from app.core.database import AsyncSessionLocal      # noqa: E402
from app.models.models import User                   # noqa: E402
from app.services.audit_log import log_event         # noqa: E402


async def set_role(db: AsyncSession, email: str, role: str) -> str | None:
    """Sets `email`'s role to `role` (either "organization" or "student"). Returns a
    human-readable outcome message, or None if no account matches `email`. Kept separate from main() so tests
    can exercise this directly (same reasoning as services/feedback_export.py::export_feedback
    being separate from its own script's main())."""
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        return None

    if user.role == role:
        return f"{email} already has role '{role}' -- nothing to do."

    old_role = user.role
    user.role = role
    # log_event commits the session itself (see audit_log.py's docstring) -- this also persists
    # the role change above, no separate commit needed.
    await log_event(db, "role_changed", user_id=user.user_id, detail=f"{old_role} -> {role}")
    return f"{email}: role changed {old_role} -> {role}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--revoke", action="store_true", help="Demote back to role 'student' instead of promoting.")
    args = ap.parse_args()
    email = args.email.lower().strip()
    role = "student" if args.revoke else "organization"

    async with AsyncSessionLocal() as db:
        outcome = await set_role(db, email, role)
        print(outcome if outcome is not None else f"No account found for {email}")


if __name__ == "__main__":
    asyncio.run(main())
