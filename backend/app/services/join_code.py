"""Private organisation join codes (added 2026-10-12) -- how a student links their account to an
organisation before enrolling their reference photo, closing the real multi-tenancy gap: before
this, the organisation dashboard showed every student to every organisation account.

Codes are generated here, stored raw (no dashes) on User.join_code, and formatted with a dash
only at display/comparison time -- avoids two representations of the same value living in the DB.
"""

import secrets

# Excludes 0/O, 1/I/L -- a human reading this off a screen (or hearing it read aloud) shouldn't
# have to guess which character was meant.
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_LENGTH = 8


def generate_join_code() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_LENGTH))


def normalize_join_code(raw: str) -> str:
    """Strips whitespace/dashes and uppercases, so "a3f9-k7m2", "A3F9 K7M2" and "A3F9K7M2" all
    compare equal -- a student copy-pasting or retyping a code shouldn't be tripped up by
    formatting."""
    return "".join(ch for ch in raw.upper() if ch.isalnum())


def format_join_code(code: str) -> str:
    """Display-only grouping, e.g. "A3F9K7M2" -> "A3F9-K7M2". The stored value is always raw."""
    if len(code) != _LENGTH:
        return code
    return f"{code[:4]}-{code[4:]}"
