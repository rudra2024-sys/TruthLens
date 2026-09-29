"""Start-up safety checks and helpers for deployment.

The app is developer-friendly by default (DEBUG on, permissive CORS, a dev signing key). Those defaults must never reach a
real deployment, so when DEBUG is off the server refuses to start with an unsafe configuration instead of silently
running with it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_SECRET = "change-this-development-secret-key"      # must match core/security.py's fallback
MIN_SECRET_LENGTH = 32


def parse_origins(value: str | None) -> list[str]:
    """CORS_ORIGINS="https://a.example,https://b.example" -> list; empty/"*" -> ["*"] (the historical default)."""
    origins = [o.strip().rstrip("/") for o in (value or "*").split(",") if o.strip()]
    return origins or ["*"]


def production_problems(debug: bool, secret: str, cors_origins: list[str]) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors block start-up when DEBUG is off; in DEBUG mode nothing is enforced."""
    if debug:
        return [], []
    errors, warnings = [], []
    if secret == DEFAULT_SECRET:
        errors.append("AUTH_SECRET_KEY is unset (using the public development key). Anyone could forge login tokens. "
                      "Set AUTH_SECRET_KEY to a long random value, e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`.")
    elif len(secret) < MIN_SECRET_LENGTH:
        errors.append(f"AUTH_SECRET_KEY is only {len(secret)} characters; use at least {MIN_SECRET_LENGTH}.")
    if cors_origins == ["*"]:
        warnings.append("CORS_ORIGINS is '*' (any website may call the API from a browser). Set CORS_ORIGINS to your "
                        "site's origin, e.g. https://truthlens.example.")
    return errors, warnings


def enforce_startup_checks(debug: bool, secret: str, cors_origins: list[str]) -> None:
    errors, warnings = production_problems(debug, secret, cors_origins)
    for w in warnings:
        logger.warning("Configuration warning: %s", w)
    if errors:
        raise RuntimeError("Refusing to start with an unsafe configuration (DEBUG is off):\n  - " + "\n  - ".join(errors))
