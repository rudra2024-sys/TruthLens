"""Shared exception types for the detection pipelines.

UnprocessableMediaError distinguishes "the uploaded file is bad" (corrupt,
empty, wrong format, undecodable) from genuine server-side failures. The
detect.py route maps it to a clean HTTP 422 with the message given here,
instead of leaking a raw Python exception string to the client.
"""
from __future__ import annotations


class UnprocessableMediaError(Exception):
    """Raised when a media file cannot be decoded/processed, independent of
    any model. The message is written to be safe to return to an API client
    directly (no file paths, no internal exception text)."""
