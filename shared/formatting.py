"""Shared formatting helpers for user-facing workflow artifacts."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def format_energy(value) -> str:
    """Format an energy-like scalar for markdown/report output."""
    return f"{float(value):.10f}" if value is not None else "N/A"


def shell_safe_artifact_token(label: str) -> str:
    """Return an ASCII/shell-safe token for file and scratch artifact names."""
    text = str(label)
    replacements = {
        "↑": "up",
        "↓": "down",
        "α": "alpha",
        "β": "beta",
        "(": "_",
        ")": "_",
        "|": "_",
        "+": "_plus_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ",": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    if safe:
        return safe

    suffix = hashlib.sha1(str(label).encode("utf-8")).hexdigest()[:8]
    return f"artifact_{suffix}"
