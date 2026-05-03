"""Shared helpers for files stored under ``shared/knowledge_base``."""

from __future__ import annotations

import os

_SHARED_KB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "knowledge_base")
)


def data_file(filename: str) -> str:
    """Return the full path to a file inside the shared knowledge base."""
    return os.path.join(_SHARED_KB, filename)
