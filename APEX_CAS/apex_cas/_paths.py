"""Internal path utilities — YAML knowledge base lives in shared/."""

import os

_SHARED_KB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "shared", "knowledge_base")
)


def get_data_dir() -> str:
    """Return the path to the shared knowledge base directory."""
    return _SHARED_KB


def data_file(filename: str) -> str:
    """Return the full path to a YAML file in the shared knowledge base."""
    return os.path.join(get_data_dir(), filename)
