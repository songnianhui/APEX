"""Regression coverage for the intended cross-package dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path("/Users/snh/Projects/APEX")


def _cross_package_imports(root: Path) -> dict[Path, list[tuple[str, str | None]]]:
    findings: dict[Path, list[tuple[str, str | None]]] = {}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[tuple[str, str | None]] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("apex_cas") or alias.name.startswith("apex_filter"):
                        imports.append((alias.name, None))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("apex_cas") or module.startswith("apex_filter"):
                    for alias in node.names:
                        imports.append((module, alias.name))
        if imports:
            findings[path] = imports
    return findings


def test_apex_filter_keeps_only_intentional_apex_cas_runtime_dependency():
    findings = _cross_package_imports(REPO_ROOT / "APEX_Filter" / "apex_filter")
    assert findings == {
        REPO_ROOT / "APEX_Filter" / "apex_filter" / "CAS_loader.py": [
            ("apex_cas.state_io", "load_cas_state"),
        ],
    }


def test_apex_cas_runtime_does_not_depend_on_apex_filter():
    findings = _cross_package_imports(REPO_ROOT / "APEX_CAS" / "apex_cas")
    assert findings == {}
