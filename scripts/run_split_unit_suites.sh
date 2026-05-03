#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/3] ruff check"
ruff check APEX_CAS APEX_Filter shared

echo "[2/3] APEX_CAS test suite"
pytest -q APEX_CAS/tests

echo "[3/3] APEX_Filter test suite"
pytest -q APEX_Filter/tests
