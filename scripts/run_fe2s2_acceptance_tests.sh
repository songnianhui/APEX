#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Fe2S2 parameter-authority audit"
python "$ROOT/scripts/audit_fe2s2_parameter_authority.py"

echo "[2/4] APEX_CAS Fe2S2 example contract"
pytest -q "$ROOT/APEX_CAS/tests/test_apex_cas_fe2s2_example_contract.py"

echo "[3/4] APEX_Filter Fe2S2 HDF5 contract"
pytest -q "$ROOT/APEX_Filter/tests/test_fe2s2_hdf5_contract.py"

echo "[4/4] APEX_Filter Fe2S2 example contract"
pytest -q "$ROOT/APEX_Filter/tests/test_fe2s2_example_contract.py"
