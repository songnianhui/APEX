#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="$ROOT/examples/fe2s2"
SESSION_DIR="$CASE_DIR/filter_session"

check_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    echo "[ok] file: $path"
  else
    echo "[missing] file: $path"
    return 1
  fi
}

check_dir() {
  local path="$1"
  if [[ -d "$path" ]]; then
    echo "[ok] dir:  $path"
  else
    echo "[missing] dir:  $path"
    return 1
  fi
}

echo "== Fe2S2 pre-rerun check =="
echo "repo:    $ROOT"
echo "case:    $CASE_DIR"
echo "session: $SESSION_DIR"
echo

status=0

echo "-- Required inputs --"
check_file "$CASE_DIR/inputs/fe2s2.xyz" || status=1
check_file "$CASE_DIR/inputs/fe2s2_cas_settings.yaml" || status=1
check_file "$CASE_DIR/inputs/fe2s2_filter_settings.yaml" || status=1
check_file "$CASE_DIR/inputs/fe2s2_cluster_info.yaml" || status=1
check_file "$CASE_DIR/inputs/fe2s2_cluster_info_draft.csv" || status=1
check_file "$CASE_DIR/inputs/fe2s2_structure_labeled.png" || status=1
check_file "$SESSION_DIR/method_controls.yaml" || status=1
echo

echo "-- Acceptance / compare tools --"
check_file "$ROOT/scripts/run_fe2s2_acceptance_tests.sh" || status=1
check_file "$ROOT/scripts/compare_fe2s2_runs.py" || status=1
check_file "$ROOT/plans/fe2s2_rerun_and_compare_playbook_20260430.md" || status=1
echo

echo "-- Current generated directories --"
for path in \
  "$CASE_DIR/outputs" \
  "$SESSION_DIR/step3_uhf" \
  "$SESSION_DIR/step6_ccsdt" \
  "$SESSION_DIR/step7_dmrg_basis" \
  "$SESSION_DIR/step8_dmrg" \
  "$SESSION_DIR/step9_extrapolate" \
  "$SESSION_DIR/step10_report"
do
  check_dir "$path" || true
done
echo

echo "-- Notes --"
echo "Current example already contains generated outputs."
echo "A fresh rerun will refresh many files under:"
echo "  $CASE_DIR/outputs"
echo "  $SESSION_DIR"
echo
echo "Recommended order when rerun starts:"
echo "  1. follow $ROOT/plans/fe2s2_rerun_and_compare_playbook_20260430.md"
echo "  2. run $ROOT/scripts/run_fe2s2_acceptance_tests.sh"
echo "  3. only then run compare_fe2s2_runs.py against APEX_bk"
echo

if [[ "$status" -ne 0 ]]; then
  echo "pre-rerun check: FAILED"
  exit 1
fi

echo "pre-rerun check: OK"
