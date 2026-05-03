#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="$ROOT/examples/fe2s2"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST_ROOT="$CASE_DIR/bk"
DEST_DIR="$DEST_ROOT/rerun_snapshot_$STAMP"

mkdir -p "$DEST_DIR"

echo "== Archive current Fe2S2 state =="
echo "source: $CASE_DIR"
echo "dest:   $DEST_DIR"
echo

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    cp -R "$src" "$dst"
    echo "[copied] $src -> $dst"
  else
    echo "[skip]   missing: $src"
  fi
}

copy_if_exists "$CASE_DIR/inputs" "$DEST_DIR/"
copy_if_exists "$CASE_DIR/chan_ref" "$DEST_DIR/"
copy_if_exists "$CASE_DIR/outputs" "$DEST_DIR/"
copy_if_exists "$CASE_DIR/filter_session" "$DEST_DIR/"

cat > "$DEST_DIR/README_snapshot.txt" <<EOF
Fe2S2 rerun snapshot created at: $STAMP

Purpose:
- preserve the current local Fe2S2 example state before a fresh rerun
- keep a same-tree snapshot separate from APEX_bk

Captured directories:
- inputs
- chan_ref
- outputs
- filter_session
EOF

echo
echo "snapshot complete: $DEST_DIR"
