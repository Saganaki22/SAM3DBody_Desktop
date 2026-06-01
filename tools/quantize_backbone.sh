#!/usr/bin/env bash
# tools/quantize_backbone.sh
#
# Venv wrapper for tools/quantize_backbone.py.
#
# Creates a local venv (tools/.venv) on first run, installs the required
# packages, then runs quantize_backbone.py with all arguments forwarded.
#
# Usage:
#   tools/quantize_backbone.sh --onnx-dir ./onnx
#   tools/quantize_backbone.sh --onnx-dir ./onnx --mode static --calib-dir /path/to/images
#
# All flags are forwarded verbatim to quantize_backbone.py; run with --help
# for the full option list.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
REQS="onnx onnxruntime numpy psutil sympy"

# ── Bootstrap venv on first run ───────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python3" ]]; then
    echo "[quantize_backbone] Creating venv at $VENV …"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    echo "[quantize_backbone] Installing: $REQS"
    "$VENV/bin/pip" install --quiet $REQS
    echo "[quantize_backbone] Venv ready."
    echo
fi

# ── Run ───────────────────────────────────────────────────────────────────────
exec "$VENV/bin/python3" "$SCRIPT_DIR/quantize_backbone.py" "$@"
