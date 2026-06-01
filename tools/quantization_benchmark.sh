#!/usr/bin/env bash
# tools/quantization_benchmark.sh
#
# Compare backbone inference speed: backbone.onnx (FP32) vs backbone_int8.onnx (INT8).
#
# Runs fast_sam_3dbody_run in --headless --skip-body mode to isolate backbone
# timing.  Both models are exercised on the same input video; results are
# reported as mean / min / max / stdev with a speedup ratio.
#
# NOTE on expected results:
#   This backbone model requires CUDA — it will not load on ORT CPU EP.
#   On ORT CUDA EP, MatMulInteger dequantizes INT8 weights to FP32 at runtime
#   before the GEMM, so GPU inference speed is unchanged vs FP32.
#   The value of INT8 quantization here is a 3× reduction in model size on
#   disk and in VRAM (4.8 GB → 1.6 GB), freeing VRAM for larger batches.
#   For real INT8 Tensor Core speedup, use TensorRT EP (--trt) with INT8
#   calibration — that requires the tensorrt Python package separately.
#
# Usage:
#   tools/quantization_benchmark.sh boom.mp4
#   tools/quantization_benchmark.sh boom.mp4 --cuda -1          # CPU benchmark
#   tools/quantization_benchmark.sh boom.mp4 --onnx-dir ./onnx --warmup 10 --frames 100
#
# Options:
#   --onnx-dir DIR    Directory containing backbone*.onnx files (default: ./onnx)
#   --cuda N          CUDA device passed to the binary (-1 = CPU; default: 0)
#   --warmup N        Frames to discard from the start of each run (default: 5)
#   --frames N        Max frames to time per run; 0 = full video (default: 0)
#   --timeout N       Per-run wall-clock timeout in seconds (default: 300)
#   --fp32-only       Skip INT8 run (useful before quantizing)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO/build/fast_sam_3dbody_run"

# ── Defaults ──────────────────────────────────────────────────────────────────
ONNX_DIR="./onnx"
CUDA_DEV=0     # passed through to binary; -1 = CPU
WARMUP=5
MAX_FRAMES=0   # 0 = whole video
TIMEOUT=300
FP32_ONLY=false
VIDEO=""

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx-dir)  ONNX_DIR="$2";     shift 2 ;;
        --cuda)      CUDA_DEV="$2";     shift 2 ;;
        --warmup)    WARMUP="$2";        shift 2 ;;
        --frames)    MAX_FRAMES="$2";    shift 2 ;;
        --timeout)   TIMEOUT="$2";       shift 2 ;;
        --fp32-only) FP32_ONLY=true;     shift   ;;
        -h|--help)   usage 0 ;;
        -*)          echo "Unknown flag: $1"; usage 1 ;;
        *)           VIDEO="$1";         shift   ;;
    esac
done

[[ -z "$VIDEO" ]]     && { echo "Error: no input file given."; usage 1; }
[[ ! -f "$VIDEO" ]]   && { echo "Error: '$VIDEO' not found.";  exit 1; }
[[ ! -x "$BINARY" ]]  && {
    echo "Error: $BINARY not found."
    echo "  Build first:  cmake --build \"$REPO/build\" -j\$(nproc)"
    exit 1
}

FP32="$ONNX_DIR/backbone.onnx"
INT8="$ONNX_DIR/backbone_int8.onnx"

[[ ! -f "$FP32" ]] && { echo "Error: $FP32 not found."; exit 1; }

RUN_INT8=true
if $FP32_ONLY; then
    RUN_INT8=false
elif [[ ! -f "$INT8" ]]; then
    echo "Warning: $INT8 not found."
    echo "  Quantize first:  tools/quantize_backbone.sh --onnx-dir $ONNX_DIR"
    echo "  Continuing with FP32 only."
    echo
    RUN_INT8=false
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# run_pass BACKBONE_FILENAME -> path to temp file with one ms value per line
run_pass() {
    local backbone_name="$1"
    local tmp
    tmp=$(mktemp /tmp/bmark_XXXXXX.txt)

    printf "  %-24s  " "[$backbone_name]" >&2

    # --headless: no display window needed
    # --skip-body: skip LBS/body model so we time backbone in isolation
    timeout "$TIMEOUT" "$BINARY" \
        --onnx-dir "$ONNX_DIR" \
        --cuda     "$CUDA_DEV" \
        --from     "$VIDEO"    \
        --backbone "$backbone_name" \
        --headless \
        --skip-body \
        2>&1 \
    | awk '/\[FSB\] backbone:/ { print $(NF-1); fflush() }' \
    > "$tmp" || true   # timeout exits non-zero; that's fine

    local n
    n=$(wc -l < "$tmp")
    printf "%d frames captured\n" "$n" >&2
    echo "$tmp"   # only this goes to stdout → captured by $()
}

# stats_of TMPFILE -> prints "N MEAN MIN MAX STDEV" space-separated
stats_of() {
    python3 - "$1" "$WARMUP" "$MAX_FRAMES" <<'PYEOF'
import sys, statistics

path  = sys.argv[1]
skip  = int(sys.argv[2])
maxf  = int(sys.argv[3])

try:
    with open(path) as f:
        vals = [float(l) for l in f if l.strip()]
except Exception:
    print("0 0 0 0 0")
    sys.exit(0)

vals = vals[skip:]
if maxf > 0:
    vals = vals[:maxf]

if not vals:
    print("0 0 0 0 0")
    sys.exit(0)

n     = len(vals)
mean  = sum(vals) / n
stdev = statistics.stdev(vals) if n > 1 else 0.0
print(f"{n} {mean:.2f} {min(vals):.2f} {max(vals):.2f} {stdev:.2f}")
PYEOF
}

# ── Runs ──────────────────────────────────────────────────────────────────────
echo
echo "Backbone benchmark — $(basename "$VIDEO")"
echo "  Binary:   $BINARY"
echo "  ONNX dir: $ONNX_DIR"
echo "  Device:   $([ "$CUDA_DEV" -ge 0 ] 2>/dev/null && echo "CUDA $CUDA_DEV" || echo "CPU")"
echo "  Warmup:   $WARMUP frames (excluded from stats)"
echo "  Cap:      $([ "$MAX_FRAMES" -eq 0 ] && echo "full video" || echo "$MAX_FRAMES frames")"
echo "  Timeout:  ${TIMEOUT}s per pass"
echo

TMP_FP32=$(run_pass "backbone.onnx")
TMP_INT8=""
$RUN_INT8 && TMP_INT8=$(run_pass "backbone_int8.onnx")

# ── Stats ─────────────────────────────────────────────────────────────────────
read -r N32 MEAN32 MIN32 MAX32 STD32 <<< "$(stats_of "$TMP_FP32")"

N8=0; MEAN8=0; MIN8=0; MAX8=0; STD8=0
$RUN_INT8 && read -r N8 MEAN8 MIN8 MAX8 STD8 <<< "$(stats_of "$TMP_INT8")"

rm -f "$TMP_FP32" "${TMP_INT8:-}"

# ── Table ─────────────────────────────────────────────────────────────────────
if [[ "$N32" -eq 0 ]]; then
    echo "Error: no backbone timings captured for FP32 run."
    echo "  Check that the binary is built and the video file is readable."
    exit 1
fi

echo
printf "  %-26s  %8s  %9s  %9s  %9s  %9s\n" \
       "Model" "N frames" "Mean" "Min" "Max" "StdDev"
printf "  %-26s  %8s  %9s  %9s  %9s  %9s\n" \
       "$(printf '%.0s-' {1..26})" "--------" "---------" "---------" "---------" "---------"
printf "  %-26s  %8s  %8sms  %8sms  %8sms  ±%7sms\n" \
       "backbone.onnx (FP32)" "$N32" "$MEAN32" "$MIN32" "$MAX32" "$STD32"

if $RUN_INT8 && [[ "$N8" -gt 0 ]]; then
    printf "  %-26s  %8s  %8sms  %8sms  %8sms  ±%7sms\n" \
           "backbone_int8.onnx (INT8)" "$N8" "$MEAN8" "$MIN8" "$MAX8" "$STD8"

    SPEEDUP=$(python3 -c "
m32, m8 = float('$MEAN32'), float('$MEAN8')
if m8 > 0:
    s = m32 / m8
    direction = 'faster' if s >= 1 else 'slower'
    print(f'{s:.2f}x {direction}')
else:
    print('N/A')
")
    printf "  %-26s  %8s  %9s\n" "$(printf '%.0s-' {1..26})" "--------" "---------"
    printf "  Speedup (mean):  %s\n" "$SPEEDUP"
elif $RUN_INT8; then
    echo "  Warning: no timings captured for INT8 run — check that backbone_int8.onnx is valid."
fi

echo
