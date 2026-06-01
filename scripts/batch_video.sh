#!/usr/bin/env bash
# scripts/batch_video.sh FILE [FILE ...] [-- extra flags]
#
# Run scripts/video.sh --save --bvh on every supplied file, serially.
#
# For each file the derived outputs are:
#   borat.mkv  →  --from borat.mkv  --save borat_rendered.mp4  --bvh borat.bvh
#
# Anything after the first --flag argument is forwarded verbatim to every
# video.sh call.  Both invocation styles work:
#
#   scripts/batch_video.sh *.mkv                      # shell expands glob
#   scripts/batch_video.sh "*.mkv"                    # script expands glob
#   scripts/batch_video.sh *.mkv --butterworth         # extra flags at end
#   scripts/batch_video.sh *.mkv --cuda -1 --bw-cutoff 4.0

set -euo pipefail

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_SH="$THISDIR/video.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") FILE [FILE ...] [extra flags for video.sh]"
    echo "  Examples:"
    echo "    $(basename "$0") *.mkv"
    echo "    $(basename "$0") *.mkv --butterworth --bw-cutoff 4.0"
    exit 1
fi

# ── Split args: files (before first --flag) vs extra flags (from first --flag) ─
FILES=()
EXTRA=()
in_extra=false

for arg in "$@"; do
    if $in_extra; then
        EXTRA+=("$arg")
    elif [[ "$arg" == --* ]]; then
        in_extra=true
        EXTRA+=("$arg")
    else
        # Could be a quoted glob or a literal path — expand either way.
        shopt -s nullglob
        expanded=($arg)
        shopt -u nullglob
        if [[ ${#expanded[@]} -gt 0 ]]; then
            FILES+=("${expanded[@]}")
        else
            # Literal path (may not exist yet, let video.sh error on it).
            FILES+=("$arg")
        fi
    fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No input files found."
    exit 1
fi

echo "Batch: ${#FILES[@]} file(s)"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "Extra flags: ${EXTRA[*]}"
echo

PASS=0
FAIL=0
FAILED_FILES=()

for FILE in "${FILES[@]}"; do
    base=$(basename "$FILE")
    stem="${base%.*}"

    BVH_OUT="${stem}.bvh"
    RENDERED_OUT="${stem}_rendered.mp4"

    echo "──────────────────────────────────────────────"
    echo "  Input:    $FILE"
    echo "  BVH:      $BVH_OUT"
    echo "  Rendered: $RENDERED_OUT"
    echo "──────────────────────────────────────────────"

    if bash "$VIDEO_SH" \
            --from "$FILE" \
            --save "$RENDERED_OUT" \
            --bvh  "$BVH_OUT" \
            "${EXTRA[@]}"; then
        PASS=$((PASS + 1))
        echo "  ✓ done: $FILE"
    else
        FAIL=$((FAIL + 1))
        FAILED_FILES+=("$FILE")
        echo "  ✗ FAILED: $FILE (continuing)"
    fi
    echo
done

echo "══════════════════════════════════════════════"
echo "  Batch complete: $PASS succeeded, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed files:"
    for f in "${FAILED_FILES[@]}"; do echo "    $f"; done
fi
echo "══════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]
