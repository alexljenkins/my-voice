#!/usr/bin/env bash
# Batch ASR evaluation: transcribe WAVs and compare against ground-truth labels.
#
# Usage: ./tools/eval.sh <wav_dir> <labels_file>
#
# labels_file format: one line per file, tab-separated:
#   filename.wav<TAB>ground truth text
#
# Outputs a markdown table and a summary.

set -euo pipefail

BINARY="${BINARY:-./target/release/my-voice}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <wav_dir> <labels_file>" >&2
    exit 1
fi

WAV_DIR="$1"
LABELS_FILE="$2"

if [[ ! -d "$WAV_DIR" ]]; then
    echo "error: wav directory not found: $WAV_DIR" >&2
    exit 1
fi

if [[ ! -f "$LABELS_FILE" ]]; then
    echo "error: labels file not found: $LABELS_FILE" >&2
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary not found or not executable: $BINARY" >&2
    echo "  Run: cargo build --release" >&2
    exit 1
fi

# Load labels into associative array: filename → expected text
declare -A LABELS
while IFS=$'\t' read -r filename expected; do
    # Skip blank lines and lines without a tab
    [[ -z "$filename" ]] && continue
    LABELS["$filename"]="$expected"
done < "$LABELS_FILE"

# Collect WAV files
mapfile -t WAV_FILES < <(find "$WAV_DIR" -maxdepth 1 -iname "*.wav" | sort)

if [[ ${#WAV_FILES[@]} -eq 0 ]]; then
    echo "No WAV files found in $WAV_DIR" >&2
    exit 1
fi

total=0
correct=0
unlabeled=()

echo ""
echo "| File | Expected | Got | Match |"
echo "|------|----------|-----|-------|"

for wav_path in "${WAV_FILES[@]}"; do
    filename="$(basename "$wav_path")"

    if [[ -z "${LABELS[$filename]+_}" ]]; then
        unlabeled+=("$filename")
        continue
    fi

    expected="${LABELS[$filename]}"

    # Capture transcription; treat non-zero exit as empty result
    got="$("$BINARY" --wav "$wav_path" 2>/dev/null || true)"

    # Normalize: trim whitespace, lowercase
    expected_norm="$(echo "$expected" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    got_norm="$(echo "$got" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ "$expected_norm" == "$got_norm" ]]; then
        match="✓"
        (( correct++ )) || true
    else
        match="✗"
    fi

    (( total++ )) || true

    # Escape pipe characters in table cells
    expected_display="${expected//|/\\|}"
    got_display="${got//|/\\|}"
    echo "| $filename | $expected_display | $got_display | $match |"
done

echo ""
echo "## Summary"
echo ""
echo "Evaluated: $total files"
echo "Correct:   $correct"
if [[ $total -gt 0 ]]; then
    pct=$(( correct * 100 / total ))
    echo "Accuracy:  ${pct}%"
else
    echo "Accuracy:  n/a"
fi

if [[ ${#unlabeled[@]} -gt 0 ]]; then
    echo ""
    echo "### Unlabeled (skipped)"
    for f in "${unlabeled[@]}"; do
        echo "  - $f"
    done
fi
