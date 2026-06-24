#!/usr/bin/env bash
# Materialise a deterministic English benchmark set from LibriSpeech test-other
# (OpenSLR-12) into samples/ for tools/bench-wer.sh and the WER regression test.
#
# The audio is gitignored (large, and the manifest is what matters); this script
# downloads + converts it on demand. With the default SEED + counts it selects
# the SAME clips on any machine, so the committed samples/expected.txt and the
# downloaded audio always line up. Run it once after cloning, then run the bench.
#
# Each clip -> 16 kHz mono pcm_s16le WAV. Two duration buckets: mostly short, plus
# a few long. The 330 MB tarball is cached + md5-verified. Re-runs are cheap:
# existing wavs are kept, expected.txt is rewritten identically. Changing SEED or
# the counts selects a different set (leftover ls_*.wav are harmless; `rm
# samples/ls_*.wav` to reset).
#
# License: LibriSpeech is CC BY 4.0 (Panayotov et al. 2015). References are kept
# VERBATIM — ALL-CAPS, numbers spelled out. The harness normalizes both sides for
# its `WER` column; track that, not `strict` (which the caps + spelled-out numbers
# inflate even on an acoustically perfect transcript).
#
# Env (defaults reproduce the committed set): COUNT=100 LONG_COUNT=10
#   SHORT_MAX=16 LONG_MAX=60 MINDUR=2 SEED=42 SPLIT=test-other
#   CACHE=~/.cache/my-voice-bench

set -euo pipefail
cd "$(dirname "$0")/.."

COUNT="${COUNT:-100}"            # total clips
LONG_COUNT="${LONG_COUNT:-10}"   # of those, how many from the long bucket
SHORT_MAX="${SHORT_MAX:-16}"     # short bucket: MINDUR..SHORT_MAX seconds
LONG_MAX="${LONG_MAX:-60}"       # long  bucket: SHORT_MAX..LONG_MAX seconds
MINDUR="${MINDUR:-2}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-test-other}"
CACHE="${CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/my-voice-bench}"

SAMPLES=samples
EXPECTED="$SAMPLES/expected.txt"
BASE_URL="https://www.openslr.org/resources/12"
SHORT_TARGET=$(( COUNT - LONG_COUNT ))

command -v ffmpeg  >/dev/null || { echo "need ffmpeg"  >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "need ffprobe" >&2; exit 1; }

mkdir -p "$CACHE" "$SAMPLES"
tarball="$CACHE/$SPLIT.tar.gz"
root="$CACHE/LibriSpeech/$SPLIT"

# 1. Download (cached) + md5 verify
if [[ ! -f "$tarball" ]]; then
    echo "downloading $SPLIT.tar.gz (~330 MB)..." >&2
    curl -fL --retry 3 -o "$tarball" "$BASE_URL/$SPLIT.tar.gz"
fi
md5line="$(curl -fsL "$BASE_URL/md5sum.txt" 2>/dev/null | grep "$SPLIT.tar.gz" || true)"
if [[ -n "$md5line" ]]; then
    want="${md5line%% *}"; have="$(md5sum "$tarball" | cut -d' ' -f1)"
    [[ "$want" == "$have" ]] || { echo "md5 mismatch ($have != $want) — rm $tarball and retry" >&2; exit 1; }
fi

# 2. Extract once
[[ -d "$root" ]] || { echo "extracting..." >&2; tar xzf "$tarball" -C "$CACHE"; }

# 3. Deterministic order: every utterance, sorted (C locale), then seeded-shuffled
mapfile -t candidates < <(find "$root" -name '*.flac' | LC_ALL=C sort | python3 -c \
    "import sys,random;l=[x for x in sys.stdin.read().splitlines() if x];random.seed($SEED);random.shuffle(l);print(chr(10).join(l))")

# 4. Fill the two buckets in that order; materialize any missing wav as we go
manifest="$(mktemp)"
short_added=0; long_added=0
for flac in "${candidates[@]}"; do
    (( short_added >= SHORT_TARGET && long_added >= LONG_COUNT )) && break
    uid="$(basename "$flac" .flac)"                       # e.g. 367-130732-0000
    out="ls_${uid}.wav"
    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$flac" 2>/dev/null || echo 0)"
    [[ -z "$dur" || "$dur" == 0 ]] && continue

    if   awk "BEGIN{exit !($dur>=$MINDUR && $dur<=$SHORT_MAX)}" && (( short_added < SHORT_TARGET )); then
        bucket=short
    elif awk "BEGIN{exit !($dur>$SHORT_MAX && $dur<=$LONG_MAX)}" && (( long_added < LONG_COUNT )); then
        bucket=long
    else
        continue
    fi

    trans="$(dirname "$flac")/${uid%-*}.trans.txt"        # <spk>-<chap>.trans.txt
    ref="$(grep -m1 "^$uid " "$trans" | cut -d' ' -f2-)"
    [[ -z "$ref" ]] && continue

    [[ -f "$SAMPLES/$out" ]] || ffmpeg -nostdin -loglevel error -y -i "$flac" \
        -ar 16000 -ac 1 -c:a pcm_s16le "$SAMPLES/$out"
    printf '%s\t%s\n' "$out" "$ref" >> "$manifest"
    if [[ "$bucket" == short ]]; then short_added=$(( short_added + 1 )); else long_added=$(( long_added + 1 )); fi
done

# 5. Write expected.txt sorted by filename (stable git diffs)
LC_ALL=C sort "$manifest" -o "$EXPECTED"
rm -f "$manifest"

echo "samples/: $short_added short + $long_added long = $(( short_added + long_added )) clips → $EXPECTED" >&2
