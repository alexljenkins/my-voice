#!/usr/bin/env bash
# The standard benchmark: accuracy + warm transcription time + memory, measured
# cleanly enough to compare PR-to-PR and model-to-model. See ../TIMING.md.
#
# Two phases:
#   1. Accuracy — one `cargo test` run. WER is greedy/deterministic, so one run
#      is the whole truth; no need to repeat it.
#   2. Perf — drive the release binary directly with `--bench-iters`, which loads
#      + warms the model ONCE per sample then re-times N warm passes. We report
#      the per-sample MIN (the run least disturbed by box load = the code's true
#      cost) and the median/p95 RTF across samples, plus peak RSS.
#
# Usage:   ./tools/bench-wer.sh
# Env:     ITERS=5  MODEL=moonshine-base  CORES=1-6  QUIET=0.5  QUIET_TIMEOUT=60
#          SKIP_GOVERNOR=1  SKIP_ACCURACY=1

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-5}"
CORES="${CORES:-1-6}"
MODEL="${MODEL:-moonshine-base}"
QUIET="${QUIET:-0.5}"                  # start only once 1-min loadavg is below this
QUIET_TIMEOUT="${QUIET_TIMEOUT:-60}"   # ...or give up waiting after this many seconds

BIN=./target/release/my-voice
SAMPLES=samples
EXPECTED="$SAMPLES/expected.txt"

# Pin to a fixed core set if taskset is available (away from core 0 / other load).
PIN=()
command -v taskset >/dev/null 2>&1 && PIN=(taskset -c "$CORES")

# Pin clocks high so runs don't throttle. Restore the original governor on exit
# so the box isn't left in performance mode. Needs sudo; warns (doesn't fail) if
# it can't — set SKIP_GOVERNOR=1 to opt out entirely.
orig_gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
if [[ -z "${SKIP_GOVERNOR:-}" && "$orig_gov" != "performance" && "$orig_gov" != "unknown" ]]; then
    if sudo cpupower frequency-set -g performance >/dev/null 2>&1; then
        trap 'sudo cpupower frequency-set -g "$orig_gov" >/dev/null 2>&1 || true' EXIT
    else
        echo "WARN: couldn't set governor to 'performance' (sudo cpupower failed) — clocks will float. See TIMING.md." >&2
    fi
fi

# Don't measure on a busy box: wait until 1-min loadavg settles below $QUIET
# (set QUIET=0 to skip). Bursty browser/editor load is the main jitter source.
if awk "BEGIN{exit !($QUIET > 0)}"; then
    waited=0
    while load1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"; \
          awk "BEGIN{exit !($load1 > $QUIET)}"; do
        if (( waited >= QUIET_TIMEOUT )); then
            echo "WARN: loadavg still $load1 (>$QUIET) after ${QUIET_TIMEOUT}s — measuring anyway, noisier." >&2
            break
        fi
        echo "waiting for box to settle: loadavg $load1 > $QUIET (${waited}/${QUIET_TIMEOUT}s)" >&2
        sleep 3; waited=$(( waited + 3 ))
    done
fi

echo "bench: model=$MODEL  iters=$ITERS  cores=$CORES ${PIN:+(pinned)}  gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"
cfg="$(mktemp --suffix=.toml)"
printf 'model = "%s"\n' "$MODEL" > "$cfg"
trap 'rm -f "$cfg"' EXIT

# ── Phase 1: accuracy (one deterministic run) ──────────────────────────────
wer_line=""
if [[ -z "${SKIP_ACCURACY:-}" ]]; then
    echo "── accuracy ──"
    acc="$(MY_VOICE_WER_MODEL="$MODEL" MY_VOICE_WER_QUIET=1 \
        "${PIN[@]}" cargo test --features debug-tools --test wer -- \
        --ignored --nocapture 2>&1)" || { echo "$acc" | tail -8; exit 1; }
    wer_line="$(grep -E '^aggregate WER' <<<"$acc" || true)"
    echo "${wer_line:-<no WER line>}"
fi

# ── Phase 2: warm perf + memory ────────────────────────────────────────────
echo "── perf (warm, $ITERS iters/sample) ──"
[[ -x "$BIN" ]] || { echo "build first: cargo build --release --features debug-tools" >&2; exit 1; }

min() { printf '%s\n' "$@" | sort -n | head -1; }

sum_enc=0; sum_dec=0; total_audio=0; max_rss=0
rtfs=()   # per-sample min RTF, for the across-sample distribution

while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"             # ltrim
    [[ -z "$line" || "$line" == \#* ]] && continue
    file="${line%%[[:space:]]*}"                          # first token = filename
    wav="$SAMPLES/$file"
    [[ -f "$wav" ]] || { echo "  skip missing $file" >&2; continue; }

    out="$("${PIN[@]}" "$BIN" -v --config "$cfg" --bench-iters "$ITERS" --wav "$wav" 2>&1)"
    mapfile -t encs < <(grep -oE 'encode [0-9]+' <<<"$out" | grep -oE '[0-9]+')
    mapfile -t decs < <(grep -oE 'decode [0-9]+' <<<"$out" | grep -oE '[0-9]+')
    if [[ ${#encs[@]} -eq 0 ]]; then
        echo "  $file: FAILED — no timings:" >&2; tail -3 <<<"$out" >&2; exit 1
    fi
    audio="$(grep -oE 'audio [0-9.]+' <<<"$out" | head -1 | grep -oE '[0-9.]+')"
    rss="$(grep -oE 'peak RSS [0-9]+' <<<"$out" | grep -oE '[0-9]+' || echo 0)"

    me="$(min "${encs[@]}")"; md="$(min "${decs[@]}")"
    sum_enc=$(( sum_enc + me )); sum_dec=$(( sum_dec + md ))
    (( rss > max_rss )) && max_rss="$rss"
    rtf="$(awk "BEGIN{printf \"%.3f\", ($me+$md)/1000/($audio>0?$audio:0.001)}")"
    rtfs+=("$rtf")
    total_audio="$(awk "BEGIN{printf \"%.2f\", $total_audio+$audio}")"
    printf "  %-22s %5.1fs  enc %4dms  dec %4dms  RTF %s\n" "$file" "$audio" "$me" "$md" "$rtf"
done < "$EXPECTED"

# Median / p95 of the per-sample RTFs (small N, nearest-rank).
mapfile -t sorted < <(printf '%s\n' "${rtfs[@]}" | sort -n)
n=${#sorted[@]}
p() { local q=$1; local idx; idx=$(awk "BEGIN{i=int(($q*$n)+0.999)-1; print (i<0?0:(i>=$n?$n-1:i))}"); echo "${sorted[$idx]}"; }
agg_rtf="$(awk "BEGIN{printf \"%.3f\", ($sum_enc+$sum_dec)/1000/($total_audio>0?$total_audio:0.001)}")"
xrt="$(awk "BEGIN{printf \"%.1f\", 1/($agg_rtf>0?$agg_rtf:0.001)}")"

echo "────────────────────────────────────────"
[[ -n "$wer_line" ]] && echo "$wer_line"
echo "warm encode ${sum_enc}ms + decode ${sum_dec}ms over ${total_audio}s — RTF ${agg_rtf} (${xrt}x realtime)"
echo "per-sample RTF: min ${sorted[0]}  median $(p 0.5)  p95 $(p 0.95)"
printf "peak RSS: %d kB (%.0f MB)\n" "$max_rss" "$(awk "BEGIN{print $max_rss/1024}")"
