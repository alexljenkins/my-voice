#!/usr/bin/env bash
# The standard validator: per-sample accuracy (WER) + warm perf (time/mem/CPU)
# for every model, in ONE inference pass per sample. Run it before and after an
# optimization and diff the tables to check for regressions. See ../TIMING.md.
#
# Single pass: each sample is transcribed once (the model is loaded+warmed once,
# then --bench-iters re-times the warm steady state). That same run yields BOTH
# the text — printed to stdout → WER vs samples/expected.txt — and the timings —
# logged to stderr → enc/dec/RTF/RSS, plus whole-process CPU% from GNU time. No
# separate accuracy run, so no model is transcribed twice.
#
# Output (terminal + $RESULTS): a per-model table (one row per sample, then an
# AGGREGATE row) and a final cross-model SUMMARY table. $RESULTS is overwritten.
#
# Usage:   ./tools/bench-wer.sh
# Env:     ITERS=5  CORES=1-6  QUIET=0.5  QUIET_TIMEOUT=30  RESULTS=tools/zz_results.txt
#          MODELS="m1 m2 ..."   MODEL=<one>   SKIP_GOVERNOR=1

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-5}"
CORES="${CORES:-1-6}"
QUIET="${QUIET:-0.5}"                   # start only once 1-min loadavg is below this
QUIET_TIMEOUT="${QUIET_TIMEOUT:-30}"    # ...or give up waiting after this many seconds
RESULTS="${RESULTS:-tools/zz_results.txt}"
MODELS="${MODELS:-moonshine-base moonshine-streaming-small moonshine-streaming-medium}"
[[ -n "${MODEL:-}" ]] && MODELS="$MODEL"   # MODEL=<one> overrides the whole list

BIN=./target/release/my-voice
SAMPLES=samples
EXPECTED="$SAMPLES/expected.txt"

[[ -x "$BIN" ]] || { echo "build first: cargo build --release --features debug-tools" >&2; exit 1; }
[[ -f "$EXPECTED" ]] || { echo "missing $EXPECTED" >&2; exit 1; }

# Pin to a fixed core set, and wrap each run in GNU time for whole-process CPU%
# (summed across cores: 600% = 6 cores busy). Both optional, degrade cleanly.
PIN=();  command -v taskset       >/dev/null 2>&1 && PIN=(taskset -c "$CORES")
GNUTIME=""; command -v /usr/bin/time >/dev/null 2>&1 && GNUTIME=/usr/bin/time

# Pin clocks high so runs don't throttle (sudo; warns, doesn't fail, if it can't
# — set SKIP_GOVERNOR=1 to opt out). The original governor is restored on exit.
orig_gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
if [[ -z "${SKIP_GOVERNOR:-}" && "$orig_gov" != "performance" && "$orig_gov" != "unknown" ]]; then
    if sudo cpupower frequency-set -g performance >/dev/null 2>&1; then
        restore_gov=1
    else
        echo "WARN: couldn't set governor to 'performance' (sudo cpupower failed) — clocks will float. See TIMING.md." >&2
    fi
fi

cfg="$(mktemp --suffix=.toml)"; tf="$(mktemp)"; errf="$(mktemp)"
cleanup() {
    rm -f "$cfg" "$tf" "$errf"
    [[ -n "${restore_gov:-}" ]] && sudo cpupower frequency-set -g "$orig_gov" >/dev/null 2>&1 || true
}
trap cleanup EXIT

TIME=(); [[ -n "$GNUTIME" ]] && TIME=("$GNUTIME" -v -o "$tf")

# Don't measure on a busy box: wait (quietly) up to $QUIET_TIMEOUT for the 1-min
# loadavg to fall below $QUIET. Only the first and final states print (QUIET=0
# skips). Bursty browser/editor load is the main jitter source.
if awk "BEGIN{exit !($QUIET > 0)}"; then
    load1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
    if awk "BEGIN{exit !($load1 > $QUIET)}"; then
        echo "settling: loadavg $load1 > $QUIET — waiting up to ${QUIET_TIMEOUT}s..." >&2
        waited=0
        while awk "BEGIN{exit !($load1 > $QUIET)}" && (( waited < QUIET_TIMEOUT )); do
            sleep 3; waited=$(( waited + 3 ))
            load1="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
        done
        if awk "BEGIN{exit !($load1 > $QUIET)}"; then
            echo "WARN: loadavg still $load1 (>$QUIET) after ${QUIET_TIMEOUT}s — measuring anyway, noisier." >&2
        else
            echo "settled: loadavg $load1 after ${waited}s" >&2
        fi
    fi
fi

# ── helpers ──────────────────────────────────────────────────────────────────
min() { printf '%s\n' "$@" | sort -n | head -1; }

# Word Error Rate vs a reference, mirroring tests/wer.rs exactly:
#   normalized — lowercase, then per whitespace-split word keep only
#                alphanumerics + apostrophes (the gated, text-blind score);
#   strict     — whitespace-split only, case + punctuation preserved.
# Prints: norm_errs norm_words strict_errs strict_words norm_wer strict_wer
wer_calc() {
    ref="$1" hyp="$2" awk '
    function lev(a,na,b,nb,   i,j,cost,t) {
        for (j=0;j<=nb;j++) P[j]=j
        for (i=1;i<=na;i++) {
            C[0]=i
            for (j=1;j<=nb;j++) {
                cost=P[j-1]+(a[i]!=b[j]?1:0)
                t=P[j]+1;   if (t<cost) cost=t
                t=C[j-1]+1; if (t<cost) cost=t
                C[j]=cost
            }
            for (j=0;j<=nb;j++) P[j]=C[j]
        }
        return P[nb]
    }
    BEGIN {
        apos=sprintf("%c",39); cls="[^a-z0-9" apos "]"
        ref=ENVIRON["ref"]; hyp=ENVIRON["hyp"]
        rn=split(ref,p,/[ \t\n\r]+/); k=0
        for(i=1;i<=rn;i++){w=tolower(p[i]); gsub(cls,"",w); if(w!="")RA[++k]=w}; rn=k
        hn=split(hyp,p,/[ \t\n\r]+/); k=0
        for(i=1;i<=hn;i++){w=tolower(p[i]); gsub(cls,"",w); if(w!="")HA[++k]=w}; hn=k
        rs=split(ref,p,/[ \t\n\r]+/); k=0
        for(i=1;i<=rs;i++){if(p[i]!="")SA[++k]=p[i]}; rs=k
        hs=split(hyp,p,/[ \t\n\r]+/); k=0
        for(i=1;i<=hs;i++){if(p[i]!="")SB[++k]=p[i]}; hs=k
        ne=lev(RA,rn,HA,hn); se=lev(SA,rs,SB,hs)
        printf "%d %d %d %d %.3f %.3f\n", ne, rn, se, rs, ne/(rn>0?rn:1), se/(rs>0?rs:1)
    }'
}

# Mirror every report line to the terminal and to $RESULTS.
say()  { printf '%s\n' "$*" | tee -a "$RESULTS"; }
sayf() { printf "$@" | tee -a "$RESULTS"; }

ROW_H="  %-22s %7s %7s %7s %6s %6s %5s %6s %6s\n"
ROW_F="  %-22s %6.1fs %5dms %5dms %6.3f %4dMB %4d%% %6.3f %6.3f\n"

# ── run ──────────────────────────────────────────────────────────────────────
: > "$RESULTS"
say "bench-wer  $(date '+%Y-%m-%d %H:%M:%S')  iters=$ITERS  cores=$CORES ${PIN:+(pinned)}  gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"

s_name=(); s_rss=(); s_encdec=(); s_rtf=(); s_wer=(); s_strict=()

for model in $MODELS; do
    printf 'model = "%s"\n' "$model" > "$cfg"
    say ""
    say "── $model ──"
    sayf "$ROW_H" sample audio enc dec RTF RSS CPU WER strict

    sum_enc=0; sum_dec=0; total_audio=0; max_rss=0; max_cpu=0
    tne=0; trn=0; tse=0; trs=0

    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"              # ltrim
        [[ -z "$line" || "$line" == \#* ]] && continue
        file="${line%%[[:space:]]*}"                           # first token = filename
        ref="${line#"$file"}"; ref="${ref#"${ref%%[![:space:]]*}"}"   # rest = reference text
        wav="$SAMPLES/$file"
        [[ -f "$wav" ]] || { echo "  skip missing $file" >&2; continue; }

        # One run: stdout → hypothesis text; stderr → timings/RSS; $tf → CPU%.
        hyp="$("${TIME[@]}" "${PIN[@]}" "$BIN" -v --config "$cfg" --bench-iters "$ITERS" --wav "$wav" 2>"$errf")"
        mapfile -t encs < <(grep -oE 'encode [0-9]+' "$errf" | grep -oE '[0-9]+')
        mapfile -t decs < <(grep -oE 'decode [0-9]+' "$errf" | grep -oE '[0-9]+')
        if [[ ${#encs[@]} -eq 0 ]]; then
            echo "  $file: FAILED — no timings:" >&2; tail -3 "$errf" >&2; exit 1
        fi
        audio="$(grep -oE 'audio [0-9.]+' "$errf" | head -1 | grep -oE '[0-9.]+')"
        rss="$(grep -oE 'peak RSS [0-9]+' "$errf" | grep -oE '[0-9]+' || echo 0)"
        cpu="$(grep -oE 'Percent of CPU this job got: [0-9]+' "$tf" 2>/dev/null | grep -oE '[0-9]+$' || echo 0)"

        me="$(min "${encs[@]}")"; md="$(min "${decs[@]}")"
        read -r ne rn se rs wer strict < <(wer_calc "$ref" "$hyp")

        sum_enc=$(( sum_enc + me )); sum_dec=$(( sum_dec + md ))
        (( rss > max_rss )) && max_rss="$rss"
        (( ${cpu:-0} > max_cpu )) && max_cpu="${cpu:-0}"
        tne=$(( tne + ne )); trn=$(( trn + rn )); tse=$(( tse + se )); trs=$(( trs + rs ))
        total_audio="$(awk "BEGIN{printf \"%.2f\", $total_audio+$audio}")"
        rtf="$(awk "BEGIN{printf \"%.3f\", ($me+$md)/1000/($audio>0?$audio:0.001)}")"
        mb="$(awk "BEGIN{printf \"%.0f\", $rss/1024}")"

        sayf "$ROW_F" "$file" "$audio" "$me" "$md" "$rtf" "$mb" "${cpu:-0}" "$wer" "$strict"
    done < "$EXPECTED"

    agg_rtf="$(awk "BEGIN{printf \"%.3f\", ($sum_enc+$sum_dec)/1000/($total_audio>0?$total_audio:0.001)}")"
    agg_wer="$(awk "BEGIN{printf \"%.3f\", $tne/($trn>0?$trn:1)}")"
    agg_strict="$(awk "BEGIN{printf \"%.3f\", $tse/($trs>0?$trs:1)}")"
    mb="$(awk "BEGIN{printf \"%.0f\", $max_rss/1024}")"
    say "  ----------------------------------------------------------------------------"
    sayf "$ROW_F" "AGGREGATE" "$total_audio" "$sum_enc" "$sum_dec" "$agg_rtf" "$mb" "$max_cpu" "$agg_wer" "$agg_strict"

    s_name+=("$model"); s_rss+=("$mb"); s_encdec+=("$(( sum_enc + sum_dec ))")
    s_rtf+=("$agg_rtf"); s_wer+=("$agg_wer"); s_strict+=("$agg_strict")
done

# ── cross-model summary ──────────────────────────────────────────────────────
say ""
say "── SUMMARY (peak RSS, total enc+dec, aggregate RTF/WER) ──"
sayf "  %-26s %6s %9s %7s %7s %7s %7s\n" model RSS enc+dec RTF xRT WER strict
for i in "${!s_name[@]}"; do
    xrt="$(awk "BEGIN{printf \"%.1f\", 1/(${s_rtf[$i]}>0?${s_rtf[$i]}:0.001)}")"
    sayf "  %-26s %4dMB %7dms %7.3f %6.1fx %7.3f %7.3f\n" \
        "${s_name[$i]}" "${s_rss[$i]}" "${s_encdec[$i]}" "${s_rtf[$i]}" "$xrt" "${s_wer[$i]}" "${s_strict[$i]}"
done
say ""
say "wrote $RESULTS"
