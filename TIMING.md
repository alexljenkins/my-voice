# The measurement standard

How to measure **accuracy**, **transcription time**, and **memory** cleanly
enough to compare PR-to-PR and model-to-model. The harness is split so each
dimension is measured the way that dimension actually behaves.

```sh
sudo cpupower frequency-set -g performance   # once per session — see "noise" below
cargo build --release --features debug-tools
./tools/bench-wer.sh
```

Knobs: `ITERS=5 MODEL=moonshine-base CORES=1-6 QUIET=0.5 ./tools/bench-wer.sh`
(`SKIP_ACCURACY=1` for perf only, `SKIP_GOVERNOR=1` to leave the governor alone).

## What it reports

```
aggregate WER 0.077 (13/169 words), strict WER 0.107 (18/169 words)
warm encode 729ms + decode 3548ms over 71.51s — RTF 0.060 (16.7x realtime)
per-sample RTF: min 0.043  median 0.062  p95 0.116
peak RSS: 334732 kB (327 MB)
```

- **WER** — normalized (lowercased, punctuation-stripped) word error rate. The
  gate. **strict WER** preserves case + punctuation, so text-quality changes
  (capitalization, commas, ITN) are visible even when normalized WER is flat.
- **warm encode/decode + RTF** — pure model time on a warmed model, the cost a
  daemon user feels after the first dictation. `Nx realtime` = how many seconds
  of audio it processes per second of compute.
- **per-sample RTF min/median/p95** — the spread across sample *types* (short
  greetings vs long-form vs numbers). p95 = the slowest a user is likely to hit.
- **peak RSS** — process memory high-water mark (`VmHWM`): model + ONNX arenas +
  buffers = the daemon's real footprint.

## Why it's split into two phases

**Accuracy and performance don't behave the same, so they aren't measured the
same.**

- **Accuracy = one `cargo test` run.** Greedy decode is deterministic — WER is
  byte-identical run to run. One run is the whole truth, and it stays a `cargo
  test` so it gates in CI (asserts `WER ≤ MY_VOICE_WER_MAX`, default 0.25).
- **Perf = the release binary driven directly**, because timing is
  environment-dependent and must never be a pass/fail assertion (box variance
  makes that flaky). It loads + warms the model **once per sample**, then
  re-times N warm passes and keeps the **min** — the pass least disturbed by box
  load is the closest thing to the code's true cost.

This is also why perf doesn't re-run `cargo test` N times like the old script:
that re-paid model load on every measurement (~60 loads for 5×12). Now it's one
load per sample (~12), and the repeats are pure warm inference.

## What the harness already excludes, and the noise that's left

The `encode`/`decode` numbers come from `Instant` spans *inside*
`Moonshine::transcribe` (`src/transcriber/moonshine.rs`) — they exclude process
spawn, ONNX init, model load, WAV I/O, and resampling. So fixed box overhead is
already out. What's left is **variance**, from three sources:

1. **CPU governor.** This box defaults to `powersave`, so core clock floats with
   load — back-to-back runs of identical code execute at different MHz (a single
   `bench-wer.sh` run swung encode 1284ms→729ms purely from this). The
   `cpupower ... performance` line above is the biggest single fix; the script
   sets it when it can `sudo` and restores your original governor on exit.
2. **Box contention.** The script pins to a fixed core set (`taskset -c 1-6`,
   off core 0 where IRQs land) and **waits until 1-min loadavg drops below
   `QUIET`** before measuring, so it won't run on a busy box.
3. **Cold-start.** Handled by the model warming once at load
   (`model_cache.rs`), so every timed pass is warm.

The within-sample `min` absorbs whatever variance survives.

## Establishing a baseline

Run it on `main` and record the four headline numbers — that's the standard a
change is measured against. A change is an improvement if WER/strict don't
regress and warm RTF or peak RSS drops (or it trades one for another
deliberately). For a model swap, watch per-sample RTF *and* the per-category
spread, not just the aggregate.

## Possible next steps (not built)

- **Per-category accuracy** — tag `samples/expected.txt` with `short`/`long`/
  `numbers`/`dates` and aggregate WER per category, to see *where* a change
  helped or hurt. Highest-value accuracy upgrade for model swaps.
- **S/I/D breakdown** — backtrace the edit-distance DP for
  substitution/insertion/deletion counts (deletions = dropped words, insertions
  = hallucination).
- **Felt latency** — also time resample + `process_capture` + `post_process`
  around the model call, for the full press-to-text number.
