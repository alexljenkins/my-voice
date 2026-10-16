# Parakeet audio processing experiment

Base: `de2958d`. All inference uses the app's Parakeet Rust backend and the pinned CUDA runtime.
The model, decoding, app text processing, empty corrections, references, and scorer stay fixed.

## Acceptance

Keep processing only if normalized errors decrease across all 30 recordings and neither cohort regresses.
Strict errors are secondary. A strict-only improvement does not qualify.
Check whole recordings and daemon segmentation before selecting a winner.
If no processing qualifies, use raw resampled audio for Parakeet. Preserve Moonshine processing.

## Variants

- `raw` passes the resampled waveform unchanged.
- `current` uses moderate WebRTC noise suppression, adaptive gain, peak normalization, and trimming.
- `peak` applies only the existing peak normalization. This tests level correction without adaptive processing.
- `peak-trim` adds the existing silence trimming to `peak`. This tests silence removal without WebRTC.

The temporary `experiment.patch` installs a measurement hook at both processing entry points.
It is not production configuration. All variants use the same binary.
`worker.py` passes only the audio path and fixed config to inference. References stay in the scorer.
Each invocation performs 2 passes. The second supplies the scored transcript. Timings retain both passes.
The segmented path uses the daemon's 200 ms cadence, overlap processing, gates, and text joiner.

## Reproduce

Run screening from base `de2958d` with this experiment directory copied into that checkout.
The final branch selects raw audio before the temporary hook, so screen variants on the base.
Source WAVs remain read-only in the parent checkout.
The scripts reject duplicate or changed references, missing recordings, and changed WAV hashes.
Use a fresh `EXPERIMENT_RUN` directory for each run.

```bash
git apply docs/experiments/parakeet-audio/experiment.patch
TRANSCRIBE_DIR=/home/alex/code/MODELS/asr-eval/transcribe-install \
LD_LIBRARY_PATH=/home/alex/code/MODELS/asr-eval/transcribe-install/lib \
PATH=/home/alex/code/MODELS/asr-eval/venv/bin:$PATH \
CARGO_TARGET_DIR=$PWD/target/parakeet-audio \
cargo build --locked --release --features debug-tools -j 4
python3 docs/experiments/parakeet-audio/run.py
```

The supervisor polls every 0.5 seconds. It retains the prior RAM, swap-out, memory-pressure, and GPU guards.
The workload has 4 GiB RAM, zero swap, 4 cores, and a 400% CPU quota.
GPU preflight requires 3 GiB free. Execution stops below the 1 GiB reserve.
Only 1 inference process runs at a time. No desktop processes are stopped.
Raw events, stderr, RSS, resource samples, source hashes, and commands remain beside this file.

## Final app check

Build the final branch without the temporary patch. Then run:

```bash
EXPERIMENT_RUN=final EXPERIMENT_VARIANTS=raw python3 docs/experiments/parakeet-audio/run.py
python3 docs/experiments/parakeet-audio/analyze.py final
```

The final app ignores `MY_VOICE_AUDIO_EXPERIMENT`. Its model choice selects raw Parakeet audio.

## Evidence archives

`screen/` has 240 scored invocations. `final/` has 60 final-app invocations.
Each invocation retains 2 inference passes. Diagnostic archives preserve exact stderr and RSS files.
Extract an archive when inspecting an event's `stderr` field:

```bash
tar -xzf docs/experiments/parakeet-audio/screen/diagnostics.tar.gz -C docs/experiments/parakeet-audio/screen
gzip -cd docs/experiments/parakeet-audio/screen-resources.jsonl.gz
```
