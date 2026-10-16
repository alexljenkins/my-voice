# Personal recording tests

Use Alex's recordings in `samples/my-samples/` for accuracy checks. Git tracks only `expected.txt`, the corrected transcripts. WAV files stay local.

## Run the checks

```sh
cargo test
cargo test --features debug-tools
cargo test --features debug-tools --test wer -- --ignored --nocapture
cargo clippy -- -D warnings
```

The accuracy test compares both whole and segmented transcription against the same references. It fails above 2% word error rate. It requires the local WAV files and downloaded `moonshine-base` model. Missing files fail the check instead of reducing the dataset silently.

## Save a comparison report

```sh
cargo build --release --features debug-tools
RESULTS=docs/reviews/latest.txt SKIP_GOVERNOR=1 ./tools/bench-wer.sh
```

The default model is `moonshine-base:int8`. Reports include each reference, whole transcript, boundary comparison, merged transcript, error counts, timings and peak memory. Reports stay local under the ignored `docs/` directory.

Use separate paths when comparing versions. The script replaces its selected report file.

```sh
RESULTS=docs/reviews/v1.txt SKIP_GOVERNOR=1 ./tools/bench-wer.sh
```

Options are `ITERS`, `CORES`, `MODEL`, `MODELS`, `SAMPLES`, `RESULTS`, and `SKIP_GOVERNOR`. The accuracy test accepts `MY_VOICE_WER_SAMPLES`, `MY_VOICE_WER_MODEL`, `MY_VOICE_WER_QUANTIZED`, and `MY_VOICE_WER_MAX`.

## Check Parakeet v2 F16

Use the existing GGUF file directly with the same accuracy gate:

```sh
MY_VOICE_WER_MODEL=/home/alex/code/MODELS/parakeet-tdt-0.6b-v2-F16.gguf \
  cargo test --features debug-tools --test wer -- --ignored --nocapture
```

The 2026-10-16 CUDA check passed all 30 recordings. Whole and segmented transcription both made 14/821 normalized errors, or 1.71% WER.
Both made 49/821 strict errors. The CPU smoke check also transcribed a recording successfully, with 1569 MiB peak process RAM.
The normal build uses CPU inference. The `cuda` feature builds NVIDIA acceleration when the toolkit is available.

## Add recordings

```sh
./target/release/my-voice --record samples/my-samples/
```

Quit the running daemon from its tray menu before starting recording mode. Hold the push-to-talk key, speak, then release. Each hold saves one raw WAV and appends the model transcript to both `expected.txt` and `transcribed.txt`. Correct only `expected.txt` before using it as a reference. Keep `transcribed.txt` as the original model output.

Both files append new entries and preserve existing text. Recording mode plays each saved raw WAV after transcription finishes. It stops playback when you start the next recording. Playback uses PipeWire's `pw-play` command. Stop the recorder with Ctrl+C.

To turn playback off:

```sh
./target/release/my-voice --record samples/my-samples/ --no-record-playback
```

Include recordings longer than 60 seconds to evaluate the 50–60 second split window. Existing short recordings check release behavior but cannot measure long boundary accuracy.

## Inspect one recording

```sh
./target/release/my-voice --wav samples/my-samples/1788393118159_1_raw.wav
./target/release/my-voice --wav samples/my-samples/1788393118159_1_raw.wav --segmented
```

The segmented command uses the daemon's split policy. Its stderr includes boundary times and `| old || new |` comparisons. Those markers are diagnostic text, not the delivered transcript.
