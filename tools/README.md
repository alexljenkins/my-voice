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

## Add recordings

```sh
./target/release/my-voice --record samples/my-samples/
```

Hold the push-to-talk key, speak, then release. Each hold saves one raw WAV and appends a model transcript to `expected.txt`. Listen to the recording and correct that transcript before using it as a reference. Stop the recorder with Ctrl+C.

Include recordings longer than 60 seconds to evaluate the 50–60 second split window. Existing short recordings check release behavior but cannot measure long boundary accuracy.

## Inspect one recording

```sh
./target/release/my-voice --wav samples/my-samples/1788393118159_1_raw.wav
./target/release/my-voice --wav samples/my-samples/1788393118159_1_raw.wav --segmented
```

The segmented command uses the daemon's split policy. Its stderr includes boundary times and `| old || new |` comparisons. Those markers are diagnostic text, not the delivered transcript.
