# Audio Pre-Processing Evaluation Harness

Dev tool for benchmarking capture quality. Does not require a running daemon.

## 1. Build release binary

```bash
cargo build --release
```

## 2. Record a single sample

Records 5 seconds from the default mic, applies the full pipeline (resample → WebRTC APM → peak-normalize), and writes two files:

- `samples/sample_01.wav` — processed 16 kHz mono (feed this to the transcriber)
- `samples/sample_01_raw.wav` — raw native-rate mono (before any processing)

```bash
mkdir -p samples
./target/release/my-voice --record samples/sample_01.wav --duration 5
```

Output printed to stdout:
```
duration:  5.00s
peak:      0.9423
processed: samples/sample_01.wav
raw:       samples/sample_01_raw.wav
```

## 3. Record multiple samples

```bash
mkdir -p samples
for i in 01 02 03 04 05; do
    echo "Recording sample_${i}.wav — speak now..."
    ./target/release/my-voice --record samples/sample_${i}.wav --duration 5
    sleep 1
done
```

## 4. Write a labels file

`labels.txt` — one line per file, filename and expected text separated by a **tab**:

```
sample_01.wav	the quick brown fox jumps over the lazy dog
sample_02.wav	hello world this is a test
sample_03.wav	open the pod bay doors hal
```

No header line. Filenames are basenames only (no path). Lines without a tab are skipped.

## 5. Run the evaluation

```bash
./tools/eval.sh samples/ labels.txt
```

Example output:

```
| File | Expected | Got | Match |
|------|----------|-----|-------|
| sample_01.wav | the quick brown fox jumps over the lazy dog | the quick brown fox jumps over the lazy dog | ✓ |
| sample_02.wav | hello world this is a test | hello world this is a test | ✓ |
| sample_03.wav | open the pod bay doors hal | open the pod bay doors hal | ✓ |

## Summary

Evaluated: 3 files
Correct:   3
Accuracy:  100%
```

Match comparison is case-insensitive and trims leading/trailing whitespace. Files in the directory that have no entry in `labels.txt` are listed separately as unlabeled.

## Using a specific audio device

```bash
./target/release/my-voice --list-devices
./target/release/my-voice --record samples/sample_01.wav --duration 5 --config /path/to/config.toml
```

Or set `audio_device` in `~/.config/my-voice/config.toml` before recording.
