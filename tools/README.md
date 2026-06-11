# Audio Pre-Processing Evaluation Harness

Dev tool for benchmarking capture quality. Requires the binary but no running daemon.

## 1. Build release binary

```bash
cargo build --release --features debug-tools
```

## 2. Record samples

`--record <DIR>` runs the normal PTT daemon and saves every utterance to `<DIR>/<timestamp>.wav`
(and `<timestamp>_raw.wav`). Hold CapsLock, speak, release — repeat as many times as you want.
Press **Ctrl+C** when done.

```bash
mkdir -p samples
./target/release/my-voice --record samples/
```

Each completed hold-to-talk produces:
- `<timestamp>.wav` — processed 16 kHz mono (what goes to the transcriber)
- `<timestamp>_raw.wav` — raw native-rate mono (before the APM pipeline)

## 3. Transcribe and compare

Transcribe a single wav file directly (bypasses the mic, requires a downloaded model):

```bash
# build with debug-tools to get --wav
./target/release/my-voice --wav samples/1234567890.wav
```

## 4. Write a labels file

`labels.txt` — one line per file, tab-separated:

```
1234567890.wav	the quick brown fox jumps over the lazy dog
1234567891.wav	hello world this is a test
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
| 1234567890.wav | the quick brown fox jumps over the lazy dog | the quick brown fox jumps over the lazy dog | ✓ |
| 1234567891.wav | hello world this is a test | hello world this is a test | ✓ |

## Summary

Evaluated: 2 files
Correct:   2
Accuracy:  100%
```

Match comparison is case-insensitive and trims leading/trailing whitespace. Files with no
entry in `labels.txt` are listed separately as unlabeled.

## Using a specific audio device

```bash
./target/release/my-voice --list-devices
./target/release/my-voice --record samples/ --config /path/to/config.toml
```

Or set `audio_device` in `~/.config/my-voice/config.toml` before recording.
