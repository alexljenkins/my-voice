# my-voice

Hold-to-talk local voice typing. English. Fast. Lean.

> **Status: Phase 1.** The daemon captures audio while the hotkey is held and
> writes a playable WAV on release. There is **no transcription and no text
> injection yet** — those land in Phase 2+. Use this build to verify the audio
> and hotkey paths.

## Requirements

- Rust toolchain (stable) — install via [rustup](https://rustup.rs)
- Linux or macOS
- A working microphone

## Build

```sh
cargo build            # debug
cargo build --release  # optimized; use this for real hotkey latency
```

## Quickstart

The fastest way to confirm the app works — no hotkey permissions needed:

```sh
# 1. See what input devices are detected
cargo run -- --list-devices

# 2. Record a fixed 3-second clip and dump a WAV
cargo run -- --test
```

`--test` records 3s from the default mic, prints duration / sample rate / peak
level, and writes the clip to `/tmp/my-voice-test.wav`. Play it back to confirm
capture works:

```sh
# duration, rate, peak printed to stdout; e.g.
# 3.00s @ 16000 Hz, peak 0.184 → /tmp/my-voice-test.wav

aplay /tmp/my-voice-test.wav     # Linux
afplay /tmp/my-voice-test.wav    # macOS
```

A non-trivial `peak` (well above `0.000`) means the mic is being captured. Near
silence means the wrong device or a muted mic — check `--list-devices`.

## Running the daemon (hotkey mode)

```sh
cargo run --release
```

This starts the push-to-talk daemon. **Hold `CapsLock`** to record; release to
stop. Each utterance is logged and written to `/tmp/my-voice-test.wav` (Phase 2
will transcribe and type instead).

### Linux permissions

The hotkey path reads `/dev/input` and creates a uinput virtual keyboard. On a
fresh box you'll lack permission and the daemon falls back to ungrabbed mode (or
finds no keyboard). To grant access:

```sh
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
sudo modprobe uinput
# then log out and back in
```

If you can't set this up (e.g. a CI/dev box with no input group), stick to
`--test` — it exercises the full audio path without touching the keyboard.

## Flags

| Flag | Effect |
|------|--------|
| `--test` | Record 3s, write a WAV, print stats, exit. No hotkey needed. |
| `--list-devices` | Print input device names and exit. |
| `--download` | Fetch the model — **not available until Phase 2.** |
| `--config <PATH>` | Use an alternate config file. |
| `-v` / `-vv` | Increase logging (`info` / `debug`). |

## Configuration

Optional. Defaults are baked in; no config file is required. To override, create
`~/.config/my-voice/config.toml`:

```toml
hotkey = "CapsLock"      # push-to-talk key
grab = true              # exclusively grab the key (uinput passthrough)
audio_device = ""        # empty = system default
trailing_silence_ms = 150
min_speech_ms = 300
```

Unknown keys warn and are ignored.

## Test

```sh
cargo test
```
