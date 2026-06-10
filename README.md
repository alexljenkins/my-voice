# my-voice

Hold-to-talk local voice typing. English. Fast. Lean.

Push and hold **CapsLock**, speak, release — text appears in the focused window. No cloud, no GUI, no subscription. Everything runs locally on CPU.

## How it works

1. Hotkey held → mic opens.
2. Release → trailing 150ms captured, audio pre-processed (HPF + noise suppression + AGC via WebRTC APM), transcribed by [Moonshine](https://github.com/usefulsensors/moonshine) tiny quantized (~50 MB, ~10× real-time on CPU).
3. Text injected into the focused window via `wtype` / `ydotool` / `xdotool` (Linux) or CGEvent unicode injection (macOS). Hold **Shift+CapsLock** to copy to clipboard instead.

## Requirements

- Rust stable — install via [rustup](https://rustup.rs)
- Linux or macOS
- A working microphone
- `cmake` + a C++ toolchain **only** if building with `--features whisper` (optional)

## Install

```sh
cargo install --git https://github.com/alexljenkins/my-voice
my-voice --download   # fetch the default model (~50 MB)
my-voice              # start the daemon
```

Or from source:

```sh
git clone https://github.com/alexljenkins/my-voice
cd my-voice
cargo build --release
./target/release/my-voice --download
./target/release/my-voice
```

### Linux permissions

The daemon reads `/dev/input` and creates a uinput virtual keyboard for passthrough. On a fresh system you'll need:

```sh
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
sudo modprobe uinput
# to persist across reboots: add 'uinput' to /etc/modules-load.d/modules
# then log out and back in (or reboot)
```

Until you do this, the daemon falls back to **ungrabbed mode** — the hotkey still works but CapsLock also toggles the caps-lock LED/state as a side effect.

Without read access to `/dev/input`, you'll see:

```
no accessible keyboard reporting 'CapsLock'
Fix: sudo usermod -aG input $USER   (then re-login)
```

### macOS permissions

The daemon remaps CapsLock → F18 via `hidutil` (mapping is active only while the daemon runs; restored on exit). It creates a CGEvent tap to intercept F18 keystrokes.

Grant two permissions in **System Settings → Privacy & Security**:
- **Input Monitoring** — enable the terminal or binary
- **Accessibility** — enable the terminal or binary

If either permission is missing, the daemon prints an error and exits with instructions.

> **Note:** If the daemon crashes without a clean exit, the hidutil remap may stay active. Restore it manually:
> ```sh
> hidutil property --set '{"UserKeyMapping":[]}'
> ```

## Usage

```
my-voice                 # run the daemon (hold CapsLock to record)
my-voice --download      # fetch the configured model, then exit
my-voice --test          # record 3s from mic, transcribe, print, exit
my-voice --list-devices  # print audio input device names, exit
my-voice --config PATH   # use an alternate config file
my-voice -v / -vv        # info / debug logging (RUST_LOG also respected)
```

**Shift+CapsLock** copies to clipboard instead of typing — useful in terminals and apps where direct injection is unreliable (e.g. GNOME Wayland without ydotool).

## Configuration

Optional — defaults work out of the box. Create `~/.config/my-voice/config.toml` to override:

```toml
model = "moonshine-tiny"   # "moonshine-tiny" | "moonshine-base" | /path/to/dir | /path/to/file.gguf
model_dir = "~/.local/share/my-voice/models"
quantized = true           # prefer quantized .onnx files (smaller, faster, negligible WER cost)
threads = 0                # 0 = auto (min(cpu_count, 4))
load_timeout_secs = 300    # idle eviction; -1 = never unload, 0 = reload every use
hotkey = "CapsLock"        # evdev key name on Linux (macOS: only CapsLock in v1)
clipboard_hotkey = true    # Shift+hotkey → clipboard instead of typing
grab = true                # Linux: exclusive grab + uinput passthrough
audio_device = ""          # substring match against device name; "" = system default
min_speech_ms = 300        # discard holds shorter than this (accidental taps)
trailing_silence_ms = 150  # extra silence captured after release (catches last word tail)
injection = "auto"         # auto | wtype | xdotool | ydotool | clipboard
```

Unknown config keys warn and are ignored.

## Models

| Name | Download size (quantized) | CPU speed | Notes |
|---|---|---|---|
| `moonshine-tiny` | ~50 MB | ~10× real-time | Default; good for clear speech |
| `moonshine-base` | ~200 MB | ~4× real-time | Better for noisy mic or accents |

Models are fetched from HuggingFace (`onnx-community/moonshine-*-ONNX`) and cached in `~/.local/share/my-voice/models/`. Run `my-voice --download` after changing the `model` config.

### whisper.cpp backend (optional)

For `.gguf` model files, build with the whisper feature:

```sh
cargo build --release --features whisper
```

Then point `model` at a `.gguf` file:

```toml
model = "/path/to/ggml-base.en.gguf"
```

Use the `.en` (English-only) variants — they're faster and more accurate at the same size than the multilingual models. Recommended: `ggml-base.en.gguf` or `ggml-small.en.gguf` from [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp).

Requires `cmake` and a C++ toolchain at build time only.

## Troubleshooting

**GNOME Wayland — text doesn't appear:**
`wtype` fails on GNOME (compositor lacks the virtual-keyboard protocol). The daemon automatically falls back to the next method. Install `ydotool` for universal injection:

```sh
sudo apt install ydotool
ydotoold &   # start the ydotool daemon (add to autostart)
```

Or use clipboard mode: `injection = "clipboard"` in config, then paste with Ctrl+V.

**Wrong microphone:**
Run `my-voice --list-devices` and set `audio_device = "Headset"` (case-insensitive substring match) in config.

**Keyboard unresponsive after crash (Linux):**
The evdev grab is released when file descriptors close (normal exit or SIGTERM). A hard `kill -9` skips this — the kernel holds the grab until the FD is garbage-collected, which typically happens within seconds. If the keyboard stays stuck, switch to a TTY with Ctrl+Alt+F2 and run:

```sh
pkill my-voice
```

**Hotplug:**
If your keyboard disconnects and reconnects, restart the daemon. Hotplug detection is out of scope for v1.

**Model not found:**
```
model file missing: /path/to/model — run: my-voice --download
```
Run `my-voice --download` to fetch the configured model.

## Development

```sh
cargo test                         # unit tests (no audio/model/network required)
cargo clippy -- -D warnings        # lint
cargo fmt                          # format
cargo build --features whisper     # opt-in backend (needs cmake)
```

Tests are pure-logic only (resample math, config round-trip, quote normalization, key-name parsing, max-token clamping, injection chain selection). No microphone, model files, or network access needed in CI.
