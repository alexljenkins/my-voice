# my-voice

Hold **CapsLock**, speak, release — your words appear in whatever app is focused. No cloud, no subscription. Everything runs locally on your computer.

## Requirements

- A working microphone
- Linux or macOS
- [Rust](https://rustup.rs) — the build tool (one install, then mostly forgotten)

---

## Install

### Linux

```sh
# 1. Install Rust (skip if you already have it)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. Install my-voice
cargo install --git https://github.com/alexljenkins/my-voice

# 3. Download the voice model (~660 MB default, one-time)
my-voice --download

# 4. Start
my-voice
```

A mic icon will appear in your system tray. Right-click it to change your microphone, model, and other settings.

**New to Linux or hitting permission errors?** You may need to grant keyboard access first:

```sh
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
sudo modprobe uinput
```

Then **log out and back in**, and run `my-voice` again.

> Without this step, my-voice still works — but CapsLock will also toggle the caps-lock state as a side-effect.

<details>
<summary>Linux — more details and troubleshooting</summary>

### What the permissions do

- `input` group — lets my-voice read keyboard events from `/dev/input`
- `udev` rule — gives the group write access to the `uinput` virtual keyboard
- `uinput` module — kernel module for creating virtual input devices (loaded automatically on most distros)

To persist `uinput` across reboots, add it to your modules list:
```sh
echo 'uinput' | sudo tee -a /etc/modules-load.d/modules.conf
```

### Text doesn't appear (GNOME / Wayland)

`wtype` doesn't work on GNOME Wayland. my-voice automatically falls back to **AT-SPI** (GNOME's accessibility bus), which is on by default and needs no setup.

If AT-SPI is disabled or a specific app ignores it, install `ydotool`:

```sh
sudo apt install ydotool
ydotoold &   # start the daemon (add to autostart)
```

Or switch to clipboard mode via the **Paste mode** submenu in the tray, then paste with Ctrl+V.

### Keyboard unresponsive after a crash

A hard `kill -9` can leave the keyboard grabbed. The kernel releases it automatically within a few seconds. If it stays stuck:

```sh
pkill my-voice
```

Or switch to a TTY with **Ctrl+Alt+F2** and run the command there.

### Wrong microphone selected

The tray **Microphone** submenu lists all detected input devices — click the one you want.

Alternatively, run `my-voice --list-devices` to see device names and set `audio_device = "Headset"` (or any substring) in the config file.

### Model not found

```
model file missing — run: my-voice --download
```

Run `my-voice --download` to fetch the configured model.

</details>

---

### macOS

```sh
# 1. Install Rust (skip if you already have it)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. Install my-voice
cargo install --git https://github.com/alexljenkins/my-voice

# 3. Download the voice model (~660 MB default, one-time)
my-voice --download

# 4. Start
my-voice
```

macOS will ask for two permissions — grant both:

1. Open **System Settings → Privacy & Security → Input Monitoring** — find your terminal and enable it
2. Open **System Settings → Privacy & Security → Accessibility** — find your terminal and enable it

Then run `my-voice` again. Use **Ctrl+C** in the terminal to stop it.

> **Note:** The macOS tray icon is not yet available. All settings are managed via the [config file](#configuration) for now.

<details>
<summary>macOS — more details and troubleshooting</summary>

### How the hotkey works on macOS

The daemon remaps CapsLock → F18 via `hidutil` while it's running. The original CapsLock behavior is restored when you quit. If the app crashes without a clean exit, restore it manually:

```sh
hidutil property --set '{"UserKeyMapping":[]}'
```

### Permissions not sticking

If you granted permissions but the daemon still exits with an error, try:
1. Quitting and reopening your terminal
2. Removing and re-adding the permission entry in System Settings
3. Running `my-voice` directly from `/usr/local/bin/my-voice` instead of through a shell wrapper

### Wrong microphone

Run `my-voice --list-devices` to see device names, then set `audio_device = "Headset"` (or any substring match) in the [config file](#configuration).

</details>

---

## Usage

| Action | Result |
|---|---|
| Hold **CapsLock** → speak → release | Text typed into focused window |
| Hold **Shift+CapsLock** → speak → release | Text copied to clipboard |

On Linux, right-click the tray icon to adjust model, microphone, paste mode, and startup settings — no config file needed.

---

<details>
<summary>Configuration file (optional)</summary>

Defaults work out of the box. To override, create `~/.config/my-voice/config.toml`:

```toml
model = "moonshine-streaming-medium"  # tiny | base | streaming-small | streaming-medium | /path/to/model
model_dir = "~/.local/share/my-voice/models"
quantized = true            # smaller, faster — negligible accuracy cost
threads = 0                 # 0 = auto (up to 4)
load_timeout_secs = 1800    # idle eviction; -1 = never unload, 0 = reload every use
hotkey = "CapsLock"         # evdev key name (Linux); macOS only supports CapsLock in v1
clipboard_hotkey = true     # Shift+hotkey → clipboard
grab = true                 # Linux: exclusive grab + virtual keyboard passthrough
audio_device = ""           # substring match against device name; "" = system default
min_speech_ms = 300         # discard holds shorter than this (prevents accidental triggers)
trailing_silence_ms = 150   # extra audio captured after release (catches word endings)
injection = "auto"          # auto | wtype | xdotool | ydotool | atspi | clipboard
```

Unknown keys are warned and ignored.

Run `my-voice --config /path/to/file.toml` to use an alternate config file.

</details>

<details>
<summary>Models</summary>

| Model | Size (quantized) | Speed | Best for |
|---|---|---|---|
| `moonshine-tiny` | ~50 MB | ~10× real-time | Clear speech, weak CPUs |
| `moonshine-base` | ~200 MB | ~4× real-time | Noisy mic or accents |
| `moonshine-streaming-small` | ~350 MB | ~15× real-time | Good accuracy, lighter download |
| `moonshine-streaming-medium` | ~660 MB | ~12× real-time | **Default**; best accuracy |

Every model is Moonshine (ONNX, English-only). The `streaming-*` variants are
run as a single push-to-talk pass over the whole utterance, not chunk-by-chunk.

Switch models from the **Model** submenu in the tray (Linux), or by editing `model` in the config file. Run `my-voice --download` after changing the model.

Models are downloaded from HuggingFace and cached in `~/.local/share/my-voice/models/`.

</details>

<details>
<summary>Troubleshooting</summary>

**Model not found**
```
model file missing — run: my-voice --download
```
Run `my-voice --download`.

**Keyboard disconnected and reconnected**
Restart the daemon. Hotplug detection isn't supported in v1.

**Verify your microphone works**
```sh
my-voice --test   # records 3 seconds and prints the transcription
```

</details>

<details>
<summary>Development</summary>

```sh
cargo test                          # unit tests (no audio, model, or network needed)
cargo clippy -- -D warnings         # lint
cargo fmt                           # format
cargo build --features debug-tools  # enable --test/--wav/--record diagnostics
```

Tests cover: resample math, config round-trip, quote normalization, key-name parsing, max-token clamping, injection chain selection.

</details>
