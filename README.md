# my-voice

Local push-to-talk dictation for Linux.

Hold **CapsLock** and talk. my-voice types each completed phrase into the app you are using. It keeps listening while it transcribes.

- Your voice stays on your computer.
- Text appears when you pause. You do not need to release the key.
- You can keep talking for as long as you need.
- There is no account or subscription.

## Get started

You need a microphone and [Rust](https://rustup.rs).

```bash
# Install the Linux build tools
sudo apt install libasound2-dev pkg-config

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Install and start my-voice
cargo install --git https://github.com/alexljenkins/my-voice
my-voice --download
my-voice
```

A microphone icon appears in the system tray. The default model download is about 345 MB.
While you hold the recording key, a small orb at the bottom of the display reacts to your voice.

<details>
<summary>Complete the one-time Linux keyboard setup</summary>

These commands let my-voice read CapsLock without changing its normal state.

```bash
sudo usermod -aG input "$USER"
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
sudo modprobe uinput
```

Log out and back in after you run the commands.

GNOME Wayland also needs `ydotool` to type directly into apps.

```bash
sudo apt install ydotool
ydotoold &
```

You can use clipboard mode from the tray if direct typing is unavailable.

</details>

my-voice supports Linux only. macOS users can use [OpenSuperWhisper](https://github.com/Starmel/OpenSuperWhisper).

## Use it

1. Hold **CapsLock** and speak.
2. Pause naturally. Each completed phrase appears while recording continues.
3. Release **CapsLock**. The final phrase appears and recording stops.

Hold **Shift+CapsLock** to copy the full dictation to the clipboard instead.

Long dictation has no total duration limit. my-voice uses pauses to split speech into smaller sections. Continuous speech also splits automatically.

## Settings and help

You can change the microphone, model, listening orb, typing mode, and startup behavior from the tray menu.

The listening orb uses X11. It also works through XWayland when `DISPLAY` is available.
If the overlay cannot start, dictation and the tray continue to work.

<details>
<summary>Configuration file</summary>

The default file is `~/.config/my-voice/config.toml`. You only need this file when the defaults do not fit.

```toml
model = "moonshine-streaming-small"
model_dir = "~/.local/share/my-voice/models"
quantized = true
threads = 0
load_timeout_secs = 1800
hotkey = "CapsLock"
clipboard_hotkey = true
grab = true
audio_device = ""
min_speech_ms = 300
trailing_silence_ms = 300
segment_pause_ms = 300
segment_max_ms = 30000
injection = "auto"
indicator_style = "neutral"
corrections = []
```

`indicator_style` accepts `calm`, `agreeable`, `thoughtful`, `neutral`, `cold`,
`defensive`, `anxious`, `frustrated`, `angry`, or `random`. The `random` option
chooses a new style for each recording.

Run with another file when you want separate settings.

```bash
my-voice --config /path/to/config.toml
```

</details>

<details>
<summary>Models</summary>

The default model is the best balance for most computers.

| Model | Download | Use it when |
|---|---:|---|
| `moonshine-tiny` | 31 MB | The computer has little memory |
| `moonshine-base` | 64 MB | You want a small download |
| `moonshine-streaming-small` | 345 MB | You want the recommended model |
| `moonshine-streaming-medium` | 566 MB | You want the best accuracy |

All models run locally and support English only.

</details>

<details>
<summary>Troubleshooting</summary>

### Text does not appear

Open the tray menu and switch to clipboard mode. On GNOME Wayland, install and start `ydotool`.

### The wrong microphone is active

Choose another microphone from the tray menu. You can also list device names.

```bash
my-voice --list-devices
```

### The model is missing

```bash
my-voice --download
```

### The microphone does not work

Build the diagnostic commands, then record a 3-second test.

```bash
cargo install --git https://github.com/alexljenkins/my-voice --features debug-tools
my-voice --test
```

### The keyboard stops responding after a crash

```bash
pkill my-voice
```

The kernel normally releases the keyboard within a few seconds.

</details>

<details>
<summary>Command line tools</summary>

```bash
my-voice --status
my-voice --list-devices
my-voice --download
my-voice --completions bash
```

</details>

<details>
<summary>Development</summary>

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt
cargo build --features debug-tools
```

The debug build adds `--test`, `--wav`, and `--record` diagnostics.

</details>
