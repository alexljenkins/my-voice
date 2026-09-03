# my-voice — Settings & Options Audit

A complete inventory of everything a user can control or observe, split by where it
lives today. Purpose: hand this to a UX redesign so we know the full surface area —
what's already exposed, what's buried in a config file, and what the app *knows* but
never tells the user.

The core interaction never changes: **hold a key, speak, release → text appears.**
Everything below is scaffolding around that one act.

---

## 1. What a non-technical user can reach today

### 1a. The tray menu (Linux only — right-click the mic icon)

This is the *only* graphical surface. Everything here is point-and-click; no file
editing. Each submenu marks the current choice with a green dot.

| Menu item | Choices | What it does | Applies |
|---|---|---|---|
| **Status line** (top, greyed) | — | Shows current state: `Ready` / `Listening…` / `Transcribing…` / `Downloading… N%` / `⚠ <error>` | live |
| **Model** ▸ | Faster (tiny) · Balanced (base) · Accurate (small) · Most accurate (medium) | Pick the speech model. Undownloaded ones show "— not downloaded" and download on selection. | live (between utterances) |
| **Microphone** ▸ | System default + every detected input device | Which mic to record from. | live |
| **Listening orb** ▸ | 9 named styles + random | Changes the bottom-center voice-reactive indicator. Random chooses once per recording. | live |
| **Hotkeys** ▸ → Recording key | *(display only)* | Shows the current trigger key. | — |
| **Hotkeys** ▸ → Set keybind… | opens capture popup | Press a new key to rebind. | restart |
| **Hotkeys** ▸ → Clipboard shortcut | on/off | Toggles whether **Shift+hotkey** copies to clipboard instead of typing. | live |
| **Hotkeys** ▸ → Reserve the hotkey | on/off | Exclusive-grab the key so it stops doing its normal job (e.g. CapsLock won't toggle caps). | restart |
| **Paste mode** ▸ | Paste at cursor / Copy to clipboard | How text is delivered. "Paste at cursor" auto-falls-back to clipboard if it can't type; shows a "Locked" hint with unlock steps when no typing tool exists. | live |
| **Start at login** | on/off | Installs/removes an autostart entry. | live |
| **Quit** | — | Stops the daemon. | — |

### 1b. Command-line flags (terminal)

Discoverable via `my-voice --help`, but a non-technical user generally only ever
runs the first one or two.

| Flag | Purpose |
|---|---|
| `my-voice` | Start the app (daemon + tray). |
| `--download` | Fetch the configured model, then exit (one-time setup). |
| `--status` | Print whether a daemon is running, its PID, and current model. |
| `--list-devices` | Print microphone device names (to set `audio_device` by hand). |
| `--config <PATH>` | Use an alternate config file. |
| `--record <DIR>` | Save every recording as a WAV while running normally (sample collection). |
| `--completions <shell>` | Emit a shell-completion script. |
| `-v` / `-vv` | More logging (info / debug). |

**Hidden / internal flags** (not for users): `--man` (packaging), `--set-hotkey`
(spawned by the tray's "Set keybind…"), and — only in `debug-tools` builds —
`--test`, `--wav <PATH>`, `--bench-iters <N>`.

---

## 2. Backend settings that exist but are NOT in any menu

These live only in `~/.config/my-voice/config.toml`. A non-technical user has no way
to reach them without editing a text file. **Strong candidates for the new UX.**

| Config key | Default | What it controls | Why it matters to a user | UX exposure today |
|---|---|---|---|---|
| `quantized` | `true` | Use the smaller/faster int8 model files vs full-precision. | Speed vs. accuracy tradeoff (negligible accuracy cost). | **None** — also note tiny/base have full variants; streaming models are quantized-only. |
| `threads` | `0` (auto, ≤8) | CPU threads for inference. | Power users on big/small machines could tune speed. | **None** |
| `load_timeout_secs` | `1800` | Idle seconds before the model is evicted from RAM. `-1` = never unload, `0` = reload every use. | RAM vs. first-word latency tradeoff. After eviction the next utterance is slow again. | **None** |
| `min_speech_ms` | `300` | Holds shorter than this are silently discarded. | Prevents accidental taps from transcribing. A user fighting "my short words vanish" can't find this. | **None** |
| `trailing_silence_ms` | `150` | Extra audio captured after key release. | Catches word endings cut off on release. | **None** |
| `corrections` | `[]` | Whole-word, case-insensitive find→replace pairs (e.g. `["git hub","GitHub"]`). | Custom vocabulary / proper nouns / jargon the model never learns. Arguably the single most-requested power feature, fully invisible. | **None** |
| `model_dir` | `~/.local/share/my-voice/models` | Where model files are stored. | Disk-location control. | **None** |
| `injection` (granular) | `"auto"` | Full set: `auto · wtype · xdotool · ydotool · atspi · clipboard`. | The menu only exposes **auto** ("Paste at cursor") and **clipboard**. Forcing a *specific* typing backend (e.g. pin `ydotool`) is config-only. | **Partial** — 2 of 6 values reachable |
| `model` = custom path | named models | `model` can be any `/path/to/model` dir, not just the 4 named ones. | Bring-your-own-model. | **None** — menu lists only the 4 registry models |
| `hotkey` | `"CapsLock"` | The trigger key. Menu can rebind via capture popup, but arbitrary evdev key names / combos are only fully expressible in config. | — | **Partial** (capture popup) |

Unknown keys in the config are warned-and-ignored, not rejected.

---

## 3. Information the app HAS but does not show the user

The app knows a lot about what's happening that never reaches a non-technical user,
or only appears in log files / debug builds. These are UX opportunities (feedback,
trust, troubleshooting).

### 3a. Recording & audio pipeline
Every capture runs through: native-rate capture → FFT resample to 16 kHz → WebRTC
noise-suppression + auto-gain (APM) → peak-normalize → silence-trim. The user sees
only "Listening…" then "Transcribing…". Not surfaced:
- The bottom-center listening orb reacts to the live microphone level while recording.
- **Silence gate**: if the captured peak is below `0.01`, the utterance is dropped with *no feedback* — looks identical to a transcription failure.
- **Too-short gate**: holds under `min_speech_ms` are dropped silently (see §2).
- **Auto-gain / normalization** is applied (up to 8× gain on quiet input) but never indicated.
- **Max capture length** is 60 s; longer holds are truncated silently.
- Raw vs. processed audio, capture duration, and peak levels exist only in debug logs / `--wav` output.

### 3b. Model download (partially surfaced)
- Tray shows `Downloading… N%` and a notification on start/complete/fail. Good.
- **Not shown**: download *size* per model up front in the menu (the registry knows `approx_mb`: ~31 / ~64 / ~345 / ~566 MB), per-file progress, SHA-256 integrity verification happening, retry/backoff attempts (3 attempts, exponential), or which HuggingFace repo it's pulling from.
- A user picking "Most accurate" has no idea they just triggered a ~566 MB download until it starts.

### 3c. Model lifecycle (invisible)
- Model is **lazy-loaded** on first use and **pre-warmed** ~2 s after startup / on key-down, to kill cold-start latency. User never told the first utterance after launch may be slower.
- Model is **evicted from RAM** after idle timeout (§2) — the *next* utterance then pays a reload cost with no warning.

### 3d. Text post-processing (invisible)
Every transcription is silently: trimmed, curly-quotes→ASCII, newlines→spaces, then
user corrections applied. The newline-stripping is a safety feature (a stray newline
would press Enter in a terminal) the user never learns about.

### 3e. Errors & notifications (one-shot, easy to miss)
Desktop notifications fire **once per session per kind** for: no microphone, hotkey
setup needed, model missing, download failed, injection failed, transcriber crash.
Miss the toast and there's no in-app history. The tray status line shows the latest
error but nothing persistent/reviewable.

### 3f. Logs (hidden from non-technical users)
A rolling log is written to `~/.local/state/my-voice/my-voice.log`. No menu item
points to it, opens it, or surfaces recent errors.

### 3g. Headless sessions
- On Linux with no D-Bus / tray host, the daemon runs without a tray icon.
- The listening orb needs X11 or XWayland through `DISPLAY`. Its failure never stops dictation.

---

## 4. Quick gap summary for the redesign

**Exposed & fine:** model choice, mic choice, listening orb, hotkey rebind, clipboard shortcut,
grab toggle, paste mode, start-at-login, quit, download progress.

**Exists, buried in config (should consider promoting):** custom-vocabulary
corrections, quantized toggle, RAM/idle-unload behavior, min-speech & trailing-silence
tuning, granular injection backend, custom model paths, threads, model storage dir.

**Known but never shown (feedback gaps):** live mic level, why an utterance was
dropped (too quiet / too short / too long), download size before commit, model
warm-up & eviction state, post-processing, a place to review past errors/logs.

**Headless sessions:** Linux sessions without a tray host have no graphical settings surface.
