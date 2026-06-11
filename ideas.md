# Ideas

Sorted roughly by estimated value-to-effort ratio — highest ROI first.

---

## 1. macOS menu-bar tray

**What:** Implement the `ui/macos.rs` stub (currently a no-op). Use `tray-icon` + `winit` or `tao`. The macOS event loop must own the main thread, so the daemon loop moves to a background thread — the architecture already notes this.

**Why it matters:** Right now macOS users have zero feedback: no icon, no way to switch models or mics, no status during transcription. The Linux tray is fully featured; parity here turns macOS from "works if you know what you're doing" to "works for anyone".

**Effort:** Medium. The Linux tray is a solid reference implementation. The main challenge is the thread inversion (event loop on main, daemon on background).

---

## 2. macOS start-at-login

**What:** Implement `autostart::set_enabled` for macOS via `launchctl` and a `LaunchAgents` plist, mirroring the Linux XDG autostart entry.

**Why it matters:** For macOS users to actually use this daily it needs to survive a reboot without manual terminal intervention. This is table-stakes UX for a productivity tool.

**Effort:** Low. The Linux path is a clean template; plist generation is ~30 lines.

---

## 3. macOS desktop notification

**What:** `notify/send_platform` on macOS currently falls back to `tracing::warn!`. Wire up `notify-rust`'s macOS backend (it uses `osascript` / `UNUserNotificationCenter`) or use `osascript` directly to send real desktop banners.

**Why it matters:** First-run download progress, "model ready", and error recovery all surface as invisible tracing logs on macOS. These are key onboarding moments.

**Effort:** Very low. `notify-rust` already supports macOS; it likely just needs the right feature flag or a `#[cfg]` branch.

---

## 4. Hotplug mic detection

**What:** When a USB microphone (or Bluetooth headset) is plugged in after the daemon starts, detect it and update the mic submenu — optionally auto-switch if the user had configured that device previously.

**Why it matters:** The current note in the README ("Restart the daemon. Hotplug detection isn't supported in v1.") is the most jarring paper-cut for wireless headset users.

**Effort:** Medium. On Linux: inotify or udev rules watching `/dev/input`. On macOS: `CoreAudio` property listeners. The hot-swap itself (rebuilding `AudioRecorder`) is already coded in `apply_reload`.

---

## 5. Configurable hotkey on macOS

**What:** The macOS hotkey is hard-wired to CapsLock (remapped to F18 via `hidutil`). Support arbitrary keys the way Linux does (evdev lets you grab any key).

**Why it matters:** Power users on macOS can't use CapsLock for other things. Many prefer `Fn`, `Right Option`, or a media key.

**Effort:** Medium-high. The `hidutil` remap approach is CapsLock-specific; an arbitrary key would need a different mechanism (Karabiner elements API, or a raw HID approach). A simpler short-term win: support a small fixed set of "safe" remap targets (F13–F19 are unused on most Macs).

---

## 6. Word-level confidence / retry on low confidence

**What:** After a transcription, if the raw token log-probabilities (available from the ONNX decoder output) are below a threshold, show a brief "low confidence" tray state instead of silently injecting possibly-wrong text.

**Why it matters:** The model sometimes hallucinates short utterances or mishears in noisy environments. The user currently has no signal that something went wrong until they look at the screen.

**Effort:** Low-medium. The token scores are already computed in `moonshine.rs`; the main work is surfacing them and deciding on a UX response (tray icon, notification, or nothing below a threshold).

---

## 7. Custom vocab / hotwords

**What:** A config list of words/phrases the model consistently gets wrong (proper nouns, tech jargon, brand names). A post-processing pass replaces them: `transcription_corrections = [["kubernetes", "Kubernetes"], ["git hub", "GitHub"]]`.

**Why it matters:** Moonshine English is general-purpose; developers say niche terms constantly. A simple string-replace dict in config.toml is zero-latency and zero-model-change.

**Effort:** Very low. A few lines in `text.rs`, one new config field, and a test.

---

## 8. Auto-punctuation / sentence capitalisation

**What:** Optionally capitalize the first word of each injection and/or append a period when the utterance ends without one. Configurable: `auto_capitalize = true`, `auto_period = false`.

**Why it matters:** Moonshine outputs lowercase without punctuation. Users dictating into documents, emails, or chat get bare lowercase streams — extra editing. Even simple first-word capitalization is a substantial UX improvement for writing use cases.

**Effort:** Low. Add to `text.rs`; a few config fields.

---

## 9. Streaming / live preview (long-form mode)

**What:** For long utterances (30s+), emit incremental partial transcriptions while the user is still holding the hotkey, showing them in a floating overlay or the tray tooltip. The streaming decoder architecture already decodes token-by-token; this would surface those partial results.

**Why it matters:** Dictating a full paragraph blind is uncomfortable. Real-time partial transcription turns my-voice into a genuine dictation tool, not just a PTT keyword launcher.

**Effort:** High. Requires a live display surface (overlay window), a threading change to run inference on partial audio while recording continues, and careful UX decisions about when/how to commit text. But the ONNX decoder already emits tokens one at a time — the inference side is already there.

---

## 10. Shell completion and `--status` flag

**What:** Generate shell completions for Bash/Zsh/Fish via clap's `generate` subcommand. Add `--status` to print whether the daemon is running (read the lockfile PID) and what model is loaded.

**Why it matters:** Small but high-frequency DX improvement. `my-voice <Tab>` and `my-voice --status` are the kinds of things that make a CLI feel finished.

**Effort:** Very low. Clap has built-in completion generation; `--status` is a lockfile read.

---

## 11. Packaging: `.deb` / `.pkg` / Homebrew formula

**What:** Provide pre-built binaries so users don't need Rust installed. A GitHub Actions release workflow that cross-compiles and publishes a `.deb` (Linux x86_64/ARM64) and a macOS `.pkg` or DMG. A Homebrew formula pointing at the release binary.

**Why it matters:** "Install Rust" is the single biggest barrier for non-developers. A `brew install my-voice` or `apt install my-voice` install path opens the audience by an order of magnitude.

**Effort:** Medium. The Rust binary is statically linked (no `.so` deps beyond `libasound2` on Linux). The main work is the Actions workflow, signing (macOS notarization), and the Homebrew tap setup.
