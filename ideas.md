# Ideas

Sorted roughly by estimated value-to-effort ratio — highest ROI first.

---

## 4. Hotplug mic detection

**What:** When a USB microphone (or Bluetooth headset) is plugged in after the daemon starts, detect it and update the mic submenu — optionally auto-switch if the user had configured that device previously.

**Why it matters:** The current note in the README ("Restart the daemon. Hotplug detection isn't supported in v1.") is the most jarring paper-cut for wireless headset users.

**Effort:** Medium. Use inotify or udev rules watching `/dev/input`. The hot-swap itself is already coded in `apply_reload`.

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

## 11. Packaging: `.deb`

**What:** Provide pre-built Linux binaries so users do not need Rust installed. Publish `.deb` packages for x86_64 and ARM64.

**Why it matters:** "Install Rust" is the single biggest barrier for non-developers. An `apt install my-voice` path removes that barrier.

**Effort:** Medium. The Rust binary has no shared dependencies beyond `libasound2`. The main work is the release workflow and package metadata.
