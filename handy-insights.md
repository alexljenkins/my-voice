# Handy → my-voice: mined insights

[Handy](https://github.com/cjpais/Handy) is a heavier, multilingual desktop voice-typing app (Tauri: Rust backend `src-tauri/` + React frontend, whisper/parakeet, OpenCC, i18n, SQLite history, overlay windows). my-voice is the opposite: a lean, offline, English-only, CPU-only PTT daemon — single static Rust binary, no async, no GUI framework.

The lens here is narrow: mine Handy's **Rust backend** for techniques that port to a lean offline English CPU PTT daemon, and reject anything Tauri-coupled, multilingual-coupled, GPU/sqlite/tokio-dragging, or too heavy. Each insight below was red-teamed against my-voice's actual source. Kept = strong/worthwhile (incl. ones that net out worthwhile despite a "marginal" tag); rejected ones are listed at the bottom with one-line reasons.

## Quick reference

| Rank | Insight | Benefit | Effort | Verdict |
|------|---------|---------|--------|---------|
| 1 | Strip invisible Unicode chars from output | robustness, ux | low | worthwhile |
| 2 | Handle I32 (and I8) sample formats | robustness | low | worthwhile |
| 3 | Verify byte count vs Content-Length | robustness | low | worthwhile |
| 4 | Score sample formats instead of first-match | robustness | low | worthwhile |
| 5 | KDE-Wayland: skip wtype | ux, latency, robustness | low | worthwhile |
| 6 | Classify mic-permission-denied errors | ux, robustness | low | worthwhile |
| 7 | Cancel/abort key for in-flight utterance | ux, robustness | low–med | worthwhile |
| 8 | Max-recording watchdog (beat Handy) | robustness, ux | low | worthwhile |
| 9 | Atomic directory install (temp dir + rename) | robustness | low | worthwhile |
| 10 | N-gram custom-vocab (multi-word → token) | accuracy, ux | low–med | worthwhile |
| 11 | Tolerant settings load (parse-error → default) | robustness, ux | low | worthwhile |
| 12 | In-memory "re-inject last" ring | ux, robustness | low | worthwhile |
| 13 | Optional toggle (tap-on/tap-off) mode | ux | low | worthwhile |
| 14 | HTTP byte-range resume for downloads | robustness, ux | medium | worthwhile |
| 15 | Start/stop audio cues (synthesized beep) | ux | medium | worthwhile |
| 16 | wl-copy on Wayland clipboard fallback | robustness, ux | low | marginal |
| 17 | Configurable trailing-space on injected text | ux | low | marginal |
| 18 | Throttled download progress | ux | low | marginal |
| 19 | Stutter collapse (text post-process) | accuracy, ux | low | marginal |
| 20 | Evict-thread shutdown flag (kill zombie threads) | maintainability, robustness | low | marginal |

---

## 1. Strip invisible Unicode chars from model output

**Tags:** robustness, ux

**Handy:** `strip_invisible_chars()` (actions.rs:55-58) removes U+200B/200C/200D (ZWSP/ZWJ) and U+FEFF (BOM) from output before paste, on every post-process return path.

**my-voice today:** `post_process()` normalizes curly quotes + newlines but does NOT strip zero-width / BOM chars. A stray U+FEFF from the decoder passes straight into wtype/xdotool injection.

**Change:** In `src/text.rs:11-18`, refactor the per-char loop so it `continue`s on `'\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}'` instead of always pushing. Add tests: `pp("\u{FEFF}ls") == "ls"`, `pp("a\u{200B}b") == "ab"`.

**Fit & red-team:** Perfect fit. The loop already exists for exactly this charter ("neutralize chars that break wtype / execute the line"). A leading U+FEFF silently turns injected `ls` into a non-command — the worst failure mode (silent corruption of an executed shell line). Honest caveat: defensive against a hypothetical Moonshine artifact (no evidence it actually emits these), so frame the commit as hardening, not a bug fix. Cost is ~4 match arms, zero deps, zero alloc. Consider folding into any other text.rs change.

**Effort:** low

## 2. Handle I32 (and I8) sample formats

**Tags:** robustness

**Handy:** `build_stream` dispatches on U8/I8/I16/I32/F32 (recorder.rs:105-149).

**my-voice today:** `src/audio.rs:77-103` matches only F32/I16/U16/U8; anything else hits `other => Err("unsupported sample format")` and capture fails outright. `select_stream_config` (audio.rs:150) blindly adopts whatever format the device reports.

**Change:** Add `SampleFormat::I32` and `SampleFormat::I8` arms at `src/audio.rs:84`, mirroring the I16 arm (`data: &[i32]` / `&[i8]`). `append_mono` is already generic `T: Sample, f32: FromSample<T>` and `f32: FromSample<i32>`/`i8` exist, so each arm is a one-line copy. Keep the `other =>` catch-all.

**Fit & red-team:** I32 is the native format on many USB/pro interfaces and some ALSA paths, so a hard capture failure on "plug in a nicer mic" is a real silent brick. No deps, ~8 lines. Honest caveat: typical laptop/consumer mics report F32/I16, so this is pure tail robustness — no latency/WER win. Worth it only because cost is near-zero.

**Effort:** low

## 3. Verify downloaded byte count against Content-Length

**Tags:** robustness

**Handy:** After the stream loop, before SHA verify, compares actual file size to expected total; on mismatch deletes the partial and errors "Download incomplete" (model.rs:1178-1190). Catches a silently truncated stream (connection dropped at EOF without error).

**my-voice today:** `stream_to()` trusts `n == 0` as completion (download.rs:166-175) and never compares total bytes to the already-parsed `content_length` (download.rs:156-159). A clean truncation passes straight to the checksum step.

**Change:** After the loop in `stream_to` (download.rs ~175): `if content_length != 0 && total != content_length { fs::remove_file(part).ok(); bail!("download incomplete: expected {content_length} got {total}"); }`. `bail` propagates into `with_retry` (download.rs:204-210) → retried, then clear error.

**Fit & red-team:** ~3 lines, content_length already in hand. The load-bearing reason: **tokenizer.json has no checksum row** (models.rs checksums cover only `.onnx` names), so this is its ONLY truncation guard — without it a truncated tokenizer.json renames into place and fails confusingly at load. Must keep the `content_length == 0` escape hatch (HF may omit it; gzip/chunked encoding makes Content-Length != decoded length) or it false-positives. Add a mock-reader unit test.

**Effort:** low

## 4. Score sample formats instead of taking first match

**Tags:** robustness (NOT accuracy — see below)

**Handy:** `get_preferred_config` scores formats F32=4 > I16=3 > I32=2 > others (recorder.rs:300-327), falls back to `default_input_config`.

**my-voice today:** `select_stream_config` (audio.rs:151-158) `find()`s the FIRST 16kHz-straddling range, whatever format it happens to be. If a device exposes an unsupported format (e.g. I32) first, my-voice picks it and `build_stream` hard-errors at audio.rs:102.

**Change:** In `select_stream_config`, among configs straddling `TARGET_RATE`, `max_by_key` on a score that (a) only counts formats the match arm supports, (b) ranks F32>I16>U16>U8. Keep `default_input_config` fallback. ~10 lines.

**Fit & red-team:** Reframe the rationale — the win is **robustness, NOT precision**. I16→f32 via `from_sample` is lossless, so "F32 = best precision" is bogus. The real win: scoring lets us skip 16kHz configs in formats `build_stream` can't handle, turning a hard capture failure into a working capture. Pairs naturally with #2 (add I32/I8 arms first, or score around them). Marginal-to-zero WER impact; value is "a device that should work, works."

**Effort:** low

## 5. KDE-Wayland awareness: skip wtype

**Tags:** ux, latency, robustness

**Handy:** Gates wtype behind `!is_kde_wayland()` ("wtype doesn't work on KDE — no zwp_virtual_keyboard_manager_v1"), prefers kwtype there (clipboard.rs:86-92, 160-173).

**my-voice today:** `build_auto_chain` (linux.rs:351-367) pushes `WtypeInjector` first on ANY Wayland session whenever wtype is on PATH — no KDE check. On KDE Plasma, wtype is installed-but-broken, so the chain wastes the first attempt then demotes to ydotool.

**Change:** Add `fn is_kde_wayland()` checking `XDG_CURRENT_DESKTOP` contains "KDE" (case-insensitive) or `KDE_FULL_SESSION=="true"`. In the `Session::Wayland` arm (linux.rs:359-361), guard the `WtypeInjector` push with `&& !is_kde_wayland()`. ydotool already follows, so it becomes first. Adopt ONLY the skip-half — reject Handy's kwtype/dotool branch (extra tools my-voice deliberately omitted).

**Fit & red-team:** Pure env read, ~6 lines, no deps. Honest sizing of the win: `ChainInjector` demotion is PERSISTENT (cursor += 1, never reset, linux.rs:244), so the wasted wtype attempt is paid only ONCE on the first dictation of the process — it's a one-time fork+handshake (tens of ms) plus one scary log line removed, not recurring latency. Correctness is already fine (wtype exits non-zero on KDE → clean demotion; not the silent-Ok AT-SPI/Mutter trap at linux.rs:355-358). Don't over-generalize: sway/Hyprland DO support wtype, so keep the gate KDE-only. Worth bundling with another small linux.rs tweak.

**Effort:** low

## 6. Classify microphone-permission-denied errors

**Tags:** ux, robustness

**Handy:** `is_microphone_access_denied()` string-matches "access is denied" / "permission denied" / "0x80070005" (recorder.rs:338-343), maps to a distinct error so the UI says "grant mic access" instead of a cryptic backend string; unit-tested.

**my-voice today:** `AudioRecorder::new` failure (main.rs:363-373) fires a hardcoded "No microphone found" toast; the mid-capture path (main.rs:531-540) fires "Microphone disconnected".

**Change:** Add `notify::ErrorKind::MicPermissionDenied` (src/notify.rs:12-19) and a free `classify()` function in audio.rs. Match Linux permission errors. Apply it at startup and in the `AudioFailed` handler.

**Fit & red-team:** Linux PipeWire denial is rare and usually presents as no-device, which `NoMicrophone` already covers. Skip this until a Linux error needs classification.

**Effort:** low

## 7. Cancel/abort key to discard an in-flight PTT utterance

**Tags:** ux, robustness

**Handy:** Registers a dynamic "cancel" shortcut (Escape) only while recording (shortcut/handler.rs:55-62), `Command::Cancel` resets to Idle. Notably DISABLES it on Linux "due to instability with dynamic shortcut registration" — an honest signal that runtime register/unregister of a second global grab is fragile.

**my-voice today:** Absent. Every CapsLock Release runs `handle_utterance` and injects (main.rs:474-512). A sneeze, wrong window, or changed mind types garbage with no escape.

**Change:** Add `HotkeyEvent::Cancel`. In evdev `run_device` (linux.rs:415-450), emit Cancel when `active && ev.code()==Key::KEY_ESC.code()`. Handle `(State::Recording, Cancel)` in main.rs:455 and discard the recording.

**Fit & red-team:** The X11 fallback cannot see Escape without a second global grab. Keep cancel limited to evdev.

**Effort:** low–medium

## 8. Max-recording watchdog (the stuck-key safety Handy lacks)

**Tags:** robustness, ux

**Handy:** Confirmed ABSENT — no force-release/max-duration logic anywhere. This is a place to BEAT Handy, not copy it.

**my-voice today:** Absent. A physically stuck CapsLock or a dropped Release (e.g. evdev device disconnect mid-hold — DeviceGuard drop ungrabs but synthesizes no Release, linux.rs:363-369) leaves `State::Recording` indefinitely, pinning the recorder and growing the audio buffer unbounded (a direct threat to the <500MB RSS target).

**Change:** On entering Recording (main.rs:472), spawn a dedicated timer thread holding a `daemon_tx` clone (pattern already used for preload at main.rs:464-469) that sleeps a generous cap (~120s) then sends `DaemonMsg::Hotkey(HotkeyEvent::Release)`. Cancel/supersede on a real Release (generation counter or per-recording cancel flag so a stale watchdog can't fire). On fire: **DISCARD and return to Idle** — do NOT transcribe a 120s buffer (wasted CPU, likely noise injection).

**Fit & red-team:** One Instant and a timer thread cover device disconnects where no later event arrives. Pairs with #13 because toggle mode needs this cap.

**Effort:** low

## 9. Atomic directory install via temp dir + rename

**Tags:** robustness

**Handy:** Stages a directory model into `{name}.extracting`, then `fs::rename`s to final only after full success; sweeps leftover temp dirs on startup (model.rs:734-743, 1222-1297). The final dir is never half-populated.

**my-voice today:** Downloads N loose files directly into `{model_dir}/{name}/`, each via its own .part+rename — atomicity is **per-file, not per-model**. The encoder (sentinel) downloads FIRST (models.rs:39,49,...), so a crash after file 1 of the 3-4-file streaming models leaves encoder-present-but-no-decoder/tokenizer. `is_model_downloaded()` checks ONLY the sentinel (config.rs:154-159) → returns TRUE for the half-dir; worse, the load path `transcriber/mod.rs:54` gates re-download on `path.exists()` of the DIR, so the half-dir skips re-download and `Moonshine::load` fails. Default model is streaming-small (3 files) — hits the common case.

**Change:** In both `run()` (download.rs:75-92) and `run_with_progress()` (109-133), build dest as a `{model}.partial` sibling under `model_dir`, download+verify all files there, then a single `fs::rename(partial, final)`. Add a stale `{model}.partial` sweep at the top. ~15-25 lines. `fs::rename` of a dir requires same filesystem — guaranteed (siblings).

**Fit & red-team:** No deps — skip Handy's tar/flate2 entirely (my-voice fetches loose HF files). Prefer atomic-dir over a 2-line band-aid: the band-aid (make `create()` check all files) leaves a stale half-dir that never self-cleans, whereas atomic-dir makes "dir exists" a true all-or-nothing signal and fixes BOTH call sites at once. Wrinkle: staging dir isn't reused across runs (re-run re-downloads everything) — but there's already no resume, so cost is unchanged. Lower urgency than #14 but cheaper to land.

**Effort:** low

## 10. N-gram custom-vocab matching (multi-word → single token)

**Tags:** accuracy, ux

**Handy:** `apply_custom_words` slides longest-first 3→1 n-grams, concatenates+cleans into a nospace string (`build_ngram`, text.rs:102-156), matches against custom words with spaces stripped. Turns "Charge B"→"ChargeBee", "Chat G P T"→"ChatGPT". (Drops Handy's fuzzy strsim/soundex.)

**my-voice today:** `apply_corrections` (src/text.rs:26-72) does whole-word/whole-phrase exact match longest-first. It CAN match "git hub"→"GitHub" if the user lists it with spaces, but CANNOT collapse a model-invented word-internal split ("Charge B") without the user pre-enumerating every spoken variant.

**Change:** Add a separate word-token pre-pass before `apply_corrections` (NOT inside the char-based scanner at text.rs:46-71 — different granularity). Build a `HashMap<nospace_lower, replacement>` ONLY from corrections whose from-key has no spaces (single-token targets). Try 3/2/1-grams against it, longest-first, exact lowercased concatenation. Add test: from-key "ChargeBee" fires on input "charge b". Drop strsim/soundex/OpenCC entirely.

**Fit & red-team:** Real gap for an English-only daemon with no ITN — this is exactly what custom-vocab exists for. Exact-equality keeps it dep-free, CPU-trivial (n≤3). Risks: false-positive collisions ("to do"→"todo", "an droid"→"android") — bounded by building keys only from spaceless from-keys the user opted into. Effort is low-to-medium (new code path + collision-guard tests), not trivially low.

**Effort:** low–medium

## 11. Tolerant settings load: parse error → default instead of bricking

**Tags:** robustness, ux

**Handy:** On parse error, logs a warning and overwrites with defaults rather than failing (settings.rs:873-879); new fields use `#[serde(default)]`.

**my-voice today:** `Config::load` (config.rs:59-82) uses `#[serde(default)]` on the whole struct, so partial/old configs already keep defaults (tested) — that half is solid. BUT a config that parses to the WRONG TOML type (`min_speech_ms = "loud"`) returns Err, `?`-propagates at main.rs:142, and the daemon **refuses to start**. Meanwhile the reload path (main.rs:731-738) already degrades gracefully to a "Bad config" tray Error.

**Change:** In `Config::load` (config.rs:59-82), when `path.is_none()` (default path) and `toml::from_str` returns Err, log a warning and return `Self::default()` WITHOUT rewriting the file (preserve the user's typo so they can fix it — diverge from Handy's overwrite). Keep the hard `?`/bail when an explicit `--config PATH` was supplied. ~3-5 lines + 1 test. Skip the serde forward-merge work — already covered by `#[serde(default)]` + `warn_unknown_keys` + the `partial_config_keeps_defaults` test.

**Fit & red-team:** Makes first-boot match the already-tolerant reload path. A voice-typing tool that won't start over one bad TOML line is the worst failure mode. Narrow trigger (only wrong-type TOML on the default path, since missing/unknown keys are already tolerated), so felt benefit is rare — rank below the audio/VAD/text ports.

**Effort:** low

## 12. In-memory "re-inject last" ring (no SQLite)

**Tags:** ux, robustness

**Handy:** Persists every transcription to a rusqlite DB and exposes `get_latest_completed_entry` (skip-empty rows) for re-show/re-copy (history.rs:530-555). (Reject the DB; keep the concept.)

**my-voice today:** `handle_utterance` post-processes, injects, and drops the string (main.rs:936-953). If injection lands in the wrong window or focus was fat-fingered, the text is gone and the user must re-speak.

**Change:** Add `let mut last_text: Option<String>` local to the single-threaded daemon loop (no Mutex needed — the loop is the only injector caller). Store `text` after a successful inject (the empty case is already gated at main.rs:939-942, so only non-empty transcripts get stored — this folds in the "latest completed = skip-empty" semantics for free). Add `UiCommand::ReinjectLast` + a tray menu item that calls `typer.inject(&text)` when `Some`. Start with N=1 (single Option); a VecDeque of 5 is speculative YAGNI.

**Fit & red-team:** Pure in-RAM, no deps, no disk, gone on quit (no privacy surface), single-binary intact. **Do NOT add a second hotkey in the first cut** — that drags in the 24KB evdev/uinput/x11rb/CGEvent hotkey layer across two platforms, where "low effort" dies. Ship tray-only first. Document the tray-focus-steal caveat: re-inject lands in whatever window has focus after the tray interaction (acceptable for "click into the right field then re-inject"). Additive to — not redundant with — the clipboard fallback (which only fires on injection *failure*, not silent success-into-wrong-window).

**Effort:** low

## 13. Optional toggle (tap-on/tap-off) mode beside hold-to-talk

**Tags:** ux

**Handy:** Supports PTT and toggle from the same input stream via a `push_to_talk` flag through one state machine (transcription_coordinator.rs:72-92).

**my-voice today:** PTT only. Release always ends the utterance (main.rs:474). Long-form dictation requires physically holding CapsLock the whole time — ergonomically hostile.

**Change:** Add `hotkey_mode: String` ("hold"|"toggle") to Config (config.rs:13-49) + the `known[]` array (lines 86-100); no new deps. In the main.rs match (455-529): in toggle mode, `(Recording, Press)` runs the current Release body (stop+transcribe) and Release is ignored. Default = hold (simple path untouched).

**Fit & red-team:** Real PTT pain, zero heavy baggage. Two non-trivial wrinkles make the "minimal" framing slightly dishonest: (1) **autorepeat** — a held key emits repeat presses (evdev value 2) that could falsely re-toggle; confirm the hotkey layer collapses these or toggle instantly stops itself. (2) **stuck-recording failure mode** — a missed second tap records forever. Therefore **do NOT ship toggle without #8 (max-recording cap)** and a distinct "armed/Listening" tray state so the user knows it's hot. With those, still low effort and inside all hard constraints.

**Effort:** low (paired with #8)

## 14. HTTP byte-range resume for interrupted downloads

**Tags:** robustness, ux

**Handy:** Checks for a `.partial`, sends `Range: bytes={size}-`, appends; guards the server-doesn't-support-ranges case (asked to resume but got 200 not 206 → delete partial, restart fresh) (model.rs:1014-1104).

**my-voice today:** No resume. `stream_to()` truncates the .part on every attempt (download.rs:161-162); the comment is explicit "there is no resume" (182-184). A 345MB download that dies at 90% restarts from zero. HF resolve URLs DO honor Range.

**Change:** In `stream_to`: (1) stat existing .part, send `Range: bytes={n}-`; (2) re-read+re-hash the prefix into the Sha256 before appending (cheap sequential read — the streaming-hash design requires this); (3) open with append not create; (4) if Range requested but resp is 200 not 206, truncate+restart (Handy's guard); (5) **FIX progress math** — under 206 `content-length` is REMAINING bytes, so `total` must be `partial_size + content_length` and `on_chunk` must report `partial_size + streamed` (download.rs:128 and the CLI counter both assume content-length == total). Pairs with `with_retry` so a retry resumes.

**Fit & red-team:** Pure ureq + std::fs, no deps. Honest framing: model is downloaded once ever, inference unaffected — payoff is bandwidth saving on repeated transient drops on a flaky link (real but infrequent). The cost is THREE correctness branches (re-hash prefix, 200-vs-206, remaining-vs-total) added to what is currently the simplest, most-obviously-correct module — trades dead-simple correctness for an edge-case win. Sequence AFTER the cheaper correctness items (#3, #9).

**Effort:** medium

## 15. Start/stop audio cues via a synthesized beep (no rodio)

**Tags:** ux

**Handy:** Plays a short WAV on record start/stop on a detached thread (audio_feedback.rs:48-95), with output-device enumeration (skip that part).

**my-voice today:** Absent — only tray-color changes + notifications. PTT gives zero confirmation the daemon grabbed keydown until text appears.

**Change:** New `src/cue.rs` synthesizing ONE short (~60-80ms) sine-with-envelope buffer; two fns (rising/falling), each spawning a detached thread that opens the **default** output device, feeds the buffer via cpal `build_output_stream`, then drops after N samples. Hook a rising blip at `TrayState::Listening` (main.rs:471), falling at Transcribing/Ready (main.rs:502/509). Gate behind a config flag (default on). DO NOT pull in rodio (heavy decode+mixer for one tone) — cpal is already a dep. Skip Handy's output-device enumeration (GUI baggage).

**Fit & red-team:** Highest UX-per-byte in the feedback subsystem — an audible "I'm listening"/"got it" is exactly what PTT lacks. No new deps, no asset file, offline, English-agnostic. Effort is **medium not low**: sine math is trivial but cpal output-stream open/teardown timing and concurrent input+output on non-dmix ALSA (exclusive/JACK) are the real work and need real-hardware testing. Detached thread keeps the keydown path non-blocking; swallow output errors (debug-log only, never notify). **Verification gap:** no mic on the dev box and the --wav/WER harness bypasses live capture, so this needs manual on-hardware test before merge.

**Effort:** medium

## 16. wl-copy on Wayland for the clipboard fallback

**Tags:** robustness, ux

**Handy:** Prefers `wl-copy -- <text>` over in-process clipboard on Wayland; runs `.status()` with null'd fds because wl-copy forks a daemon that inherits piped fds and `.output()` hangs forever (clipboard.rs:28-43, 386-402).

**my-voice today:** `ArboardInjector` (linux.rs:213-223) uses arboard for the clipboard fallback on all sessions. arboard runs an in-process Wayland selection owner; `set_text` creates a fresh Clipboard each call (linux.rs:216) and can race ownership on some compositors (KDE/wlroots).

**Change:** In `build_clipboard()` (linux.rs:427): when `detect_session()==Wayland` AND wl-copy on PATH, return a `WlCopyInjector` running `wl-copy --` via Command with stdout/stderr nulled; else `ArboardInjector`. Keep arboard for X11/no-session.

**Fit & red-team:** Demoted to marginal. **Reject the "umlauts" justification** — multilingual, N/A to English-only (text.rs already ASCII-normalizes). **Reject the "Stdio::null gotcha is the prize" framing** — `run_argv` ALREADY uses `.status()` (linux.rs:130), so the documented hang is already avoided. Real merit = arboard's Wayland selection-ownership fragility vs wl-copy's persistent owner daemon. But this touches only the clipboard FALLBACK (last resort, rarely hit), so impact ceiling is low. Worth a brain-wiki note on the wl-copy forked-daemon-inherits-fds behavior regardless.

**Effort:** low

## 17. Configurable trailing-space on injected text

**Tags:** ux

**Handy:** Optionally appends a trailing space (`format!("{} ", text)`) so consecutive dictations don't run together (clipboard.rs:596-601).

**my-voice today:** Injector takes the final string as-is; `post_process` trims both ends, so back-to-back utterances inject "send emailthanks Bob".

**Change:** Add `config.append_trailing_space` (default false) to Config (config.rs:28/46). In `src/text.rs` `post_process` (line 9), append a single `' '` after `apply_corrections` when the flag is set (thread the flag through the signature or a small wrapper). Keep injectors dumb sinks.

**Fit & red-team:** Real for burst dictation, trivial one-liner, no deps. Marginal because: default-off means most users won't discover it; always-on adds a stray space before user-typed punctuation. Do NOT attempt "smart" spacing (track last injected char) — that needs cross-utterance state the daemon doesn't keep and breaks the pure-function model. YAGNI: ship only if Alex feels the friction.

**Effort:** low

## 18. Throttled download progress

**Tags:** ux

**Handy:** Throttles progress emission to ~10/sec via an `Instant` gate + guaranteed final 100% emit (model.rs:1121-1173).

**my-voice today:** `on_chunk` fires every 64KB (download.rs:174), so the chain on_chunk → on_progress(u8) → mpsc → ksni repaint runs ~5000× for a 345MB file (main.rs:432, 558).

**Change:** Add `let mut last = Instant::now()` inside `download_file`'s on_chunk wrapper; skip emit unless `elapsed >= 100ms`, plus a guaranteed final emit. Drop the byte-accurate "312/345 MB" half (pure polish for a once-per-install screen; not worth widening DownloadEvent/DaemonMsg/TrayState).

**Fit & red-team:** Marginal. One-time event, off the PTT hot path (the listed "latency" benefit is bogus). my-voice already collapses progress to u8 0-99 at download.rs:128, so consecutive chunks mostly emit the SAME percent — much redundancy is already absorbed. Reality check before bothering: verify ksni `set_state` actually repaints on identical state, else the throttle buys almost nothing. Strictly hygiene.

**Effort:** low

## 19. Stutter collapse (text post-process)

**Tags:** accuracy, ux

**Handy:** `collapse_stutters` reduces 3+ consecutive identical words to one ("I I I I think"→"I think"); separately strips filler words (uh/um) via regex (text.rs:236-271, 288-320).

**my-voice today:** Absent. `post_process` does only quote/newline normalization + corrections. my-voice fights repetition audio-side, but those gates DISCARD whole utterances — they don't surgically fix a 3+ repeat inside an otherwise-good transcript.

**Change:** Port ONLY the stutter-collapse half into `src/text.rs` (`split_whitespace` + a counter loop, no regex). Wire into `post_process` (text.rs:9-20), behind a config bool defaulted OFF (config.rs already has the bool pattern at lines 16-21). Do NOT add the `regex` crate.

**Fit & red-team:** Marginal. Stutter collapse is dep-free and complementary to the existing audio-side gates (gates discard; collapse fixes in-place). **DROP the filler-removal half** until someone shows Moonshine actually emits "uh/um" on samples/ — the strong WER baselines (streaming-small 0.154) imply it isn't spewing fillers, so filler removal may be dead on arrival, plus over-deletion risk on a context-blind injector. Cheap validation gate: grep WER-harness transcripts for `\b(uh|um|uhm|umm|hmm)\b`; if ~0 hits, only stutter collapse earns its place.

**Effort:** low

## 20. Evict-thread shutdown flag (kill zombie threads on model switch)

**Tags:** maintainability, robustness

**Handy:** A strong-count-gated Drop joins the watcher thread only when the last Arc drops (transcription.rs:830-854). (Wrong mechanism for my-voice — see below.)

**my-voice today:** **Real defect, opposite shape.** On model switch, `actions.model => ModelCache::new + start_evict_thread; *cache = c` (main.rs:758-761). The old ModelCache is NOT dropped because its evict thread holds its own `Arc::clone` (model_cache.rs:113) and loops forever with no shutdown signal (114-127). Every model switch leaks one ModelCache + factory closure + one zombie 30s-ticking thread.

**Change:** Do NOT port Handy's strong_count Drop (the surviving thread IS the strong ref, so a Drop guard can't stop it; my-voice never used Drop to join). Lean fix: add a shutdown `Arc<AtomicBool>` the evict loop polls each `EVICT_TICK`; signal it when `*cache` is replaced in apply_reload. ~15 lines, no deps. Bonus: makes evict-thread shutdown testable.

**Fit & red-team:** Marginal — severity LOW. The evicted model's RAM is reclaimed on the next tick (slot → None), so RSS impact is tiny (struct + closure, well under 500MB); only the thread leaks, bounded by switch count per session, and users rarely toggle models repeatedly. Pure thread-lifecycle hygiene, not user-facing. File as a small bug.

**Effort:** low

---

## Not worth porting

Rejected against my-voice's constraints (lean / offline / English / CPU / single-binary / PTT-not-streaming). One-line reason each.

- **catch_unwind around the engine call** — already covered by mutex-poison recovery (model_cache.rs:55-71, tested); adopting it doubles up recovery mechanisms in the single documented concurrency path.
- **GPU execution providers** — explicitly "do NOT add"; self-confirming no-op, any actual adoption violates CPU-only + single-binary. (Capture the FMA3/SIGILL footnote in the brain wiki, not src/.)
- **Single-thread command coordinator** — my-voice's `daemon_rx` loop ALREADY IS this (single owner, unified DaemonMsg, state enum, busy-drop at main.rs:528-529); synchronous inline transcribe makes the press-while-busy race impossible by construction.
- **Idle watcher refresh during recording** — race exists but self-heals via inline reload in `load_locked` (model_cache.rs:97/134); worst case ~1s reload on one utterance, only reachable with a tuned-down timeout + multi-minute hold (pathological for PTT).
- **Silero VAD content filter** — adds a stateful 3rd ONNX model + registry/eviction surface; interior frame-dropping risks splice artifacts in Moonshine's raw-waveform encoder; robustness win unmeasurable until noisy WER fixtures exist.
- **Hysteresis VAD state machine (onset/hangover)** — `trim_silence` does boundary-only trimming (no interior cuts), so hangover/mid-word-chopping benefits don't apply; audio already APM-cleaned. At most a 2-frame onset on the boundary search, gated on WER.
- **Pre-roll (prefill) buffer** — PTT already holds the full buffer at keyup; the 80ms PAD already restores look-back. Pre-roll only helps streaming onset detection my-voice deliberately doesn't have.
- **VAD-as-endpointing guardrail** — validation only; keyup is already the sole stop. Fold the "filter-only, never auto-stop" caveat onto any future VAD-import proposal.
- **Silero threshold 0.3 tuning** — conditional on importing Silero (which my-voice shouldn't); probability anecdote doesn't map onto an energy threshold. At most: sweep `SPEECH_RMS` on the harness.
- **Pre-flight "no input device" check** — `--list-devices` + friendly `input_devices()` + the "(see --list-devices)" hint already cover discovery; cosmetic polish on a cold path.
- **Pad short captures to a floor** — Whisper-specific (fixed 30s mel window); Moonshine eats variable-length raw waveform (moonshine.rs:2), so padding ADDS latency + hallucination risk on a "yes" for no gain.
- **Case-pattern preservation in corrections** — conflicts with the fixed-casing intent of custom vocab; would break test src/text.rs:111 ("ask CLAUDE" → "ask Claude").
- **Length-gap prefilter before fuzzy compare** — optimizes a fuzzy path my-voice doesn't have and shouldn't add (strsim/natural deps + risk of silently corrupting correct text on a blind injector).
- **Punctuation prefix/suffix preservation** — already free in the boundary-aware char scan (text.rs:48-70); only "needed" if the n-gram pass is adopted, and even then letter-only windows sidestep it (and Handy's GPT-44 double-count bug).
- **Skip post-process on blank input** — no LLM exists, so Handy's failure mode can't occur; `post_process` is already a no-op on blank input.
- **Optional local-LLM cleanup** — remote violates offline+async; localhost violates single-binary; rule-based caps/punct is a different feature that's a likely regression on a context-blind injector ("ls -la" → "Ls -la.").
- **Per-shortcut behavior variants (struct + flag)** — no second behavior exists; the only realistic one (LLM post-process) is out of bounds; `post_process` is mandatory safety, not a toggle site.
- **Clipboard save/restore** — inseparable from synthesized Ctrl+V auto-paste, which my-voice rejects; both clipboard uses WANT to clobber (the text IS the deliverable).
- **enigo cross-platform crate** — my-voice injects TEXT (layout-immune), so enigo's layout-keycode value is categorically inapplicable; adopting it fattens the binary for zero benefit.
- **RAII download-cleanup guard** — no shared in-flight flag to guard; the stateless background-thread + events model already prevents the bug class.
- **Self-healing model registry** — realistic cases already self-heal (missing registry → auto background download main.rs:413; missing path → download then clear error); silently swapping to a different model violates config-as-source-of-truth.
- **Time-based press debounce** — the synchronous single-consumer loop + immediate Idle→Recording + existing dupe-drop already collapse duplicate presses; adds a 4th debounce mechanism for a non-event, with a regression risk of swallowing a legit fast re-press.
- **Persist backend fallback** — selection is a deterministic env probe with in-launch fallback; caching a transiently-failed path goes stale exactly when env changes (session type).
- **Cache "feedback enabled" AtomicBool** — guards a non-existent visualizer; the cpal callback is already a bare buffer-append, config applied between utterances. The WebKit-leak motivation is Tauri-specific.
- **FFT log-bucketed visualizer** — needs a rendering surface (overlay/GTK-layer-shell/NSPanel) that's the exact GUI weight the project rejects; PTT has no live level to render.
- **Tray state-dependent Cancel item** — PTT keyup already ends recording (Cancel-during-Recording is redundant); Cancel-during-Transcribing is blocked by the synchronous loop (handle_utterance inline at main.rs:508). A decode-step cap is the right primitive, not a tray item.
- **Output-device selection for cues** — parasitic on a cue subsystem that doesn't exist; the 6-line fallback shape is obvious if cues are ever built.
- **Blocking vs async cue variant** — a footnote to a non-existent feature; fold one sentence ("prefer detached playback") into the cue proposal if/when it lands.
- **Remote-control socket + --toggle** — `--toggle` smuggles in a latch paradigm contradicting PTT; `--status` already proves the lean IPC answer is "read the lockfile pid." Salvage only Cancel — as a hotkey chord (#7), not a socket.
- **SIGUSR1/2 trigger** — a signal can't express PTT hold; would require a whole new non-hold capture mode + self-pipe plumbing (most of what a socket needs anyway).
- **Versioned settings migration via custom Deserializer** — `#[serde(default)]` + per-field defaults + `warn_unknown_keys` already handle add/remove/rename/missing; only a populated-field type change breaks (hypothetical), and the fix is then a localized custom Deserialize — no banking needed (YAGNI).
- **Runtime-only --debug override** — my-voice's verbosity (-v/RUST_LOG) is already non-persisted, so the in-memory-override problem is structurally absent.
- **Persistent SQLite history + WAV-on-disk** — pulls rusqlite + rusqlite_migration + chrono, default-on audio persistence is a privacy regression, and the feature is meaningless without a GUI list view. `--wav` + `--record DIR` already serve inspect + retry-from-audio.
- **"Latest completed = skip-empty" semantics** — dependent footnote to #12; the empty case is already gated at main.rs:939-942, folded into #12 for free.

## Notes

- **Measurement gates before merge:**
  - #15 (audio cues) cannot be exercised by the --wav/WER harness (it bypasses live capture) and there's no mic on the dev box — needs manual on-hardware test, plus real-hardware ALSA testing for input+output stream contention.
  - #19 (stutter/filler) — grep WER-harness transcripts for `\b(uh|um|uhm|umm|hmm)\b` first; if ~0 hits, ship only stutter collapse and drop filler removal entirely.
  - #10 (n-gram vocab), #17 (trailing space), any `SPEECH_RMS`/`PAD_MS` tuning — gate on a WER run over samples/ (baselines: streaming-small 0.154, streaming-medium 0.077) to confirm no regression.
- **Decisions for Alex:**
  - #13 (toggle mode) MUST ship with #8 (max-recording cap) and a distinct "armed" tray state — toggle-without-cap is a worse bug than the problem it solves. Verify autorepeat doesn't false-toggle.
  - #7 (cancel key) is evdev-only; X11 XGrabKey cannot see Escape without a fragile second grab. Accept the X11 gap.
  - #14 (resume) vs #3/#9 — land the cheap correctness items (#3 byte-count, #9 atomic dir) first; resume trades the download module's dead-simple correctness for an edge-case bandwidth win.
- **Brain-wiki notes worth capturing even without code:** the GPU FMA3/SIGILL footnote; the wl-copy forked-daemon-inherits-fds behavior; the load-bearing rule "context-blind injection forbids speculative auto-formatting" (the reason the whole LLM/caps/punct class is rejected).
