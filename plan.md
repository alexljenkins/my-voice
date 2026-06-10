# my-voice — implementation plan

Architectural decisions live in `ideas.md`. This file is the build sequence and the
implementation spec. It is written so an implementer can build each phase without
re-deriving any research. Where exact behavior matters (tensor names, token IDs,
key codes, fallback order), it is spelled out — do not improvise alternatives.

Reference implementation: [voxtype](https://github.com/peteonrails/voxtype) (MIT).
Patterns below marked `[voxtype]` are lifted from its source and verified against
`src/transcribe/moonshine.rs`, `src/hotkey/evdev_listener.rs`, `src/output/mod.rs`,
`src/audio/cpal_capture.rs` as of v0.7.5. Steal patterns, not code structure —
voxtype supports 10 engines, meetings, OSDs; we support exactly one job.

**Scope guardrail:** English only. Push-to-talk only. No VAD, no GUI, no tray, no
meeting mode, no cloud, no multilingual. If a feature is not in this plan, it is
out of scope for v1.

---

## Cargo.toml (complete)

```toml
[package]
name = "my-voice"
version = "0.1.0"
edition = "2021"
description = "Hold-to-talk local voice typing. English. Fast. Lean."
license = "MIT"

[dependencies]
# Audio
cpal = "0.15"
hound = "3.5"            # debug wav dumps + --test mode only

# Moonshine inference (primary backend, always compiled)
# ort 2.x is still rc; pin the rc voxtype ships with. Default features include
# `download-binaries` which fetches a prebuilt static ONNX Runtime at build
# time — keeps the shipped binary self-contained.
ort = { version = "2.0.0-rc.12", features = ["ndarray"] }
ndarray = "0.16"
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }

# whisper.cpp backend — OPT-IN feature (compiles C++ via cmake, slow build,
# most users never need it). `cargo build --features whisper` to enable.
whisper-rs = { version = "0.16", optional = true }

# CLI + config
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
dirs = "5"

# Model download
ureq = "2"

# Util
anyhow = "1"
thiserror = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
num_cpus = "1.16"

[target.'cfg(target_os = "linux")'.dependencies]
evdev = "0.12"           # kernel-level hotkey + uinput passthrough
libc = "0.2"

[target.'cfg(target_os = "macos")'.dependencies]
core-graphics = "0.24"
core-foundation = "0.10"

[features]
default = []
whisper = ["dep:whisper-rs"]

[profile.release]
lto = true
codegen-units = 1
strip = true
opt-level = 3

[profile.dev]
opt-level = 1            # debug-mode inference is unusably slow otherwise
```

Build prerequisites to document in README: Rust stable; `cmake` + a C++ toolchain
only when building `--features whisper`.

---

## Runtime architecture (one page)

Plain threads + `std::sync::mpsc`. **No tokio** — nothing here needs an async
runtime, and dropping it keeps binary and mental footprint small.

```
hotkey thread(s)  --HotkeyEvent-->  main thread (state machine)  ---> injector
(evdev/CGEvent)                          |            ^
                                         v            |
                                   AudioRecorder   ModelCache (lazy load,
                                   (cpal stream)    idle eviction thread)
```

```rust
enum HotkeyEvent {
    Press { clipboard_only: bool },   // clipboard_only = shift held at press
    Release,
}

// Main-thread state machine. Transcription runs synchronously on the main
// thread (~300ms); events arriving meanwhile queue in the channel and are
// processed afterward, in order.
enum State { Idle, Recording { clipboard_only: bool } }
```

Transition rules (implement exactly — these kill the edge cases):
- `Idle + Press` → start recorder, spawn a throwaway thread that calls
  `model_cache.ensure_loaded()` (so model load overlaps with speech), → `Recording`.
- `Recording + Release` → sleep `trailing_silence_ms` (default 150 — PTT radio
  pattern, catches the tail of the last word), stop recorder, gate (below),
  transcribe, post-process, inject (or clipboard), → `Idle`.
- `Recording + Press` → ignore (duplicate; key autorepeat is filtered at the
  source but be defensive).
- `Idle + Release` → ignore (stale release from before startup or after an
  ignored press).

Gate before transcribing — both checks, in order:
1. duration < `min_speech_ms` (default 300) → discard, log debug. Accidental taps.
2. peak amplitude < 0.01 → discard, log debug. Dead mic / pure silence. ASR
   models hallucinate text on silence; never feed them empty air.

Single instance: at startup, take an exclusive `flock` on
`$XDG_RUNTIME_DIR/my-voice.lock` (fallback `/tmp/my-voice-$UID.lock`). If held,
print "my-voice is already running (pid N)" and exit 1. Two daemons grabbing the
same keyboard is chaos; this is ~15 lines, do not skip it.

---

## Module layout (final)

```
src/
  main.rs            # CLI (clap), state machine loop, wiring
  config.rs          # Config struct, TOML load, model path resolution
  audio.rs           # AudioRecorder: cpal stream, mono downmix, resample
  download.rs        # HuggingFace model fetcher (--download)
  model_cache.rs     # lazy load + idle eviction
  text.rs            # post-processing (quote normalization, trim)
  hotkey/
    mod.rs           # HotkeyEvent, spawn_listener() platform dispatch
    linux.rs         # evdev grab + uinput passthrough
    macos.rs         # hidutil remap + CGEvent tap
  transcriber/
    mod.rs           # Transcriber trait + backend factory
    moonshine.rs     # ONNX encoder/decoder, tokenizer, greedy decode
    whisper.rs       # whisper-rs wrapper [cfg(feature = "whisper")]
  injector/
    mod.rs           # Injector trait + detection
    linux.rs         # wtype / xdotool / ydotool / wl-copy / xclip
    macos.rs         # CGEvent unicode injection + pbcopy
```

---

## Config (complete spec)

`~/.config/my-voice/config.toml`, optional — every field has a hardcoded default.
Unknown keys: warn and continue (serde `deny_unknown_fields` OFF, but log keys we
didn't recognize via a post-parse diff if cheap; otherwise skip the diff).

```toml
model = "moonshine-tiny"   # "moonshine-tiny" | "moonshine-base" | /path/to/model/dir | /path/to/file.gguf
model_dir = "~/.local/share/my-voice/models"
quantized = true           # prefer *_quantized.onnx files when present (smaller, faster, negligible WER cost)
threads = 0                # 0 = auto: min(num_cpus, 4)  [voxtype]
load_timeout_secs = 300    # idle eviction; -1 = never unload, 0 = reload every use
hotkey = "CapsLock"        # evdev key name on Linux; fixed CapsLock-via-F18 on macOS for v1
clipboard_hotkey = true    # Shift+hotkey routes result to clipboard instead of typing
grab = true                # Linux: exclusive grab + uinput passthrough (see hotkey spec)
audio_device = ""          # substring match against device name; "" = system default input
min_speech_ms = 300
trailing_silence_ms = 150
injection = "auto"         # auto | wtype | xdotool | ydotool | clipboard
```

Model resolution (`config.rs`):
- `"moonshine-tiny"` / `"moonshine-base"` → `{model_dir}/{name}/` → Moonshine backend.
- Path to a directory → Moonshine backend, expect the three files below inside it.
- Path ending `.gguf` → whisper backend. If built without `--features whisper`,
  fail with: `this binary was built without whisper support; rebuild with
  --features whisper, or use a Moonshine model`.
- Tilde-expand `model_dir` and any paths (use `dirs::home_dir()`, not shellexpand).

CLI (clap derive):
```
my-voice                 # run the daemon
my-voice --download      # fetch the configured model, then exit
my-voice --test          # record 3s from mic, transcribe, print, exit (no hotkey, no injection)
my-voice --list-devices  # print audio input device names, exit
my-voice --config PATH   # alternate config file
my-voice -v / -vv        # info / debug logging (tracing EnvFilter; RUST_LOG also respected)
```

---

## Phase 1 — scaffold + audio + hotkey

Goal: daemon starts, records while hotkey held, saves a playable wav on release.
No model, no injection.

### 1a. Scaffold
`cargo new my-voice`, add deps **except** ort/ndarray/tokenizers/whisper-rs
(keep Phase 1 builds fast), `config.rs` with defaults + TOML load, clap skeleton,
tracing init, flock single-instance check.

### 1b. Audio (`audio.rs`)

```rust
pub struct AudioRecorder { /* device, stream handle, shared buffer */ }
impl AudioRecorder {
    pub fn start(&mut self) -> anyhow::Result<()>;
    pub fn stop(&mut self) -> Vec<f32>;   // 16 kHz mono f32 in [-1, 1]
}
```

- Device selection: `audio_device` substring match (case-insensitive) over
  `host.input_devices()`, else `host.default_input_device()`.
- Build the input stream with the device's **default config** — do NOT request
  16 kHz; most hardware is 44.1/48 kHz and cpal will error. Match on sample
  format and convert: support `F32`, `I16`, `U16` via `cpal::FromSample` [voxtype].
- In the stream callback: convert to f32, **downmix to mono** (average across
  channels), append to `Arc<Mutex<Vec<f32>>>`. Pre-allocate ~16000×60 capacity.
  Hard cap the buffer at 60s of audio (at device rate); drop samples beyond it.
- `stop()`: drop the stream, take the buffer, **resample to 16 kHz** with linear
  interpolation [voxtype — verbatim algorithm, ~25 lines, no rubato dependency]:

```rust
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() { return samples.to_vec(); }
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        out.push(if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else {
            samples.get(idx).copied().unwrap_or(0.0)
        });
    }
    out
}
```

- The stream is created on `start()` and dropped on `stop()`. Mic-open latency
  (~50–100 ms on PipeWire) is covered by human reaction time — people press,
  then inhale, then speak.

### 1c. Hotkey — Linux (`hotkey/linux.rs`)

**The trap this design avoids:** `evdev` grab is per-DEVICE, not per-key. A naive
exclusive grab silences the entire keyboard. The fix is a passthrough proxy:
grab the real device, create a uinput virtual keyboard, and re-emit every event
except the hotkey.

When `grab = true` (default):
1. Enumerate `/dev/input/event*`. A device qualifies if its `supported_keys()`
   contains the target key (KEY_CAPSLOCK) **and** KEY_A (filters out lid
   switches and buttons that report weird key sets). **Skip any device whose
   name starts with `"my-voice"`** — that's our own virtual device; without this
   check, a daemon restart grabs its own output and feeds back forever.
2. For each qualifying device, spawn a thread:
   - `device.grab()` — exclusive.
   - Create a uinput device via `evdev::uinput::VirtualDeviceBuilder`, name
     `"my-voice passthrough"`, registering the source device's full key set.
   - Loop on `device.fetch_events()`:
     - Target key, value 1 (down) → send `Press { clipboard_only: shift_down }`.
     - Target key, value 0 (up) → send `Release`.
     - Target key, value 2 (autorepeat) → swallow, send nothing.
     - **Any other event → `vdev.emit(&[event])` verbatim.** Track shift state
       (KEY_LEFTSHIFT/KEY_RIGHTSHIFT down/up) from events passing through.
3. uinput note: when shift is used for `clipboard_hotkey`, the shift keydown has
   already been forwarded before we see the hotkey. That is fine — a dangling
   shift press+release types nothing.
4. On shutdown (and on panic — use a drop guard): `device.ungrab()`, drop the
   virtual device. A stuck grab means a dead keyboard until reboot; treat
   cleanup as load-bearing.

When `grab = false`, or when uinput open fails (`/dev/uinput` permission):
fall back to **non-exclusive listening** — same event loop, no grab, no uinput,
and CapsLock will also toggle the OS caps state. Log a warning naming the fix:

```
uinput unavailable; running ungrabbed (CapsLock will still toggle).
Fix: sudo usermod -aG input $USER
     echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
     sudo modprobe uinput   # and add 'uinput' to /etc/modules-load.d/
then re-login.
```

Hotplug: out of scope for v1 (voxtype uses inotify on /dev/input; we don't).
If the keyboard reconnects, restart the daemon. Note it in README.

Key-name parsing: accept the evdev names without prefix, case-insensitive:
`"CapsLock"` → `KEY_CAPSLOCK`, `"F12"` → `KEY_F12`, etc. Map via
`evdev::Key::from_str` on `"KEY_" + upper(name)` with a tiny alias table for
CapsLock/ScrollLock spellings.

### 1d. Hotkey — macOS (`hotkey/macos.rs`)

**The trap this design avoids:** CapsLock on macOS does not produce
keyDown/keyUp — it arrives as a `flagsChanged` event, and many keyboards
debounce it in firmware, making hold/release detection unreliable.

Strategy: remap CapsLock → F18 at the HID level on startup, listen for F18.
1. On startup, shell out:
   `hidutil property --set '{"UserKeyMapping":[{"HIDKeyboardModifierMappingSrc":0x700000039,"HIDKeyboardModifierMappingDst":0x70000006D}]}'`
   (0x39 = CapsLock usage, 0x6D = F18 — a key no physical Mac keyboard has).
   On clean shutdown: `hidutil property --set '{"UserKeyMapping":[]}'`.
2. Create a CGEvent tap (`CGEventTapCreate`, session tap, default/active so we
   can suppress) for `keyDown | keyUp | flagsChanged`. Match virtual keycode
   **79** (F18). For Shift detection read `CGEventFlags` shiftKey bit off the event.
3. keyDown with autorepeat field set (`kCGKeyboardEventAutorepeat != 0`) → swallow.
   keyDown → `Press`, keyUp → `Release`. **Return null from the callback** for
   matched events to suppress them; pass everything else through untouched.
4. Run the tap in a `CFRunLoop` on a dedicated thread; re-enable the tap if it
   gets disabled by timeout (`kCGEventTapDisabledByTimeout` arrives as an event
   type — handle it by calling `CGEventTapEnable` again).
5. If `CGEventTapCreate` returns null: print a friendly error — the binary needs
   **Input Monitoring** and **Accessibility** in System Settings → Privacy &
   Security — and exit. Don't limp along silently.

### 1e. Wire it (`main.rs`)
`Press` → `recorder.start()`. `Release` → sleep 150ms → `recorder.stop()` →
write `/tmp/my-voice-test.wav` via hound (16 kHz mono, f32→i16).

**Done when:** hold CapsLock → speak → release produces a wav that plays back
clean at correct pitch, on both a 48 kHz and (if available) 44.1 kHz input
device; other keys keep working while the daemon runs; Ctrl-C restores the
keyboard (Linux) and the CapsLock mapping (macOS).

---

## Phase 2 — Moonshine inference + model download

Goal: transcription printed to stdout. The whole phase is a faithful port of
voxtype's `src/transcribe/moonshine.rs` — when in doubt, read it.

### 2a. Model files & download (`download.rs`)

HuggingFace repos (English-only, MIT):
- `onnx-community/moonshine-tiny-ONNX`
- `onnx-community/moonshine-base-ONNX`

Files per model — exactly these, into `{model_dir}/{moonshine-tiny|moonshine-base}/`:

| repo path | local name |
|---|---|
| `onnx/encoder_model.onnx` (and `onnx/encoder_model_quantized.onnx`) | same basename |
| `onnx/decoder_model_merged.onnx` (and `_quantized` variant) | same basename |
| `tokenizer.json` | `tokenizer.json` |

URL form: `https://huggingface.co/{repo}/resolve/main/{path}`. Download with
`ureq`, stream to `{file}.part`, rename on completion (atomic-ish; survives
Ctrl-C), byte-count progress to stderr. When `quantized = true` download only
the quantized pair + tokenizer; full-precision pair otherwise.

Never auto-download. If the model dir is missing at startup, exit with:
`model not found at {path} — run: my-voice --download`.

### 2b. Transcriber trait (`transcriber/mod.rs`)

```rust
pub trait Transcriber: Send {
    /// audio: 16 kHz mono f32 in [-1, 1]
    fn transcribe(&mut self, audio: &[f32]) -> anyhow::Result<String>;
}
pub fn create(config: &Config) -> anyhow::Result<Box<dyn Transcriber>>;
```

(`&mut self` — ort `Session::run` needs mut; no internal Mutexes needed since
the main loop owns the transcriber exclusively. Simpler than voxtype here.)

### 2c. Moonshine backend (`transcriber/moonshine.rs`) — full algorithm [voxtype]

Constants:
```rust
const DECODER_START_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;
const MAX_TOKENS_PER_SECOND: f32 = 8.0;
const ABSOLUTE_MAX_TOKENS: usize = 512;
```

**Load:**
- Pick `encoder_model_quantized.onnx` / `decoder_model_merged_quantized.onnx`
  when `quantized` and both exist; warn + fall back to full precision otherwise.
- `Session::builder().with_intra_threads(threads)` for each, CPU EP only
  (no GPU providers in v1 — Moonshine tiny on CPU already beats real-time 10×).
- Load `tokenizer.json` with `tokenizers::Tokenizer::from_file`.
- Cache decoder input/output names from session metadata.
- Detect `num_heads` / `head_dim` from the first decoder input named
  `past_key_values*`: its tensor shape is `[batch, num_heads, seq_len, head_dim]`;
  read dims 1 and 3 where positive. Fallback if undetectable: warn and use
  (8, 52) (base); tiny is (6, ~44) but detection always works on the
  onnx-community exports, so don't sweat the fallback.

**Encode:**
- Input tensor: f32, shape `[1, audio_len]`, the raw samples. No mel
  spectrogram, no padding — Moonshine eats raw waveform at native length.
- Run encoder; take its single output (name from metadata, expect
  `last_hidden_state`); keep it for every decoder step.

**Decode (greedy, autoregressive, merged decoder with KV cache):**
- `max_tokens = (duration_secs * 8.0) as usize, clamped to [16, 512]`.
- Partition decoder KV I/O names: inputs starting `past_key_values`, outputs
  starting `present`; within each, split by substring `".decoder."` vs
  `".encoder."`; sort each list (pairing is positional after sorting — keep
  all four lists sorted the same way).
- `tokens = [DECODER_START_TOKEN_ID]`. For `step in 0..max_tokens`:
  - `input_ids`: step 0 → shape `[1, tokens.len()]` with all tokens; step ≥ 1 →
    shape `[1, 1]` with only the last token.
  - `encoder_hidden_states`: the encoder output, every step.
  - Decoder-side KV inputs: step 0 → dummy zero tensors shape
    `[1, num_heads, 1, head_dim]`; step ≥ 1 → the `present.*.decoder.*` outputs
    of the previous step.
  - Encoder-side (cross-attention) KV inputs: step 0 → same dummy zeros;
    step ≥ 1 → the `present.*.encoder.*` outputs **captured at step 0** and
    reused forever. (The merged model emits empty encoder KV on later steps —
    cross-attention KV is only computed once. Caching step 0's is mandatory.)
  - `use_cache_branch`: bool tensor `[1]`, `false` at step 0, `true` after.
  - Run; read `logits` (shape `[1, seq_len, vocab]`), take the last position,
    argmax → next token. If `EOS_TOKEN_ID`: stop. Else push and continue.
- Decode text: `tokenizer.decode(&tokens[1..] as u32s, /*skip_special*/ true)`.

### 2d. Post-processing (`text.rs`)

Applied to every transcription result, all backends:
1. Trim leading/trailing whitespace.
2. Normalize curly quotes to ASCII [voxtype — these literally break wtype]:
   `‘ ’ ‛ ′` → `'`, `“ ” ‟ ″` → `"`.
3. Collapse internal newlines to spaces (PTT output is one utterance; a stray
   `\n` presses Enter in the target app — in a terminal that *executes* it).

### 2e. Wire + verify
`Release` path: gate → transcribe → post-process → `println!`. Log timing at
info: audio seconds, encode ms, decode ms, token count.

**Done when:** `my-voice --test` and the hotkey path both print accurate text;
a ~5s utterance transcribes in < 500 ms on Intel CPU (tiny quantized).

---

## Phase 3 — whisper.cpp backend (`--features whisper`) ✅ DONE

Goal: `.gguf` path in config works. Entirely `#[cfg(feature = "whisper")]`.

1. `transcriber/whisper.rs`: `WhisperContext::new_with_params` (default ctx
   params), one `WhisperState`, run `full()` per call.
2. `FullParams`: `Greedy { best_of: 1 }`, `language = Some("en")`,
   `no_timestamps`, `single_segment = false`, `suppress_blank = true`,
   `n_threads = threads`. Print no whisper.cpp logs (install the no-op log hook
   `whisper_rs::install_whisper_log_trampoline` + tracing).
3. Collect all segment texts, join with a space, then the same `text.rs`
   post-processing.
4. README: recommend `ggml-base.en.gguf` / `ggml-small.en.gguf` (English-only
   variants only — never the multilingual ones, they're slower and worse at
   English at the same size).

**Done when:** pointing `model` at a `.en.gguf` file transcribes correctly;
default build (no feature) still compiles with no cmake installed.

---

## Phase 4 — text injection ✅ DONE

Goal: text lands in the focused window.

### Trait (`injector/mod.rs`)
```rust
pub trait Injector: Send {
    fn inject(&mut self, text: &str) -> anyhow::Result<()>;
    fn name(&self) -> &'static str;
}
pub fn detect(config: &Config) -> Box<dyn Injector>;       // typing path
pub fn clipboard() -> Box<dyn Injector>;                   // clipboard_hotkey path
```

**Rule for all external tools: spawn with `std::process::Command` and pass text
as a single argv element or via stdin. Never go through a shell — transcribed
text can contain quotes, `$`, backticks, anything.** Prefer stdin where the tool
supports it (wl-copy, xclip read stdin; wtype/xdotool/ydotool take an arg after
a `--` separator).

### Linux (`injector/linux.rs`) — session-aware chain

Detect the session: `WAYLAND_DISPLAY` set → Wayland; else `DISPLAY` set → X11;
neither → clipboard-only. Probe = binary exists on PATH (`which`-equivalent).
Selection happens once at startup; log the chosen method at info. But keep the
chain: if a method **fails at runtime** (nonzero exit — e.g. wtype on GNOME,
whose compositor lacks the virtual-keyboard protocol), log a warning, demote it
for the rest of the session, and retry the same text with the next method.

Wayland order: `wtype -- <text>` → `ydotool type -- <text>` → clipboard.
X11 order: `xdotool type --clearmodifiers -- <text>` → clipboard.

ydotool detail [voxtype]: it needs its daemon socket. Before selecting ydotool,
find the socket: `$YDOTOOL_SOCKET` → `$XDG_RUNTIME_DIR/.ydotool_socket` →
`/tmp/.ydotool_socket` → `/run/user/$UID/.ydotool_socket`; pass the found path
as `YDOTOOL_SOCKET` env to the child. No socket → skip ydotool in the chain.

Clipboard fallback (also the `clipboard_hotkey` path): `wl-copy` (Wayland) or
`xclip -selection clipboard` (X11), text via stdin. Then log at info:
`copied to clipboard (no typing tool available — paste with Ctrl+V)`. Do NOT
synthesize a Ctrl+V press: if we had a key-synthesis tool, we'd have typed the
text with it in the first place.

### macOS (`injector/macos.rs`)

Primary — CGEvent unicode injection (no per-character keycode tables, works
with every layout):
- Split text into chunks of ≤ 20 UTF-16 code units (CGEvent's limit).
- Per chunk: `CGEventCreateKeyboardEvent(source, 0, true)` →
  `CGEventKeyboardSetUnicodeString(chunk)` → post; then the matching
  keyUp event, also with the string set. Post to the HID event tap location.
  1 ms sleep between chunks (slow targets drop events otherwise).

Fallback if event posting fails (no Accessibility permission): `pbcopy` via
stdin + the same "paste with Cmd+V" info log. `clipboard()` on macOS = pbcopy.

### Wire
Replace `println!`. `Press { clipboard_only: true }` routes the result to
`clipboard()` instead of the typing injector. Empty post-processed text →
inject nothing, log debug.

**Done when:** text appears in a GUI editor AND a terminal on: Linux/X11,
Linux/Wayland (sway or GNOME — on GNOME expect the wtype→clipboard demotion to
kick in unless ydotool is set up), macOS. Apostrophes, quotes, `$HOME`, and
backticks inject literally.



---

## Phase 5 - audio pre-processing ✅ DONE

Goal: improve the audio input from the mics with as little latency add as possible.

solution: add sonora (Pure Rust WebRTC)
The sonora crate is a complete, pure Rust port of the WebRTC Audio Processing module (M145).  It is the industry standard for a reason: it is battle-tested, extremely fast, and handles edge cases better than custom code.

Why it fits:
Zero External Dependencies: No C++ toolchain, no libwebrtc linking. It compiles as a single static binary.
Performance: Uses SIMD (AVX2 on Linux/x86, NEON on Mac/ARM) for sub-millisecond latency. Benchmarks show it processes a 10ms frame in ~4–13μs.
Features: Includes Noise Suppression (Wiener filter), Automatic Gain Control (AGC), and High-Pass Filtering (removes rumble).
Platform: Native support for Linux and macOS. 
Usage: Add sonora to your Cargo.toml. It provides a simple Processor struct where you push PCM frames.
Integrate it into the system so audio is processed before sent to the transcriber.


---

## Phase 6 — model keep-alive

Goal: lazy load, transparent reload, idle eviction.

```rust
pub struct ModelCache {
    slot: Arc<Mutex<Option<Box<dyn Transcriber>>>>,
    last_used: Arc<Mutex<Instant>>,
    timeout_secs: i64,                     // -1 never evict, 0 always reload
    // + whatever Config bits are needed to (re)construct the transcriber
}
impl ModelCache {
    pub fn ensure_loaded(&self);           // lock slot; if None, build; update last_used
    pub fn transcribe(&self, audio: &[f32]) -> anyhow::Result<String>;
    pub fn start_evict_thread(&self);      // skip entirely if timeout_secs == -1
}
```

- `transcribe()`: lock slot → `ensure_loaded` inline if `None` → run → update
  `last_used` → if `timeout_secs == 0`, drop the transcriber before returning.
- Evict thread: wake every 30s; lock slot (this is the whole concurrency story —
  if a transcription is mid-flight the lock blocks until it finishes); re-check
  `last_used` **after** acquiring; if idle > timeout, `*slot = None`, log info
  `model unloaded after {n}s idle`.
- On `Press`, spawn a thread calling `ensure_loaded()` — cold-start model load
  (~1s) overlaps with the user's speech. `transcribe()` later blocks on the same
  mutex until the load completes, so there is no race and no double load.

**Done when:** first use after idle-evict has only mid-speech load cost; RSS
drops after eviction; rapid press-during-evict doesn't deadlock or double-load
(verify with `-vv` logs).

---

## Phase 7 — polish, errors, packaging

1. **Friendly failures** (each names its fix):
   - no mic → list available devices, mention `audio_device` config + `--list-devices`
   - model missing → the `--download` hint
   - `/dev/input` permission denied → `sudo usermod -aG input $USER` + re-login
   - uinput missing → the udev rule block from Phase 1c
   - no injection tool found → name the packages: `wtype` (Wayland), `xdotool`
     (X11), `ydotool` (universal); meanwhile clipboard mode is active
   - macOS tap creation failed → System Settings path for Input Monitoring/Accessibility
2. **Signal handling**: SIGINT/SIGTERM → ungrab evdev devices, drop uinput,
   restore hidutil mapping (macOS), release flock, exit 0. Use a small signal
   thread + atomic flag; the drop guards from Phase 1 do the heavy lifting.
   This is the difference between "quit" and "keyboard dead until reboot".
3. **Unit tests** (pure logic only — no audio/model/network in CI):
   `resample` (identity, 48k→16k length, sine continuity), config defaults +
   round-trip + model path resolution, quote normalization, key-name parsing,
   max-token clamp, injection chain selection given a fake env/PATH probe result.
4. **README.md**: install, the udev/input-group block, macOS permissions
   walkthrough (incl. that hidutil remap is active while running), config
   reference table, model table (tiny vs base, quantized sizes), troubleshooting
   (GNOME Wayland → install ydotool or live with clipboard).
5. **CI** (GitHub Actions): `fmt --check`, `clippy -D warnings`, `cargo test`,
   `cargo build --release` on ubuntu-latest + macos-latest; one job with
   `--features whisper` on ubuntu only (cmake preinstalled there).
6. `Cargo.toml` metadata (description, license, repo) for `cargo install`.

---

## Phase 8 (stretch, post-v1) — streaming decode

Parked deliberately. Moonshine v2's streaming encoder (Feb 2026) would let us
decode during the hold and flush only the tail on release. Not in v1 because:
batch decode of a 10s utterance is already ~300 ms on Intel — streaming buys
back at most that, in exchange for the only component with **no proven Rust/ONNX
reference implementation** (voxtype has none; the v2 streaming ONNX export needs
verification it even exists on HF). Revisit when v1 is shipped and someone
actually feels the keyup latency. Design sketch stays in git history / ideas.md.

---

## Phase exit criteria summary

| Phase | Deliverable | Key risk (mitigation) |
|---|---|---|
| 1 | hotkey-gated wav capture, both OSes | uinput permissions (ungrabbed fallback); macOS tap permission (hard error with instructions) |
| 2 | Moonshine → stdout | decoder KV protocol (fully specified above; diff against voxtype moonshine.rs when stuck) |
| 3 ✅ | whisper `.gguf` opt-in | cmake build friction (feature-gated, default off) |
| 4 ✅ | text in focused window | GNOME Wayland wtype failure (runtime demotion to next method) |
| 5 ✅ | WebRTC audio pre-processing (HPF + NS + AGC2) | batch post-processing on stop(); no latency added during recording |
| 6 | lazy load + eviction | evict-vs-transcribe race (single mutex owns both — blocked, not raced) |
| 7 | errors, signals, CI, README | stuck keyboard on crash (drop guards + signal handler) |
