# Segmented dictation (streaming feel + unbounded hold length)

## Outcome

While the push-to-talk key remains held, completed phrases appear after natural pauses. Recording continues while earlier phrases transcribe. Releasing the key flushes the final phrase. A hold may be arbitrarily long without silent audio loss or unbounded audio memory.

Injected chunks are final; this is not a preview/replace UI and not true incremental model streaming.

## Current gaps

1. No feedback until release: text appears only after the full hold.
2. `AudioRecorder` silently stops appending after `MAX_SECONDS` (`src/audio.rs:14`).
3. Running transcription in the daemon loop would block release, UI, reload, and audio-error handling while the live stream continues filling its buffer.

## Approach

Split a hold into bounded audio segments:

- Prefer a natural boundary after a sustained silence.
- Force a boundary at a maximum segment duration, even during continuous speech.
- Drain audio without stopping the cpal stream.
- Send drained segments to one ordered transcription worker.
- Return text to the daemon for mode-aware joining and injection.

Pipeline:

```text
cpal capture -> bounded live buffer -> ordered segment queue -> transcription worker
             -> daemon result message -> hold accumulator -> typing / clipboard
```

One worker preserves segment order and keeps the mutable model serialized. The daemon never performs inference inline while recording.

## Configuration

Add:

```toml
segment_pause_ms = 800
segment_max_ms = 30000
```

- `segment_pause_ms`: independent from `trailing_silence_ms`. Segments through 3s require at least 800ms of silence.
- `segment_max_ms`: 30s soft maximum. From 3s to 48s, the required pause falls linearly from 800ms to 120ms. At 48s (`segment_max_ms + 18000ms`), split unconditionally.
- Poll capture state every 200ms while recording (see the `recv_timeout` note under `src/main.rs` — no ticker thread).

Both keys must also be added to `Config` + `Default` and to the `known` array in `warn_unknown_keys` (`src/config.rs:100`), or a config that sets them logs spurious unknown-key warnings, and to the config block in `README.md`. Neither belongs in `reload_actions`: they are read live at drain time, so a reload needs no rebuild.

These are initial values requiring an ear-check across quiet/loud microphones. Do not reuse `trim_silence`'s `SPEECH_RMS = 0.02` directly: that threshold applies after NS, AGC, and normalization, while segmentation observes raw native-rate samples.

## Design

### `src/audio.rs`

Add a non-stopping drain API, conceptually:

```rust
pub enum DrainReason {
    Pause,
    MaxDuration,
    Release,
}

pub struct DrainedSegment {
    pub raw: Vec<f32>,
    pub raw_rate: u32,
    pub observed_speech_ms: u64,
    pub reason: DrainReason,
}
```

The drain returns **raw samples only**. Resampling and `process_capture` happen on the transcription worker, not on the daemon thread: `apply_audio_processing` builds a fresh WebRTC APM and walks the segment in 10ms frames, which would block the daemon loop — hotkey release, tray updates, audio-error handling — on every drain. `mem::take` and return.

`AudioRecorder::try_drain_segment(segment_pause_ms, segment_max_ms)`:

- Inspect only enough trailing raw audio to update windowed RMS/silence state; do not clone the growing buffer on every poll.
- Maintain whether speech has actually been observed plus consecutive silent duration. Total buffer duration is not proof of speech.
- Use a raw-input RMS threshold that is separately named, tested, and logged. If one fixed threshold is unreliable during microphone testing, replace this detector with an adaptive noise-floor/VAD implementation; do not borrow the post-processing threshold.
- Before 3s, drain when speech was observed and trailing silence reaches 800ms.
- From 3s to `segment_max_ms + 18000ms`, lower the required pause linearly from 800ms to 120ms. At the endpoint, drain unconditionally during uninterrupted speech.
- `mem::take` the buffer under its mutex, then release the lock immediately. No processing under the lock, and none on the daemon thread.
- A `MaxDuration` drain with no speech observed (peak below the raw silence floor) is discarded in the drain and never enqueued — holding the key silently must not spend inference on pure noise.
- Keep the cpal stream alive throughout.
- Reset per-buffer detector state after each drain.

`stop_with_raw` remains the release path in concept: stop the stream first and take the final raw buffer, but enqueue processing on the worker like every other segment. Silent/too-short tails may be discarded downstream.

Remove silent truncation from `append_mono`. Preallocate 60s as emergency capacity, but do not treat capacity as a write cap: `append_mono` continues appending if it is exceeded and sets an overrun flag for the next poll to log and force-drain. At 48kHz mono f32, 60s is about 11MB. Temporary growth is safer than audio loss. Normal operation hard-splits at 48s.

### Native-rate capture

`select_stream_config` already prefers 16 kHz native capture, in which case `resample` is a no-op and segments concatenate cleanly. On a device that only offers 48 kHz, each segment goes through its own `FftFixedOut` with a zero-padded tail (`src/audio.rs:344`), so boundaries carry a small resampler discontinuity. Accepted: the 16 kHz-native path is the supported one, and the artifact sits inside a pause in the common case.

### Ordered transcription worker

Add one worker with bounded request/result channels.

Request fields:

- `hold_id`: monotonically increasing identifier.
- `segment_index`: monotonically increasing within the hold.
- raw native-rate samples plus their rate.

Result fields:

- `hold_id` and `segment_index`.
- `Result<Option<String>, String>` where `None` means gated silence/short audio or empty transcription.

The worker owns `Arc<ModelCache>` and performs, in order: `resample` → `process_capture` → gates → `cache.transcribe` → `post_process`. It does not inject or mutate UI state. A single worker guarantees ordered results without reorder buffering, and `ModelCache`'s existing `Mutex<Option<Box<dyn Transcriber>>>` already serializes it against the keydown preload and the evict thread — no new locking.

`load_timeout_secs = 0` means "drop the model after every `transcribe`" (`src/model_cache.rs:114`), which under segmentation becomes a full reload *per segment*. The worker must hold the model resident for the duration of a hold and only honour the 0-timeout drop after the last segment of that hold completes.

### Cumulative hold gate

`handle_utterance`'s `min_speech_ms` check is currently per utterance (`src/main.rs:924`). Applying it independently per segment silently deletes words: a forced split immediately followed by release can produce a short but speech-bearing tail. Evaluating only the first segment is also wrong because it can delete a legitimate short opening phrase before later speech makes the hold valid.

- Track cumulative `observed_speech_ms` across the hold from the raw detector.
- Transcribe speech-bearing segments immediately to preserve worker throughput, but defer delivery of their text until cumulative observed speech reaches `min_speech_ms`.
- Once the hold qualifies, deliver all deferred non-empty results in order, then deliver later results normally.
- If keyup and all pending results arrive before the hold qualifies, discard the deferred text as a short hold.
- Individual segments retain peak / speech-observed gates but have no duration floor.

### Per-segment normalization is intentional

`apply_audio_processing` builds a fresh APM per call and `normalize_peak` picks a fresh gain per call (`src/audio.rs:298`). Per segment this is an improvement, not a regression: today one loud bang anywhere in a 60s hold sets the peak for the entire recording and pushes all the speech down. Segmenting confines that to the segment containing the bang, and the very next segment recovers full level. Keep `process_capture` exactly as-is, called once per segment. The cost is a possible small level/noise-floor step at a boundary, which is the cheaper side of that trade.

Use a bounded queue sized for a small number of segments. Normal inference should outrun segments. If the queue fills, keep recording and surface/log backpressure; never block the cpal callback or silently discard audio. Before implementation, choose one explicit overflow behavior: temporarily retain one pending drained segment in daemon memory, then report an error and stop the hold if capacity is still exhausted on the next poll.

### `src/main.rs`

Extend `DaemonMsg` with `SegmentComplete`. There is **no** `CaptureTick` message and no ticker thread.

Replace `for msg in daemon_rx` (`src/main.rs:453`) with a loop that blocks on `recv()` while `Idle` and on `recv_timeout(200ms)` while `Recording`, treating the timeout as the poll. This removes the ticker thread, the tick message, the coalescing requirement, and the possibility of a tick backlog outright — and it leaves an idle daemon fully asleep instead of waking five times a second forever, which matters on battery.

- On press: allocate `hold_id`, reset segment index and accumulated text, start capture, preload model, enter `Recording`.
- On poll timeout while recording: try to drain; enqueue any segment without waiting for inference.
- On release: preserve the existing `trailing_silence_ms` sleep, stop/drain the tail, enqueue it, and mark the hold released. Do not return to fully idle/ready until every queued segment for that hold has completed.
- On segment completion: ignore results from cancelled/stale hold IDs; otherwise join and inject according to mode.
- Defer config reload until capture has stopped and all results belonging to the hold are complete.
- Audio failure stops capture but does not invalidate already queued segments; finish those, then show the microphone error.

Recommended state shape:

```rust
Recording {
    hold_id: u64,
    clipboard_only: bool,
    next_segment: u32,
    pending_segments: usize,
    observed_speech_ms: u64,
    deferred_text: String,
    accumulated_text: String,
    delivery_mode: DeliveryMode,
    clipboard_deferred: bool,
    released: bool,
}
```

Factor the current `handle_utterance` into transcription and injection stages. `Result<Option<String>>` is required: current `Result<()>` cannot distinguish successful injection from a discarded or empty segment, so it cannot safely control spacing/accumulation.

### Joining and injection

Join non-empty segment text with one ASCII space. Do not add a leading space before the first non-empty result.

- Change `Injector::inject` to return the effective `DeliveryMode::{Typed, Clipboard}`. This reports what actually happened on every platform, including Linux chain demotion and macOS CGEvent → pbcopy fallback.
- **Typing mode:** inject only the newly joined chunk, prefixed with a space after the first successfully injected chunk.
- **External typing speed:** use 2ms key hold and delay for `ydotool`. Keep 1ms spacing for `wtype` and `xdotool`. Zero-delay typing drops events in some target apps.
- **Clipboard-only mode:** append to `accumulated_text`, then overwrite the clipboard with the complete accumulated hold after every result. On release, clipboard therefore contains the full dictation rather than only the last segment.
- **Automatic clipboard fallback:** when `inject` first returns `Clipboard`, mark the hold clipboard-deferred. Keep accumulating subsequent results but do not inject them. After keyup and all segment results complete, overwrite the clipboard once with the complete accumulated hold. The fallback call may have temporarily copied only its current chunk; the final write must replace it with the full hold.

Append every accepted non-empty result to `accumulated_text` before attempting delivery; this is the recovery source if typing later falls back. Update delivered-spacing state only after delivery succeeds. In clipboard-deferred mode, accumulation continues without delivery. An empty/discarded segment changes neither.

### Post-processing boundary trade-off

Independent transcription changes semantics at segment boundaries:

- Model punctuation/capitalization may show seams.
- Multi-word corrections can fail when their phrase spans two segments (`src/text.rs:51`).
- Boundary words may be less accurate without surrounding audio context.

Accept these as v1 trade-offs, but do not claim `post_process` is fully boundary-independent. Natural 300ms pauses may expose more seams; forced splits may still expose them too. If testing shows visible regressions, retain a small audio overlap for forced splits or add a boundary-aware text joiner as a later change.

### `--record` diagnostics

Segment draining must not lose earlier diagnostic audio.

Write one raw and one processed WAV per drained segment, sharing `hold_id` and zero-padded `segment_index` in filenames. Do not accumulate an entire hold merely to preserve the old single-WAV shape; that would reintroduce unbounded memory.

## Failure semantics

- Transcription failure affects that segment, reports an error, and does not reorder later results.
- Injection failure stops further delivery for the hold and retains/logs accumulated text where possible; it must not silently claim success.
- Queue overflow stops the current hold with a visible error after the single pending-segment allowance is exhausted.
- Release during inference is handled immediately by the daemon; final UI state waits for pending results.
- A new press while the previous released hold is still completing is ignored until delivery finishes (v1 simplicity). This window is longer than today's, so the tray must stay in `Transcribing` for its whole duration — otherwise the dead keypress reads as a dropped hotkey.

## Verification

### Unit tests

- Pause drain requires observed speech plus sustained raw silence.
- Short silence does not drain.
- The pause threshold shrinks from 300ms to 120ms across the 30s soft window.
- At 30s, a 120ms voice gap drains continuous speech. At 48s, uninterrupted speech drains unconditionally.
- Drain resets detector state and leaves stream logically active.
- Empty/all-silent segments do not alter first-segment spacing.
- Segment results remain ordered and stale `hold_id` results are ignored.
- Explicit clipboard mode contains the complete accumulated hold after each result.
- Automatic clipboard fallback defers subsequent delivery and contains the complete hold after keyup/all results.
- Typing receives only incremental chunks with exactly one join space.
- Multi-word correction split across segments is documented by a regression test as an accepted limitation.
- Queue overflow cannot grow memory without bound.
- Cumulative speech crossing `min_speech_ms` releases deferred text in order.
- A hold that never reaches cumulative `min_speech_ms` delivers nothing.
- A short speech-bearing tail after a forced split is delivered once the hold has qualified.
- A `MaxDuration` drain with no observed speech is discarded without enqueuing.
- Exceeding 60s emergency capacity continues capture, forces a drain, logs, and never drops samples.

### Manual acceptance

- Hold, speak two phrases separated by ~1s: first text appears before release; final text reads naturally with one space.
- Speak continuously beyond 60s: no audio disappears; forced segments appear and capture continues.
- Release while a segment transcribes: tail is captured and delivered in order.
- Repeat with Shift+hotkey and configured clipboard injection: clipboard ends with the complete hold.
- Exercise automatic typing failure/fallback: clipboard still contains the complete hold.
- Test quiet mic, loud mic, background fan/noise, short hesitation, and a 1s deliberate pause.
- Run with `--record`: every segment has matching raw/processed WAVs and no prior segment is lost.
- On the weakest target laptop, a forced `segment_max_ms` split's text lands before the next forced split does — if inference cannot outrun capture, lower `segment_max_ms` rather than growing the queue.
- Clap or knock mid-hold: only the segment containing it is quiet; the following segment recovers full level.
- Idle daemon shows no periodic wakeups (no ticker thread).

## Out of scope

- Preview overlay or replacement of already injected text.
- Parallel transcription; it complicates ordering and competes for the same model.
- Frame-level incremental encoder/decoder with persistent KV cache. The current Moonshine backend still runs a full transcription pass per segment (`src/transcriber/moonshine.rs:9-11`).
