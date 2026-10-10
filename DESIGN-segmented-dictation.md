# Segmented dictation

## Current behavior

Recordings under 50 seconds stay together until release. Between 50 and 60 seconds, the required trailing pause decreases linearly from 800 ms to 120 ms. At 60 seconds, recording continues in a new segment even without a pause.

`segment_pause_ms` can raise the starting pause above 800 ms. The 50–60 second window is fixed. Older `segment_max_ms` settings are ignored with a warning and disappear when configuration is saved.

Release retains the configured `trailing_silence_ms` capture delay, then queues the tail. It never waits for the 50-second threshold.

## Capture and inference

The capture callback appends native-rate audio. The daemon checks for a segment every 200 ms while recording. A fixed deadline prevents incoming events from postponing the check. A single ordered worker resamples, processes and transcribes drained audio. Inference never runs in the daemon loop or capture callback.

Segment duration includes copied overlap. A pause before the final spoken word selects up to 2 seconds of overlap for the next segment. Continuous speech without a suitable pause has no overlap.

A forced split drains exactly 60 seconds of native audio, counting the overlap. A poll usually arrives up to 200 ms late. The next buffer keeps the new overlap, then every sample past 60 seconds. The WAV diagnostic path uses the same split. A release that arrives after 60 seconds also splits at the limit before the final tail. Speech counts come from the drained samples, so the kept overflow is counted once, in the next segment.

Every model call accepts at most 60 seconds of 16 kHz audio. Record mode, whole-WAV runs and a release merged with a queued segment can still create longer buffers. For those, the inference wrapper divides the buffer into balanced chunks without discarding samples. This is a safety fallback, not the normal split path.

Capture preallocates 62 seconds. It reports overflow only past 62 seconds, which means a stalled poll, not normal 200 ms poll delay. Capture never truncates. The worker queue retains one pending segment before stopping an overloaded hold with a visible error. Stopping retains the captured tail and merges it into the pending segment for transcription.

## Text delivery

The joiner emits prior words and holds final punctuation until the next segment. It aligns up to 2 overlapping words and removes at least 1 leading word when audio overlaps. This can delete a word when the model disagrees about the overlap.

The September 2026 personal comparison tested preserving unmatched words. It increased segmented errors from 35 to 46 across 395 reference words. The more accurate existing rule remains. Revisit this choice with naturally recorded long dictations and corrected labels.

Typed delivery sends new text in order. Clipboard mode accumulates the full hold. Automatic clipboard fallback copies the accumulated hold after all results complete. A new press waits until the previous hold finishes delivery. A failed transcription reports a visible error and clears boundary context. Later successful segments still reach the app.

## Main verification flow

[Personal recording tests](tools/README.md) use `samples/my-samples/`. Unit tests cover the 50-second minimum, linear pause thresholds, 60-second forced split, capped drains after a late poll or release, the capture overflow threshold, overlap, release tails and bounded inference without sample loss.

The current personal clips include a 56.4-second recording. They verify whole-recording accuracy and release behavior. Synthetic long audio checks the timing and sample-preservation rules, but does not establish recognition quality on natural long speech. A 68.2-second repeated recording split at 52.2 seconds and peaked at 568 MiB in the diagnostic process. This exceeds the previous 500 MB target. The requested 50–60 second window remains; daemon memory needs a separate live measurement.

The 2026-10-10 check rejected a 58-second minimum. Keeping the 56.4-second eval recording whole produced repeated text and lost its ending. The existing 52.2-second pause split kept the ending with 2 errors across 72 reference words. A 620.4-second repeated replay reached its final segment. Repeated audio checks continuation, not natural-rant accuracy.
