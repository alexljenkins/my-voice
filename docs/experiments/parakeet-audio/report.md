# Parakeet audio processing result

No processing variant qualifies. Keep raw resampled audio for Parakeet.
Current processing ties raw normalized errors. Its punctuation and case gain does not meet the acceptance rule.

All variants completed 30/30 recordings in both whole and daemon-segmented paths.
Coverage is 560.567 seconds and 821 reference words.
The original cohort has 16 recordings and 395 words. The added cohort has 14 recordings and 426 words.

## Accuracy

These counts apply to both paths. Every variant had identical whole and segmented scores per recording.
WER means word error rate. The supplied scorer remains unchanged.

| Audio | Normalized errors | Normalized WER | Strict errors | Strict WER | Original normalized | Added normalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 14/821 | 1.705% | 55/821 | 6.699% | 5/395 | 9/426 |
| Current processing | 14/821 | 1.705% | 49/821 | 5.968% | 5/395 | 9/426 |
| Peak normalization | 15/821 | 1.827% | 51/821 | 6.212% | 6/395 | 9/426 |
| Peak normalization and trimming | 20/821 | 2.436% | 60/821 | 7.308% | 7/395 | 13/426 |

Current processing changes no recording's normalized error count.
Peak normalization adds 1 error in the original cohort.
Adding trimming adds 2 errors in the original cohort and 4 in the added cohort.
Raw strict errors split into 20 original and 35 added errors.
Current processing splits into 18 and 31. Peak normalization splits into 20 and 31.
Normalization with trimming splits into 25 and 35.
The full per-recording differences, including strict differences, are in `screen/per-recording.tsv`.
Transcripts and both score breakdowns remain in `screen/events.jsonl`.

## Time and memory

Times below sum the second pass for all 30 recordings. They exclude loading and audio processing.
Each segmented second pass includes all inference calls for that recording.
The longest recording exercised 2 segments. All other recordings exercised 1 inference segment.

| Audio | Whole inference seconds | Segmented inference seconds | Whole peak RSS MiB | Segmented peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Raw | 8.497 | 8.393 | 379.8 | 383.8 |
| Current processing | 8.003 | 8.192 | 379.8 | 382.7 |
| Peak normalization | 8.352 | 9.588 | 379.6 | 383.6 |
| Peak normalization and trimming | 8.149 | 8.216 | 380.1 | 383.0 |

RSS is the maximum resident process memory reported by GNU time.
These are CUDA diagnostic processes, not measured live-daemon or CPU-only memory figures.
Variants ran in blocks. Timing differences include desktop load and do not establish a speed improvement.
Events also retain process wall times, individual encode/decode timings, and per-recording RSS.

## Controls

The experiment starts at `de2958d` and uses the new Rust Parakeet backend.
The runtime is transcribe.cpp `18947cf405f957195dd457db4e34c1cff9e88889` through the existing CUDA install.
The model hash is `8440125213cb92625e3993ec751c3919dd329da9ae9149951cd6f95dd8bafe81`.
No decoding, text processing, corrections, references, or scoring rules change.
The inference process receives no labels. All variants use the same patched binary.
Input hashes, source hashes, binary hash, build command, and runtime hash are in `sources.json`.

Only 1 inference process ran at a time under 4 GiB RAM, zero swap, and 4 pinned cores.
`screen-cgroup.txt` records the applied limits. The supervisor completed without a pressure stop.
The sampled minimum GPU reserve was 4,343 MiB. Minimum available system RAM was 18,707 MiB.
Maximum sampled full memory pressure was 0%. The unit reported zero swap.
No desktop applications or installed daemon were stopped or changed.

## Implementation

`Config::uses_parakeet` matches the named model and custom GGUF route.
The daemon worker and diagnostic paths select raw resampled audio for that backend.
The shared audio choice preserves every input sample, including overlap and silence.
Moonshine retains the existing persistent processing path.
Capture queues, overflow reporting, speech gates, segmentation, and text joining remain unchanged.
The temporary environment hook exists only in `experiment.patch`, not in the final source.

## Limits

This is a measured no-improvement result for these 30 personal recordings and these generic processing variants.
It does not establish an advantage on other microphones, noise conditions, models, or weights.
Using raw audio gives up the current path's 6 strict-error improvement, as required by the normalized acceptance rule.

## Final verification

The final app completed another 30 whole and 30 segmented checks without the measurement hook.
All 60 transcripts, normalized scores, and strict scores exactly match the screened raw baseline.
`final-comparison.json` records that comparison. `final-sources.json` pins the final source and binary.
Both cohorts retain 5/395 and 9/426 normalized errors, with 55/821 strict errors overall.

The final diagnostic peaks were 379.5 MiB whole and 383.3 MiB segmented.
Second-pass inference totals were 14.113 seconds whole and 15.492 seconds segmented.
Compilation ran alongside this accuracy verification, so these times are not a performance comparison.
Minimum sampled GPU reserve was 4,366 MiB. Available system RAM stayed above 17,491 MiB.
The supervisor completed without a pressure stop or swap-out. Maximum sampled full memory pressure was 0%.

The package tests passed 145 tests. The live overlay check and default-model accuracy check remain ignored.
The dedicated runner above supplies the complete Parakeet accuracy check.
Both diagnostic and default feature sets passed Clippy with warnings denied. Formatting and diff checks passed.

```bash
cargo test --locked --release --features debug-tools -j 4
cargo clippy --locked --release --all-targets --features debug-tools -j 4 -- -D warnings
cargo clippy --locked --release --all-targets -j 4 -- -D warnings
cargo fmt --check
git diff --check
```

These Cargo commands use the runtime and target environment in `README.md`.
Raw stderr and GNU time files remain byte-for-byte in each run's `diagnostics.tar.gz`.
`diagnostic-hashes.json` records each archived file's hash. Resource logs use gzip compression.
