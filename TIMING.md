# Accuracy and performance checks

Use `samples/my-samples/` and its corrected `expected.txt` references. Audio remains local. [tools/README.md](tools/README.md) owns recording instructions and test commands.

```sh
cargo build --release --features debug-tools
RESULTS=docs/reviews/latest.txt SKIP_GOVERNOR=1 ./tools/bench-wer.sh
```

## Report fields

- WER is word error rate after lowercasing and removing punctuation except apostrophes.
- Strict WER also counts differences in case and punctuation.
- `segWER` and `segStrict` measure the segmented path against the same references.
- Encode and decode timings measure model work after warmup. They exclude process startup, loading, audio processing and delivery.
- RTF is model processing time divided by audio duration. Lower values are faster.
- RSS is peak process memory. The benchmark reports memory for the whole-recording pass.
- Boundary comparisons show both transcriptions of overlapping audio. The merged line shows delivered text.

The benchmark reports accuracy but does not enforce an accuracy threshold. The ignored Cargo accuracy test enforces 2% for both whole and segmented paths.

## Compare versions

Keep the dataset, model, precision, build profile, CPU affinity and iteration count unchanged. Save each run to a separate report. CPU affinity can change the automatic thread count, so do not compare pinned release results directly with unpinned debug results.

The script normally uses 5 warm passes and reports minimum encode and decode times separately. `SKIP_GOVERNOR=1` leaves the CPU governor unchanged. Otherwise, the script attempts to select performance mode and restores the previous setting on exit.

Timing varies with CPU load and clock speed. It is reported, not asserted. Review per-file errors as well as the aggregate before accepting a boundary change.

Saved historical reports under `tools/` describe their original datasets and settings. They are not the current accuracy baseline.
