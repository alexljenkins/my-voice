# my-voice optimization plan

Offline English-only PTT voice typing. Latency that matters is perceived keyup→text; lean single static binary, CPU-only, <500MB RSS are hard constraints. Below is the master list after deduplication + red-team filtering. The recurring theme: **decode (greedy autoregressive, ~18-25ms/token graph EXEC) dominates perceived latency; encode is only ~10-17% of model time (already measured in TIMING.md), so the big plays are decoder-side RAM/code wins, cheap text-quality wins, and the eventual streaming-encoder export — NOT micro-tuning the inference knobs.**

## Quick reference

| Rank | Optimization | Dimensions | Est. impact | Effort | Verdict |
|------|--------------|-----------|-------------|--------|---------|
| 1 | Merge split decoder → one graph (RAM + LOC + disk) | memory, package, latency | ~125MB RSS small / ~200MB medium; −85 LOC; medium 566→~400MB disk | high (offline re-export) | ✅ **DONE** (2026-06-24) |
| 2 | Heuristic capitalization + terminal punctuation | accuracy | measurable strict-WER drop; large perceived-quality | low | worthwhile |
| 3 | Spoken-form punctuation commands ("comma","period") | accuracy | unlocks punctuation; niche but real | low | worthwhile |
| 4 | Upgrade streaming models to Moonshine v2 | accuracy, memory, package, latency | none — already on v2 | — | ✅ **closed** — already v2 streaming; only #10 remains
| 5 | Capture avg-logprob/confidence; log it (don't gate yet) | accuracy | enables hallucination gate; ~0 runtime cost | low | worthwhile |
| 6 | Arena shrinkage after each transcribe | memory | idle RSS toward weights floor; gate on >30MB | low | worthwhile |
| 7 | Suppress `<<ST_n>>`/`<unk>` special tokens in argmax | accuracy | latent correctness; sub-noise WER | low | worthwhile |
| 8 | Inverse text normalization (numbers→digits) | accuracy | readability win for dev dictation | medium | worthwhile |
| 9 | Default to never-evict for the small model | latency, memory | kills post-idle cold-start cliff | low | worthwhile |
| 10 | Chunked/streaming-encoder export (encode-during-capture) | latency | ~15-17% encode ceiling; constant-TTFT on long clips | high (re-export, not research) | ⏸️ **PARKED** — feasible (v2 streaming arch already in use); gated on causal preproc |
| 11 | A/B NS=Moderate vs Off vs Low | accuracy | possibly free WER on clean dictation; likely small | low | marginal |
| 12 | Soften peak-normalize to downward limiter | accuracy | removes gain-staging confound; clip-dependent | low | marginal |
| 13 | No-repeat-3gram in-loop block | latency, accuracy | shaves a couple cycles on bad clips only | medium | marginal |
| 14 | Spin control off (global pool) | latency | daemon CPU hygiene, NOT latency | low | marginal |
| 15 | Hysteresis on silence-trim start/end | accuracy | correctness hardening on noise-spike heads | low | marginal |

---

## 1. Merge split decoder into one graph — ✅ DONE (2026-06-24)
**Tags:** memory, package, latency (latency ≈ neutral)

> **Shipped.** Merged the already-int8 split pair directly (no re-quant → WER cannot
> move; validated bit-exact, |Δlogits|~1e-5). Re-hosted the 3-file merged layout on
> `Immortalizer/moonshine-streaming-{small,medium}-onnx` (MIT, attributed to Useful
> Sensors / Moonshine). Disk: small 345→~220 MB, medium 566→~366 MB. `src/models.rs`
> re-pinned (sha256 + file lists + approx_mb); `DecoderGraph::Split` + `decode_split`
> (~150 LOC) deleted from `src/transcriber/moonshine.rs`, enum collapsed to a struct.
> WER on samples/ (merged, current harness): small **0.124** agg / 0.160 strict;
> medium **0.024** agg / 0.095 strict — no regression (the merge is bit-exact vs the
> split pair, so it equals split on the same harness; the older 0.154/0.077 snapshot
> predates other branch changes). The merge toolchain (`model_scripts/`) has served
> its purpose and was removed. The historical analysis below is kept for the record.

**Convergence:** This is the single most-proposed idea — it surfaced independently under decoder-algo, alt-models, memory, and download-size lenses. All four agree on the same core change; they disagree only on which dimension to sell it as. The honest framing is **RAM + code-deletion + one-fewer-file**, with disk as a bonus and latency as ~zero.

**What & mechanism:** streaming-small (default) and streaming-medium ship two decoder files — `decoder_model_quantized.onnx` (no-past, step 0) and `decoder_with_past_model_quantized.onnx` (steps 1..N) — both built eagerly at load (`Moonshine::load`, moonshine.rs:121-122) and held resident for the process lifetime. The two graphs are ~90% identical weights, so keeping both roughly doubles resident decoder weight RSS. A merged-decoder export (use_cache_branch flag) — exactly what tiny/base already use via `DecoderGraph::Merged` / `decode_merged` — replaces both with one weight set and lets `decode_split` (~85 lines) be deleted. The loader already falls through to the merged path when no with-past graph is present.

**Expected impact:** RSS: ~125MB saved (small), ~200MB (medium) — material toward <500MB, decisive for medium which is already ~562MB of resident weights. Disk: medium 566→~400-450MB (red-team flags the brief's ~365 figure as optimistic; merged ≈ the larger no-past graph, not half). Code: ~85+ lines deleted, dual-graph branch collapses to one decode path. Latency: net zero (a tiny step-0 use_cache_branch dummy-KV pass, immaterial; per-step exec unchanged because graph EXEC dominates and the math is identical). **The per-step-exec "win" some lenses chased does NOT exist — do not sell it as latency.**

**Effort:** high — and the effort is OUT of the repo. Models are self-hosted (alexjenkins89/*, re-exported from Mazino0 split-only source; no upstream merged form exists). Requires: re-export from PyTorch/optimum to merged → re-quantize int8 → re-host on HF → re-pin 6 sha256 in models.rs. The Rust diff is the easy 10%.

**Red-team take:** Worthwhile and constraint-aligned (helps memory + lean, deletes code, offline/English/CPU untouched, merged path already proven in-binary for tiny/base so ort 2.0.0-rc.12 runtime risk is nil). Load-bearing risks: (1) **WER must be re-validated, not assumed** — re-quant to a merged graph changes QDQ/If-node boundaries and can shift WER even with nominally identical weights; gate hard on streaming-small 0.154 / medium 0.077 holding on samples/. (2) A hash mismatch bricks downloads for all users. (3) Measure step-0 latency and peak RSS post-export rather than trusting the headline numbers. Priority depends on current medium RSS headroom; if <500MB is already comfortable for the default small model, this drops toward a lean/disk win.

**Where in code:** `src/transcriber/moonshine.rs` DecoderGraph::Split (72-77), decode_split (375-459, deletable), load split-detection branch (115-136); `src/models.rs` STREAMING_FILES_SMALL/MEDIUM file lists + sha256 (137-145, 165-173).

---

## 2. Heuristic capitalization + terminal-punctuation pass
**Tags:** accuracy

**What & mechanism:** Moonshine emits lowercase with little punctuation. Add a pure-Rust pass in `src/text.rs::post_process` AFTER curly-quote/newline normalization and BEFORE `apply_corrections`: (1) capitalize first alphabetic char; (2) capitalize after `.`/`?`/`!` + space (guard against abbreviations/decimals by requiring prior run isn't a single letter or digit); (3) upcase standalone `i` and `i`+contraction via the existing word-boundary scan already in apply_corrections (lines 48-70). Optional auto-period gated behind a config bool, **default off** (wrong for terminals/search boxes).

**Expected impact:** Large perceived-usability gain for short PTT dictation ("Push the commit." vs "push the commit"). **Correction to the original proposal:** this DOES move a metric — tests/wer.rs already computes a strict (case+punct-preserving) WER (`normalize_strict`, line 54) and samples/expected.txt references are fully cased+punctuated, so case is currently the dominant strict-error source; this should measurably reduce strict WER. The gated/normalized WER won't move (expected).

**Effort:** low — <20 lines, no dep, no binary growth, <10us/utterance.

**Red-team take:** Worthwhile, clean fit (English-specific text cleanup is exactly the constraint the app leans into; no inference/ort/tensor interaction). Order BEFORE apply_corrections so user corrections like "GitHub" win. Use strict WER as the acceptance check. Keep auto-period default-off. Add unit tests for i'm/i've/i'll/i'd, decimal "3.14", multi-sentence. Validated only on 12 clips; multi-sentence mid-utterance punctuation stays unhandled (acceptable — bookends are the bulk of the gap).

**Where in code:** `src/text.rs::post_process` (insert at lines 9-19); reuse boundary scan from apply_corrections (48-70); config gate in `src/config.rs` (11-29, warn_unknown_keys 86-100).

---

## 3. Spoken-form punctuation commands
**Tags:** accuracy

**What & mechanism:** PTT users currently cannot produce punctuation/newlines — post_process collapses `\n`/`\r` to spaces. Add an opt-in spoken-command map applied as a phrase-level replace before newline-collapse and before apply_corrections, reusing the EXISTING longest-first word-boundary matcher in apply_corrections (multi-word, case-insensitive, boundary-safe). **Ship the punctuation half first** ("comma"→",", "period"→".", "question mark"→"?", parens, etc.) — clean and low-risk.

**Expected impact:** Unlocks punctuation users can't otherwise dictate. Medium value for code/markdown/terminal dictation, negligible for prose. Not a WER mover. Niche but real, and a capability gap that exists *because* of the lean/English/PTT design.

**Effort:** low for punctuation; the newline half is more.

**Red-team take:** Worthwhile, split it. The punctuation half is near-zero-risk (ordinary printable chars every injector handles), reuses the matcher as-is, default-OFF config bool to avoid literal-dictation collision ("add a comma here"). **DEFER the newline/paragraph half:** verified `AtSpiInjector` (`src/injector/linux.rs:189`) maps `\n` (0x0A) via the Latin-1 fallthrough, NOT the Return keysym 0xFF0D — so "new line"→`\n` SILENTLY no-ops on the AT-SPI backend while working on wtype/ydotool. Newlines need a backend-aware Return path before shipping, and re-introduce the terminal-Enter-executes hazard text.rs deliberately guards. Watch spacing artifacts on paren/bracket tokens.

**Where in code:** `src/text.rs::post_process` (intercept before newline-collapse, line 15) reusing apply_corrections matcher (26-72); config in `src/config.rs`; newline backend fix in `src/injector/linux.rs:189`.

---

## 4. Upgrade streaming models to Moonshine v2
**Tags:** accuracy, memory, package, latency

> **Corrected 2026-06-24 — effectively closed.** We are **already on Moonshine v2
> streaming** (`UsefulSensors/moonshine-streaming-{small,medium}` → Mazino0 int8 ONNX).
> The "current models are v1-era split-decoder" premise below is **wrong** — there is no
> v1→v2 accuracy/size upgrade left to do. The only v2-related work remaining is the
> streaming-encoder **re-export** tracked in #10. Keep this section only as the pointer
> to #10.

**Convergence:** Pairs naturally with #1 (do the merged export at the same time, never re-export twice) and with #10 (v2 is the streaming-native architecture that makes encode-during-capture feasible).

**What & mechanism:** Current default/-medium are self-hosted int8 exports of v1-era split-decoder Moonshine. v2 (arXiv 2602.12241) ships official English streaming checkpoints at smaller fp sizes with better benchmark WER. v2 is still encoder + (no-past/with-past) decoder with step-0 cross-attn KV — fits `decode_split` as-is. Action: export v2 small/medium to int8 ONNX, verify the encoder still ingests raw `[1,N]` (frontend is conv-based 50Hz CMVN/asinh, historically embedded in transformers exports — must confirm), re-host, swap checksums + approx_mb.

**Expected impact:** Likely net-positive on two axes: download/RAM shrink (~345MB→~130-160MB for small, estimated) and probably-improved WER. **Honest caveat:** the "halve the WER" pitch is apples-to-oranges — it compares v2-small fp benchmark WER against the repo's int8-on-samples number. Fair comparison is v2-int8-on-samples vs v1-int8-on-samples on the 12 short PTT clips, which nobody has run; for short utterances the streaming-context advantage compresses. Best independent guess: meaningful size win, small-to-moderate accuracy win. All unverified until bench-wer.sh runs.

**Effort:** medium — and it's export work, not a re-host: **v2 streaming ONNX exports are not yet published.** Must export + int8-quant + validate.

**Red-team take:** Worthwhile, directionally correct, two-dimensional upside, zero constraint violations. Gates: (a) confirm exported encoder takes raw `[1,N]` (else run_encoder needs a feature frontend — lean strain); (b) split-decoder I/O names match decode_split; (c) **license must be MIT/Apache** before re-hosting. Sequence: export+int8 → verify encoder signature on one clip → bench v1-int8 vs v2-int8 head-to-head on samples/ BEFORE swapping defaults.

**Where in code:** `src/models.rs` MODELS[] (hf_repo, checksums, approx_mb); `src/transcriber/moonshine.rs` run_encoder() (verify raw vs feature input) + decode_split().

---

## 5. Capture avg-logprob / confidence and log it
**Tags:** accuracy

**Convergence:** Five hallucination-lens candidates collapse here (avg-logprob gate, top-2 margin, compression-ratio, confidence UX, negative test set). The red-team consensus: **the cheap, safe, high-value move is INSTRUMENTATION, not a live gate.** The gate itself is deferred until calibration data exists.

**What & mechanism:** Both decode loops compute argmax over the materialized vocab slice and DISCARD the winning logit (`moonshine.rs:342`, `:420`, argmax at `:587`). Accumulate the chosen token's log-softmax (one max + sum-exp over the in-cache slice — sub-0.1ms/token, off the 18-25ms/token critical path). Change `Transcriber::transcribe` to return a small struct `{text, avg_logprob, n_tokens}` and LOG it on every transcribe (and in --wav/bench output). Optionally also log top-2 margin as a diagnostic.

**Expected impact:** Zero runtime cost, no behavior change, no regression risk. Unlocks a data-driven threshold for a future hallucination gate instead of guessing −1.0 (a Whisper-float32 convention that doesn't port to int8 Moonshine, which also has no no_speech token).

**Effort:** low. Trait return-type change ripples to mod.rs warm(), both decode fns, ModelCache::transcribe (model_cache.rs:95), main.rs gate (936), and ~4 test stubs — small but touches the hot trait.

**Red-team take:** Worthwhile as instrumentation; **do NOT ship a live injection-skip gate yet.** The dominant Moonshine failure mode (repetition loops) is already caught by truncate_loop/collapse_runaway; pure silence is caught by min_speech_ms(300) + peak<0.01 pre-gates. The residual a logprob gate catches is the narrow "quiet-but-real audio that passes peak" slice, whose size is unknown. A false-positive silent drop on quiet real speech is the worst PTT UX (user spoke, nothing appeared). Promote to an active gate only after a recorded silence/noise corpus exists and shows clean separation. Build that negative corpus alongside (a few real noise/cough WAVs with expected="" — note run_wav at main.rs:238 currently BYPASSES the discard gates, so testing the gate needs plumbing or asserting on raw output).

**Where in code:** `src/transcriber/moonshine.rs:342/420/587`, return at :213; `src/transcriber/mod.rs:38`; `src/model_cache.rs:95`; `src/main.rs:936`.

---

## 6. Arena shrinkage after each transcribe
**Tags:** memory

**What & mechanism:** Keep the ORT CPU arena but return unused blocks to the OS at the end of each Run via `RunOptions` config entry `memory.enable_memory_arena_shrinkage=cpu:0`. Wire per-keyup (NOT per-token) at the existing `.run()` sites. PTT has long idle gaps, so shrinking after each keyup brings idle RSS back toward the weights floor between dictations.

**Expected impact:** Idle RSS drops from per-utterance peak toward weights+baseline. Likely tens of MB for streaming-small (may be immaterial vs the 500MB target); potentially more meaningful for medium where weights alone strain the target. Zero latency benefit; tiny per-keyup shrink cost. Magnitude unverified.

**Effort:** low — `RunOptions::add_config_entry` (run_options.rs:355) and `Session::run_with_options` (mod.rs:253) are both confirmed present in ort 2.0.0-rc.12; it's a per-call wrap at 4 sites.

**Red-team take:** Worthwhile, real, not a re-tread of the rejected memory_pattern(false) (different knob). Caveat: the default split decoder holds growing KV in owned Rust DynValues that drop at transcribe end, so the arena's retained peak is activation scratch only — reclaim may be small. **Gate on measurement:** long noisy clip, VmRSS (not VmHWM) before/after; ship if reclaim > ~30MB, else reject. A cheaper substitute for idle RAM is just lowering the eviction timeout, but shrinkage wins in the active back-to-back-dictation window.

**Where in code:** `src/transcriber/moonshine.rs` run() calls at 202 (encoder), 338 (merged), 402/415 (split).

---

## 7. Suppress `<<ST_n>>` / `<unk>` special tokens in argmax
**Tags:** accuracy

**What & mechanism:** Vocab is 32000 text tokens + 768 `<<ST_0..767>>` at ids 32000-32767 plus `<unk>=0`. argmax (`moonshine.rs:587`) ranges over the full 32768 logits, so on edge audio the model can argmax onto a special token. Restrict argmax to `logits[1..32000]` (skip `<unk>=0` and the ST tail), keeping EOS=2 reachable. One helper applied at both call sites.

**Expected impact:** **Correction to the original proposal:** the visible-garbage symptom it claims to fix is ALREADY handled — `decode(&ids, true)` at moonshine.rs:253 uses skip_special_tokens=true, and all ST tokens + `<unk>` are flagged special, so they're stripped before injection today. The real residual benefit is preventing a stripped-but-KV-poisoning special token from derailing subsequent real tokens (it still consumes a decode step and feeds back as the next input_id + KV state). Safe correctness hardening; WER movement within noise. Speed benefit (argmax over 32000 vs 32768) is negligible — argmax isn't the bottleneck.

**Effort:** low — pure slice indexing, no dep, no constraint strain.

**Red-team take:** Worthwhile as cheap correctness hardening, **not** as an accuracy win — bundle opportunistically with another moonshine.rs change (e.g. #5's argmax-site edit) rather than a standalone PR/validation cycle. Add a unit test asserting masked argmax never returns id 0 or ≥32000. Fix the changelog framing (don't claim it fixes visible garbage).

**Where in code:** `src/transcriber/moonshine.rs:587` argmax + call sites :342 (merged), :420 (split).

---

## 8. Inverse text normalization (numbers → digits)
**Tags:** accuracy

**What & mechanism:** Moonshine emits spoken-style numbers ("twenty twenty six", "three point one four", "first"). Add a small rule-based, table-driven English number-words→digits converter in `src/text.rs` running over the word stream (cardinals/ordinals/decimals first; punt dates/times/currency). Pure Rust, no heavy WFST/pynini dep.

**Expected impact:** Medium readability win for a frequent dev-dictation annoyance ("port eight thousand" → "port 8000"). Zero latency/RAM. Opt-in (default-off) caps realized impact. No guaranteed WER movement (possibly negative against spoken-form refs) — treat readability as the metric.

**Effort:** medium — ITN is edge-case-heavy ("twenty twenty" date vs cardinal, "and three" contexts); a conservative cardinal/ordinal/decimal-only v1 with a real test matrix is ~150-250 lines.

**Red-team take:** Worthwhile, on-constraint (CPU/offline/English-aligned, no cargo-cult). **Corrections:** text.rs does NOT do capitalization restoration, so the real ordering constraint is ITN BEFORE apply_corrections (so a user "8000" rule doesn't re-fire). No objective regression gate exists in-repo (samples/ WER can't validate readability), so ship default-OFF with an explicit `#[cfg(test)]` matrix and HARD-punt dates/times/currency/ranges — that tail is where lean dies. Lower priority than #1/#4 since those move the numbers that define the product.

**Where in code:** new fn in `src/text.rs` invoked from post_process (9-19); config flag in `src/config.rs` (13-29, 86-100).

---

## 9. Default to never-evict for the small model
**Tags:** latency, memory

**What & mechanism:** Cold-start is hidden by a startup preload (main.rs:438) and a keydown preload thread (main.rs:464) that races the user's speech. Eviction defaults to 1800s idle (config.rs:38). After a genuine 30+ minute gap the model is evicted and the next keydown can pay ~1s load + a full warm() inference pass inline if the utterance is too short for preload to finish. `-1` never-evict is already wired (config.rs:38 guard) — flip the default for streaming-small.

**Expected impact:** Eliminates the post-idle cold-start cliff on the first short utterance after a long break. **Correction:** NOT a recurring tax — every transcribe resets last_used, so intermittent all-day dictation never trips the 30-min timer; the cliff is one hit after a real gap. Costs permanent baseline RSS = loaded-model footprint (safe for streaming-small under 500MB).

**Effort:** low — one-line default change.

**Red-team take:** Worthwhile mainly because effort is ~zero and downside is bounded. **REJECT the footprint-conditional-default sub-idea** (adds branching against lean); ship a flat `-1` default and let medium users (566MB, opt-in heavy) override their own timeout. The "skip inline warm if recently warmed" half is a no-op — load_locked already warms once per load and never re-warms a resident model. Pairs with #1 (merged decoder halves medium RAM, making never-evict safer there).

**Where in code:** `src/config.rs:38` load_timeout_secs default; `src/model_cache.rs` start_evict_thread (109-128); preload threads `src/main.rs` 438, 464.

---

## 10. Chunked / streaming-encoder export (encode-during-capture)
**Tags:** latency

> **Verified 2026-06-24 — feasible, PARKED.** The models we run
> (`moonshine-streaming-{small,medium}`) are already Moonshine **v2**: an
> **ergodic sliding-window-attention encoder** (windows `(16,4)`/`(16,0)`; ~16-frame
> ≈320 ms left context, ≤4-frame ≈80 ms lookahead; no encoder positional embeds),
> **NOT** full bidirectional attention. So the encoder is streamable *by
> architecture* — only the int8 ONNX we ship (Mazino0) is a whole-waveform **static**
> graph with no cached-state I/O. This is a **re-export, not a research project**:
> float `model.safetensors` is public + **MIT**, and upstream `moonshine-ai/moonshine`
> already caches the encoding and adds audio incrementally (working reference). No
> retraining, no architecture change, **no WER risk from the encoder side**. That
> corrects the "needs a custom KV-cached streaming-state export — an ML-export project"
> framing in the red-team below (overstated for these specific models). Real remaining
> cost: (1) export a chunked/stateful encoder ONNX exposing the local-context cache →
> int8 re-quant → WER-validate vs current; (2) the **still-true causal-preprocessing
> blocker** — `normalize_peak` (global whole-utterance gain) + `trim_silence` run at
> keyup and must be made causal, else mid-hold encode sees a different input
> distribution. Ceiling unchanged: encode is ~15-17% of model time (TIMING.md); decode
> stays 100% on the critical path → a long-clip / constant-TTFT win, small on short PTT.
> Refs: arXiv 2410.15608 (v1, full-attn), arXiv 2602.12241 (v2 ergodic streaming),
> HF `UsefulSensors/moonshine-streaming-small`, `Mazino0/moonshine-streaming-small-onnx`,
> GitHub `moonshine-ai/moonshine`.

**Convergence:** The most-proposed latency idea (encoder, alt-models, pipeline-overlap lenses all land here). The brief calls it "biggest latency lever" — **the red-team consensus is that this is overstated and it must be gated.**

**What & mechanism:** Today the encoder runs once over the whole waveform at keyup (`run_encoder`, moonshine.rs:179), fully on the critical path. With a v2 sliding-window encoder exported to carry incremental KV state, feed audio chunks during the hold so at keyup only the final tail + decode remain.

**Expected impact:** **Capped low.** TIMING.md already records encode at ~17% of model time (warm encode 729ms / decode 3548ms); MEMORY witnesses base at ~10%. So the ceiling on this whole lever is ~15% of *model* time, and even less of *perceived* latency (which also carries the fixed 150ms trailing buffer + batch APM). Decode (greedy autoregressive, ~18-25ms/token) stays 100% on the critical path. Local measurement on streaming-small: 14.75s clip = enc ~610 / dec ~970; encode ~40% of inference on long clips but the dominant half is still decode.

**Effort:** high, with hard prerequisites.

**Red-team take:** Marginal NOW; becomes worthwhile only AFTER v2 (#4) lands. Three blockers: (1) the v1 exports are full-attention — the naive "re-encode the growing buffer every 500ms" version is O(prefix²) and contends for the shared 8-thread pool, can net WORSE on CPU; the correct version needs a custom KV-cached streaming-state encoder export (an ML-export project). (2) **Preprocessing causality conflict (load-bearing):** `normalize_peak` (audio.rs:233) applies a GLOBAL gain from the whole-utterance peak (unknown until keyup), and `trim_silence` reshapes frame boundaries — so any mid-hold encode runs on a different input distribution than the final pass, risking WER regression, unless preprocessing is made causal (its own design project). (3) Strains lean (worker thread + streaming buffer + feed_chunk/finalize trait surface). Gate behind: v2 int8 ONNX exists + streaming-state export proven feasible + a measured encode_ms fraction that justifies it. Do NOT build now.

**Where in code:** `src/transcriber/moonshine.rs:179-205` run_encoder; `src/audio.rs:198-233` buffer/normalize/trim; `src/main.rs:474-512` keyup handler.

---

## 11. A/B NS=Moderate vs Off vs Low
**Tags:** accuracy

**What & mechanism:** `NoiseSuppressionLevel::Moderate` is hardcoded (audio.rs:325) and never isolated (only NS=High-vs-Moderate was compared). Moderate-vs-Off is the untested lever. Run bench-wer.sh in three configs holding AGC+normalize+trim fixed; if clean clips regress with NS on, drop or condition NS.

**Expected impact:** Most likely 0 to ~1-2 WER points; possible regression-removal worth a few points only if NS is actively hurting; possibly net-zero (gentle WebRTC NS ≈ neutral on real captures).

**Effort:** low — but no env toggle exists, so it's edit-line-325-and-rebuild per config (3 builds) or wire a one-line override.

**Red-team take:** Marginal — the A/B is cheap and answers an open question, but the cited "de-noising hurts" literature is from heavier neural denoisers / cloud contexts that only partially transfer to gentle WebRTC NS at Moderate. samples/ (one user, ~homogeneous SNR) is a weak basis for a global default flip. **REJECT the SNR-gated middle path** as the outcome (adds a tunable heuristic fighting lean) — if Off wins, just delete NS (leaner); if neutral/loses, keep Moderate and close it.

**Where in code:** `src/audio.rs:325`; validate via tools/bench-wer.sh.

---

## 12. Soften peak-normalize to a downward limiter
**Tags:** accuracy

**What & mechanism:** `normalize_peak` (audio.rs:233) scales the whole buffer so the loudest sample hits 0.95 (up to 8x), on TOP of AGC2 — a second gain stage. A transient outlier (click/breath) quietens real speech; quiet clips get 8x noise floor. Replace upward gain with a downward-only limiter (clamp ~0.99) + optional gentle RMS target, preserving the clip guarantee for the --wav 16-bit write.

**Expected impact:** Small and clip-dependent (~0.00-0.02 abs WER), concentrated on peaky/quiet clips. Primary value is removing a gain-staging confound so #11's NS A/B is interpretable.

**Effort:** low.

**Red-team take:** Marginal. **Drop the "fights int8 calibration" justification** — verified the encoder feeds plain `[-1,1]` f32 with no 32768 scaling, so that argument is bogus. Keep the double-gain-staging + transient-outlier reasoning. Must NOT fully remove normalization (keep the downward limiter for APM overshoot + WAV clip). Change one gain knob per bench run. Best sequenced with #11 since they share the gain chain. Expected WER delta bounded by Moonshine's own gain-robustness.

**Where in code:** `src/audio.rs` normalize_peak (233-243), order in process_capture (218).

---

## 13. No-repeat-3gram in-loop block
**Tags:** latency, accuracy

**What & mechanism:** Add an in-loop no-repeat-3gram mask at the argmax sites (moonshine.rs:342/420): if emitting `next` would complete a recently-seen 3-gram, mask to −inf and take 2nd-best. Operates on the logits slice already in hand (near-free).

**Expected impact:** Tens to low-hundreds of ms saved ONLY on clips that already trip the repetition guard (a minority); zero change on clean speech (output byte-identical). Far smaller than the "stop paying for the whole tail" framing — truncate_loop already runs in-loop and breaks after 3 reps, so today's wasted tail is small, and max_tokens=duration*8 caps total work.

**Effort:** medium.

**Red-team take:** Marginal. Implement ONLY the no-repeat-3gram block; **drop the confidence-EOS half** (cargo-culted from Whisper's no_speech-token design that doesn't port to int8 Moonshine; route silence-hallucination to the existing audio gate instead). Keep truncate_loop/collapse_runaway as belt-and-suspenders (don't delete — they guard a different case). Verify how often samples/ clips even trip the guard before spending effort; addressable win may be near zero.

**Where in code:** `src/transcriber/moonshine.rs:342,420` (argmax sites), 587 (argmax), 493 (truncate_loop), 465 (collapse_runaway).

---

## 14. Spin control off on the global thread pool
**Tags:** latency (really: CPU/power hygiene)

**What & mechanism:** The global ORT intra-op pool busy-waits (spins) before blocking. For a daemon idle 99% then bursting once per keyup, spinning wastes idle/burst CPU. Set `GlobalThreadPoolOptions::with_spin_control(false)` in `init_thread_pool` (the only correct layer — global pool is committed).

**Expected impact:** ~0 to sub-millisecond per transcribe (within noise floor) — **NOT a latency win.** Real but minor secondary win: lower idle/burst CPU and power. The proposal's `spin_duration_us` sweep is fictional — only a boolean exists in this stack.

**Effort:** low — one line.

**Red-team take:** Marginal. ort's own doc recommends spin OFF for infrequent use = exactly PTT, and edge builds may already ship it off. The repo's own KEY FINDING (exec dominates, copies aren't the bottleneck) plus the noise floor that rejected inter-threads/memory_pattern predict the latency effect lands in noise. Worth a near-free A/B (spin on vs off CPU-during-idle), set explicitly to false if not already; bundle into a larger threading touch rather than its own PR. Do NOT sell as latency.

**Where in code:** `src/transcriber/mod.rs` init_thread_pool.

---

## 15. Hysteresis on silence-trim start/end
**Tags:** accuracy

**What & mechanism:** `trim_silence` takes the first/last RMS-window crossing (audio.rs:369-407), so a single head noise spike defeats leading trim. Require 2-3 consecutive speech windows before marking start/end.

**Expected impact:** Low — correctness hardening on noise-spike heads; plausibly <0.01 aggregate WER, could be 0/noise.

**Effort:** low.

**Red-team take:** Marginal but a clean correctness fix on its own merit, cheaper and safer than the rejected Silero-VAD swap. **Do NOT bundle the noise-floor-relative k*floor threshold** — trim runs AFTER normalize so the 0.02 threshold is already semi-relative; the incremental win is small and gated behind quiet-capture tails. Ship hysteresis alone; only pursue adaptive-k if the WER harness shows quiet-clip edge truncation actually exists in samples/.

**Where in code:** `src/audio.rs` trim_silence (369-407).

---

## Rejected / low-value

- **Tune spin_duration_us sweep** — knob is fictional in ort 2.0.0-rc.12 (boolean only); see #14 for the descoped version.
- **Persist ORT-optimized graph to disk (with_optimized_model_path)** — felt win gated behind eviction + ultra-short utterance, which keydown-preload + speech already hide; doubles cache disk + portability footgun. Cheaper: raise eviction timeout (#9).
- **Bump ORT to 1.22-1.24 via api-* feature** — already on 1.24 via default download-binaries; api-* gates Rust API surface, doesn't choose the binary. No-op; verify with `cargo tree -i ort`, then close.
- **XNNPACK execution provider** — ONNX guidance: quantized → MLAS/CPU EP, XNNPACK is float-first and *discouraged* for int8. Fights the committed global pool. Expected net-zero-to-negative on x86 int8 decoder GEMM.
- **Disable memory-pattern for RSS** — arena shrinkage can't lower VmHWM (the only RSS metric tracked); disabling pattern doesn't cap length-driven peak. Likely 0MB on the measured metric.
- **Pin GraphOptimizationLevel::Level3 / ship .ort / offline transformers.optimizer** — ort default is already ENABLE_ALL (Level3 ≠ All in rc.12; pinning Level3 would *downgrade* to ENABLE_LAYOUT). Offline fusion duplicates work ORT already does at load. Keep at most a one-off graph dump to confirm QDQ fusions fire; pin `All` not `Level3` if pinning at all.
- **Per-channel dynamic int8 requant** — gain below the 12-clip harness resolution; per-channel under dynamic quant only touches weights. Do the cheap diagnostic (inspect existing scale shapes) first; full re-quant unjustified at current eval scale.
- **Mixed-precision fp32 lm_head** — *worthwhile-leaning but demoted on latency risk:* lm_head is in the with-past graph on the default split decoder (runs every token), so fp32 there adds per-token cost on the critical path. Dynamic quant already keeps fp32 accumulation, shrinking the recoverable error. If pursued, dump lm_head quant params first and A/B latency, not just WER. Re-promote only if a one-model spike shows WER gain with no latency regression.
- **int4/uint4 (MatMulNBits) for download size** — ~10x slower CPU decode hits the dominant per-token loop; hardware gaps hard-error with no GPU fallback. Strictly dominated by #1 (decoder merge) on size/speed/accuracy.
- **fp16 CPU path** — slower than int8 on commodity CPUs (cast overhead, weaker fusion). Non-action; fold into a code comment.
- **N-gram self-speculative decoding** — needs ONNX re-export for dynamic decoder query length, reintroduces KV copies the codebase deliberately eliminated, ~120-180ms only on long clips, near-zero on the common short PTT case.
- **Lower MAX_TOKENS_PER_SECOND 8→6** — *validate-don't-ship:* truncate_loop already catches typical runaways before the budget fires; risk of mid-utterance truncation (6 tokens/sec ≠ 6 words/sec). Ship only on 0.000 WER delta.
- **int8 KV cache** — KV is ~1-2MB at PTT seq lengths (negligible vs 500MB); no fused CPU int8-KV kernel in ort 2.0; would add dequant to the exec-dominated loop. Correctly rejected.
- **Batch start-prompt / skip step-0 dummy KV** — dummy already hoisted out of the step loop; default model uses split (not merged) path; sub-microsecond.
- **Encoder zero-pad/f32-copy elimination** — ~30us memcpy vs 18-25ms/token; trait-signature ripple to save noise.
- **Encode-overlap with the 150ms sleep / speculative partial encode** — global normalize_peak + trim + whole-utterance non-chunked encode make any pre-stop encode non-reusable; collapses into #10.
- **Streaming resample/APM during capture** — single-digit ms (preprocessing is <1% of perceived latency); normalize+trim are unstreamable anyway; --wav parity risk.
- **Persistent APM/resampler instance** — ~1-3ms saved on a multi-hundred-ms path; resampler half is moot on native-16k; AGC2 cross-utterance state bleed risk.
- **Eliminate post-keyup buffer copies** — microseconds vs decode; trait Cow-ification adds complexity exceeding the payoff.
- **Parakeet-TDT-0.6b-v2 backend** — sherpa reports ~1.2GB RAM = ~2.4x over the <500MB target; needs mel frontend + TDT loop + sentencepiece (~400-600 LOC). Route its real draws (low-hallucination → #5; punctuation → #2) to lighter levers. Reconsider only if 500MB is relaxed.
- **Whisper / distil-whisper / turbo backend** — 30s mel-pad + larger decoder = worse short-utterance latency; foreclosed by the Moonshine lineage + English/lean constraints. Memo, not an optimization.
- **Silero VAD endpointing** — adds a stateful ONNX session + segmentation glue for a batch-trim problem PTT doesn't have (full buffer in hand, key brackets the utterance). A/B the RMS threshold (#15) first.
- **Dither before encoder** — dither cures log(0) singularities in log-mel; Moonshine eats raw waveform (no log). The HPF half is already available via sonora's `high_pass_filter` config field — one-liner A/B, not custom DSP.
- **Rubato FFT tail-artifact fix** — kept-frames truncation + silence-trim already absorb the transient; only the cold (non-16k) resample path, which baselines don't exercise. Cheap diff only if a 48k user reports tail garbage.
- **VAD-replace the 150ms trailing sleep** — the sleep compensates for cpal *capture latency* (audio still in the driver ring buffer at keyup), not for waiting on the user; an RMS check on "the tail" inspects bytes that aren't there yet → would clip the last word on the common case. The salvageable thread is device-latency-calibrated trailing, a separate low-priority task.
- **Token-streaming / incremental injection** — Linux injectors spawn a subprocess per call (per-chunk spawn > per-token decode); retroactive loop guards mean you can't un-type. Negative EV.
- **Decode on a worker thread / warm injector during decode** — injector handles are acquired ONCE at startup (verified injector/linux.rs), so there's nothing per-utterance to warm; PTT FSM has no concurrent work to service. Adds async-shaped complexity against the no-async constraint for ~0.
- **Pipeline post-capture audio chain off critical path** — APM is <~10% of perceived latency and sits behind the 150ms sleep + decode; streaming APM risks WER via AGC2 trajectory change. Native-16k already no-ops resample.
- **Lazy-load with-past decoder** — both graphs needed per utterance >1 token, so zero peak-RSS win; lazy/overlap variant either no-ops against warm-up or regresses cold-start. Superseded by #1.
- **External-data + mmap weights** — int8 prepacking re-copies weights into private dirty heap regardless of mmap (and prepacking feeds the fast kernels — disabling it regresses the dominant graph-exec). Re-host burden for sub-MB gain.
- **Disable CPU memory arena** — proposed config key is fictional in rc.12; needs unsafe ortsys FFI; saving cargo-culted from a large fp model. If ever pursued, measurement spike behind the existing peak_rss harness first; ranks below #1.
- **Disable prepacking on cold decoder** — MS's 1436MB figure is fp/1.85GB cloud; int8 saving likely single-MB and lands a latency adder on the heaviest step-0. Moot after #1 deletes the separate initial graph.
- **Shorten eviction to 120-300s for idle RAM** — pessimizes the common bursty case (short utterance after a pause eats full cold load+warm); RSS target already met by the small model. Opposite of #9; offer a documented low-RAM profile instead of changing the default.
- **Custom-build ORT with op reduction / .ort minimal build** — optimizes binary .text (~12MB ort_sys) which is invisible behind a 345MB model download; minimal_build disables runtime graph opt (latency risk); permanent op-config maintenance. If size ever matters, only the low-risk op-reduction-keeping-.onnx path (~2-5MiB).
- **ort default-features=false to drop tracing+tls-native** — native-tls is build-dep-only (ort-sys build script), never in the shipped binary; dropping tls-native without tls-rustls breaks the build-time download. At most a tls-native→tls-rustls build-hygiene swap.
- **TLS duplication (drop rustls+ring)** — no runtime TLS duplication exists (the second stack is ort-sys's build-time fetch); system-TLS swap breaks the single-static-binary constraint. 0.6MB of a 345MB-model app.
- **Feature-gate keybind popup / tray stack** — ~2.3% of a binary that's 99% ORT; default-off dead-ends the only UX surface for non-technical users. The one clean sub-win: drop winit's `wayland-csd-adwaita` feature (~178KiB tiny_skia + tiny-xlib, no UX cost) — take just that.
- **opt-level="s"** — ORT C++ is clang-built (untouched by Rust opt-level); ~<1MB off a 25MB binary already fat-LTO'd, with a small resample/tokenizer latency risk. Scoped per-crate stanzas add config bloat.
- **panic="abort"** — the daemon has TESTED unwind/mutex-poison panic recovery (model_cache.rs:50-71, audio.rs:187-203); abort turns a single bad utterance into a hard SIGABRT killing the always-on daemon. Sub-percent size win on a model-dominated download. Tests pass under the test profile so CI wouldn't catch the regression.
- **zstd-precompressed ONNX download** — measured ratio on the REAL streaming int8 files is ~18-20% (not the proposal's 30-33% from the tiny encoder), a one-time ~70MB saving bought with permanent dual-artifact HF-sync burden + a new crate. Shipping merged-decoder-only (#1) is the better download lever.
- **q4f16 encoder** — encoder is ~21% of the package; ~3% download saving on the accuracy-sensitive graph, likely fp16 up-convert on CPU EP (no runtime gain, maybe slower). Dominated by #1.
- **Shared tokenizer cache file** — 3.6MB (~0.6-1%) only for multi-model users; breaks "rm -rf model_dir/<name> uninstalls a model" and adds orphan-GC. Skip unless models.rs is already being rewritten for #1.
- **HTTP Range resume** — *real but low-priority:* one-time first-run-only, conditional on a mid-transfer stall; no runtime/latency/size benefit. Only worth bundling into a broader download-robustness pass.
- **int4 decoder as default** — verified HF tree: int8 decoder is the SMALLEST variant; q4/bnb4 are ~73% LARGER. Premise (int4 beats int8 on size) is false for these exports. Self-rejecting.
- **Confidence-driven tray "not typed" / clipboard route** — dependent on an unbuilt calibrated metric; clipboard auto-route silently clobbers the clipboard. Fold the tray-state half into #5's gate (when/if built); drop auto-route.
- **Compression-ratio hallucination gate** — largely redundant with the in-loop guards + duration*8 cap; the cited exact-cycle example is already caught by truncate_loop; silent-drop on false positive is worse than a deletable hallucination. Prefer #5.
- **Temperature/beam-fallback re-decode** — no confidence gate exists yet to trigger it; existing loop guards cover the repetition mode; doubles decode latency on exactly the worst inputs; beam KV bookkeeping fights the move-based loop. Reject.
- **Top-2 margin as primary gate** — feasible/cheap but no logprob gate exists to piggyback on (proposal's premise is false), unvalidated for int8 Moonshine, per-model threshold burden, no noise corpus to calibrate. Log it as a diagnostic under #5 only.
- **Decode-time logit biasing (shallow fusion) for custom vocab** — greedy decode has one hypothesis, so it can't rerank; to recover a misheard term you must boost first-token unconditionally = the false-insertion bug. Cheaper alternative: fuzzy/phonetic post-hoc matching on the emitted string.
- **Tiny INT8 punct+truecase ONNX head** — news-trained, over-punctuates terse PTT; needs SentencePiece (new dep) + RSS headroom; adds latency to the perceived-critical path. Do the heuristic (#2) first; only escalate if it proves insufficient on long-form.

## Notes — needs real measurement to decide

- **#1, #4 (re-export work):** every accuracy claim is unverified until v1-int8 vs new-int8 runs head-to-head on samples/ via tools/bench-wer.sh. Hard WER gate (0.154 / 0.077) before swapping defaults or re-pinning checksums.
- **#6 (arena shrinkage):** decide on a long-noisy-clip VmRSS-before/after measurement (not VmHWM); ship only if reclaim > ~30MB.
- **#10 (chunked encode):** encode_ms fraction is already ~15% per TIMING.md — confirm on-box and gate the whole multi-day lever on it; do not build before v2 + a proven streaming-state export.
- **#11/#12 (audio gain/NS):** run as a sequenced A/B, ONE gain knob per bench run, or results are uninterpretable; samples/ is one user — be cautious about a global default flip.
- **#5 gate promotion + #13 confidence-EOS:** both need a recorded silence/noise corpus that does not exist today (no mic on dev box per MEMORY; use the --wav permanent-capture path elsewhere). Until it exists, instrument and log only.
- **General:** dev box has no mic — all live-path claims validate via `--wav` + WER harness; box load drifts, so interleave variants and report medians (per the ASR-verify protocol).
