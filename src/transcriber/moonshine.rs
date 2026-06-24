//! Moonshine ONNX backend: encoder + autoregressive greedy decode over the raw
//! 16 kHz waveform (no mel spectrogram, no 30s padding).
//!
//! Two decoder graph shapes are supported, detected from the files on disk:
//!
//! * **Merged** (`moonshine-tiny`/`-base`): one `decoder_model_merged.onnx`
//!   switched by a `use_cache_branch` flag. KV names are `past_key_values.*` /
//!   `present.*`. Faithful port of voxtype's backend.
//! * **Split** (streaming `-small`/`-medium`): a no-past `decoder_model_quantized.onnx`
//!   for step 0 and a `decoder_with_past_model_quantized.onnx` for later steps. Self-attn
//!   KV (`past_self_*` / `present_self_*`) grows each step; cross-attn KV is
//!   computed once at step 0 and fed back as `present_cross_*_orig`. The
//!   streaming encoder also takes an `attention_mask` input. We run it as a
//!   single push-to-talk pass over the whole utterance, not chunk-by-chunk.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{DynValue, Tensor, ValueType};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use super::Transcriber;
use crate::config::Config;

const DECODER_START_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;
/// Streaming encoders reshape the waveform into 80-sample (5 ms) frames.
const STREAMING_FRAME: usize = 80;
const MAX_TOKENS_PER_SECOND: f32 = 8.0;
const ABSOLUTE_MAX_TOKENS: usize = 512;
const SAMPLE_RATE: f32 = 16_000.0;

/// ort's `Error<R>` embeds non-`Send`/`Sync` context (the builder, the rejected
/// array), so it can't cross `?` into `anyhow`. Flatten it to its `Display`.
trait OrtExt<T> {
    fn ort(self) -> Result<T>;
}
impl<T, E: std::fmt::Display> OrtExt<T> for std::result::Result<T, E> {
    fn ort(self) -> Result<T> {
        self.map_err(|e| anyhow!("ort: {e}"))
    }
}

pub struct Moonshine {
    encoder: Session,
    tokenizer: Tokenizer,
    encoder_input_name: String,
    encoder_output_name: String,
    /// Present only on streaming encoders — fed all-ones (full audio, no pad).
    encoder_mask_input: Option<String>,
    decoder: DecoderGraph,
}

enum DecoderGraph {
    /// One merged graph switched by `use_cache_branch`.
    Merged {
        session: Session,
        /// `past_key_values.*` names partitioned + sorted; pairing with the
        /// matching `present.*` outputs is positional after sorting.
        decoder_kv_input_names: Vec<String>,
        encoder_kv_input_names: Vec<String>,
        decoder_kv_output_names: Vec<String>,
        encoder_kv_output_names: Vec<String>,
        num_heads: usize,
        head_dim: usize,
    },
    /// Separate no-past (step 0) and with-past (later) graphs.
    Split {
        initial: Session,
        with_past: Session,
        n_layers: usize,
    },
}

impl Moonshine {
    pub fn load(dir: &Path, config: &Config) -> Result<Self> {
        let threads = config.resolved_threads();

        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            bail!(
                "model file missing: {} — run: my-voice --download",
                tok_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tok_path.display()))?;

        // --- Encoder.
        let enc_path = pick(dir, encoder_candidates(config.quantized))
            .ok_or_else(|| anyhow!("no encoder .onnx found in {}", dir.display()))?;
        let encoder = build_session(&enc_path, threads)?;
        let encoder_input_name = encoder
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| anyhow!("encoder has no inputs"))?;
        let encoder_output_name = encoder
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| anyhow!("encoder has no outputs"))?;
        let encoder_mask_input = encoder
            .inputs()
            .iter()
            .map(|i| i.name())
            .find(|n| *n == "attention_mask")
            .map(str::to_string);

        // --- Decoder: split if a with-past graph is present, else merged.
        let decoder = if let Some(with_past_path) =
            pick(dir, &["decoder_with_past_model_quantized.onnx"])
        {
            let initial_path = pick(dir, &["decoder_model_quantized.onnx"]).ok_or_else(|| {
                anyhow!("split decoder missing no-past graph in {}", dir.display())
            })?;
            let initial = build_session(&initial_path, threads)?;
            let with_past = build_session(&with_past_path, threads)?;
            let n_layers = initial
                .outputs()
                .iter()
                .filter(|o| o.name().starts_with("present_self_key_"))
                .count();
            if n_layers == 0 {
                bail!("split decoder exposes no present_self_key_* outputs");
            }
            info!("loading moonshine streaming ({n_layers} layers, {threads} threads)");
            DecoderGraph::Split {
                initial,
                with_past,
                n_layers,
            }
        } else {
            let dec_path = pick(dir, decoder_merged_candidates(config.quantized))
                .ok_or_else(|| anyhow!("no decoder .onnx found in {}", dir.display()))?;
            let session = build_session(&dec_path, threads)?;
            let (num_heads, head_dim) = detect_kv_dims(&session);

            let collect =
                |sess: &Session, get: fn(&Session) -> Vec<String>, prefix: &str, side: &str| {
                    let mut v: Vec<String> = get(sess)
                        .into_iter()
                        .filter(|n| n.starts_with(prefix) && n.contains(side))
                        .collect();
                    v.sort();
                    v
                };
            let in_names = |s: &Session| s.inputs().iter().map(|i| i.name().to_string()).collect();
            let out_names =
                |s: &Session| s.outputs().iter().map(|o| o.name().to_string()).collect();
            info!("loading moonshine ({threads} threads)");
            DecoderGraph::Merged {
                decoder_kv_input_names: collect(&session, in_names, "past_key_values", ".decoder."),
                encoder_kv_input_names: collect(&session, in_names, "past_key_values", ".encoder."),
                decoder_kv_output_names: collect(&session, out_names, "present", ".decoder."),
                encoder_kv_output_names: collect(&session, out_names, "present", ".encoder."),
                num_heads,
                head_dim,
                session,
            }
        };

        Ok(Self {
            encoder,
            tokenizer,
            encoder_input_name,
            encoder_output_name,
            encoder_mask_input,
            decoder,
        })
    }

    /// Raw waveform `[1, len]` → encoder hidden states, kept as an owned ort
    /// value so decode steps can feed it back as a view without copying.
    fn run_encoder(&mut self, audio: &[f32]) -> Result<DynValue> {
        // The streaming encoder reshapes the waveform into 80-sample (5 ms)
        // frames and rejects lengths that don't divide evenly. Zero-pad the
        // tail; <5 ms of silence on audio that already ends in silence.
        let mut samples = audio.to_vec();
        if self.encoder_mask_input.is_some() {
            samples.resize(audio.len().next_multiple_of(STREAMING_FRAME), 0.0);
        }
        let len = samples.len();
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![(
            Cow::Borrowed(self.encoder_input_name.as_str()),
            Tensor::<f32>::from_array(([1usize, len], samples))
                .ort()?
                .into(),
        )];
        if let Some(mask) = &self.encoder_mask_input {
            inputs.push((
                Cow::Borrowed(mask.as_str()),
                Tensor::<i64>::from_array(([1usize, len], vec![1i64; len]))
                    .ort()?
                    .into(),
            ));
        }
        let mut out = self.encoder.run(inputs).ort()?;
        take_output(&mut out, &self.encoder_output_name)
    }
}

fn compute_max_tokens(audio_len: usize) -> usize {
    let duration = audio_len as f32 / SAMPLE_RATE;
    ((duration * MAX_TOKENS_PER_SECOND) as usize).clamp(16, ABSOLUTE_MAX_TOKENS)
}

impl Transcriber for Moonshine {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let duration = audio.len() as f32 / SAMPLE_RATE;
        let max_tokens = compute_max_tokens(audio.len());

        let t_enc = Instant::now();
        let enc = self.run_encoder(audio)?;
        let encode_ms = t_enc.elapsed().as_millis();

        let t_dec = Instant::now();
        let tokens = match &mut self.decoder {
            DecoderGraph::Merged {
                session,
                decoder_kv_input_names,
                encoder_kv_input_names,
                decoder_kv_output_names,
                encoder_kv_output_names,
                num_heads,
                head_dim,
            } => decode_merged(
                session,
                &enc,
                max_tokens,
                decoder_kv_input_names,
                encoder_kv_input_names,
                decoder_kv_output_names,
                encoder_kv_output_names,
                *num_heads,
                *head_dim,
            )?,
            DecoderGraph::Split {
                initial,
                with_past,
                n_layers,
            } => decode_split(initial, with_past, *n_layers, &enc, max_tokens)?,
        };
        let decode_ms = t_dec.elapsed().as_millis();

        let ids: Vec<u32> = tokens[1..].iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?;

        info!(
            "audio {duration:.2}s, encode {encode_ms}ms, decode {decode_ms}ms, {} tokens",
            ids.len()
        );
        debug!(?text, "decoded");
        Ok(text)
    }
}

/// Greedy decode over the merged decoder (`use_cache_branch`).
///
/// KV tensors are moved between steps as owned ort values (no extract/copy
/// round-trip); the encoder hidden states and the static cross-attention KV
/// are fed as views of values held outside the loop.
#[allow(clippy::too_many_arguments)]
fn decode_merged(
    decoder: &mut Session,
    enc: &DynValue,
    max_tokens: usize,
    decoder_kv_input_names: &[String],
    encoder_kv_input_names: &[String],
    decoder_kv_output_names: &[String],
    encoder_kv_output_names: &[String],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<i64>> {
    let dummy = Tensor::<f32>::from_array((
        [1usize, num_heads, 1usize, head_dim],
        vec![0.0f32; num_heads * head_dim],
    ))
    .ort()?;

    let mut tokens = vec![DECODER_START_TOKEN_ID];
    let mut hit_eos = false;
    // Previous step's `present.*.decoder.*` (feeds next `past_key_values`), and
    // the `present.*.encoder.*` captured at step 0 and reused forever.
    let mut past_decoder: Vec<DynValue> = Vec::new();
    let mut encoder_kv: Vec<DynValue> = Vec::new();

    for step in 0..max_tokens {
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::new();

        // input_ids: step 0 = all tokens; later = only the last token.
        let ids: Vec<i64> = if step == 0 {
            tokens.clone()
        } else {
            vec![*tokens.last().unwrap()]
        };
        inputs.push((
            Cow::Borrowed("input_ids"),
            Tensor::<i64>::from_array(([1usize, ids.len()], ids))
                .ort()?
                .into(),
        ));
        inputs.push((Cow::Borrowed("encoder_hidden_states"), enc.into()));

        // Decoder-side KV: dummy zeros at step 0, else previous present (moved).
        // Encoder-side (cross-attention) KV: dummy at step 0, else views of the
        // values captured at step 0 — the merged model emits empty encoder KV later.
        if step == 0 {
            for name in decoder_kv_input_names.iter().chain(encoder_kv_input_names) {
                inputs.push((Cow::Borrowed(name.as_str()), (&dummy).into()));
            }
        } else {
            for (name, v) in decoder_kv_input_names
                .iter()
                .zip(std::mem::take(&mut past_decoder))
            {
                inputs.push((Cow::Borrowed(name.as_str()), v.into()));
            }
            for (name, v) in encoder_kv_input_names.iter().zip(&encoder_kv) {
                inputs.push((Cow::Borrowed(name.as_str()), v.into()));
            }
        }

        inputs.push((
            Cow::Borrowed("use_cache_branch"),
            Tensor::<bool>::from_array(([1usize], vec![step > 0]))
                .ort()?
                .into(),
        ));

        let mut outputs = decoder.run(inputs).ort()?;

        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().ort()?;
        let vocab = *lshape.last().context("logits has no dims")? as usize;
        let next = argmax(&logits[logits.len() - vocab..]);
        if next == EOS_TOKEN_ID {
            hit_eos = true;
            break;
        }
        tokens.push(next);
        if truncate_loop(&mut tokens) {
            debug!("repetition loop detected at step {step}; truncating");
            break;
        }

        // Take KV for the next step out of the outputs (owned, no copy).
        past_decoder = decoder_kv_output_names
            .iter()
            .map(|n| take_output(&mut outputs, n))
            .collect::<Result<_>>()?;
        if step == 0 {
            encoder_kv = encoder_kv_output_names
                .iter()
                .map(|n| take_output(&mut outputs, n))
                .collect::<Result<_>>()?;
        }
    }
    if !hit_eos && collapse_runaway(&mut tokens) {
        debug!("runaway decode collapsed to {} tokens", tokens.len() - 1);
    }
    Ok(tokens)
}

/// Greedy decode over the split (no-past + with-past) streaming decoder.
///
/// Self-attn KV is moved between steps as owned ort values; the static
/// cross-attn KV and encoder hidden states are fed as views.
fn decode_split(
    initial: &mut Session,
    with_past: &mut Session,
    n_layers: usize,
    enc: &DynValue,
    max_tokens: usize,
) -> Result<Vec<i64>> {
    let mut tokens = vec![DECODER_START_TOKEN_ID];
    let mut hit_eos = false;
    // Self-attn KV, length 2*n_layers as key0,val0,key1,val1,…; grows each step.
    let mut past_self: Vec<DynValue> = Vec::new();
    // Cross-attn KV: static, captured at step 0, fed back every later step.
    let mut cross_kv: Vec<DynValue> = Vec::new();

    for step in 0..max_tokens {
        let last = *tokens.last().unwrap();
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (
                Cow::Borrowed("decoder_input_ids"),
                Tensor::<i64>::from_array(([1usize, 1usize], vec![last]))
                    .ort()?
                    .into(),
            ),
            (Cow::Borrowed("encoder_hidden_states"), enc.into()),
        ];

        let mut outputs = if step == 0 {
            initial.run(inputs).ort()?
        } else {
            for (i, v) in std::mem::take(&mut past_self).into_iter().enumerate() {
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((Cow::Owned(format!("past_self_{kind}_{}", i / 2)), v.into()));
            }
            for (i, v) in cross_kv.iter().enumerate() {
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((
                    Cow::Owned(format!("present_cross_{kind}_{}_orig", i / 2)),
                    v.into(),
                ));
            }
            with_past.run(inputs).ort()?
        };

        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().ort()?;
        let vocab = *lshape.last().context("logits has no dims")? as usize;
        let next = argmax(&logits[logits.len() - vocab..]);
        if next == EOS_TOKEN_ID {
            hit_eos = true;
            break;
        }
        tokens.push(next);
        if truncate_loop(&mut tokens) {
            debug!("repetition loop detected at step {step}; truncating");
            break;
        }

        // Take self-attn KV for the next step out of the outputs (owned, no copy).
        let mut next_self = Vec::with_capacity(2 * n_layers);
        for i in 0..n_layers {
            next_self.push(take_output(&mut outputs, &format!("present_self_key_{i}"))?);
            next_self.push(take_output(
                &mut outputs,
                &format!("present_self_value_{i}"),
            )?);
        }
        // Cross-attn KV depends only on the encoder output: capture once.
        if step == 0 {
            for i in 0..n_layers {
                cross_kv.push(take_output(
                    &mut outputs,
                    &format!("present_cross_key_{i}"),
                )?);
                cross_kv.push(take_output(
                    &mut outputs,
                    &format!("present_cross_value_{i}"),
                )?);
            }
        }
        past_self = next_self;
    }
    if !hit_eos && collapse_runaway(&mut tokens) {
        debug!("runaway decode collapsed to {} tokens", tokens.len() - 1);
    }
    Ok(tokens)
}

/// A decode that ran out of token budget without emitting EOS is usually a
/// repetition loop too short-lived for `truncate_loop` to prove (short clips
/// give small budgets). Find the shortest period whose cycle covers the tail
/// at least twice — partial final cycle allowed — and keep a single cycle.
fn collapse_runaway(tokens: &mut Vec<i64>) -> bool {
    let len = tokens.len();
    for period in 1..=len / 2 {
        let mut run = 0;
        while run + period < len && tokens[len - 1 - run] == tokens[len - 1 - run - period] {
            run += 1;
        }
        if run >= period {
            tokens.truncate(len - run);
            return true;
        }
    }
    false
}

/// Remove a named output from the session outputs as an owned value.
fn take_output(outputs: &mut SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    outputs
        .remove(name)
        .ok_or_else(|| anyhow!("model output '{name}' missing"))
}

/// Greedy Moonshine loops on noisy audio ("…amazing job this is amazing job
/// this is…"). When the tail of the sequence is the same short token cycle
/// repeated several times, the decode is stuck: collapse the run to a single
/// cycle and stop. Single-token cycles need more repeats before tripping —
/// legitimate dictation repeats one word ("hello hello hello") far more often
/// than it repeats a whole phrase three times verbatim.
fn truncate_loop(tokens: &mut Vec<i64>) -> bool {
    /// Longest cycle (in tokens) we look for; hallucination loops are short phrases.
    const MAX_PERIOD: usize = 8;
    /// Consecutive repeats of a multi-token / single-token cycle before tripping.
    const MIN_REPS: usize = 3;
    const MIN_REPS_SINGLE: usize = 6;

    for period in 1..=MAX_PERIOD {
        let reps = if period == 1 {
            MIN_REPS_SINGLE
        } else {
            MIN_REPS
        };
        let need = period * reps;
        if tokens.len() < need {
            break;
        }
        let tail = &tokens[tokens.len() - need..];
        if !(period..need).all(|i| tail[i] == tail[i - period]) {
            continue;
        }
        // Extend the periodic run as far back as it goes, then keep one cycle.
        let mut start = tokens.len() - need;
        while start > 0 && tokens[start - 1] == tokens[start - 1 + period] {
            start -= 1;
        }
        tokens.truncate(start + period);
        return true;
    }
    false
}

/// First of `names` that exists inside `dir`.
fn pick(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

fn encoder_candidates(quantized: bool) -> &'static [&'static str] {
    if quantized {
        &["encoder_model_quantized.onnx", "encoder_model.onnx"]
    } else {
        &["encoder_model.onnx", "encoder_model_quantized.onnx"]
    }
}

fn decoder_merged_candidates(quantized: bool) -> &'static [&'static str] {
    if quantized {
        &[
            "decoder_model_merged_quantized.onnx",
            "decoder_model_merged.onnx",
        ]
    } else {
        &[
            "decoder_model_merged.onnx",
            "decoder_model_merged_quantized.onnx",
        ]
    }
}

fn build_session(path: &Path, threads: usize) -> Result<Session> {
    Session::builder()
        .ort()?
        .with_intra_threads(threads)
        .ort()?
        .commit_from_file(path)
        .ort()
        .with_context(|| format!("loading {}", path.display()))
}

/// Detect `num_heads` / `head_dim` from the first `past_key_values` input's
/// shape `[batch, num_heads, seq, head_dim]`. Detection always works on the
/// onnx-community exports; the fallback just avoids a panic.
fn detect_kv_dims(decoder: &Session) -> (usize, usize) {
    decoder
        .inputs()
        .iter()
        .find(|i| i.name().starts_with("past_key_values"))
        .and_then(|i| match i.dtype() {
            ValueType::Tensor { shape, .. } => {
                let d: &[i64] = shape;
                if d.len() == 4 && d[1] > 0 && d[3] > 0 {
                    Some((d[1] as usize, d[3] as usize))
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap_or_else(|| {
            warn!("could not detect KV head dims; using (8, 52)");
            (8, 52)
        })
}

fn argmax(v: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut best_val = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_tokens_floor() {
        // Even a single sample must produce at least 16 tokens.
        assert_eq!(compute_max_tokens(1), 16);
        assert_eq!(compute_max_tokens(0), 16);
    }

    #[test]
    fn max_tokens_normal() {
        // 10s at 16 kHz → 10 * 8 = 80 tokens.
        assert_eq!(compute_max_tokens(16_000 * 10), 80);
    }

    #[test]
    fn loop_detection_collapses_cycle() {
        // Prefix X, then cycle A B C repeated 3×: collapse to X + one cycle.
        let mut t = vec![99, 10, 20, 30, 10, 20, 30, 10, 20, 30];
        assert!(truncate_loop(&mut t));
        assert_eq!(t, vec![99, 10, 20, 30]);
    }

    #[test]
    fn loop_detection_allows_double_phrase() {
        // A phrase said twice ("the more you know the more you grow" pattern)
        // is legitimate — only 2 repeats, gate needs 3.
        let mut t = vec![10, 20, 30, 40, 10, 20, 30, 40];
        assert!(!truncate_loop(&mut t));
        assert_eq!(t.len(), 8);
    }

    #[test]
    fn loop_detection_allows_repeated_word() {
        // "hello hello hello hello hello" (5×) must not trip; 6+ collapses.
        let mut t = vec![7, 7, 7, 7, 7];
        assert!(!truncate_loop(&mut t));
        assert_eq!(t.len(), 5);
        let mut t = vec![7, 7, 7, 7, 7, 7];
        assert!(truncate_loop(&mut t));
        assert_eq!(t, vec![7]);
    }

    #[test]
    fn runaway_collapse_keeps_one_cycle() {
        // Prefix then cycle A B C ×2 + partial: collapse to prefix + one cycle.
        let mut t = vec![99, 10, 20, 30, 10, 20, 30, 10, 20];
        assert!(collapse_runaway(&mut t));
        assert_eq!(t, vec![99, 10, 20, 30]);
        // Two full cycles, no partial.
        let mut t = vec![99, 10, 20, 10, 20];
        assert!(collapse_runaway(&mut t));
        assert_eq!(t, vec![99, 10, 20]);
    }

    #[test]
    fn runaway_collapse_leaves_normal_text() {
        let mut t = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert!(!collapse_runaway(&mut t));
        assert_eq!(t.len(), 8);
    }

    #[test]
    fn loop_detection_allows_normal_text() {
        let mut t = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(!truncate_loop(&mut t));
        // Too short to ever loop.
        let mut short = vec![1, 2, 3];
        assert!(!truncate_loop(&mut short));
    }

    #[test]
    fn max_tokens_ceiling() {
        // Very long audio must be clamped to 512.
        assert_eq!(compute_max_tokens(16_000 * 100), ABSOLUTE_MAX_TOKENS);
        assert_eq!(compute_max_tokens(usize::MAX / 4), ABSOLUTE_MAX_TOKENS);
    }
}
