//! Moonshine ONNX backend: encoder + autoregressive greedy decode over the raw
//! 16 kHz waveform (no mel spectrogram, no 30s padding).
//!
//! Two decoder graph shapes are supported, detected from the files on disk:
//!
//! * **Merged** (`moonshine-tiny`/`-base`): one `decoder_model_merged.onnx`
//!   switched by a `use_cache_branch` flag. KV names are `past_key_values.*` /
//!   `present.*`. Faithful port of voxtype's backend.
//! * **Split** (streaming `-small`/`-medium`): a no-past `decoder_model.onnx`
//!   for step 0 and a `decoder_with_past_model.onnx` for later steps. Self-attn
//!   KV (`past_self_*` / `present_self_*`) grows each step; cross-attn KV is
//!   computed once at step 0 and fed back as `present_cross_*_orig`. The
//!   streaming encoder also takes an `attention_mask` input. We run it as a
//!   single push-to-talk pass over the whole utterance, not chunk-by-chunk.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, ValueType};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use super::Transcriber;
use crate::config::Config;

const DECODER_START_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;
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
        let decoder = if let Some(with_past_path) = pick(
            dir,
            &[
                "decoder_with_past_model_int8.onnx",
                "decoder_with_past_model.onnx",
            ],
        ) {
            let initial_path = pick(dir, &["decoder_model_int8.onnx", "decoder_model.onnx"])
                .ok_or_else(|| {
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

    /// Raw waveform `[1, len]` → encoder hidden states `(shape, data)`.
    fn run_encoder(&mut self, audio: &[f32]) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![(
            Cow::Borrowed(self.encoder_input_name.as_str()),
            Tensor::<f32>::from_array(([1usize, audio.len()], audio.to_vec()))
                .ort()?
                .into(),
        )];
        if let Some(mask) = &self.encoder_mask_input {
            inputs.push((
                Cow::Borrowed(mask.as_str()),
                Tensor::<i64>::from_array(([1usize, audio.len()], vec![1i64; audio.len()]))
                    .ort()?
                    .into(),
            ));
        }
        let out = self.encoder.run(inputs).ort()?;
        let (shape, data) = out[self.encoder_output_name.as_str()]
            .try_extract_tensor::<f32>()
            .ort()?;
        Ok((shape.to_vec(), data.to_vec()))
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
        let (enc_shape, enc_data) = self.run_encoder(audio)?;
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
                &enc_shape,
                &enc_data,
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
            } => decode_split(
                initial, with_past, *n_layers, &enc_shape, &enc_data, max_tokens,
            )?,
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
#[allow(clippy::too_many_arguments)]
fn decode_merged(
    decoder: &mut Session,
    enc_shape: &[i64],
    enc_data: &[f32],
    max_tokens: usize,
    decoder_kv_input_names: &[String],
    encoder_kv_input_names: &[String],
    decoder_kv_output_names: &[String],
    encoder_kv_output_names: &[String],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<i64>> {
    let dummy = vec![0.0f32; num_heads * head_dim];
    let dummy_shape = [1usize, num_heads, 1usize, head_dim];

    let mut tokens = vec![DECODER_START_TOKEN_ID];
    // Previous step's `present.*.decoder.*` (feeds next `past_key_values`), and
    // the `present.*.encoder.*` captured at step 0 and reused forever.
    let mut past_decoder: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    let mut encoder_kv: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();

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
        inputs.push((
            Cow::Borrowed("encoder_hidden_states"),
            Tensor::<f32>::from_array((enc_shape.to_vec(), enc_data.to_vec()))
                .ort()?
                .into(),
        ));

        // Decoder-side KV: dummy zeros at step 0, else previous present.
        for (i, name) in decoder_kv_input_names.iter().enumerate() {
            let tensor = if step == 0 {
                Tensor::<f32>::from_array((dummy_shape, dummy.clone())).ort()?
            } else {
                let (sh, d) = &past_decoder[i];
                Tensor::<f32>::from_array((sh.clone(), d.clone())).ort()?
            };
            inputs.push((Cow::Borrowed(name.as_str()), tensor.into()));
        }
        // Encoder-side (cross-attention) KV: dummy at step 0, else the values
        // captured at step 0 — the merged model emits empty encoder KV later.
        for (i, name) in encoder_kv_input_names.iter().enumerate() {
            let tensor = if step == 0 {
                Tensor::<f32>::from_array((dummy_shape, dummy.clone())).ort()?
            } else {
                let (sh, d) = &encoder_kv[i];
                Tensor::<f32>::from_array((sh.clone(), d.clone())).ort()?
            };
            inputs.push((Cow::Borrowed(name.as_str()), tensor.into()));
        }

        inputs.push((
            Cow::Borrowed("use_cache_branch"),
            Tensor::<bool>::from_array(([1usize], vec![step > 0]))
                .ort()?
                .into(),
        ));

        let outputs = decoder.run(inputs).ort()?;

        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().ort()?;
        let vocab = *lshape.last().context("logits has no dims")? as usize;
        let next = argmax(&logits[logits.len() - vocab..]);
        if next == EOS_TOKEN_ID {
            break;
        }
        tokens.push(next);

        // Capture KV for the next step while `outputs` is still alive.
        let mut next_decoder = Vec::with_capacity(decoder_kv_output_names.len());
        for name in decoder_kv_output_names {
            let (sh, d) = outputs[name.as_str()].try_extract_tensor::<f32>().ort()?;
            next_decoder.push((sh.to_vec(), d.to_vec()));
        }
        if step == 0 {
            for name in encoder_kv_output_names {
                let (sh, d) = outputs[name.as_str()].try_extract_tensor::<f32>().ort()?;
                encoder_kv.push((sh.to_vec(), d.to_vec()));
            }
        }
        drop(outputs);
        past_decoder = next_decoder;
    }
    Ok(tokens)
}

/// Greedy decode over the split (no-past + with-past) streaming decoder.
fn decode_split(
    initial: &mut Session,
    with_past: &mut Session,
    n_layers: usize,
    enc_shape: &[i64],
    enc_data: &[f32],
    max_tokens: usize,
) -> Result<Vec<i64>> {
    let mut tokens = vec![DECODER_START_TOKEN_ID];
    // Self-attn KV, length 2*n_layers as key0,val0,key1,val1,…; grows each step.
    let mut past_self: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    // Cross-attn KV: static, captured at step 0, fed back every later step.
    let mut cross_kv: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();

    for step in 0..max_tokens {
        let last = *tokens.last().unwrap();
        let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (
                Cow::Borrowed("decoder_input_ids"),
                Tensor::<i64>::from_array(([1usize, 1usize], vec![last]))
                    .ort()?
                    .into(),
            ),
            (
                Cow::Borrowed("encoder_hidden_states"),
                Tensor::<f32>::from_array((enc_shape.to_vec(), enc_data.to_vec()))
                    .ort()?
                    .into(),
            ),
        ];

        let outputs = if step == 0 {
            initial.run(inputs).ort()?
        } else {
            for i in 0..n_layers {
                let (sk, dk) = &past_self[2 * i];
                let (sv, dv) = &past_self[2 * i + 1];
                inputs.push((
                    Cow::Owned(format!("past_self_key_{i}")),
                    Tensor::<f32>::from_array((sk.clone(), dk.clone()))
                        .ort()?
                        .into(),
                ));
                inputs.push((
                    Cow::Owned(format!("past_self_value_{i}")),
                    Tensor::<f32>::from_array((sv.clone(), dv.clone()))
                        .ort()?
                        .into(),
                ));
            }
            for i in 0..n_layers {
                let (sk, dk) = &cross_kv[2 * i];
                let (sv, dv) = &cross_kv[2 * i + 1];
                inputs.push((
                    Cow::Owned(format!("present_cross_key_{i}_orig")),
                    Tensor::<f32>::from_array((sk.clone(), dk.clone()))
                        .ort()?
                        .into(),
                ));
                inputs.push((
                    Cow::Owned(format!("present_cross_value_{i}_orig")),
                    Tensor::<f32>::from_array((sv.clone(), dv.clone()))
                        .ort()?
                        .into(),
                ));
            }
            with_past.run(inputs).ort()?
        };

        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().ort()?;
        let vocab = *lshape.last().context("logits has no dims")? as usize;
        let next = argmax(&logits[logits.len() - vocab..]);
        if next == EOS_TOKEN_ID {
            break;
        }
        tokens.push(next);

        // Capture self-attn KV for the next step.
        let mut next_self = Vec::with_capacity(2 * n_layers);
        for i in 0..n_layers {
            let (sk, dk) = outputs[format!("present_self_key_{i}").as_str()]
                .try_extract_tensor::<f32>()
                .ort()?;
            let (sv, dv) = outputs[format!("present_self_value_{i}").as_str()]
                .try_extract_tensor::<f32>()
                .ort()?;
            next_self.push((sk.to_vec(), dk.to_vec()));
            next_self.push((sv.to_vec(), dv.to_vec()));
        }
        // Cross-attn KV depends only on the encoder output: capture once.
        if step == 0 {
            for i in 0..n_layers {
                let (sk, dk) = outputs[format!("present_cross_key_{i}").as_str()]
                    .try_extract_tensor::<f32>()
                    .ort()?;
                let (sv, dv) = outputs[format!("present_cross_value_{i}").as_str()]
                    .try_extract_tensor::<f32>()
                    .ort()?;
                cross_kv.push((sk.to_vec(), dk.to_vec()));
                cross_kv.push((sv.to_vec(), dv.to_vec()));
            }
        }
        drop(outputs);
        past_self = next_self;
    }
    Ok(tokens)
}

/// First of `names` that exists inside `dir`.
fn pick(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

/// Encoder filename preference. Streaming repos ship only the `_int8` variant;
/// the merged repos ship `_quantized` + full.
fn encoder_candidates(quantized: bool) -> &'static [&'static str] {
    if quantized {
        &[
            "encoder_model_quantized.onnx",
            "encoder_model_int8.onnx",
            "encoder_model.onnx",
        ]
    } else {
        &[
            "encoder_model.onnx",
            "encoder_model_quantized.onnx",
            "encoder_model_int8.onnx",
        ]
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
    fn max_tokens_ceiling() {
        // Very long audio must be clamped to 512.
        assert_eq!(compute_max_tokens(16_000 * 100), ABSOLUTE_MAX_TOKENS);
        assert_eq!(compute_max_tokens(usize::MAX / 4), ABSOLUTE_MAX_TOKENS);
    }
}
