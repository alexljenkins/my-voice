//! Moonshine ONNX backend: encoder + merged KV-cache decoder, greedy decode.
//!
//! Faithful port of voxtype's `src/transcribe/moonshine.rs`. Moonshine eats the
//! raw 16 kHz waveform (no mel spectrogram, no 30s padding); the decoder is
//! autoregressive greedy over a merged graph with a KV cache.

use std::borrow::Cow;
use std::path::Path;
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
    decoder: Session,
    tokenizer: Tokenizer,
    encoder_input_name: String,
    encoder_output_name: String,
    /// KV input names starting `past_key_values`, partitioned and each sorted —
    /// pairing with the matching `present` outputs is positional after sorting.
    decoder_kv_input_names: Vec<String>,
    encoder_kv_input_names: Vec<String>,
    decoder_kv_output_names: Vec<String>,
    encoder_kv_output_names: Vec<String>,
    num_heads: usize,
    head_dim: usize,
}

impl Moonshine {
    pub fn load(dir: &Path, config: &Config) -> Result<Self> {
        // Prefer the quantized pair only when both files are present.
        let use_quant = config.quantized
            && dir.join("encoder_model_quantized.onnx").exists()
            && dir.join("decoder_model_merged_quantized.onnx").exists();
        if config.quantized && !use_quant {
            warn!("quantized model files not found; falling back to full precision");
        }
        let (enc_file, dec_file) = if use_quant {
            (
                "encoder_model_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
            )
        } else {
            ("encoder_model.onnx", "decoder_model_merged.onnx")
        };

        let enc_path = dir.join(enc_file);
        let dec_path = dir.join(dec_file);
        let tok_path = dir.join("tokenizer.json");
        for p in [&enc_path, &dec_path, &tok_path] {
            if !p.exists() {
                bail!(
                    "model file missing: {} — run: my-voice --download",
                    p.display()
                );
            }
        }

        let threads = config.resolved_threads();
        info!(
            "loading moonshine ({}, {threads} threads)",
            if use_quant { "quantized" } else { "full" }
        );

        let encoder = build_session(&enc_path, threads)?;
        let decoder = build_session(&dec_path, threads)?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tok_path.display()))?;

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

        let (num_heads, head_dim) = detect_kv_dims(&decoder);

        let collect = |names: &[String], prefix: &str, side: &str| -> Vec<String> {
            let mut v: Vec<String> = names
                .iter()
                .filter(|n| n.starts_with(prefix) && n.contains(side))
                .cloned()
                .collect();
            v.sort();
            v
        };
        let dec_in: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let dec_out: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            encoder_input_name,
            encoder_output_name,
            decoder_kv_input_names: collect(&dec_in, "past_key_values", ".decoder."),
            encoder_kv_input_names: collect(&dec_in, "past_key_values", ".encoder."),
            decoder_kv_output_names: collect(&dec_out, "present", ".decoder."),
            encoder_kv_output_names: collect(&dec_out, "present", ".encoder."),
            num_heads,
            head_dim,
        })
    }
}

impl Transcriber for Moonshine {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let duration = audio.len() as f32 / SAMPLE_RATE;
        let max_tokens =
            ((duration * MAX_TOKENS_PER_SECOND) as usize).clamp(16, ABSOLUTE_MAX_TOKENS);

        // --- Encode: raw waveform [1, len] → hidden states, kept for every step.
        let t_enc = Instant::now();
        let enc_inputs: Vec<(Cow<str>, SessionInputValue)> = vec![(
            Cow::Borrowed(self.encoder_input_name.as_str()),
            Tensor::<f32>::from_array(([1usize, audio.len()], audio.to_vec()))
                .ort()?
                .into(),
        )];
        let enc_out = self.encoder.run(enc_inputs).ort()?;
        let (enc_shape, enc_data) = enc_out[self.encoder_output_name.as_str()]
            .try_extract_tensor::<f32>()
            .ort()?;
        let enc_shape: Vec<i64> = enc_shape.to_vec();
        let enc_data: Vec<f32> = enc_data.to_vec();
        drop(enc_out);
        let encode_ms = t_enc.elapsed().as_millis();

        // --- Decode: greedy, autoregressive, merged decoder with KV cache.
        let dummy = vec![0.0f32; self.num_heads * self.head_dim];
        let dummy_shape = [1usize, self.num_heads, 1usize, self.head_dim];

        let mut tokens = vec![DECODER_START_TOKEN_ID];
        // Previous step's `present.*.decoder.*` (feeds next `past_key_values`),
        // and the `present.*.encoder.*` captured at step 0 and reused forever.
        let mut past_decoder: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
        let mut encoder_kv: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();

        let t_dec = Instant::now();
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
                Tensor::<i64>::from_array(([1usize, ids.len()], ids)).ort()?.into(),
            ));
            inputs.push((
                Cow::Borrowed("encoder_hidden_states"),
                Tensor::<f32>::from_array((enc_shape.clone(), enc_data.clone()))
                    .ort()?
                    .into(),
            ));

            // Decoder-side KV: dummy zeros at step 0, else previous present.
            for (i, name) in self.decoder_kv_input_names.iter().enumerate() {
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
            for (i, name) in self.encoder_kv_input_names.iter().enumerate() {
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
                Tensor::<bool>::from_array(([1usize], vec![step > 0])).ort()?.into(),
            ));

            let outputs = self.decoder.run(inputs).ort()?;

            // Last position's logits → argmax. Row-major with vocab as the final
            // dim, so the trailing `vocab` slice is the last token regardless of
            // rank (2D or 3D).
            let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().ort()?;
            let vocab = *lshape.last().context("logits has no dims")? as usize;
            let last = &logits[logits.len() - vocab..];
            let next = argmax(last);
            if next == EOS_TOKEN_ID {
                break;
            }
            tokens.push(next);

            // Capture KV for the next step while `outputs` is still alive.
            let mut next_decoder = Vec::with_capacity(self.decoder_kv_output_names.len());
            for name in &self.decoder_kv_output_names {
                let (sh, d) = outputs[name.as_str()].try_extract_tensor::<f32>().ort()?;
                next_decoder.push((sh.to_vec(), d.to_vec()));
            }
            if step == 0 {
                for name in &self.encoder_kv_output_names {
                    let (sh, d) = outputs[name.as_str()].try_extract_tensor::<f32>().ort()?;
                    encoder_kv.push((sh.to_vec(), d.to_vec()));
                }
            }
            drop(outputs);
            past_decoder = next_decoder;
        }
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
