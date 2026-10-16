//! Parakeet TDT F16 GGUF inference through the pinned transcribe.cpp runtime.
use std::path::Path;

use anyhow::{ensure, Context, Result};
use transcribe_cpp::{Model, RunOptions, Session, SessionOptions};

use super::Transcriber;
use crate::config::Config;

pub struct Parakeet {
    session: Session,
}

impl Parakeet {
    pub fn load(path: &Path, config: &Config) -> Result<Self> {
        let model = Model::load(path).with_context(|| format!("loading {}", path.display()))?;
        ensure!(
            model.arch() == "parakeet",
            "GGUF model must use the Parakeet architecture"
        );
        tracing::info!(backend = model.backend(), "Parakeet F16 loaded");
        let session = model.session_with(&SessionOptions {
            n_threads: i32::try_from(config.resolved_threads())
                .context("too many inference threads")?,
            ..Default::default()
        })?;
        Ok(Self { session })
    }
}

impl Transcriber for Parakeet {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let started = std::time::Instant::now();
        let result = self
            .session
            .run(
                audio,
                &RunOptions {
                    language: Some("en".into()),
                    ..Default::default()
                },
            )
            .context("Parakeet inference failed")?;
        ensure!(
            !self.session.was_truncated(),
            "Parakeet output was truncated"
        );
        tracing::info!(
            "Parakeet transcribed in {} ms: encode {}ms, decode {}ms",
            started.elapsed().as_millis(),
            result.timings.encode_ms.round() as u64,
            result.timings.decode_ms.round() as u64
        );
        Ok(result.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_gguf_returns_a_load_error() {
        let path =
            std::env::temp_dir().join(format!("my-voice-invalid-{}.gguf", std::process::id()));
        std::fs::write(&path, b"not a GGUF model").unwrap();
        let result = Parakeet::load(&path, &Config::default());
        std::fs::remove_file(path).unwrap();
        assert!(result.is_err());
    }
}
