//! Transcriber trait + backend factory.

mod moonshine;
#[cfg(feature = "whisper")]
mod whisper;

use anyhow::{bail, Result};

use crate::config::{Backend, Config};

pub trait Transcriber: Send {
    /// `audio`: 16 kHz mono f32 in [-1, 1]. Returns raw decoded text (the caller
    /// post-processes).
    fn transcribe(&mut self, audio: &[f32]) -> Result<String>;
}

/// Build the transcriber the config's `model` resolves to. The main loop owns
/// it exclusively, so `&mut self` on `transcribe` needs no internal locking.
pub fn create(config: &Config) -> Result<Box<dyn Transcriber>> {
    let resolved = config.resolve_model();
    match resolved.backend {
        Backend::Moonshine => {
            if !resolved.path.exists() {
                tracing::info!("model not found — downloading {}...", config.model);
                crate::download::run(config)?;
            }
            Ok(Box::new(moonshine::Moonshine::load(
                &resolved.path,
                config,
            )?))
        }
        #[cfg(feature = "whisper")]
        Backend::Whisper => Ok(Box::new(whisper::WhisperTranscriber::load(
            &resolved.path,
            config,
        )?)),
        #[cfg(not(feature = "whisper"))]
        Backend::Whisper => bail!(
            "this binary was built without whisper support; rebuild with \
             --features whisper, or use a Moonshine model"
        ),
        Backend::Parakeet => bail!(
            "Parakeet transcriber is not yet implemented"
        ),
    }
}
