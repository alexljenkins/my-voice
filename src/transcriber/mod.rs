//! Transcriber trait + backend factory.

mod moonshine;

use anyhow::Result;

use crate::config::Config;

pub trait Transcriber: Send {
    /// `audio`: 16 kHz mono f32 in [-1, 1]. Returns raw decoded text (the caller
    /// post-processes).
    fn transcribe(&mut self, audio: &[f32]) -> Result<String>;

    /// Run one throwaway pass on a short silent buffer to pay ORT's first-call
    /// graph-init cost at load instead of on the user's first transcription.
    /// Default routes through `transcribe`, exercising the same encode+decode
    /// path (split decoders warm both graphs). Discards output and errors.
    fn warm(&mut self) {
        let _ = self.transcribe(&[0.0; 1600]);
    }
}

/// Build the Moonshine transcriber the config's `model` resolves to. The main
/// loop owns it exclusively, so `&mut self` on `transcribe` needs no internal
/// locking.
pub fn create(config: &Config) -> Result<Box<dyn Transcriber>> {
    let path = config.resolve_model();
    if !path.exists() {
        tracing::info!("model not found — downloading {}...", config.model);
        crate::download::run(config)?;
    }
    Ok(Box::new(moonshine::Moonshine::load(&path, config)?))
}
