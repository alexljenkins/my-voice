//! Transcriber trait + backend factory.

mod moonshine;

use anyhow::Result;
use ort::environment::GlobalThreadPoolOptions;

use crate::config::Config;

pub const MAX_AUDIO_SECONDS: usize = 60;

/// Commit ONE global ORT intra-op thread pool before any `Session` is built, so
/// the encoder + decoder graphs share it instead of each spinning up its own
/// N-thread pool (sessions run strictly sequentially, so per-session pools are
/// pure waste — up to ~16 idle threads on the medium model). The env is
/// immutable once committed: this MUST run before the first `create`, or
/// sessions silently fall back to per-session pools. Returns whether the global
/// pool committed (`false` if an env already exists) so the effect is verifiable.
pub fn init_thread_pool(config: &Config) -> bool {
    let threads = config.resolved_threads();
    let opts = match GlobalThreadPoolOptions::default().with_intra_threads(threads) {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("ort global thread pool setup failed: {e}; using per-session pools");
            return false;
        }
    };
    let committed = ort::init().with_global_thread_pool(opts).commit();
    if committed {
        tracing::info!("ort: shared global intra-op pool ({threads} threads)");
    } else {
        tracing::warn!("ort env already committed; sessions keep per-session pools");
    }
    committed
}

pub trait Transcriber: Send {
    /// `audio`: 16 kHz mono f32 in [-1, 1]. Returns raw decoded text (the caller
    /// post-processes).
    fn transcribe(&mut self, audio: &[f32]) -> Result<String>;

    /// Bound every inference call, including oversized release/record buffers.
    /// Balance emergency chunks so a late poll cannot leave a tiny extra chunk.
    fn transcribe_bounded(&mut self, audio: &[f32]) -> Result<String> {
        let limit = MAX_AUDIO_SECONDS * 16_000;
        if audio.len() <= limit {
            return self.transcribe(audio);
        }
        tracing::warn!("audio exceeds 60s; splitting without dropping samples");
        let chunk_size = audio.len().div_ceil(audio.len().div_ceil(limit));
        let mut text = String::new();
        for chunk in audio.chunks(chunk_size) {
            let part = self.transcribe(chunk)?;
            let part = part.trim();
            if !text.is_empty() && !part.is_empty() {
                text.push(' ');
            }
            text.push_str(part);
        }
        Ok(text)
    }

    /// Run one throwaway pass on a short silent buffer to pay ORT's first-call
    /// graph-init cost at load instead of on the user's first transcription.
    /// Default routes through `transcribe`, exercising the same encode+decode
    /// path. Discards output and errors.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_audio_reaches_inference_in_order_without_loss() {
        struct Recorder {
            samples: Vec<f32>,
            lengths: Vec<usize>,
        }
        impl Transcriber for Recorder {
            fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
                self.samples.extend_from_slice(audio);
                self.lengths.push(audio.len());
                Ok(self.lengths.len().to_string())
            }
        }
        let limit = MAX_AUDIO_SECONDS * 16_000;
        for size in [0, 1, limit, limit + 1, limit * 2 + 3] {
            let audio: Vec<f32> = (0..size).map(|i| (i % 97) as f32 / 97.0).collect();
            let mut recorder = Recorder {
                samples: Vec::new(),
                lengths: Vec::new(),
            };
            let text = recorder.transcribe_bounded(&audio).unwrap();
            assert_eq!(recorder.samples, audio);
            assert!(recorder.lengths.iter().all(|&len| len <= limit));
            assert_eq!(
                text,
                (1..=recorder.lengths.len())
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if size > limit {
                assert!(recorder.lengths.iter().all(|&len| len >= limit / 2));
            }
        }
    }

    /// The ort env is a process-global `OnceLock`: the first commit in this test
    /// binary must win (proving the global pool took effect), and a second must
    /// report `false` rather than silently re-committing. No model/audio/network.
    #[test]
    fn global_thread_pool_commits_once() {
        let config = Config::default();
        assert!(
            init_thread_pool(&config),
            "first commit must install the shared global pool"
        );
        assert!(
            !init_thread_pool(&config),
            "second commit must be a no-op (env already committed)"
        );
    }
}
