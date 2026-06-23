//! Transcriber trait + backend factory.

mod moonshine;

use anyhow::Result;
use ort::environment::GlobalThreadPoolOptions;

use crate::config::Config;

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
