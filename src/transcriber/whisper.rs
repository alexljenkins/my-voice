use std::os::raw::c_int;

use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;

pub struct WhisperTranscriber {
    // WhisperState holds its own Arc to the context; no need to keep the
    // WhisperContext separately.
    state: whisper_rs::WhisperState,
    threads: c_int,
}

impl WhisperTranscriber {
    pub fn load(model_path: &std::path::Path, config: &Config) -> Result<Self> {
        // Route whisper.cpp internal logs through the tracing system.
        whisper_rs::install_logging_hooks();

        let path_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;

        let ctx = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .with_context(|| format!("loading whisper model from {}", model_path.display()))?;

        let state = ctx.create_state().context("creating whisper state")?;
        let threads = config.resolved_threads() as c_int;

        Ok(Self { state, threads })
    }
}

impl super::Transcriber for WhisperTranscriber {
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_no_timestamps(true);
        params.set_single_segment(false);
        params.set_suppress_blank(true);
        params.set_n_threads(self.threads);

        self.state
            .full(params, audio)
            .context("whisper inference failed")?;

        let n = self.state.full_n_segments();
        let mut parts = Vec::with_capacity(n as usize);
        for i in 0..n {
            if let Some(seg) = self.state.get_segment(i) {
                parts.push(seg.to_string());
            }
        }

        Ok(parts.join(" "))
    }
}
