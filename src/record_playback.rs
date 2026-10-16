//! Replay saved recordings without blocking capture or transcription.
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;

use anyhow::{Context, Result};
use tracing::warn;

pub struct RecordPlayback {
    enabled: bool,
    child: Option<Child>,
}

impl RecordPlayback {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            child: None,
        }
    }

    pub fn start(&mut self, path: &Path) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.stop();
        self.child = Some(
            Command::new("pw-play")
                .arg(path)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .spawn()
                .context("starting recording playback with pw-play")?,
        );
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if let Err(error) = child.kill() {
                warn!(%error, "could not stop recording playback");
            }
            // Reap off the daemon thread. Capture must not wait for playback.
            thread::spawn(move || {
                if let Err(error) = child.wait() {
                    warn!(%error, "could not reap recording playback");
                }
            });
        }
    }
}

impl Drop for RecordPlayback {
    fn drop(&mut self) {
        self.stop();
    }
}
