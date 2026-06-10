//! Lazy model load + idle eviction.
//!
//! One `Mutex<Option<Box<dyn Transcriber>>>` is the entire concurrency story:
//! the transcribe path, the keydown preload, and the evict thread all contend
//! on it, so eviction can never race a transcription — it just waits for the
//! lock. Load is deferred to first use so the daemon starts instantly and idle
//! sessions hold no model in RAM.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{debug, info};

use crate::config::Config;
use crate::transcriber::{self, Transcriber};

/// How often the evict thread wakes to check idle time.
const EVICT_TICK: Duration = Duration::from_secs(30);

pub struct ModelCache {
    slot: Mutex<Option<Box<dyn Transcriber>>>,
    last_used: Mutex<Instant>,
    config: Config,
    /// `-1` never evict, `0` reload every use, `>0` evict after N idle seconds.
    timeout_secs: i64,
}

impl ModelCache {
    pub fn new(config: &Config) -> Arc<Self> {
        Arc::new(Self {
            slot: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
            config: config.clone(),
            timeout_secs: config.load_timeout_secs,
        })
    }

    /// Load the model if it isn't resident; cheap no-op when it is. Spawned on a
    /// throwaway thread at keydown so the ~1s cold start overlaps with speech.
    pub fn ensure_loaded(&self) -> Result<()> {
        let mut slot = self.slot.lock().unwrap();
        load_locked(&mut slot, &self.config)?;
        *self.last_used.lock().unwrap() = Instant::now();
        Ok(())
    }

    /// Transcribe, loading inline if the preload hasn't finished (or wasn't
    /// run). Blocks on the same lock the preload holds, so there's no double
    /// load and no race. With `timeout_secs == 0` the model is dropped before
    /// returning — reload-every-use mode.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut slot = self.slot.lock().unwrap();
        load_locked(&mut slot, &self.config)?;
        let text = slot.as_mut().unwrap().transcribe(audio)?;
        *self.last_used.lock().unwrap() = Instant::now();
        if self.timeout_secs == 0 {
            *slot = None;
            debug!("model unloaded (load_timeout_secs = 0)");
        }
        Ok(text)
    }

    /// Spawn the background evictor. No-op for `timeout_secs <= 0`: negative
    /// never evicts, and `0` keeps nothing resident to evict.
    pub fn start_evict_thread(self: &Arc<Self>) {
        if self.timeout_secs <= 0 {
            return;
        }
        let me = Arc::clone(self);
        thread::spawn(move || loop {
            thread::sleep(EVICT_TICK);
            let mut slot = me.slot.lock().unwrap();
            if slot.is_none() {
                continue;
            }
            // Re-check idle *after* acquiring the lock: a transcription may have
            // touched last_used while we slept or waited on the lock.
            let idle = me.last_used.lock().unwrap().elapsed();
            if idle.as_secs() as i64 >= me.timeout_secs {
                *slot = None;
                info!("model unloaded after {}s idle", idle.as_secs());
            }
        });
    }
}

/// Build the transcriber into `slot` if empty. Caller holds the slot lock.
fn load_locked(slot: &mut Option<Box<dyn Transcriber>>, config: &Config) -> Result<()> {
    if slot.is_none() {
        let t0 = Instant::now();
        *slot = Some(transcriber::create(config)?);
        info!("model loaded in {} ms", t0.elapsed().as_millis());
    }
    Ok(())
}
