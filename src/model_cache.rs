//! Lazy model load + idle eviction.
//!
//! One `Mutex<Option<Box<dyn Transcriber>>>` is the entire concurrency story:
//! the transcribe path, the keydown preload, and the evict thread all contend
//! on it, so eviction can never race a transcription — it just waits for the
//! lock. Load is deferred to first use so the daemon starts instantly and idle
//! sessions hold no model in RAM.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{debug, error, info};

use crate::config::Config;
use crate::transcriber::{self, Transcriber};

/// How often the evict thread wakes to check idle time.
const EVICT_TICK: Duration = Duration::from_secs(30);

type Factory = dyn Fn() -> Result<Box<dyn Transcriber>> + Send + Sync;

pub struct ModelCache {
    slot: Mutex<Option<Box<dyn Transcriber>>>,
    last_used: Mutex<Instant>,
    factory: Box<Factory>,
    /// `-1` never evict, `0` reload every use, `>0` evict after N idle seconds.
    timeout_secs: i64,
    /// Set by `shutdown` to stop the evict thread. Without it the thread's own
    /// `Arc` clone keeps a replaced ModelCache (and its model's RAM) alive after
    /// a model switch until the zombie's own idle-eviction eventually fired.
    shutdown: Arc<AtomicBool>,
}

impl ModelCache {
    pub fn new(config: &Config) -> Arc<Self> {
        let config = config.clone();
        Self::with_factory(
            config.load_timeout_secs,
            Box::new(move || transcriber::create(&config)),
        )
    }

    fn with_factory(timeout_secs: i64, factory: Box<Factory>) -> Arc<Self> {
        Arc::new(Self {
            slot: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
            factory,
            timeout_secs,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Signal the evict thread to exit. Called on the OLD cache during a model
    /// switch: once the thread breaks it drops its `Arc` clone, letting the old
    /// ModelCache — and the model's RAM — drop now instead of lingering until the
    /// zombie thread's own idle-eviction eventually fired.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Lock the slot, recovering from poison. A panic inside `transcribe` (on a
    /// worker thread) poisons this mutex; without recovery every later lock
    /// unwraps Err and the daemon is wedged until restart. The transcriber that
    /// was live during the panic may be mid-mutation, so drop it and rebuild on
    /// next use.
    fn lock_slot(&self) -> MutexGuard<'_, Option<Box<dyn Transcriber>>> {
        match self.slot.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.slot.clear_poison();
                let mut guard = poisoned.into_inner();
                *guard = None;
                error!("transcriber panicked earlier; model dropped, will reload on next use");
                crate::notify::once(
                    crate::notify::ErrorKind::TranscriberPanicked,
                    "Transcription error",
                    "The speech model crashed and was reset — it will reload on next use.",
                );
                guard
            }
        }
    }

    /// `last_used` is only ever assigned while held, so poison here means a
    /// panic elsewhere on the thread; the Instant itself can't be corrupt.
    fn lock_last_used(&self) -> MutexGuard<'_, Instant> {
        self.last_used.lock().unwrap_or_else(|poisoned| {
            self.last_used.clear_poison();
            poisoned.into_inner()
        })
    }

    /// Load the model if it isn't resident; cheap no-op when it is. Spawned on a
    /// throwaway thread at keydown so the ~1s cold start overlaps with speech.
    pub fn ensure_loaded(&self) -> Result<()> {
        let mut slot = self.lock_slot();
        load_locked(&mut slot, &self.factory)?;
        *self.lock_last_used() = Instant::now();
        Ok(())
    }

    /// Transcribe, loading inline if the preload hasn't finished (or wasn't
    /// run). Blocks on the same lock the preload holds, so there's no double
    /// load and no race. With `timeout_secs == 0` the model is dropped before
    /// returning — reload-every-use mode.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let mut slot = self.lock_slot();
        load_locked(&mut slot, &self.factory)?;
        let text = slot.as_mut().unwrap().transcribe(audio)?;
        *self.lock_last_used() = Instant::now();
        if self.timeout_secs == 0 {
            *slot = None;
            debug!("model unloaded (load_timeout_secs = 0)");
        }
        Ok(text)
    }

    /// Spawn the background evictor. No-op for `timeout_secs <= 0`: negative
    /// never evicts, and `0` keeps nothing resident to evict.
    pub fn start_evict_thread(self: &Arc<Self>) {
        self.spawn_evictor(EVICT_TICK);
    }

    /// Inner spawn with an injectable tick so tests can drive shutdown without a
    /// 30s wait. Returns the handle (the public caller lets it detach).
    fn spawn_evictor(self: &Arc<Self>, tick: Duration) -> Option<thread::JoinHandle<()>> {
        if self.timeout_secs <= 0 {
            return None;
        }
        let me = Arc::clone(self);
        Some(thread::spawn(move || loop {
            // Check at the top (prompt on shutdown) and again after the sleep
            // (catches a flag set during the tick) so a replaced cache exits fast.
            if me.shutdown.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(tick);
            if me.shutdown.load(Ordering::Relaxed) {
                break;
            }
            let mut slot = me.lock_slot();
            if slot.is_none() {
                continue;
            }
            // Re-check idle *after* acquiring the lock: a transcription may have
            // touched last_used while we slept or waited on the lock.
            let idle = me.lock_last_used().elapsed();
            if idle.as_secs() as i64 >= me.timeout_secs {
                *slot = None;
                info!("model unloaded after {}s idle", idle.as_secs());
            }
        }))
    }
}

/// Build the transcriber into `slot` if empty, then warm it (one throwaway pass
/// pays ORT's first-call graph-init cost here, not on the user's first real
/// transcription). Caller holds the slot lock.
fn load_locked(slot: &mut Option<Box<dyn Transcriber>>, factory: &Factory) -> Result<()> {
    if slot.is_none() {
        let t0 = Instant::now();
        let mut t = factory()?;
        t.warm();
        *slot = Some(t);
        info!("model loaded + warmed in {} ms", t0.elapsed().as_millis());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct PanicTranscriber;
    impl Transcriber for PanicTranscriber {
        fn transcribe(&mut self, _audio: &[f32]) -> Result<String> {
            panic!("transcriber blew up");
        }
        // The load-time warm must not panic; this fixture only blows up on a
        // real transcribe (the poison path under test).
        fn warm(&mut self) {}
    }

    struct OkTranscriber;
    impl Transcriber for OkTranscriber {
        fn transcribe(&mut self, _audio: &[f32]) -> Result<String> {
            Ok("ok".into())
        }
    }

    /// First build panics mid-transcribe (poisoning the slot mutex); the cache
    /// must recover, drop the dead transcriber, and rebuild on the next call.
    #[test]
    fn poisoned_slot_recovers_and_rebuilds() {
        let builds = Arc::new(AtomicUsize::new(0));
        let b = Arc::clone(&builds);
        let cache = ModelCache::with_factory(
            -1,
            Box::new(move || {
                Ok(if b.fetch_add(1, Ordering::SeqCst) == 0 {
                    Box::new(PanicTranscriber) as Box<dyn Transcriber>
                } else {
                    Box::new(OkTranscriber)
                })
            }),
        );

        let panicked =
            std::panic::catch_unwind(AssertUnwindSafe(|| cache.transcribe(&[0.0]))).is_err();
        assert!(panicked, "PanicTranscriber should have panicked");

        assert_eq!(cache.transcribe(&[0.0]).unwrap(), "ok");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "model should be rebuilt after poison"
        );
    }

    /// Loading the model warms it exactly once: `ensure_loaded` (and the inline
    /// load in `transcribe`) must run one throwaway pass so ORT's first-call cost
    /// is paid at load, and a second `ensure_loaded` on the resident model must
    /// not warm again.
    #[test]
    fn load_warms_once() {
        struct CountingTranscriber {
            warms: Arc<AtomicUsize>,
        }
        impl Transcriber for CountingTranscriber {
            fn transcribe(&mut self, _audio: &[f32]) -> Result<String> {
                Ok(String::new())
            }
            fn warm(&mut self) {
                self.warms.fetch_add(1, Ordering::SeqCst);
            }
        }

        let warms = Arc::new(AtomicUsize::new(0));
        let w = Arc::clone(&warms);
        let cache = ModelCache::with_factory(
            -1,
            Box::new(move || {
                Ok(Box::new(CountingTranscriber {
                    warms: Arc::clone(&w),
                }) as Box<dyn Transcriber>)
            }),
        );

        cache.ensure_loaded().unwrap();
        cache.ensure_loaded().unwrap();
        assert_eq!(warms.load(Ordering::SeqCst), 1, "warm runs once per load");
    }

    /// The default `warm()` routes through `transcribe`. Verify the trait
    /// default actually calls `transcribe`.
    #[test]
    fn default_warm_calls_transcribe() {
        struct DefaultWarm {
            calls: Arc<AtomicUsize>,
        }
        impl Transcriber for DefaultWarm {
            fn transcribe(&mut self, _audio: &[f32]) -> Result<String> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(String::new())
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut t = DefaultWarm {
            calls: Arc::clone(&calls),
        };
        t.warm();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// ensure_loaded must also survive a poisoned slot (keydown preload path).
    #[test]
    fn poisoned_slot_recovers_in_ensure_loaded() {
        let cache = ModelCache::with_factory(
            -1,
            Box::new(|| Ok(Box::new(PanicTranscriber) as Box<dyn Transcriber>)),
        );
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| cache.transcribe(&[0.0])));
        cache
            .ensure_loaded()
            .expect("ensure_loaded should recover from poison");
    }

    /// The evict thread must exit once `shutdown()` is signalled, so a model
    /// switch can drop the old cache (and its model) instead of leaking a zombie
    /// 30s-ticking thread. A short tick keeps the test fast; `join()` blocks
    /// until the thread observes the flag and breaks — it'd hang if it didn't.
    #[test]
    fn evict_thread_exits_on_shutdown() {
        let cache = ModelCache::with_factory(
            60, // positive timeout so the evictor actually spawns
            Box::new(|| Ok(Box::new(OkTranscriber) as Box<dyn Transcriber>)),
        );
        let handle = cache
            .spawn_evictor(Duration::from_millis(5))
            .expect("evictor must spawn for a positive timeout");
        cache.shutdown();
        handle
            .join()
            .expect("evict thread must exit after shutdown()");
    }
}
