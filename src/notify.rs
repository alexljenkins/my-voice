//! Desktop notifications.
//!
//! Call `init()` once at daemon startup to enable per-session deduplication.
//! `once()` fires at most one notification per `ErrorKind` per session.
//! `send()` always fires (info / success events, or pre-init errors like "already running").

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Error kinds used to deduplicate repeated notifications within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    NoMicrophone,
    HotkeySetupNeeded,
    ModelMissing,
    ModelDownloadFailed,
    InjectionFailed,
    TranscriberPanicked,
}

static SEEN: OnceLock<Mutex<HashSet<ErrorKind>>> = OnceLock::new();

/// Enable per-session deduplication. Call once at daemon startup.
pub fn init() {
    SEEN.get_or_init(|| Mutex::new(HashSet::new()));
}

/// Fire a notification at most once per session for this kind.
/// Falls through to `send()` if called before `init()`.
pub fn once(kind: ErrorKind, title: &str, body: &str) {
    if let Some(seen) = SEEN.get() {
        if seen.lock().unwrap().insert(kind) {
            send_platform(title, body);
        }
    } else {
        send_platform(title, body);
    }
}

/// Fire a notification unconditionally (no deduplication, no `init()` required).
pub fn send(title: &str, body: &str) {
    send_platform(title, body);
}

#[cfg(target_os = "linux")]
fn send_platform(title: &str, body: &str) {
    use notify_rust::Notification;
    if let Err(e) = Notification::new()
        .summary(title)
        .body(body)
        .appname("my-voice")
        .show()
    {
        tracing::debug!("desktop notification failed: {e}");
    }
}

#[cfg(not(target_os = "linux"))]
fn send_platform(title: &str, body: &str) {
    // macOS: notify-rust works from a signed .app bundle; dev builds fall back to tracing.
    tracing::warn!("[notify] {title}: {body}");
}
