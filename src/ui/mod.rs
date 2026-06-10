//! Platform-neutral tray/menu-bar UI.
//!
//! Two directions of flow, both over channels: the daemon pushes [`TrayState`]
//! into the UI via [`UiHandle`], and the UI sends [`UiCommand`]s back to the
//! daemon. One neutral model is rendered per backend, mirroring `hotkey/` and
//! `injector/`. Linux renders via `ksni` (StatusNotifierItem over D-Bus, on its
//! own thread — so the daemon keeps the main thread). macOS will render via
//! `tray-icon` on the main thread (event loop owns it), which inverts the
//! daemon onto a background thread — not yet implemented (see `ui/macos.rs`).

use std::sync::mpsc::Sender;

/// Visual state of the tray icon + status line. Rendered per-backend (§3 fills
/// in real icon assets; the scaffold maps each to a freedesktop themed name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayState {
    /// Running, waiting for the hotkey.
    Ready,
    /// Hotkey held, recording.
    Listening,
    /// Released, inference running.
    Transcribing,
    /// Model download in progress. Emitted by the §6 first-run download flow.
    #[allow(dead_code)]
    Downloading { pct: u8 },
    /// Something needs attention; the string is the user-facing reason.
    Error(String),
}

/// Commands the UI sends back to the daemon.
#[derive(Debug, Clone)]
pub enum UiCommand {
    /// Config file changed on disk; reload and re-apply live between utterances.
    ReloadConfig,
    /// User chose Quit.
    Quit,
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// A live handle the daemon uses to push state into the tray. Cheap to clone is
/// not needed — one handle lives in the daemon loop.
pub struct UiHandle {
    #[cfg(target_os = "linux")]
    inner: linux::LinuxUiHandle,
    #[cfg(not(target_os = "linux"))]
    _priv: (),
}

impl UiHandle {
    /// Update the tray icon + status line. Safe to call on every transition;
    /// a backend with no attached tray host drops it silently.
    pub fn set_state(&self, state: TrayState) {
        #[cfg(target_os = "linux")]
        self.inner.set_state(state);
        #[cfg(not(target_os = "linux"))]
        let _ = state;
    }
}

/// Spawn the tray on its own thread and return a handle for pushing state.
/// UI-originated commands arrive on `cmd_tx`. Failing to attach to a tray host
/// is non-fatal: the daemon runs headless (hotkey + notifications still work)
/// and a later pass re-attaches when a host appears (§6.4).
pub fn spawn(cmd_tx: Sender<UiCommand>) -> UiHandle {
    #[cfg(target_os = "linux")]
    {
        UiHandle {
            inner: linux::spawn(cmd_tx),
        }
    }
    #[cfg(target_os = "macos")]
    {
        macos::spawn(cmd_tx);
        UiHandle { _priv: () }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = cmd_tx;
        UiHandle { _priv: () }
    }
}
