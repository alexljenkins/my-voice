//! Platform-neutral tray/menu-bar UI.
//!
//! Two directions of flow, both over channels: the daemon pushes [`TrayState`]
//! and [`TrayMenuState`] into the UI via [`UiHandle`], and the UI sends
//! [`UiCommand`]s back to the daemon. One neutral model is rendered per
//! backend, mirroring `hotkey/` and `injector/`. Linux renders via `ksni`
//! (StatusNotifierItem over D-Bus, on its own thread — so the daemon keeps the
//! main thread). macOS will render via `tray-icon` on the main thread (event
//! loop owns it), which inverts the daemon onto a background thread — not yet
//! implemented (see `ui/macos.rs`).

use std::sync::mpsc::Sender;

/// Visual state of the tray icon + status line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayState {
    /// Running, waiting for the hotkey.
    Ready,
    /// Hotkey held, recording.
    Listening,
    /// Released, inference running.
    Transcribing,
    /// Model download in progress.
    Downloading { pct: u8 },
    /// Something needs attention; the string is the user-facing reason.
    Error(String),
}

/// One model entry in the tray model submenu.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelItem {
    /// Config value (e.g. "moonshine-tiny").
    pub name: String,
    /// Human-readable label shown in the menu (e.g. "Faster  •  moonshine-tiny").
    pub label: String,
    /// Currently selected model (what the config says).
    pub active: bool,
    /// Model files exist on disk and are ready to use.
    pub downloaded: bool,
}

/// One selectable input device in the microphone submenu. `value` is what gets
/// saved to config (substring-matched against the cpal device name); `label` is
/// the friendly name shown to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceItem {
    pub value: String,
    pub label: String,
}

/// Configuration state the daemon pushes into the tray to drive menu rendering.
/// All fields default to empty/off so the tray can start before the daemon
/// pushes real state.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TrayMenuState {
    pub models: Vec<ModelItem>,
    /// Curated input devices (enumerated at startup).
    pub audio_devices: Vec<DeviceItem>,
    /// Currently configured device (empty string = system default).
    pub active_device: String,
    pub hotkey: String,
    pub injection: String,
    /// Whether a real typing tool is available so "Paste at cursor" can work.
    pub inject_type_available: bool,
    /// Plain-language unlock instructions shown when typing is unavailable.
    pub inject_unlock_hint: String,
    pub grab: bool,
    pub clipboard_hotkey: bool,
    /// Whether the XDG autostart entry is currently installed.
    pub start_at_login: bool,
}

/// Commands the UI sends back to the daemon.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum UiCommand {
    /// Switch to this model (name matches a `ModelItem::name`).
    SetModel(String),
    /// Switch to this audio device (empty = system default).
    SetAudioDevice(String),
    /// Change injection mode ("auto" | "clipboard").
    SetInjection(String),
    /// Toggle grab mode — requires daemon restart to take effect.
    SetGrab(bool),
    /// Toggle whether Shift+hotkey copies to clipboard instead of typing.
    SetClipboardHotkey(bool),
    /// Open the key-capture popup, then apply the chosen hotkey (self-restart).
    CaptureHotkey,
    /// Enable/disable launching at login (XDG autostart entry).
    SetStartAtLogin(bool),
    /// Config file changed on disk; reload and re-apply live between utterances.
    #[allow(dead_code)]
    ReloadConfig,
    /// User chose Quit.
    Quit,
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// A live handle the daemon uses to push state into the tray. Cloneable so
/// background threads (e.g. auto-download) can push state independently.
pub struct UiHandle {
    #[cfg(target_os = "linux")]
    inner: linux::LinuxUiHandle,
    #[cfg(not(target_os = "linux"))]
    _priv: (),
}

impl Clone for UiHandle {
    fn clone(&self) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            inner: self.inner.clone(),
            #[cfg(not(target_os = "linux"))]
            _priv: (),
        }
    }
}

impl UiHandle {
    /// Update the tray icon + status line.
    pub fn set_state(&self, state: TrayState) {
        #[cfg(target_os = "linux")]
        self.inner.set_state(state);
        #[cfg(not(target_os = "linux"))]
        let _ = state;
    }

    /// Push new config state into the tray so the menu reflects current settings.
    pub fn set_menu(&self, menu: TrayMenuState) {
        #[cfg(target_os = "linux")]
        self.inner.set_menu(menu);
        #[cfg(not(target_os = "linux"))]
        let _ = menu;
    }
}

/// Spawn the tray on its own thread and return a handle for pushing state.
/// UI-originated commands arrive on `cmd_tx`. Failing to attach to a tray host
/// is non-fatal: the daemon runs headless and re-attaches when a host appears
/// (§6.4).
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
