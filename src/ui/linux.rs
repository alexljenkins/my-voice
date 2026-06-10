//! Linux tray via `ksni` (pure-Rust StatusNotifierItem over D-Bus). Runs on its
//! own thread, so the daemon keeps the main thread. This is the §1 scaffold:
//! a status line + Quit. The model/microphone/hotkey submenus land in §4.

use std::sync::mpsc::Sender;

use ksni::blocking::{Handle, TrayMethods};
use tracing::{info, warn};

use super::{TrayState, UiCommand};

/// The ksni tray model. `ksni` clones-and-renders this on its own thread; menu
/// activation closures get `&mut Self`, so the command sender lives here.
pub struct MyVoiceTray {
    state: TrayState,
    cmd_tx: Sender<UiCommand>,
}

impl ksni::Tray for MyVoiceTray {
    fn id(&self) -> String {
        "my-voice".into()
    }

    fn category(&self) -> ksni::Category {
        ksni::Category::ApplicationStatus
    }

    fn title(&self) -> String {
        "my-voice".into()
    }

    fn icon_name(&self) -> String {
        // Themed freedesktop names as placeholders; §3 swaps in bundled PNGs.
        match &self.state {
            TrayState::Ready => "audio-input-microphone",
            TrayState::Listening => "media-record",
            TrayState::Transcribing => "view-refresh",
            TrayState::Downloading { .. } => "emblem-downloads",
            TrayState::Error(_) => "dialog-warning",
        }
        .into()
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{MenuItem, StandardItem};
        vec![
            StandardItem {
                label: status_label(&self.state),
                enabled: false,
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            // §4 replaces this with the real settings submenus, each of which
            // writes config.toml then emits ReloadConfig. Until then a single
            // item proves the UI→daemon reload path end-to-end.
            StandardItem {
                label: "Reload settings".into(),
                activate: Box::new(|this: &mut Self| {
                    let _ = this.cmd_tx.send(UiCommand::ReloadConfig);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                activate: Box::new(|this: &mut Self| {
                    let _ = this.cmd_tx.send(UiCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

/// The non-interactive status line shown at the top of the menu (§4 mockup).
fn status_label(state: &TrayState) -> String {
    match state {
        TrayState::Ready => "● Ready".into(),
        TrayState::Listening => "● Listening…".into(),
        TrayState::Transcribing => "● Transcribing…".into(),
        TrayState::Downloading { pct } => format!("● Downloading… {pct}%"),
        TrayState::Error(msg) => format!("⚠ {msg}"),
    }
}

/// Handle for pushing state into the tray. `NoOp` when no tray host is present
/// (headless / no session bus / GNOME without the AppIndicator extension) — the
/// daemon still runs, just without a tray (§6.4).
pub enum LinuxUiHandle {
    Live(Handle<MyVoiceTray>),
    NoOp,
}

impl LinuxUiHandle {
    pub fn set_state(&self, state: TrayState) {
        if let LinuxUiHandle::Live(handle) = self {
            handle.update(move |tray| tray.state = state);
        }
    }
}

/// Spawn the ksni service on its own thread. A missing session bus or tray host
/// is non-fatal: log it and run headless rather than panic.
pub fn spawn(cmd_tx: Sender<UiCommand>) -> LinuxUiHandle {
    if std::env::var_os("DBUS_SESSION_BUS_ADDRESS").is_none() {
        warn!("no D-Bus session bus; running without a tray icon");
        return LinuxUiHandle::NoOp;
    }

    let tray = MyVoiceTray {
        state: TrayState::Ready,
        cmd_tx,
    };
    match tray.spawn() {
        Ok(handle) => {
            info!("tray registered");
            LinuxUiHandle::Live(handle)
        }
        Err(e) => {
            warn!("no tray host (running without a tray icon): {e}");
            LinuxUiHandle::NoOp
        }
    }
}
