//! Linux tray via `ksni` (pure-Rust StatusNotifierItem over D-Bus). Runs on its
//! own thread, so the daemon keeps the main thread.
//!
//! §3: icon pixmaps — programmatic microphone silhouette at 16×16 and 32×32,
//!     ARGB32 format. Themed icon names remain as the primary source for
//!     desktops with a proper icon theme; pixmaps are the fallback.
//!
//! §4: full settings menu — Model / Microphone / Hotkey / Inject as submenus
//!     plus Clipboard shortcut, Grab mode, and Quit.

use std::sync::mpsc::Sender;

use ksni::blocking::{Handle, TrayMethods};
use tracing::{info, warn};

use super::{TrayMenuState, TrayState, UiCommand};

// ── §3 icon generation ───────────────────────────────────────────────────────

/// 16×16 microphone silhouette bitmap. Each u16 is one row; bit 15 = col 0.
/// Shape: rounded capsule body (rows 1-6), neck (row 7), mount yoke (row 8),
/// stand (rows 9-10), base plate (row 11).
const MIC_16: [u16; 16] = [
    0x0000, // row 0
    0x03C0, // row 1  cols 6-9  (capsule top arc)
    0x07E0, // row 2  cols 5-10 (capsule body)
    0x07E0, // row 3
    0x07E0, // row 4
    0x07E0, // row 5
    0x03C0, // row 6  cols 6-9  (capsule bottom arc)
    0x0180, // row 7  cols 7-8  (neck)
    0x0FF0, // row 8  cols 4-11 (mount yoke)
    0x0180, // row 9  cols 7-8  (stand)
    0x0180, // row 10
    0x07E0, // row 11 cols 5-10 (base plate)
    0x0000, // row 12
    0x0000, // row 13
    0x0000, // row 14
    0x0000, // row 15
];

/// Build an ARGB32 `ksni::Icon` of `size`×`size` pixels in the given RGB color.
/// The 16×16 mic silhouette is scaled up (nearest-neighbour) for sizes > 16.
fn make_mic_icon(size: usize, r: u8, g: u8, b: u8) -> ksni::Icon {
    let scale = (size / 16).max(1);
    let actual = scale * 16;
    let mut data = vec![0u8; actual * actual * 4];
    for row in 0..16usize {
        for col in 0..16usize {
            if (MIC_16[row] >> (15 - col)) & 1 == 0 {
                continue;
            }
            for dr in 0..scale {
                for dc in 0..scale {
                    let idx = ((row * scale + dr) * actual + (col * scale + dc)) * 4;
                    data[idx] = 255; // A
                    data[idx + 1] = r;
                    data[idx + 2] = g;
                    data[idx + 3] = b;
                }
            }
        }
    }
    ksni::Icon { width: actual as i32, height: actual as i32, data }
}

fn state_color(state: &TrayState) -> (u8, u8, u8) {
    match state {
        TrayState::Ready => (200, 200, 200),
        TrayState::Listening => (220, 60, 60),
        TrayState::Transcribing => (200, 140, 0),
        TrayState::Downloading { .. } => (60, 140, 220),
        TrayState::Error(_) => (220, 150, 0),
    }
}

// ── §4 menu constants ────────────────────────────────────────────────────────

static HOTKEY_OPTIONS: &[&str] = &[
    "CapsLock", "RightCtrl", "F12", "F13", "F14", "F15", "F16", "F17", "F18",
    "F19", "F20", "ScrollLock",
];

// ── tray model ───────────────────────────────────────────────────────────────

pub struct MyVoiceTray {
    state: TrayState,
    menu: TrayMenuState,
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
        // Primary: themed freedesktop icon names (work on desktops with icon themes).
        match &self.state {
            TrayState::Ready => "audio-input-microphone",
            TrayState::Listening => "media-record",
            TrayState::Transcribing => "view-refresh",
            TrayState::Downloading { .. } => "emblem-downloads",
            TrayState::Error(_) => "dialog-warning",
        }
        .into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        // Fallback: programmatic mic silhouette in state-specific color.
        let (r, g, b) = state_color(&self.state);
        vec![make_mic_icon(16, r, g, b), make_mic_icon(32, r, g, b)]
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, MenuItem, StandardItem, SubMenu};

        let mut items: Vec<MenuItem<Self>> = Vec::new();

        // ── status line ──────────────────────────────────────────────────────
        items.push(
            StandardItem {
                label: status_label(&self.state),
                enabled: false,
                ..Default::default()
            }
            .into(),
        );
        items.push(MenuItem::Separator);

        // ── Model submenu ────────────────────────────────────────────────────
        let model_items: Vec<MenuItem<Self>> = self
            .menu
            .models
            .iter()
            .map(|m| {
                let name = m.name.clone();
                let indicator = if m.active { "●" } else if m.downloaded { "○" } else { "  " };
                let label = if m.downloaded {
                    format!("{indicator} {}", m.label)
                } else {
                    format!("{indicator} {}  (not downloaded)", m.label)
                };
                StandardItem {
                    label,
                    enabled: !m.active,
                    activate: Box::new(move |this: &mut Self| {
                        let _ = this.cmd_tx.send(UiCommand::SetModel(name.clone()));
                    }),
                    ..Default::default()
                }
                .into()
            })
            .collect();

        // Show downloading progress inside the model submenu when relevant.
        let mut model_submenu = model_items;
        if let TrayState::Downloading { pct } = &self.state {
            model_submenu.push(MenuItem::Separator);
            model_submenu.push(
                StandardItem {
                    label: format!("  Downloading…  {pct}%"),
                    enabled: false,
                    ..Default::default()
                }
                .into(),
            );
        }

        items.push(
            SubMenu {
                label: "Model".into(),
                submenu: model_submenu,
                ..Default::default()
            }
            .into(),
        );

        // ── Microphone submenu ────────────────────────────────────────────────
        let active_device = self.menu.active_device.clone();
        let mut mic_items: Vec<MenuItem<Self>> = vec![StandardItem {
            label: if active_device.is_empty() {
                "● System default".into()
            } else {
                "  System default".into()
            },
            enabled: !active_device.is_empty(),
            activate: Box::new(|this: &mut Self| {
                let _ = this.cmd_tx.send(UiCommand::SetAudioDevice(String::new()));
            }),
            ..Default::default()
        }
        .into()];

        for dev in &self.menu.audio_devices {
            let selected = *dev == active_device && !active_device.is_empty();
            let ind = if selected { "●" } else { "  " };
            let dev_clone = dev.clone();
            mic_items.push(
                StandardItem {
                    label: format!("{ind} {dev}"),
                    enabled: !selected,
                    activate: Box::new(move |this: &mut Self| {
                        let _ = this.cmd_tx.send(UiCommand::SetAudioDevice(dev_clone.clone()));
                    }),
                    ..Default::default()
                }
                .into(),
            );
        }

        items.push(SubMenu { label: "Microphone".into(), submenu: mic_items, ..Default::default() }.into());

        // ── Hotkey submenu ────────────────────────────────────────────────────
        let current_hk = self.menu.hotkey.clone();
        let hk_items: Vec<MenuItem<Self>> = HOTKEY_OPTIONS
            .iter()
            .map(|&hk| {
                let selected = hk == current_hk;
                let ind = if selected { "●" } else { "  " };
                let hk_s = hk.to_string();
                StandardItem {
                    label: format!("{ind} {hk}"),
                    enabled: !selected,
                    activate: Box::new(move |this: &mut Self| {
                        let _ = this.cmd_tx.send(UiCommand::SetHotkey(hk_s.clone()));
                    }),
                    ..Default::default()
                }
                .into()
            })
            .collect();

        items.push(SubMenu { label: "Hotkey".into(), submenu: hk_items, ..Default::default() }.into());

        // ── Inject as submenu ─────────────────────────────────────────────────
        let injection = self.menu.injection.clone();
        let inject_options = [
            ("auto", "Type directly (auto-detect)"),
            ("clipboard", "Copy to clipboard"),
        ];
        let inject_items: Vec<MenuItem<Self>> = inject_options
            .iter()
            .map(|&(val, label)| {
                let selected = val == injection;
                let ind = if selected { "●" } else { "  " };
                let val_s = val.to_string();
                StandardItem {
                    label: format!("{ind} {label}"),
                    enabled: !selected,
                    activate: Box::new(move |this: &mut Self| {
                        let _ = this.cmd_tx.send(UiCommand::SetInjection(val_s.clone()));
                    }),
                    ..Default::default()
                }
                .into()
            })
            .collect();

        items.push(SubMenu { label: "Inject as".into(), submenu: inject_items, ..Default::default() }.into());

        items.push(MenuItem::Separator);

        // ── Clipboard shortcut toggle ─────────────────────────────────────────
        items.push(
            CheckmarkItem {
                label: "Clipboard shortcut  (Shift+hotkey)".into(),
                checked: self.menu.clipboard_hotkey,
                activate: Box::new(|this: &mut Self| {
                    let new_val = !this.menu.clipboard_hotkey;
                    let _ = this.cmd_tx.send(UiCommand::SetClipboardHotkey(new_val));
                }),
                ..Default::default()
            }
            .into(),
        );

        // Start at Login — stub until §7 is implemented.
        items.push(
            StandardItem {
                label: "Start at Login".into(),
                enabled: false,
                ..Default::default()
            }
            .into(),
        );

        // Grab mode toggle (Linux only).
        items.push(
            CheckmarkItem {
                label: "Grab mode  (advanced)".into(),
                checked: self.menu.grab,
                activate: Box::new(|this: &mut Self| {
                    let new_val = !this.menu.grab;
                    let _ = this.cmd_tx.send(UiCommand::SetGrab(new_val));
                }),
                ..Default::default()
            }
            .into(),
        );

        items.push(MenuItem::Separator);

        items.push(
            StandardItem {
                label: "Quit".into(),
                activate: Box::new(|this: &mut Self| {
                    let _ = this.cmd_tx.send(UiCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        );

        items
    }
}

fn status_label(state: &TrayState) -> String {
    match state {
        TrayState::Ready => "● Ready".into(),
        TrayState::Listening => "● Listening…".into(),
        TrayState::Transcribing => "● Transcribing…".into(),
        TrayState::Downloading { pct } => format!("● Downloading… {pct}%"),
        TrayState::Error(msg) => format!("⚠ {msg}"),
    }
}

// ── handle ───────────────────────────────────────────────────────────────────

/// Handle for pushing state into the tray. `NoOp` when no tray host is present
/// (headless / no session bus / GNOME without the AppIndicator extension) — the
/// daemon still runs (§6.4).
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

    pub fn set_menu(&self, menu: TrayMenuState) {
        if let LinuxUiHandle::Live(handle) = self {
            handle.update(move |tray| tray.menu = menu);
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
        menu: TrayMenuState::default(),
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
