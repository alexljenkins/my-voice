//! Linux tray via `ksni` (pure-Rust StatusNotifierItem over D-Bus). Runs on its
//! own thread, so the daemon keeps the main thread.
//!
//! Icons are driven by our own colored pixmaps (not themed names): symbolic
//! desktop themes render the record glyph monochrome-grey, which hid the
//! "listening" state — so we render a grey mic when idle and a solid red dot
//! while recording, and leave `icon_name` empty so the host uses our pixmap.
//!
//! Menu: Model / Microphone / Hotkeys / Paste mode submenus, then Start at
//! Login and Quit. The "current" option in every submenu is marked with a green
//! dot (a generated PNG in `icon_data` — menu text can't be colored or bolded in
//! DBusMenu) and stays enabled; only genuinely unavailable options are greyed.

use std::sync::mpsc::Sender;
use std::sync::OnceLock;

use ksni::blocking::{Handle, TrayMethods};
use tracing::{info, warn};

use super::{TrayMenuState, TrayState, UiCommand};

// ── icon generation ──────────────────────────────────────────────────────────

/// 16×16 microphone silhouette. Each u16 is one row; bit 15 = col 0.
const MIC_16: [u16; 16] = [
    0x0000, 0x03C0, 0x07E0, 0x07E0, 0x07E0, 0x07E0, 0x03C0, 0x0180, 0x0FF0, 0x0180, 0x0180, 0x07E0,
    0x0000, 0x0000, 0x0000, 0x0000,
];

/// 16×16 filled circle — the universal "recording" dot for the listening state.
const CIRCLE_16: [u16; 16] = [
    0x0000, 0x07E0, 0x1FF8, 0x3FFC, 0x7FFE, 0x7FFE, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x7FFE, 0x7FFE,
    0x3FFC, 0x1FF8, 0x07E0, 0x0000,
];

/// Build an ARGB32 `ksni::Icon` of `size`×`size` from a 16-row bitmap, scaled
/// nearest-neighbour, in the given RGB color.
fn make_icon(bitmap: &[u16; 16], size: usize, r: u8, g: u8, b: u8) -> ksni::Icon {
    let scale = (size / 16).max(1);
    let actual = scale * 16;
    let mut data = vec![0u8; actual * actual * 4];
    for row in 0..16usize {
        for col in 0..16usize {
            if (bitmap[row] >> (15 - col)) & 1 == 0 {
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
    ksni::Icon {
        width: actual as i32,
        height: actual as i32,
        data,
    }
}

/// (bitmap, color) for the tray icon in each state.
fn state_icon(state: &TrayState) -> (&'static [u16; 16], (u8, u8, u8)) {
    match state {
        TrayState::Ready => (&MIC_16, (170, 170, 170)), // neutral grey mic
        TrayState::Listening => (&CIRCLE_16, (230, 40, 40)), // solid red record dot
        TrayState::Transcribing => (&MIC_16, (235, 170, 0)), // amber mic
        TrayState::Downloading { .. } => (&MIC_16, (70, 150, 230)),
        TrayState::Error(_) => (&CIRCLE_16, (235, 150, 0)),
    }
}

/// Generated green "selected" dot as PNG bytes, cached. Used as menu-item
/// `icon_data` to mark the current option in a theme-independent green.
fn green_dot() -> Vec<u8> {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| dot_png((64, 200, 96))).clone()
}

/// Anti-aliased filled circle, RGBA, encoded as PNG.
fn dot_png(rgb: (u8, u8, u8)) -> Vec<u8> {
    const S: u32 = 22;
    let mut data = vec![0u8; (S * S * 4) as usize];
    let c = (S as f32 - 1.0) / 2.0;
    let radius = S as f32 * 0.40;
    for y in 0..S {
        for x in 0..S {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            let dist = (dx * dx + dy * dy).sqrt();
            let alpha = ((radius - dist + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let i = ((y * S + x) * 4) as usize;
            data[i] = rgb.0;
            data[i + 1] = rgb.1;
            data[i + 2] = rgb.2;
            data[i + 3] = alpha;
        }
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, S, S);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = enc.write_header() {
            let _ = writer.write_image_data(&data);
        }
    }
    out
}

// ── tray model ───────────────────────────────────────────────────────────────

pub struct MyVoiceTray {
    state: TrayState,
    menu: TrayMenuState,
    cmd_tx: Sender<UiCommand>,
}

type Item = ksni::menu::MenuItem<MyVoiceTray>;

/// A selectable option row: marked with the green dot when current, greyed only
/// when unavailable (current stays clickable, never greyed).
fn option_row<F>(label: String, selected: bool, available: bool, on_activate: F) -> Item
where
    F: Fn(&mut MyVoiceTray) + Send + 'static,
{
    ksni::menu::StandardItem {
        label,
        enabled: available,
        icon_data: if selected { green_dot() } else { Vec::new() },
        activate: Box::new(on_activate),
        ..Default::default()
    }
    .into()
}

/// A non-interactive, indented helper line under an option.
fn hint_line(text: &str) -> Item {
    ksni::menu::StandardItem {
        label: format!("    {text}"),
        enabled: false,
        ..Default::default()
    }
    .into()
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
        // Deliberately empty: forces the host to use our colored pixmap rather
        // than a monochrome symbolic theme icon (which hid the listening state).
        String::new()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        let (bitmap, (r, g, b)) = state_icon(&self.state);
        vec![
            make_icon(bitmap, 16, r, g, b),
            make_icon(bitmap, 32, r, g, b),
        ]
    }

    fn menu(&self) -> Vec<Item> {
        use ksni::menu::{CheckmarkItem, MenuItem, StandardItem, SubMenu};

        let mut items: Vec<Item> = Vec::new();

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
        let mut model_submenu: Vec<Item> = self
            .menu
            .models
            .iter()
            .map(|m| {
                let name = m.name.clone();
                let label = if m.downloaded {
                    m.label.clone()
                } else {
                    format!("{}  —  not downloaded", m.label)
                };
                option_row(label, m.active, true, move |this: &mut MyVoiceTray| {
                    let _ = this.cmd_tx.send(UiCommand::SetModel(name.clone()));
                })
            })
            .collect();
        if let TrayState::Downloading { pct } = &self.state {
            model_submenu.push(MenuItem::Separator);
            model_submenu.push(hint_line(&format!("Downloading…  {pct}%")));
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
        let mut mic_items: Vec<Item> = vec![option_row(
            "System default".into(),
            active_device.is_empty(),
            true,
            |this: &mut MyVoiceTray| {
                let _ = this.cmd_tx.send(UiCommand::SetAudioDevice(String::new()));
            },
        )];
        for dev in &self.menu.audio_devices {
            let selected = dev.value == active_device && !active_device.is_empty();
            let value = dev.value.clone();
            mic_items.push(option_row(
                dev.label.clone(),
                selected,
                true,
                move |this: &mut MyVoiceTray| {
                    let _ = this.cmd_tx.send(UiCommand::SetAudioDevice(value.clone()));
                },
            ));
        }
        items.push(
            SubMenu {
                label: "Microphone".into(),
                submenu: mic_items,
                ..Default::default()
            }
            .into(),
        );

        // ── Hotkeys submenu ───────────────────────────────────────────────────
        let mut hk_items: Vec<Item> = Vec::new();
        hk_items.push(hint_line(&format!(
            "Recording key:  {}",
            display_hotkey(&self.menu.hotkey)
        )));
        hk_items.push(
            StandardItem {
                label: "Set keybind…".into(),
                activate: Box::new(|this: &mut MyVoiceTray| {
                    let _ = this.cmd_tx.send(UiCommand::CaptureHotkey);
                }),
                ..Default::default()
            }
            .into(),
        );
        hk_items.push(MenuItem::Separator);
        hk_items.push(
            CheckmarkItem {
                label: "Clipboard shortcut".into(),
                checked: self.menu.clipboard_hotkey,
                activate: Box::new(|this: &mut MyVoiceTray| {
                    let new_val = !this.menu.clipboard_hotkey;
                    let _ = this.cmd_tx.send(UiCommand::SetClipboardHotkey(new_val));
                }),
                ..Default::default()
            }
            .into(),
        );
        hk_items.push(hint_line("Hold Shift + hotkey to copy instead of type"));
        hk_items.push(MenuItem::Separator);
        hk_items.push(
            CheckmarkItem {
                label: "Reserve the hotkey".into(),
                checked: self.menu.grab,
                activate: Box::new(|this: &mut MyVoiceTray| {
                    let new_val = !this.menu.grab;
                    let _ = this.cmd_tx.send(UiCommand::SetGrab(new_val));
                }),
                ..Default::default()
            }
            .into(),
        );
        hk_items.push(hint_line(
            "Stops the key doing its normal job (e.g. Caps Lock)",
        ));
        items.push(
            SubMenu {
                label: "Hotkeys".into(),
                submenu: hk_items,
                ..Default::default()
            }
            .into(),
        );

        // ── Paste mode submenu ────────────────────────────────────────────────
        let injection = self.menu.injection.clone();
        let type_available = self.menu.inject_type_available;
        // When typing is unavailable the daemon falls back to clipboard at
        // runtime, so reflect that: clipboard is the effective current mode and
        // "Paste at cursor" is shown locked rather than selected.
        let paste_selected = injection != "clipboard" && type_available;
        let clipboard_selected = injection == "clipboard" || !type_available;
        let mut paste_items: Vec<Item> = Vec::new();
        paste_items.push(option_row(
            "Paste at cursor".into(),
            paste_selected,
            type_available,
            |this: &mut MyVoiceTray| {
                let _ = this.cmd_tx.send(UiCommand::SetInjection("auto".into()));
            },
        ));
        if type_available {
            paste_items.push(hint_line(
                "Falls back to clipboard if the cursor can't take it",
            ));
        } else {
            paste_items.push(hint_line("Locked — needs a typing tool:"));
            for line in self.menu.inject_unlock_hint.lines() {
                paste_items.push(hint_line(line));
            }
        }
        paste_items.push(option_row(
            "Copy to clipboard".into(),
            clipboard_selected,
            true,
            |this: &mut MyVoiceTray| {
                let _ = this
                    .cmd_tx
                    .send(UiCommand::SetInjection("clipboard".into()));
            },
        ));
        items.push(
            SubMenu {
                label: "Paste mode".into(),
                submenu: paste_items,
                ..Default::default()
            }
            .into(),
        );

        items.push(MenuItem::Separator);

        // ── Start at Login ────────────────────────────────────────────────────
        items.push(
            CheckmarkItem {
                label: "Start at login".into(),
                checked: self.menu.start_at_login,
                activate: Box::new(|this: &mut MyVoiceTray| {
                    let new_val = !this.menu.start_at_login;
                    let _ = this.cmd_tx.send(UiCommand::SetStartAtLogin(new_val));
                }),
                ..Default::default()
            }
            .into(),
        );

        items.push(MenuItem::Separator);

        items.push(
            StandardItem {
                label: "Quit".into(),
                activate: Box::new(|this: &mut MyVoiceTray| {
                    let _ = this.cmd_tx.send(UiCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        );

        items
    }
}

/// Prettify a hotkey config string for display (e.g. `Ctrl+Period` → `Ctrl + Period`).
fn display_hotkey(hk: &str) -> String {
    if hk.is_empty() {
        return "—".into();
    }
    hk.split('+').collect::<Vec<_>>().join(" + ")
}

fn status_label(state: &TrayState) -> String {
    match state {
        TrayState::Ready => "Ready".into(),
        TrayState::Listening => "Listening…".into(),
        TrayState::Transcribing => "Transcribing…".into(),
        TrayState::Downloading { pct } => format!("Downloading… {pct}%"),
        TrayState::Error(msg) => format!("⚠ {msg}"),
    }
}

// ── handle ───────────────────────────────────────────────────────────────────

/// Handle for pushing state into the tray. `NoOp` when no tray host is present
/// (headless / no session bus / GNOME without the AppIndicator extension) — the
/// daemon still runs (§6.4).
///
/// Cloneable: `Handle<T>` wraps an `Arc` so cloning is cheap and safe.
pub enum LinuxUiHandle {
    Live(Handle<MyVoiceTray>),
    NoOp,
}

impl Clone for LinuxUiHandle {
    fn clone(&self) -> Self {
        match self {
            LinuxUiHandle::Live(h) => LinuxUiHandle::Live(h.clone()),
            LinuxUiHandle::NoOp => LinuxUiHandle::NoOp,
        }
    }
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
