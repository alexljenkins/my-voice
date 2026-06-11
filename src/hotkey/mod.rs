//! Hotkey listening: platform dispatch + the event type the state machine consumes.

use std::sync::mpsc::Sender;

use anyhow::Result;

use crate::config::Config;

/// Events emitted by the platform listener(s) into the main state machine.
#[derive(Debug, Clone, Copy)]
pub enum HotkeyEvent {
    /// Hotkey pressed. `clipboard_only` = shift held at press time.
    Press { clipboard_only: bool },
    /// Hotkey released.
    Release,
}

/// The modifier keys that may gate a hotkey combo (e.g. `Ctrl+Period`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub sup: bool,
}

impl Mods {
    /// True if every modifier this combo requires is currently held (extra
    /// modifiers are allowed — e.g. Shift for the clipboard shortcut).
    pub fn satisfied_by(&self, held: &Mods) -> bool {
        (!self.ctrl || held.ctrl)
            && (!self.shift || held.shift)
            && (!self.alt || held.alt)
            && (!self.sup || held.sup)
    }
}

/// Split a hotkey string into its required modifiers and the main key token.
/// `"Ctrl+Period"` → (`{ctrl}`, `"Period"`); `"CapsLock"` → (`{}`, `"CapsLock"`).
pub fn parse_hotkey(s: &str) -> (Mods, &str) {
    let mut mods = Mods::default();
    let parts: Vec<&str> = s
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let Some((main, modifiers)) = parts.split_last() else {
        return (mods, s);
    };
    for m in modifiers {
        match m.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => mods.ctrl = true,
            "shift" => mods.shift = true,
            "alt" => mods.alt = true,
            "super" | "meta" | "win" | "cmd" => mods.sup = true,
            _ => {}
        }
    }
    (mods, main)
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// Restore any persistent platform state set up by the listener (macOS hidutil
/// remap). No-op on Linux, where evdev grabs are released by fd-close on exit.
pub fn restore_platform() {
    #[cfg(target_os = "macos")]
    macos::restore_mapping();
}

/// Spawn the platform listener thread(s). Returns once listeners are running;
/// events arrive asynchronously on `tx`.
pub fn spawn_listener(config: &Config, tx: Sender<HotkeyEvent>) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::spawn(config, tx)
    }
    #[cfg(target_os = "macos")]
    {
        macos::spawn(config, tx)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (config, tx);
        anyhow::bail!("unsupported platform: only Linux and macOS are supported");
    }
}
