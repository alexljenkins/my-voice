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
