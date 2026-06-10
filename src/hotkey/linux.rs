//! Linux hotkey: evdev grab + uinput passthrough, with an ungrabbed fallback.
//!
//! evdev grab is per-DEVICE, not per-key — a bare exclusive grab silences the
//! whole keyboard. The fix: grab the real device, create a uinput virtual
//! keyboard, and re-emit every event except the hotkey.

use std::sync::mpsc::Sender;
use std::thread;

use anyhow::{anyhow, Result};
use evdev::uinput::{VirtualDevice, VirtualDeviceBuilder};
use evdev::{AttributeSet, Device, EventType, Key};
use tracing::{error, info, warn};

use super::HotkeyEvent;
use crate::config::Config;

const UINPUT_HELP: &str = "\
uinput unavailable; running ungrabbed (CapsLock will still toggle).
Fix: sudo usermod -aG input $USER
     echo 'KERNEL==\"uinput\", GROUP=\"input\", MODE=\"0660\"' | sudo tee /etc/udev/rules.d/99-my-voice.rules
     sudo modprobe uinput   # and add 'uinput' to /etc/modules-load.d/
then re-login.";

pub fn spawn(config: &Config, tx: Sender<HotkeyEvent>) -> Result<()> {
    let target = resolve_key(&config.hotkey)?;
    let want_grab = config.grab;
    let clipboard_hotkey = config.clipboard_hotkey;

    let mut count = 0;
    for (path, device) in evdev::enumerate() {
        // Never grab our own virtual passthrough device (restart feedback loop).
        if device.name().is_some_and(|n| n.starts_with("my-voice")) {
            continue;
        }
        // Qualify: reports the target key AND KEY_A (filters lid switches etc.).
        let qualifies = device
            .supported_keys()
            .is_some_and(|k| k.contains(target) && k.contains(Key::KEY_A));
        if !qualifies {
            continue;
        }

        count += 1;
        let name = device.name().unwrap_or("<unknown>").to_string();
        let tx = tx.clone();
        thread::Builder::new()
            .name(format!("hotkey:{}", path.display()))
            .spawn(move || {
                run_device(name, device, target, want_grab, clipboard_hotkey, tx);
            })?;
    }

    if count == 0 {
        warn!(
            "no accessible keyboard reporting '{}'. If a keyboard is connected, \
             you likely lack permission to read /dev/input.\n  \
             Fix: sudo usermod -aG input $USER   (then re-login)",
            config.hotkey
        );
        return Err(anyhow!(
            "no accessible keyboard device for hotkey '{}'",
            config.hotkey
        ));
    }

    info!(
        "listening for hotkey '{}' on {count} device(s)",
        config.hotkey
    );
    Ok(())
}

/// Ungrabs on drop — including on panic. A stuck grab is a dead keyboard.
struct DeviceGuard {
    device: Device,
    grabbed: bool,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.grabbed {
            let _ = self.device.ungrab();
        }
    }
}

fn run_device(
    name: String,
    mut device: Device,
    target: Key,
    want_grab: bool,
    clipboard_hotkey: bool,
    tx: Sender<HotkeyEvent>,
) {
    let mut vdev: Option<VirtualDevice> = None;
    let mut grabbed = false;

    if want_grab {
        match try_grab(&mut device) {
            Ok(v) => {
                vdev = Some(v);
                grabbed = true;
                info!("grabbed '{name}' (uinput passthrough active)");
            }
            Err(e) => {
                warn!("{UINPUT_HELP}\n  ({name}: {e})");
            }
        }
    } else {
        info!("listening ungrabbed on '{name}' (grab disabled in config)");
    }

    let mut guard = DeviceGuard { device, grabbed };
    let target_code = target.code();
    let mut shift_down = false;

    loop {
        let events = match guard.device.fetch_events() {
            Ok(e) => e,
            Err(e) => {
                error!("'{name}': reading events failed: {e}");
                break;
            }
        };

        for ev in events {
            if ev.event_type() == EventType::KEY && ev.code() == target_code {
                let outgoing = match ev.value() {
                    1 => Some(HotkeyEvent::Press {
                        clipboard_only: clipboard_hotkey && shift_down,
                    }),
                    0 => Some(HotkeyEvent::Release),
                    _ => None, // autorepeat (2): swallow
                };
                if let Some(e) = outgoing {
                    if tx.send(e).is_err() {
                        return; // main hung up
                    }
                }
                continue; // never pass the hotkey through
            }

            // Track shift from events passing through.
            if ev.event_type() == EventType::KEY
                && (ev.code() == Key::KEY_LEFTSHIFT.code()
                    || ev.code() == Key::KEY_RIGHTSHIFT.code())
            {
                shift_down = ev.value() != 0;
            }

            // Passthrough verbatim when grabbed.
            if let Some(v) = vdev.as_mut() {
                if let Err(e) = v.emit(&[ev]) {
                    error!("'{name}': passthrough emit failed: {e}");
                }
            }
        }
    }
}

/// Grab the device and build a matching uinput passthrough keyboard. On any
/// failure after a successful grab, ungrab before returning the error.
fn try_grab(device: &mut Device) -> Result<VirtualDevice> {
    device.grab()?;
    let keys: AttributeSet<Key> = device
        .supported_keys()
        .map(|k| k.iter().collect())
        .unwrap_or_default();
    match build_vdev(&keys) {
        Ok(v) => Ok(v),
        Err(e) => {
            let _ = device.ungrab();
            Err(e)
        }
    }
}

fn build_vdev(keys: &AttributeSet<Key>) -> Result<VirtualDevice> {
    let vdev = VirtualDeviceBuilder::new()?
        .name("my-voice passthrough")
        .with_keys(keys)?
        .build()?;
    Ok(vdev)
}

/// Map a config key name (no prefix, case-insensitive) to an evdev `Key`.
/// Covers the keys realistically used for push-to-talk; default is CapsLock.
pub fn resolve_key(name: &str) -> Result<Key> {
    let mut n: String = name.to_uppercase();
    n.retain(|c| c != ' ' && c != '_');
    if let Some(stripped) = n.strip_prefix("KEY") {
        n = stripped.to_string();
    }

    let key = match n.as_str() {
        "CAPSLOCK" | "CAPS" => Key::KEY_CAPSLOCK,
        "SCROLLLOCK" | "SCROLL" => Key::KEY_SCROLLLOCK,
        "NUMLOCK" | "NUM" => Key::KEY_NUMLOCK,
        "RIGHTCTRL" | "RCTRL" => Key::KEY_RIGHTCTRL,
        "LEFTCTRL" | "LCTRL" => Key::KEY_LEFTCTRL,
        "RIGHTALT" | "RALT" => Key::KEY_RIGHTALT,
        "LEFTALT" | "LALT" => Key::KEY_LEFTALT,
        "RIGHTSHIFT" | "RSHIFT" => Key::KEY_RIGHTSHIFT,
        "LEFTSHIFT" | "LSHIFT" => Key::KEY_LEFTSHIFT,
        "RIGHTMETA" | "RMETA" | "RIGHTSUPER" => Key::KEY_RIGHTMETA,
        "LEFTMETA" | "LMETA" | "LEFTSUPER" | "SUPER" => Key::KEY_LEFTMETA,
        other => resolve_fkey(other).ok_or_else(|| {
            anyhow!("unrecognized hotkey name '{name}' (try CapsLock, F12, RightCtrl)")
        })?,
    };
    Ok(key)
}

fn resolve_fkey(n: &str) -> Option<Key> {
    let num: u8 = n.strip_prefix('F')?.parse().ok()?;
    Some(match num {
        1 => Key::KEY_F1,
        2 => Key::KEY_F2,
        3 => Key::KEY_F3,
        4 => Key::KEY_F4,
        5 => Key::KEY_F5,
        6 => Key::KEY_F6,
        7 => Key::KEY_F7,
        8 => Key::KEY_F8,
        9 => Key::KEY_F9,
        10 => Key::KEY_F10,
        11 => Key::KEY_F11,
        12 => Key::KEY_F12,
        13 => Key::KEY_F13,
        14 => Key::KEY_F14,
        15 => Key::KEY_F15,
        16 => Key::KEY_F16,
        17 => Key::KEY_F17,
        18 => Key::KEY_F18,
        19 => Key::KEY_F19,
        20 => Key::KEY_F20,
        21 => Key::KEY_F21,
        22 => Key::KEY_F22,
        23 => Key::KEY_F23,
        24 => Key::KEY_F24,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_capslock_variants() {
        assert_eq!(resolve_key("CapsLock").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("capslock").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("CAPS_LOCK").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("KEY_CAPSLOCK").unwrap(), Key::KEY_CAPSLOCK);
    }

    #[test]
    fn parses_fkeys() {
        assert_eq!(resolve_key("F12").unwrap(), Key::KEY_F12);
        assert_eq!(resolve_key("f18").unwrap(), Key::KEY_F18);
    }

    #[test]
    fn rejects_unknown() {
        assert!(resolve_key("PenguinKey").is_err());
        assert!(resolve_key("F99").is_err());
    }
}
