//! Linux hotkey: X11 XGrabKey (no permissions; primary for X11/XWayland sessions)
//! with evdev grab + uinput passthrough fallback (Wayland-only or X11 failure).
//!
//! Backend selection:
//!   DISPLAY set      → X11 XGrabKey (no permissions needed; falls back on failure)
//!   Wayland-only     → evdev ungrabbed (input group required; guided via §5 notification)
//!   grab = true      → evdev exclusive grab + uinput passthrough (input group + udev rule)
//!
//! The XDG GlobalShortcuts portal backend (KDE 5.27+/GNOME 48+) is the planned
//! priority-0 spike target; it is not yet implemented pending PTT hold/release
//! semantic verification on real hardware.
//!
//! Evdev grab is per-DEVICE, not per-key — a bare exclusive grab silences the
//! whole keyboard. Fix: grab the real device, create a uinput virtual keyboard,
//! and re-emit every event except the hotkey.

use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

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
    if std::env::var_os("DISPLAY").is_some() {
        match spawn_x11(config, tx.clone()) {
            Ok(()) => return Ok(()),
            Err(e) => warn!("X11 hotkey unavailable ({e:#}), falling back to evdev"),
        }
    }
    spawn_evdev(config, tx)
}

// ── X11 XGrabKey backend ─────────────────────────────────────────────────────

fn spawn_x11(config: &Config, tx: Sender<HotkeyEvent>) -> Result<()> {
    use x11rb::connection::Connection as _;
    use x11rb::protocol::xproto::{ConnectionExt as _, GrabMode, ModMask};
    use x11rb::rust_connection::RustConnection;

    let keysym = resolve_keysym(&config.hotkey)?;
    let clipboard_hotkey = config.clipboard_hotkey;

    let (conn, screen_num) =
        RustConnection::connect(None).map_err(|e| anyhow!("X11 connect: {e}"))?;

    // Extract what we need from setup/screen before any borrows prevent moving conn.
    let root = conn.setup().roots[screen_num].root;
    let keycode = keysym_to_keycode(&conn, keysym)?;

    // Passive grab: activates when our key is pressed regardless of current modifiers.
    // owner_events=false keeps other clients' key events entirely unaffected.
    // CapsLock LED still toggles (modifier processing is X server-side); F-keys or
    // RightCtrl are recommended as hotkeys if the LED behaviour is undesirable.
    conn.grab_key(false, root, ModMask::ANY, keycode, GrabMode::ASYNC, GrabMode::ASYNC)
        .map_err(|e| anyhow!("grab_key request: {e}"))?
        .check()
        .map_err(|e| anyhow!("grab_key rejected by X server (key already grabbed?): {e}"))?;

    info!(
        "listening for hotkey '{}' (keycode {keycode}) via X11 XGrabKey",
        config.hotkey
    );

    thread::Builder::new()
        .name("hotkey:x11".into())
        .spawn(move || run_x11(conn, keycode, clipboard_hotkey, tx))?;

    Ok(())
}

fn run_x11(
    conn: x11rb::rust_connection::RustConnection,
    keycode: u8,
    clipboard_hotkey: bool,
    tx: Sender<HotkeyEvent>,
) {
    use x11rb::connection::Connection as _;
    use x11rb::protocol::xproto::KeyButMask;
    use x11rb::protocol::Event;

    let mut key_down = false;

    loop {
        let event = match conn.wait_for_event() {
            Ok(e) => e,
            Err(e) => {
                error!("X11 event error: {e}");
                break;
            }
        };

        match event {
            Event::KeyPress(ev) if ev.detail == keycode => {
                if key_down {
                    // X11 auto-repeat sends synthetic press+release pairs; the
                    // key_down flag catches the press half here.
                    continue;
                }
                key_down = true;
                let shift = ev.state.contains(KeyButMask::SHIFT);
                if tx
                    .send(HotkeyEvent::Press {
                        clipboard_only: clipboard_hotkey && shift,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Event::KeyRelease(ev) if ev.detail == keycode => {
                if !key_down {
                    continue;
                }
                // X11 auto-repeat: a synthetic KeyRelease is immediately followed
                // by a KeyPress for the same key. Sleep 1 ms then peek; if a
                // matching KeyPress is already queued, discard both (auto-repeat).
                thread::sleep(Duration::from_millis(1));
                if let Ok(Some(Event::KeyPress(kp))) = conn.poll_for_event() {
                    if kp.detail == keycode {
                        continue; // auto-repeat pair; key_down stays true
                    }
                    // A different key's press arrived — it's a real release; the
                    // unrelated KeyPress is dropped (we only care about PTT events).
                }
                key_down = false;
                if tx.send(HotkeyEvent::Release).is_err() {
                    break;
                }
            }
            _ => {}
        }
    }
}

fn keysym_to_keycode(conn: &x11rb::rust_connection::RustConnection, keysym: u32) -> Result<u8> {
    use x11rb::connection::Connection as _;
    use x11rb::protocol::xproto::ConnectionExt as _;

    let setup = conn.setup();
    let first = setup.min_keycode;
    let count = setup.max_keycode.saturating_sub(first).saturating_add(1);
    let mapping = conn.get_keyboard_mapping(first, count)?.reply()?;

    let per = mapping.keysyms_per_keycode as usize;
    for (i, chunk) in mapping.keysyms.chunks(per).enumerate() {
        if chunk.iter().any(|&s| s == keysym) {
            return Ok(first.saturating_add(i as u8));
        }
    }
    anyhow::bail!("keysym {keysym:#x} not found in keyboard mapping")
}

/// Map a config key name to an X11 keysym constant (XK_* values from X11/keysymdef.h).
fn resolve_keysym(name: &str) -> Result<u32> {
    let mut n = name.to_uppercase();
    n.retain(|c| c != ' ' && c != '_');
    if let Some(rest) = n.strip_prefix("KEY") {
        n = rest.to_string();
    }
    let sym: u32 = match n.as_str() {
        "CAPSLOCK" | "CAPS" => 0xFFE5,
        "SCROLLLOCK" | "SCROLL" => 0xFF14,
        "NUMLOCK" | "NUM" => 0xFF7F,
        "RIGHTCTRL" | "RCTRL" => 0xFFE4,
        "LEFTCTRL" | "LCTRL" => 0xFFE3,
        "RIGHTALT" | "RALT" => 0xFFEA,
        "LEFTALT" | "LALT" => 0xFFE9,
        "RIGHTSHIFT" | "RSHIFT" => 0xFFE2,
        "LEFTSHIFT" | "LSHIFT" => 0xFFE1,
        "RIGHTMETA" | "RMETA" | "RIGHTSUPER" => 0xFFEC,
        "LEFTMETA" | "LMETA" | "LEFTSUPER" | "SUPER" => 0xFFEB,
        other => resolve_fkey_sym(other).ok_or_else(|| {
            anyhow!("unrecognized hotkey '{name}' (try CapsLock, F12, RightCtrl)")
        })?,
    };
    Ok(sym)
}

fn resolve_fkey_sym(n: &str) -> Option<u32> {
    let num: u8 = n.strip_prefix('F')?.parse().ok()?;
    // XK_F1 = 0xFFBE; uniform stride: XK_Fn = 0xFFBD + n (verified F1–F24).
    if (1..=24).contains(&num) {
        Some(0xFFBD + num as u32)
    } else {
        None
    }
}

// ── evdev backend ─────────────────────────────────────────────────────────────

fn spawn_evdev(config: &Config, tx: Sender<HotkeyEvent>) -> Result<()> {
    let target = resolve_key(&config.hotkey)?;
    let want_grab = config.grab;
    let clipboard_hotkey = config.clipboard_hotkey;

    let mut count = 0;
    for (path, device) in evdev::enumerate() {
        if device.name().is_some_and(|n| n.starts_with("my-voice")) {
            continue;
        }
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
        "listening for hotkey '{}' on {count} device(s) via evdev",
        config.hotkey
    );
    Ok(())
}

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
                        return;
                    }
                }
                continue;
            }

            if ev.event_type() == EventType::KEY
                && (ev.code() == Key::KEY_LEFTSHIFT.code()
                    || ev.code() == Key::KEY_RIGHTSHIFT.code())
            {
                shift_down = ev.value() != 0;
            }

            if let Some(v) = vdev.as_mut() {
                if let Err(e) = v.emit(&[ev]) {
                    error!("'{name}': passthrough emit failed: {e}");
                }
            }
        }
    }
}

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
    Ok(VirtualDeviceBuilder::new()?
        .name("my-voice passthrough")
        .with_keys(keys)?
        .build()?)
}

/// Map a config key name to an evdev `Key`. Used by the evdev backend.
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
    fn evdev_parses_capslock_variants() {
        assert_eq!(resolve_key("CapsLock").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("capslock").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("CAPS_LOCK").unwrap(), Key::KEY_CAPSLOCK);
        assert_eq!(resolve_key("KEY_CAPSLOCK").unwrap(), Key::KEY_CAPSLOCK);
    }

    #[test]
    fn evdev_parses_fkeys() {
        assert_eq!(resolve_key("F12").unwrap(), Key::KEY_F12);
        assert_eq!(resolve_key("f18").unwrap(), Key::KEY_F18);
    }

    #[test]
    fn evdev_rejects_unknown() {
        assert!(resolve_key("PenguinKey").is_err());
        assert!(resolve_key("F99").is_err());
    }

    #[test]
    fn x11_keysym_capslock() {
        assert_eq!(resolve_keysym("CapsLock").unwrap(), 0xFFE5);
        assert_eq!(resolve_keysym("CAPS_LOCK").unwrap(), 0xFFE5);
    }

    #[test]
    fn x11_keysym_fkeys() {
        assert_eq!(resolve_keysym("F1").unwrap(), 0xFFBE);
        assert_eq!(resolve_keysym("F12").unwrap(), 0xFFC9);
        assert_eq!(resolve_keysym("F13").unwrap(), 0xFFCA);
        assert_eq!(resolve_keysym("F20").unwrap(), 0xFFD1);
    }

    #[test]
    fn x11_keysym_modifiers() {
        assert_eq!(resolve_keysym("RightCtrl").unwrap(), 0xFFE4);
        assert_eq!(resolve_keysym("ScrollLock").unwrap(), 0xFF14);
    }

    #[test]
    fn x11_keysym_rejects_unknown() {
        assert!(resolve_keysym("PenguinKey").is_err());
        assert!(resolve_keysym("F99").is_err());
    }
}
