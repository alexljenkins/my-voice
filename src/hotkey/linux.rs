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
    // Under Wayland, DISPLAY is still set (XWayland) and XGrabKey *succeeds*, but
    // an X11 passive grab only receives keys routed to XWayland clients — when a
    // native Wayland surface has focus the hotkey never reaches us. evdev grabs at
    // the device layer, below the compositor, so it works regardless of focus.
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var_os("XDG_SESSION_TYPE").is_some_and(|v| v == "wayland");
    if !wayland && std::env::var_os("DISPLAY").is_some() {
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

    let (mods, main) = super::parse_hotkey(&config.hotkey);
    let keysym = resolve_keysym(main)?;
    let clipboard_hotkey = config.clipboard_hotkey;

    let (conn, screen_num) =
        RustConnection::connect(None).map_err(|e| anyhow!("X11 connect: {e}"))?;

    // Extract what we need from setup/screen before any borrows prevent moving conn.
    let root = conn.setup().roots[screen_num].root;
    let keycode = keysym_to_keycode(&conn, keysym)?;

    // owner_events=false keeps other clients' key events entirely unaffected.
    // No modifiers → grab the key for ANY modifier state (dedicated keys like
    // CapsLock/F12; the LED still toggles, that's X server-side). With modifiers
    // → grab only the specific combo (so e.g. plain `.` still types), once per
    // lock-state variant so CapsLock/NumLock don't defeat the grab.
    if mods == super::Mods::default() {
        conn.grab_key(
            false,
            root,
            ModMask::ANY,
            keycode,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .map_err(|e| anyhow!("grab_key request: {e}"))?
        .check()
        .map_err(|e| anyhow!("grab_key rejected by X server (key already grabbed?): {e}"))?;
    } else {
        let mut base = ModMask::from(0u16);
        if mods.ctrl {
            base |= ModMask::CONTROL;
        }
        if mods.shift {
            base |= ModMask::SHIFT;
        }
        if mods.alt {
            base |= ModMask::M1;
        }
        if mods.sup {
            base |= ModMask::M4;
        }
        let lock_variants = [
            ModMask::from(0u16),
            ModMask::LOCK,
            ModMask::M2,
            ModMask::LOCK | ModMask::M2,
        ];
        for v in lock_variants {
            conn.grab_key(
                false,
                root,
                base | v,
                keycode,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
            )
            .map_err(|e| anyhow!("grab_key request: {e}"))?
            .check()
            .map_err(|e| anyhow!("grab_key rejected (combo already grabbed?): {e}"))?;
        }
    }

    info!(
        "listening for hotkey '{}' (keycode {keycode}) via X11 XGrabKey",
        config.hotkey
    );

    thread::Builder::new()
        .name("hotkey:x11".into())
        .spawn(move || run_x11(conn, keycode, mods, clipboard_hotkey, tx))?;

    Ok(())
}

fn run_x11(
    conn: x11rb::rust_connection::RustConnection,
    keycode: u8,
    required: super::Mods,
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
                let held = super::Mods {
                    ctrl: ev.state.contains(KeyButMask::CONTROL),
                    shift: ev.state.contains(KeyButMask::SHIFT),
                    alt: ev.state.contains(KeyButMask::MOD1),
                    sup: ev.state.contains(KeyButMask::MOD4),
                };
                if !required.satisfied_by(&held) {
                    continue; // required modifiers not held — not our combo
                }
                key_down = true;
                let clipboard_only = clipboard_hotkey && held.shift && !required.shift;
                if tx.send(HotkeyEvent::Press { clipboard_only }).is_err() {
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
        if chunk.contains(&keysym) {
            return Ok(first.saturating_add(i as u8));
        }
    }
    anyhow::bail!("keysym {keysym:#x} not found in keyboard mapping")
}

/// Map a config key name (main key, modifiers stripped) to an X11 keysym
/// constant (XK_* values from X11/keysymdef.h).
fn resolve_keysym(name: &str) -> Result<u32> {
    let mut n = name.to_uppercase();
    n.retain(|c| c != ' ' && c != '_');
    if n.len() > 3 {
        if let Some(rest) = n.strip_prefix("KEY") {
            n = rest.to_string();
        }
    }
    if let Some(digit) = n.strip_prefix("DIGIT") {
        n = digit.to_string();
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
        // Named keys.
        "SPACE" => 0x20,
        "TAB" => 0xFF09,
        "ENTER" | "RETURN" => 0xFF0D,
        "BACKSPACE" => 0xFF08,
        "INSERT" => 0xFF63,
        "DELETE" | "DEL" => 0xFFFF,
        "HOME" => 0xFF50,
        "END" => 0xFF57,
        "PAGEUP" => 0xFF55,
        "PAGEDOWN" => 0xFF56,
        "UP" => 0xFF52,
        "DOWN" => 0xFF54,
        "LEFT" => 0xFF51,
        "RIGHT" => 0xFF53,
        // Punctuation (Latin-1 keysyms == ASCII codepoints).
        "PERIOD" | "DOT" => 0x2E,
        "COMMA" => 0x2C,
        "SLASH" => 0x2F,
        "BACKSLASH" => 0x5C,
        "SEMICOLON" => 0x3B,
        "APOSTROPHE" | "QUOTE" => 0x27,
        "LEFTBRACKET" | "LEFTBRACE" => 0x5B,
        "RIGHTBRACKET" | "RIGHTBRACE" => 0x5D,
        "MINUS" => 0x2D,
        "EQUAL" => 0x3D,
        "GRAVE" | "BACKQUOTE" => 0x60,
        other => resolve_alnum_sym(other)
            .or_else(|| resolve_fkey_sym(other))
            .ok_or_else(|| {
                anyhow!("unrecognized hotkey '{name}' (try CapsLock, F12, Ctrl+Period)")
            })?,
    };
    Ok(sym)
}

/// Single letter A–Z (lowercase keysym, the base level) or digit 0–9.
fn resolve_alnum_sym(n: &str) -> Option<u32> {
    let mut chars = n.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    if c.is_ascii_uppercase() {
        Some(c.to_ascii_lowercase() as u32) // XK_a..=XK_z
    } else if c.is_ascii_digit() {
        Some(c as u32) // XK_0..=XK_9
    } else {
        None
    }
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
    let (mods, main) = super::parse_hotkey(&config.hotkey);
    let target = resolve_key(main)?;
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
                run_device(name, device, target, mods, want_grab, clipboard_hotkey, tx);
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
    required: super::Mods,
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
    let mut held = super::Mods::default();
    // True while a hotkey press is active, so we swallow its matching release/
    // autorepeat under grab. When the required modifiers aren't held, the target
    // key passes through untouched (so e.g. `Ctrl+Period` doesn't eat plain `.`).
    let mut active = false;

    loop {
        let events = match guard.device.fetch_events() {
            Ok(e) => e,
            Err(e) => {
                error!("'{name}': reading events failed: {e}");
                break;
            }
        };

        for ev in events {
            let is_key = ev.event_type() == EventType::KEY;
            if is_key {
                update_mods(&mut held, ev.code(), ev.value() != 0);
            }

            if is_key && ev.code() == target_code {
                match ev.value() {
                    1 if required.satisfied_by(&held) => {
                        active = true;
                        let clipboard_only = clipboard_hotkey && held.shift && !required.shift;
                        if tx.send(HotkeyEvent::Press { clipboard_only }).is_err() {
                            return;
                        }
                        continue; // swallow under grab
                    }
                    2 if active => continue, // swallow autorepeat while active
                    0 if active => {
                        active = false;
                        if tx.send(HotkeyEvent::Release).is_err() {
                            return;
                        }
                        continue; // swallow
                    }
                    // Otherwise (modifiers not held, or stale): fall through to
                    // passthrough so the key behaves normally.
                    _ => {}
                }
            }

            if let Some(v) = vdev.as_mut() {
                if let Err(e) = v.emit(&[ev]) {
                    error!("'{name}': passthrough emit failed: {e}");
                }
            }
        }
    }
}

/// Update held-modifier state from a key event code/down-state.
fn update_mods(held: &mut super::Mods, code: u16, down: bool) {
    if code == Key::KEY_LEFTCTRL.code() || code == Key::KEY_RIGHTCTRL.code() {
        held.ctrl = down;
    } else if code == Key::KEY_LEFTSHIFT.code() || code == Key::KEY_RIGHTSHIFT.code() {
        held.shift = down;
    } else if code == Key::KEY_LEFTALT.code() || code == Key::KEY_RIGHTALT.code() {
        held.alt = down;
    } else if code == Key::KEY_LEFTMETA.code() || code == Key::KEY_RIGHTMETA.code() {
        held.sup = down;
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

/// Map a config key name (the main key of a hotkey, modifiers already stripped)
/// to an evdev `Key`. Used by the evdev backend.
pub fn resolve_key(name: &str) -> Result<Key> {
    let mut n: String = name.to_uppercase();
    n.retain(|c| c != ' ' && c != '_');
    if n.len() > 3 {
        if let Some(stripped) = n.strip_prefix("KEY") {
            n = stripped.to_string();
        }
    }
    if let Some(digit) = n.strip_prefix("DIGIT") {
        n = digit.to_string();
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
        // Named keys (match the keybind-capture popup's tokens).
        "SPACE" => Key::KEY_SPACE,
        "TAB" => Key::KEY_TAB,
        "ENTER" | "RETURN" => Key::KEY_ENTER,
        "BACKSPACE" => Key::KEY_BACKSPACE,
        "INSERT" => Key::KEY_INSERT,
        "DELETE" | "DEL" => Key::KEY_DELETE,
        "HOME" => Key::KEY_HOME,
        "END" => Key::KEY_END,
        "PAGEUP" => Key::KEY_PAGEUP,
        "PAGEDOWN" => Key::KEY_PAGEDOWN,
        "UP" => Key::KEY_UP,
        "DOWN" => Key::KEY_DOWN,
        "LEFT" => Key::KEY_LEFT,
        "RIGHT" => Key::KEY_RIGHT,
        // Punctuation.
        "PERIOD" | "DOT" => Key::KEY_DOT,
        "COMMA" => Key::KEY_COMMA,
        "SLASH" => Key::KEY_SLASH,
        "BACKSLASH" => Key::KEY_BACKSLASH,
        "SEMICOLON" => Key::KEY_SEMICOLON,
        "APOSTROPHE" | "QUOTE" => Key::KEY_APOSTROPHE,
        "LEFTBRACKET" | "LEFTBRACE" => Key::KEY_LEFTBRACE,
        "RIGHTBRACKET" | "RIGHTBRACE" => Key::KEY_RIGHTBRACE,
        "MINUS" => Key::KEY_MINUS,
        "EQUAL" => Key::KEY_EQUAL,
        "GRAVE" | "BACKQUOTE" => Key::KEY_GRAVE,
        other => resolve_alnum(other)
            .or_else(|| resolve_fkey(other))
            .ok_or_else(|| {
                anyhow!("unrecognized hotkey name '{name}' (try CapsLock, F12, Ctrl+Period)")
            })?,
    };
    Ok(key)
}

/// Map a single letter A–Z or digit 0–9 to its evdev `Key`.
fn resolve_alnum(n: &str) -> Option<Key> {
    let mut chars = n.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    Some(match c {
        'A' => Key::KEY_A,
        'B' => Key::KEY_B,
        'C' => Key::KEY_C,
        'D' => Key::KEY_D,
        'E' => Key::KEY_E,
        'F' => Key::KEY_F,
        'G' => Key::KEY_G,
        'H' => Key::KEY_H,
        'I' => Key::KEY_I,
        'J' => Key::KEY_J,
        'K' => Key::KEY_K,
        'L' => Key::KEY_L,
        'M' => Key::KEY_M,
        'N' => Key::KEY_N,
        'O' => Key::KEY_O,
        'P' => Key::KEY_P,
        'Q' => Key::KEY_Q,
        'R' => Key::KEY_R,
        'S' => Key::KEY_S,
        'T' => Key::KEY_T,
        'U' => Key::KEY_U,
        'V' => Key::KEY_V,
        'W' => Key::KEY_W,
        'X' => Key::KEY_X,
        'Y' => Key::KEY_Y,
        'Z' => Key::KEY_Z,
        '0' => Key::KEY_0,
        '1' => Key::KEY_1,
        '2' => Key::KEY_2,
        '3' => Key::KEY_3,
        '4' => Key::KEY_4,
        '5' => Key::KEY_5,
        '6' => Key::KEY_6,
        '7' => Key::KEY_7,
        '8' => Key::KEY_8,
        '9' => Key::KEY_9,
        _ => return None,
    })
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

    #[test]
    fn evdev_parses_popup_tokens() {
        assert_eq!(resolve_key("Period").unwrap(), Key::KEY_DOT);
        assert_eq!(resolve_key("K").unwrap(), Key::KEY_K);
        assert_eq!(resolve_key("Digit5").unwrap(), Key::KEY_5);
        assert_eq!(resolve_key("LeftBracket").unwrap(), Key::KEY_LEFTBRACE);
        assert_eq!(resolve_key("PageUp").unwrap(), Key::KEY_PAGEUP);
    }

    #[test]
    fn x11_parses_popup_tokens() {
        assert_eq!(resolve_keysym("Period").unwrap(), 0x2E);
        assert_eq!(resolve_keysym("K").unwrap(), 0x6B); // XK_k (lowercase)
        assert_eq!(resolve_keysym("Digit5").unwrap(), 0x35); // XK_5
        assert_eq!(resolve_keysym("Slash").unwrap(), 0x2F);
    }

    #[test]
    fn hotkey_combo_parses() {
        use super::super::{parse_hotkey, Mods};
        let (mods, main) = parse_hotkey("Ctrl+Period");
        assert_eq!(main, "Period");
        assert_eq!(
            mods,
            Mods {
                ctrl: true,
                ..Default::default()
            }
        );

        let (mods, main) = parse_hotkey("Ctrl+Shift+K");
        assert_eq!(main, "K");
        assert!(mods.ctrl && mods.shift && !mods.alt);

        let (mods, main) = parse_hotkey("CapsLock");
        assert_eq!(main, "CapsLock");
        assert_eq!(mods, Mods::default());
    }

    #[test]
    fn mods_satisfied_subset() {
        use super::super::Mods;
        let required = Mods {
            ctrl: true,
            ..Default::default()
        };
        let held = Mods {
            ctrl: true,
            shift: true,
            ..Default::default()
        };
        assert!(required.satisfied_by(&held)); // extra Shift is allowed
        assert!(!required.satisfied_by(&Mods::default())); // Ctrl missing
    }
}
