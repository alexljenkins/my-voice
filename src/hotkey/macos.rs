//! macOS hotkey: remap CapsLock → F18 via hidutil, then a CGEvent tap on F18.
//!
//! CapsLock on macOS arrives as `flagsChanged` (not keyDown/keyUp) and is often
//! firmware-debounced, so hold/release is unreliable. We remap it to F18 (a key
//! no physical Mac keyboard has) at the HID level and listen for that instead.
//!
//! NOTE: this module is only compiled on macOS and has not been build-verified
//! on the Linux dev host. Build with `cargo build` on macOS to validate.

use std::process::Command;
use std::sync::mpsc::Sender;
use std::thread;

use anyhow::{anyhow, Result};
use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
use core_graphics::event::{
    CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType, EventField,
};
use core_graphics::event_source::CGEventFlags;
use tracing::{error, info, warn};

use super::HotkeyEvent;
use crate::config::Config;

/// Virtual keycode for F18 (the remap destination).
const KEYCODE_F18: i64 = 79;

const REMAP_TO_F18: &str = r#"{"UserKeyMapping":[{"HIDKeyboardModifierMappingSrc":0x700000039,"HIDKeyboardModifierMappingDst":0x70000006D}]}"#;
const REMAP_CLEAR: &str = r#"{"UserKeyMapping":[]}"#;

pub fn spawn(config: &Config, tx: Sender<HotkeyEvent>) -> Result<()> {
    if config.hotkey.to_lowercase().replace([' ', '_'], "") != "capslock" {
        warn!(
            "macOS v1 only supports the CapsLock hotkey (configured: '{}'); using CapsLock",
            config.hotkey
        );
    }

    apply_remap()?;
    let clipboard_hotkey = config.clipboard_hotkey;

    thread::Builder::new()
        .name("hotkey:macos".into())
        .spawn(move || run_tap(clipboard_hotkey, tx))?;

    info!("listening for CapsLock (remapped to F18)");
    Ok(())
}

/// Restore the original CapsLock mapping. Call on clean shutdown.
pub fn restore_mapping() {
    if let Err(e) = run_hidutil(REMAP_CLEAR) {
        warn!("failed to restore CapsLock mapping: {e}");
    }
}

fn apply_remap() -> Result<()> {
    run_hidutil(REMAP_TO_F18)
}

fn run_hidutil(mapping: &str) -> Result<()> {
    let status = Command::new("hidutil")
        .args(["property", "--set", mapping])
        .status()
        .map_err(|e| anyhow!("running hidutil: {e}"))?;
    if !status.success() {
        return Err(anyhow!("hidutil exited with {status}"));
    }
    Ok(())
}

fn run_tap(clipboard_hotkey: bool, tx: Sender<HotkeyEvent>) {
    let current = CFRunLoop::get_current();

    let tap = CGEventTap::new(
        CGEventTapLocation::Session,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::Default,
        vec![
            CGEventType::KeyDown,
            CGEventType::KeyUp,
            CGEventType::FlagsChanged,
            CGEventType::TapDisabledByTimeout,
        ],
        move |_proxy, event_type, event| {
            // Re-enable if the OS disabled us for being slow.
            if event_type == CGEventType::TapDisabledByTimeout {
                return Some(event.to_owned());
            }

            let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
            if keycode != KEYCODE_F18 {
                return Some(event.to_owned()); // pass everything else through
            }

            let autorepeat = event.get_integer_value_field(EventField::KEYBOARD_EVENT_AUTOREPEAT);
            let shift = event.get_flags().contains(CGEventFlags::CGEventFlagShift);

            match event_type {
                CGEventType::KeyDown if autorepeat == 0 => {
                    let clipboard_only = clipboard_hotkey && shift;
                    let _ = tx.send(HotkeyEvent::Press { clipboard_only });
                }
                CGEventType::KeyUp => {
                    let _ = tx.send(HotkeyEvent::Release);
                }
                _ => {}
            }
            None // suppress F18 so it never reaches apps
        },
    );

    let tap = match tap {
        Ok(t) => t,
        Err(()) => {
            error!("could not create CGEvent tap — Accessibility/Input Monitoring permission likely missing");
            crate::notify::once(
                crate::notify::ErrorKind::HotkeySetupNeeded,
                "Permission needed",
                "my-voice needs Accessibility and Input Monitoring permission. Open System Settings → Privacy & Security, find my-voice under each, and turn it on. Then quit and reopen my-voice.",
            );
            return;
        }
    };

    let source = match tap.mach_port().create_runloop_source(0) {
        Ok(s) => s,
        Err(()) => {
            error!("could not create run loop source for event tap");
            return;
        }
    };
    unsafe {
        current.add_source(&source, kCFRunLoopCommonModes);
    }
    tap.enable();
    CFRunLoop::run_current();
}
