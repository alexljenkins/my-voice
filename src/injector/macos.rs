//! macOS text injection: CGEvent unicode injection + pbcopy clipboard fallback.
//!
//! NOTE: only compiled on macOS; not build-verified on the Linux dev host.
//!
//! CGEventKeyboardSetUnicodeString is not wrapped by the core-graphics crate,
//! so we declare the required symbols directly. ApplicationServices is already
//! linked transitively via the core-graphics dependency.

use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Result};
use tracing::{info, warn};

use super::Injector;

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn CGEventSourceCreate(state_id: i32) -> *mut std::ffi::c_void;
    fn CGEventCreateKeyboardEvent(
        source: *mut std::ffi::c_void,
        virtual_key: u16,
        key_down: bool,
    ) -> *mut std::ffi::c_void;
    fn CGEventKeyboardSetUnicodeString(
        event: *mut std::ffi::c_void,
        string_length: usize,
        unicode_string: *const u16,
    );
    fn CGEventPost(tap: u32, event: *mut std::ffi::c_void);
    fn CFRelease(cf: *mut std::ffi::c_void);
}

// kCGEventSourceStateCombinedSessionState = 1
const COMBINED_SESSION_STATE: i32 = 1;
// kCGHIDEventTap = 0
const HID_EVENT_TAP: u32 = 0;
// CGEvent's per-call limit for unicode code units
const MAX_CHUNK_UTF16: usize = 20;

pub struct MacOsTypingInjector;

impl Injector for MacOsTypingInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        inject_unicode(text).or_else(|e| {
            warn!("CGEvent injection failed ({e:#}); falling back to pbcopy");
            pbcopy(text)?;
            info!("copied to clipboard — paste with Cmd+V");
            Ok(())
        })
    }
    fn name(&self) -> &'static str {
        "cgevent"
    }
}

pub struct PbcopyInjector;

impl Injector for PbcopyInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        pbcopy(text)
    }
    fn name(&self) -> &'static str {
        "pbcopy"
    }
}

pub fn typing_injector() -> Box<dyn Injector> {
    info!("injection: cgevent (macOS)");
    Box::new(MacOsTypingInjector)
}

pub fn clipboard_injector() -> Box<dyn Injector> {
    Box::new(PbcopyInjector)
}

fn pbcopy(text: &str) -> Result<()> {
    let mut child = Command::new("pbcopy").stdin(Stdio::piped()).spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        bail!("pbcopy exited with {status}")
    }
}

fn inject_unicode(text: &str) -> Result<()> {
    let utf16: Vec<u16> = text.encode_utf16().collect();
    if utf16.is_empty() {
        return Ok(());
    }

    unsafe {
        let source = CGEventSourceCreate(COMBINED_SESSION_STATE);
        if source.is_null() {
            bail!("CGEventSourceCreate returned null (Accessibility permission missing?)");
        }

        for chunk in utf16.chunks(MAX_CHUNK_UTF16) {
            let key_down = CGEventCreateKeyboardEvent(source, 0, true);
            if key_down.is_null() {
                CFRelease(source);
                bail!("CGEventCreateKeyboardEvent returned null");
            }
            CGEventKeyboardSetUnicodeString(key_down, chunk.len(), chunk.as_ptr());
            CGEventPost(HID_EVENT_TAP, key_down);
            CFRelease(key_down);

            let key_up = CGEventCreateKeyboardEvent(source, 0, false);
            if key_up.is_null() {
                CFRelease(source);
                bail!("CGEventCreateKeyboardEvent returned null");
            }
            CGEventKeyboardSetUnicodeString(key_up, chunk.len(), chunk.as_ptr());
            CGEventPost(HID_EVENT_TAP, key_up);
            CFRelease(key_up);

            // Slow targets drop events without this gap.
            thread::sleep(Duration::from_millis(1));
        }

        CFRelease(source);
    }

    Ok(())
}
