//! Linux text injection: session-aware chain with runtime demotion.
//!
//! Wayland: wtype → ydotool → clipboard (arboard)
//! X11:     xdotool → AT-SPI → clipboard (arboard)
//! Neither: clipboard (arboard; may fail without a display server)
//!
//! AT-SPI keyboard synthesis is X11-only. On Wayland, Mutter discards the events
//! but the D-Bus call still returns Ok, so it can't be trusted (and would defeat
//! the clipboard fallback) — it's excluded from the Wayland chain. On GNOME
//! Wayland `wtype` is also blocked by Mutter; ydotool (uinput) is the only
//! reliable typing path, otherwise injection demotes to clipboard.
//!
//! All external tools are spawned via Command argv — never through a shell,
//! since transcribed text can contain quotes, `$`, backticks, anything.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::{debug, info, warn};

use super::Injector;
use crate::config::Config;

/// `KeySynthType::KEY_SYM` — generate a key event for a given X keysym.
const KEY_SYM: u32 = 3;

/// Map a Unicode scalar to an X keysym. Latin-1 codepoints map directly; the
/// rest use the Unicode-to-keysym convention (`0x01000000 + codepoint`).
fn char_to_keysym(c: char) -> i32 {
    let cp = c as u32;
    let ks = if cp <= 0xff { cp } else { 0x0100_0000 + cp };
    ks as i32
}

/// AT-SPI2 D-Bus plumbing. Two proxies: `org.a11y.Bus` on the session bus to
/// discover the accessibility bus address, then `DeviceEventController` on that
/// bus to synthesize keystrokes (the same mechanism Orca uses).
mod atspi {
    #[zbus::proxy(
        interface = "org.a11y.Bus",
        default_service = "org.a11y.Bus",
        default_path = "/org/a11y/bus"
    )]
    pub trait A11yBus {
        fn get_address(&self) -> zbus::Result<String>;
    }

    #[zbus::proxy(
        interface = "org.a11y.atspi.DeviceEventController",
        default_service = "org.a11y.atspi.Registry",
        default_path = "/org/a11y/atspi/registry/deviceeventcontroller"
    )]
    pub trait DeviceEventController {
        fn generate_keyboard_event(
            &self,
            keycode: i32,
            keystring: &str,
            synth_type: u32,
        ) -> zbus::Result<()>;
    }

    /// Connect to the session bus, resolve the a11y bus address, and return a
    /// blocking proxy to its DeviceEventController.
    pub fn connect() -> zbus::Result<DeviceEventControllerProxyBlocking<'static>> {
        let session = zbus::blocking::Connection::session()?;
        let addr = A11yBusProxyBlocking::new(&session)?.get_address()?;
        let a11y = zbus::blocking::connection::Builder::address(addr.as_str())?.build()?;
        // Disable property caching: the interface's `version` property doesn't
        // serve GetAll reliably, and we only ever call a method.
        DeviceEventControllerProxyBlocking::builder(&a11y)
            .cache_properties(zbus::proxy::CacheProperties::No)
            .build()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Session {
    Wayland,
    X11,
    None,
}

fn detect_session() -> Session {
    if std::env::var("WAYLAND_DISPLAY").is_ok_and(|v| !v.is_empty()) {
        Session::Wayland
    } else if std::env::var("DISPLAY").is_ok_and(|v| !v.is_empty()) {
        Session::X11
    } else {
        Session::None
    }
}

fn binary_on_path(name: &str) -> bool {
    std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .any(|dir| std::path::Path::new(dir).join(name).is_file())
}

// [voxtype] socket discovery order for ydotool
fn find_ydotool_socket() -> Option<PathBuf> {
    if let Ok(s) = std::env::var("YDOTOOL_SOCKET") {
        let p = PathBuf::from(&s);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(dir).join(".ydotool_socket");
        if p.exists() {
            return Some(p);
        }
    }
    let p = PathBuf::from("/tmp/.ydotool_socket");
    if p.exists() {
        return Some(p);
    }
    let uid = unsafe { libc::getuid() };
    let p = PathBuf::from(format!("/run/user/{uid}/.ydotool_socket"));
    if p.exists() {
        return Some(p);
    }
    None
}

fn run_argv(cmd: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .with_context(|| format!("{cmd} not found on PATH"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("{cmd} exited with {status}")
    }
}

// --- individual tool injectors ---

struct WtypeInjector;
impl Injector for WtypeInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        run_argv("wtype", &["--", text])
    }
    fn name(&self) -> &'static str {
        "wtype"
    }
}

struct YdotoolInjector {
    socket: PathBuf,
}
impl Injector for YdotoolInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        let socket_str = self.socket.to_string_lossy();
        let status = Command::new("ydotool")
            .args(["type", "--", text])
            .env("YDOTOOL_SOCKET", socket_str.as_ref())
            .status()?;
        if status.success() {
            Ok(())
        } else {
            bail!("ydotool exited with {status}")
        }
    }
    fn name(&self) -> &'static str {
        "ydotool"
    }
}

struct XdotoolInjector;
impl Injector for XdotoolInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        run_argv("xdotool", &["type", "--clearmodifiers", "--", text])
    }
    fn name(&self) -> &'static str {
        "xdotool"
    }
}

/// Injects via the AT-SPI accessibility bus, one keysym event per character.
/// Zero-setup and lower-privilege than the external typing tools.
struct AtSpiInjector {
    dec: atspi::DeviceEventControllerProxyBlocking<'static>,
}
impl Injector for AtSpiInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        for c in text.chars() {
            self.dec
                .generate_keyboard_event(char_to_keysym(c), "", KEY_SYM)
                .map_err(|e| anyhow::anyhow!("atspi generate_keyboard_event: {e}"))?;
        }
        Ok(())
    }
    fn name(&self) -> &'static str {
        "atspi"
    }
}

/// Probe AT-SPI at startup. Returns None (logged at debug) when the a11y bus is
/// unavailable or disabled, so it's simply skipped in the chain.
fn try_atspi() -> Option<Box<dyn Injector>> {
    match atspi::connect() {
        Ok(dec) => Some(Box::new(AtSpiInjector { dec })),
        Err(e) => {
            debug!("atspi unavailable ({e}); skipping in injection chain");
            None
        }
    }
}

struct ArboardInjector;
impl Injector for ArboardInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        arboard::Clipboard::new()
            .and_then(|mut cb| cb.set_text(text))
            .map_err(|e| anyhow::anyhow!("clipboard: {e}"))
    }
    fn name(&self) -> &'static str {
        "clipboard"
    }
}

// --- chain injector ---

/// Tries typing methods in order; on runtime failure logs a warning, demotes the
/// method for the session, and retries with the next. Clipboard is the final
/// fallback and is always attempted.
struct ChainInjector {
    chain: Vec<Box<dyn Injector>>,
    cursor: usize,
    clipboard: Box<dyn Injector>,
}

impl Injector for ChainInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        while self.cursor < self.chain.len() {
            match self.chain[self.cursor].inject(text) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let name = self.chain[self.cursor].name();
                    warn!("{name} failed ({e:#}); falling back to next injection method");
                    self.cursor += 1;
                }
            }
        }
        let result = self.clipboard.inject(text);
        if result.is_ok() {
            info!("copied to clipboard (no typing tool available — paste with Ctrl+V)");
        }
        result
    }
    fn name(&self) -> &'static str {
        if self.cursor < self.chain.len() {
            self.chain[self.cursor].name()
        } else {
            "clipboard-fallback"
        }
    }
}

// --- public API ---

pub fn detect(config: &Config) -> Box<dyn Injector> {
    let session = detect_session();

    if config.injection == "clipboard" {
        return clipboard_injector();
    }

    let chain = if config.injection == "auto" {
        build_auto_chain(session)
    } else {
        build_specific_chain(&config.injection, session)
    };

    let session_name = match session {
        Session::Wayland => "wayland",
        Session::X11 => "x11",
        Session::None => "no session",
    };

    if chain.is_empty() {
        warn!(
            "injection: no typing tool on {session_name} — \
             falling back to clipboard; paste with Ctrl+V"
        );
    } else {
        info!("injection: {} ({session_name})", chain[0].name());
    }

    Box::new(ChainInjector {
        chain,
        cursor: 0,
        clipboard: build_clipboard(),
    })
}

pub fn clipboard_injector() -> Box<dyn Injector> {
    build_clipboard()
}

fn build_auto_chain(session: Session) -> Vec<Box<dyn Injector>> {
    let mut chain: Vec<Box<dyn Injector>> = Vec::new();
    match session {
        Session::Wayland => {
            // AT-SPI is deliberately NOT in the Wayland chain: Mutter silently
            // drops synthesized key events, yet generate_keyboard_event still
            // returns Ok — an unverifiable false success that would swallow every
            // utterance and defeat the clipboard fallback. It stays X11-only.
            if binary_on_path("wtype") {
                chain.push(Box::new(WtypeInjector));
            }
            if binary_on_path("ydotool") {
                if let Some(socket) = find_ydotool_socket() {
                    chain.push(Box::new(YdotoolInjector { socket }));
                }
            }
        }
        Session::X11 => {
            if binary_on_path("xdotool") {
                chain.push(Box::new(XdotoolInjector));
            }
            if let Some(inj) = try_atspi() {
                chain.push(inj);
            }
        }
        Session::None => {}
    }
    chain
}

fn build_specific_chain(injection: &str, session: Session) -> Vec<Box<dyn Injector>> {
    match injection {
        "wtype" => {
            if binary_on_path("wtype") {
                vec![Box::new(WtypeInjector)]
            } else {
                warn!("injection=wtype but wtype not found; falling back to auto");
                build_auto_chain(session)
            }
        }
        "xdotool" => {
            if binary_on_path("xdotool") {
                vec![Box::new(XdotoolInjector)]
            } else {
                warn!("injection=xdotool but xdotool not found; falling back to auto");
                build_auto_chain(session)
            }
        }
        "atspi" => {
            if let Some(inj) = try_atspi() {
                vec![inj]
            } else {
                warn!("injection=atspi but AT-SPI unavailable; falling back to auto");
                build_auto_chain(session)
            }
        }
        "ydotool" => {
            if binary_on_path("ydotool") {
                if let Some(socket) = find_ydotool_socket() {
                    vec![Box::new(YdotoolInjector { socket })]
                } else {
                    warn!("injection=ydotool but ydotool socket not found; falling back to auto");
                    build_auto_chain(session)
                }
            } else {
                warn!("injection=ydotool but ydotool not found; falling back to auto");
                build_auto_chain(session)
            }
        }
        name => {
            warn!("unknown injection={name}; falling back to auto");
            build_auto_chain(session)
        }
    }
}

fn build_clipboard() -> Box<dyn Injector> {
    Box::new(ArboardInjector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_probe_absent() {
        assert!(!binary_on_path("__my_voice_nonexistent_binary_xyz__"));
    }

    #[test]
    fn binary_probe_present() {
        // `ls` is on PATH everywhere.
        assert!(binary_on_path("ls"));
    }

    #[test]
    fn detect_session_no_panic() {
        // Just verify it returns without panicking regardless of env state.
        let _ = detect_session();
    }

    #[test]
    fn chain_empty_when_no_session_and_no_tools() {
        // Session::None produces an empty chain (no X11/Wayland tools to probe).
        let chain = build_auto_chain(Session::None);
        assert!(chain.is_empty());
    }

    #[test]
    fn clipboard_injector_name() {
        let inj = build_clipboard();
        assert_eq!(inj.name(), "clipboard");
    }
}
