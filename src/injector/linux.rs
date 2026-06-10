//! Linux text injection: session-aware chain with runtime demotion.
//!
//! Wayland: wtype → ydotool → clipboard (wl-copy)
//! X11:     xdotool → clipboard (xclip)
//! Neither: clipboard (fails if no display)
//!
//! All external tools are spawned via Command argv/stdin — never through a shell,
//! since transcribed text can contain quotes, `$`, backticks, anything.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{bail, Result};
use tracing::{info, warn};

use super::Injector;
use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Session {
    Wayland,
    X11,
    None,
}

fn detect_session() -> Session {
    if std::env::var("WAYLAND_DISPLAY").map_or(false, |v| !v.is_empty()) {
        Session::Wayland
    } else if std::env::var("DISPLAY").map_or(false, |v| !v.is_empty()) {
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
    let status = Command::new(cmd).args(args).status()?;
    if status.success() {
        Ok(())
    } else {
        bail!("{cmd} exited with {status}")
    }
}

fn run_stdin(cmd: &str, args: &[&str], text: &str) -> Result<()> {
    let mut child = Command::new(cmd).args(args).stdin(Stdio::piped()).spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
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

struct WlCopyInjector;
impl Injector for WlCopyInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        run_stdin("wl-copy", &[], text)
    }
    fn name(&self) -> &'static str {
        "wl-copy"
    }
}

struct XclipInjector;
impl Injector for XclipInjector {
    fn inject(&mut self, text: &str) -> Result<()> {
        run_stdin("xclip", &["-selection", "clipboard"], text)
    }
    fn name(&self) -> &'static str {
        "xclip"
    }
}

struct NoSessionInjector;
impl Injector for NoSessionInjector {
    fn inject(&mut self, _text: &str) -> Result<()> {
        bail!(
            "no display server detected (WAYLAND_DISPLAY and DISPLAY are both unset); \
             cannot inject text"
        )
    }
    fn name(&self) -> &'static str {
        "none"
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
        if session == Session::None {
            info!("injection: clipboard only (no display server detected)");
        } else {
            info!("injection: clipboard only (no typing tool found on {session_name})");
        }
    } else {
        info!("injection: {} ({session_name})", chain[0].name());
    }

    Box::new(ChainInjector {
        chain,
        cursor: 0,
        clipboard: build_clipboard(session),
    })
}

pub fn clipboard_injector() -> Box<dyn Injector> {
    build_clipboard(detect_session())
}

fn build_auto_chain(session: Session) -> Vec<Box<dyn Injector>> {
    let mut chain: Vec<Box<dyn Injector>> = Vec::new();
    match session {
        Session::Wayland => {
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

fn build_clipboard(session: Session) -> Box<dyn Injector> {
    match session {
        Session::Wayland => Box::new(WlCopyInjector),
        Session::X11 => Box::new(XclipInjector),
        Session::None => Box::new(NoSessionInjector),
    }
}
