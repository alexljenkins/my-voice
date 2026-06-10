//! macOS tray (menu bar) — NOT YET IMPLEMENTED.
//!
//! TODO(§1/§3): render via `tray-icon` + `tao` (or winit). The macOS event loop
//! must own the **main thread**, which inverts the architecture: `run_daemon`'s
//! loop (hotkey rx, recorder, transcriber, injector) moves to a background
//! thread while the tray/event-loop owns main. The `CFRunLoop` driving the
//! CGEvent tap already lives on its own thread and is unaffected.
//!
//! Until then this is a no-op so the daemon compiles and runs on macOS without a
//! tray — verification of this path is deferred to real macOS hardware.

use std::sync::mpsc::Sender;

use super::UiCommand;

pub fn spawn(cmd_tx: Sender<UiCommand>) {
    let _ = cmd_tx;
    // No tray yet — see module docs.
}
