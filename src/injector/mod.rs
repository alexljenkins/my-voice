use anyhow::Result;

use crate::config::Config;

pub trait Injector: Send {
    fn inject(&mut self, text: &str) -> Result<()>;
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
}

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;

/// Build the typing injector (auto-detected or configured via `config.injection`).
pub fn detect(config: &Config) -> Box<dyn Injector> {
    #[cfg(target_os = "linux")]
    {
        linux::detect(config)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = config;
        macos::typing_injector()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = config;
        Box::new(NullInjector)
    }
}

/// Whether direct typing ("Paste at cursor") is available, plus an unlock hint
/// for the menu when it isn't. macOS always types via CGEvent.
pub fn typing_availability() -> (bool, String) {
    #[cfg(target_os = "linux")]
    {
        linux::typing_availability()
    }
    #[cfg(not(target_os = "linux"))]
    {
        (true, String::new())
    }
}

/// Build the clipboard injector used for the Shift+hotkey path.
pub fn clipboard() -> Box<dyn Injector> {
    #[cfg(target_os = "linux")]
    {
        linux::clipboard_injector()
    }
    #[cfg(target_os = "macos")]
    {
        macos::clipboard_injector()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Box::new(NullInjector)
    }
}

#[allow(dead_code)]
struct NullInjector;

#[allow(dead_code)]
impl Injector for NullInjector {
    fn inject(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "null"
    }
}
