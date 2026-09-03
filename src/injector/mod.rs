use anyhow::Result;

use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryMode {
    Typed,
    Clipboard,
}

pub trait Injector: Send {
    fn inject(&mut self, text: &str) -> Result<()>;
    fn delivery_mode(&self) -> DeliveryMode {
        DeliveryMode::Typed
    }
    fn inject_effective(&mut self, text: &str) -> Result<DeliveryMode> {
        self.inject(text)?;
        Ok(self.delivery_mode())
    }
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
}

pub mod linux;

/// Build the typing injector (auto-detected or configured via `config.injection`).
pub fn detect(config: &Config) -> Box<dyn Injector> {
    linux::detect(config)
}

/// Whether direct typing ("Paste at cursor") is available, plus an unlock hint
/// for the menu when it isn't.
pub fn typing_availability() -> (bool, String) {
    linux::typing_availability()
}

/// Build the clipboard injector used for the Shift+hotkey path.
pub fn clipboard() -> Box<dyn Injector> {
    linux::clipboard_injector()
}
