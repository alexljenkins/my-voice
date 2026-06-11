//! Start-at-login via the XDG autostart spec: a `.desktop` entry in
//! `~/.config/autostart/` is launched by the session at login. No root, no
//! systemd unit — just a file we create or delete. macOS is a stub for now.

#[cfg(target_os = "linux")]
mod imp {
    use std::path::PathBuf;

    use anyhow::{Context, Result};

    fn desktop_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("autostart/my-voice.desktop"))
    }

    /// True if the autostart entry is currently installed.
    pub fn is_enabled() -> bool {
        desktop_path().is_some_and(|p| p.exists())
    }

    /// Install or remove the autostart entry. Installing points `Exec` at the
    /// current binary so a moved/reinstalled binary self-corrects on next toggle.
    pub fn set_enabled(on: bool) -> Result<()> {
        let path = desktop_path().context("no config dir for autostart entry")?;
        if !on {
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("removing {}", path.display()))?;
            }
            return Ok(());
        }
        let exe = std::env::current_exe().context("resolving current executable")?;
        let entry = format!(
            "[Desktop Entry]\n\
             Type=Application\n\
             Name=my-voice\n\
             Comment=Hold-to-talk local voice typing\n\
             Exec={}\n\
             Terminal=false\n\
             NoDisplay=true\n\
             X-GNOME-Autostart-enabled=true\n",
            exe.display()
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::write(&path, entry).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use anyhow::{bail, Result};

    pub fn is_enabled() -> bool {
        false
    }

    pub fn set_enabled(_on: bool) -> Result<()> {
        bail!("start-at-login is not implemented on this platform")
    }
}

pub use imp::{is_enabled, set_enabled};
