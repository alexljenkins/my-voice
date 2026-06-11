//! Config struct, TOML load, model path resolution.
//!
//! `~/.config/my-voice/config.toml` is optional — every field has a hardcoded
//! default. Unknown keys warn and continue (no `deny_unknown_fields`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: String,
    pub model_dir: String,
    pub quantized: bool,
    pub threads: usize,
    pub load_timeout_secs: i64,
    pub hotkey: String,
    pub clipboard_hotkey: bool,
    pub grab: bool,
    pub audio_device: String,
    pub min_speech_ms: u64,
    pub trailing_silence_ms: u64,
    pub injection: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "moonshine-streaming-medium".into(),
            model_dir: "~/.local/share/my-voice/models".into(),
            quantized: true,
            threads: 0,
            load_timeout_secs: 1800,
            hotkey: "CapsLock".into(),
            clipboard_hotkey: true,
            grab: true,
            audio_device: String::new(),
            min_speech_ms: 300,
            trailing_silence_ms: 150,
            injection: "auto".into(),
        }
    }
}

impl Config {
    /// Default config file location: `~/.config/my-voice/config.toml`.
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("my-voice/config.toml"))
    }

    /// Load from `path` if given, else from the default location. A missing
    /// file is not an error — defaults are returned.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let chosen = match path {
            Some(p) => Some(p.to_path_buf()),
            None => Self::default_path(),
        };

        let Some(p) = chosen else {
            return Ok(Self::default());
        };

        if !p.exists() {
            if path.is_some() {
                anyhow::bail!("config file not found: {}", p.display());
            }
            return Ok(Self::default());
        }

        let raw = std::fs::read_to_string(&p)
            .with_context(|| format!("reading config {}", p.display()))?;
        let cfg: Config =
            toml::from_str(&raw).with_context(|| format!("parsing config {}", p.display()))?;
        cfg.warn_unknown_keys(&raw);
        Ok(cfg)
    }

    /// Best-effort diff of top-level keys we didn't recognize. Cheap; logs only.
    fn warn_unknown_keys(&self, raw: &str) {
        let known = [
            "model",
            "model_dir",
            "quantized",
            "threads",
            "load_timeout_secs",
            "hotkey",
            "clipboard_hotkey",
            "grab",
            "audio_device",
            "min_speech_ms",
            "trailing_silence_ms",
            "injection",
        ];
        if let Ok(value) = toml::from_str::<toml::Table>(raw) {
            for key in value.keys() {
                if !known.contains(&key.as_str()) {
                    tracing::warn!("unknown config key ignored: {key}");
                }
            }
        }
    }

    /// Resolve `threads`: 0 = auto = min(num_cpus, 4). [voxtype]
    pub fn resolved_threads(&self) -> usize {
        if self.threads == 0 {
            num_cpus::get().min(4)
        } else {
            self.threads
        }
    }

    /// Tilde-expanded model storage directory.
    pub fn resolved_model_dir(&self) -> PathBuf {
        expand_tilde(&self.model_dir)
    }

    /// Write the current config to disk. Creates parent directories as needed.
    /// Uses `path` if given, otherwise the default location.
    pub fn save(&self, path: Option<&Path>) -> Result<()> {
        let p = match path {
            Some(p) => p.to_path_buf(),
            None => Self::default_path()
                .ok_or_else(|| anyhow::anyhow!("cannot determine config path"))?,
        };
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {}", parent.display()))?;
        }
        let toml = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(&p, &toml).with_context(|| format!("writing config {}", p.display()))?;
        Ok(())
    }

    /// True if the model files for the configured model are present on disk.
    /// Uses the registry sentinel file; returns false for unknown/custom models.
    pub fn is_model_downloaded(&self) -> bool {
        let Some(spec) = crate::models::find(&self.model) else {
            return false;
        };
        let dir = self.resolved_model_dir().join(&self.model);
        if !dir.is_dir() {
            return false;
        }
        let sentinel = if self.quantized {
            spec.sentinel_quantized
        } else {
            spec.sentinel_full
        };
        dir.join(sentinel).exists()
    }

    /// Map `model` → the Moonshine model directory. Does not check existence.
    /// Named registry models live under `model_dir/<name>`; anything else is
    /// treated as a custom (tilde-expanded) path to a model directory.
    pub fn resolve_model(&self) -> PathBuf {
        if crate::models::find(&self.model).is_some() {
            self.resolved_model_dir().join(&self.model)
        } else {
            expand_tilde(&self.model)
        }
    }
}

/// Expand a leading `~` using `dirs::home_dir()` (not shellexpand). A bare `~`
/// or `~/...` is expanded; everything else is returned unchanged.
pub fn expand_tilde(s: &str) -> PathBuf {
    if s == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    } else if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip() {
        let cfg = Config::default();
        let toml = toml::to_string(&cfg).unwrap();
        let back: Config = toml::from_str(&toml).unwrap();
        assert_eq!(back.model, "moonshine-streaming-medium");
        assert_eq!(back.load_timeout_secs, 1800);
        assert_eq!(back.min_speech_ms, 300);
        assert_eq!(back.trailing_silence_ms, 150);
        assert!(back.quantized);
    }

    #[test]
    fn partial_config_keeps_defaults() {
        let cfg: Config = toml::from_str("min_speech_ms = 500").unwrap();
        assert_eq!(cfg.min_speech_ms, 500);
        assert_eq!(cfg.model, "moonshine-streaming-medium"); // default preserved
    }

    #[test]
    fn model_resolution_named() {
        let cfg = Config::default();
        assert!(cfg.resolve_model().ends_with("moonshine-streaming-medium"));
    }

    #[test]
    fn model_resolution_custom_path() {
        let cfg = Config {
            model: "/models/custom-moonshine".into(),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolve_model(),
            PathBuf::from("/models/custom-moonshine")
        );
    }

    #[test]
    fn tilde_expansion() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/x/y"), home.join("x/y"));
        assert_eq!(expand_tilde("/abs"), PathBuf::from("/abs"));
    }

    #[test]
    fn threads_auto_caps_at_four() {
        let cfg = Config::default();
        assert!(cfg.resolved_threads() >= 1);
        assert!(cfg.resolved_threads() <= 4);
    }
}
