//! HuggingFace model fetcher — used by `--download` and the first-run auto-download.
//!
//! Streams each file to `{name}.part` and renames on completion so a Ctrl-C or
//! crash mid-download never leaves a truncated file masquerading as complete.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::config::Config;
use crate::models::{self, ModelSpec};

/// Events emitted by a background download.
pub enum DownloadEvent {
    /// Download progress, 0–99 while in progress.
    Progress(u8),
    /// All files verified and renamed to their final paths.
    Complete,
    /// Download failed; string is a user-facing reason.
    Failed(String),
}

/// CLI download — called by `--download` flag. Prints progress to stderr.
pub fn run(config: &Config) -> Result<()> {
    let Some(spec) = models::find(&config.model) else {
        bail!(
            "--download supports only known model names; \
             '{}' is a custom path — place the model files there yourself",
            config.model
        );
    };

    let dest = config.resolved_model_dir().join(&config.model);
    fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    for &(remote, base) in files_for(spec, config.quantized) {
        let url_display = format!(
            "https://huggingface.co/{}/resolve/main/{remote}",
            spec.hf_repo
        );
        eprintln!("downloading {url_display}");
        download_file(spec, remote, &dest.join(base), |done, _total| {
            eprint!("\r  {} KiB", done / 1024);
        })?;
        eprintln!();
    }

    info!("model ready at {}", dest.display());
    Ok(())
}

/// Spawn a background thread to download the configured model.
///
/// Fires [`DownloadEvent`]s through `on_event`. Callers should check
/// `config.is_model_downloaded()` first; per-file downloads are still
/// idempotent (skip if already present) so duplicate calls are safe.
pub fn start_background(config: Config, on_event: impl Fn(DownloadEvent) + Send + 'static) {
    std::thread::spawn(move || {
        match run_with_progress(&config, |pct| on_event(DownloadEvent::Progress(pct))) {
            Ok(()) => on_event(DownloadEvent::Complete),
            Err(e) => on_event(DownloadEvent::Failed(format!("{e:#}"))),
        }
    });
}

fn run_with_progress(config: &Config, on_progress: impl Fn(u8)) -> Result<()> {
    let Some(spec) = models::find(&config.model) else {
        bail!(
            "auto-download: '{}' is not a known model name",
            config.model
        );
    };

    let dest = config.resolved_model_dir().join(&config.model);
    fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    let files = files_for(spec, config.quantized);
    let n = files.len() as u8;

    for (i, &(remote, base)) in files.iter().enumerate() {
        let base_pct = (i as u8 * 100) / n;
        let range = (100u8 / n).max(1);
        download_file(spec, remote, &dest.join(base), |done, total| {
            let within = (done * range as u64).checked_div(total).unwrap_or(0) as u8;
            on_progress((base_pct + within).min(99));
        })?;
    }
    Ok(())
}

fn files_for(spec: &ModelSpec, quantized: bool) -> &[crate::models::FileEntry] {
    if quantized {
        spec.files_quantized
    } else {
        spec.files_full
    }
}

/// Download one file. Calls `on_chunk(bytes_done, content_length)` after each
/// write. Skips if `dest` already exists. Writes to `{dest}.part`, verifies
/// SHA-256 for pinned files, then renames on success.
fn download_file(
    spec: &ModelSpec,
    remote: &str,
    dest: &Path,
    on_chunk: impl Fn(u64, u64),
) -> Result<()> {
    if dest.exists() {
        info!("have {}", dest.display());
        return Ok(());
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{remote}",
        spec.hf_repo
    );
    let resp = ureq::get(&url)
        .call()
        .with_context(|| format!("GET {url}"))?;

    let content_length = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let part = dest.with_extension("part");
    let mut file =
        fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut hasher = Sha256::new();
    loop {
        let n = reader.read(&mut buf).context("reading response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        total += n as u64;
        on_chunk(total, content_length);
    }
    file.sync_all().ok();

    let base = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if let Some(&(_, expected)) = spec.checksums.iter().find(|&&(name, _)| name == base) {
        let got = format!("{:x}", hasher.finalize());
        if got != expected {
            let _ = fs::remove_file(&part);
            bail!(
                "SHA-256 mismatch for {base}: expected {expected}, got {got} \
                 — possible corrupt download or changed upstream file"
            );
        }
    }

    fs::rename(&part, dest).with_context(|| format!("renaming to {}", dest.display()))?;
    Ok(())
}
