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

/// Events emitted by a background download.
pub enum DownloadEvent {
    /// Download progress, 0–99 while in progress.
    Progress(u8),
    /// All files verified and renamed to their final paths.
    Complete,
    /// Download failed; string is a user-facing reason.
    Failed(String),
}

/// Pinned SHA-256 (HuggingFace git-LFS oid) for the large model binaries.
/// `tokenizer.json` is intentionally unpinned (non-LFS, may be reformatted upstream).
fn expected_sha(model: &str, base: &str) -> Option<&'static str> {
    match (model, base) {
        ("moonshine-tiny", "encoder_model_quantized.onnx") => Some("c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a"),
        ("moonshine-tiny", "decoder_model_merged_quantized.onnx") => Some("eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367"),
        ("moonshine-tiny", "encoder_model.onnx") => Some("cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5"),
        ("moonshine-tiny", "decoder_model_merged.onnx") => Some("4131cef00b62942e9cdef691101f2cc7dbbcd828d71eee8c6c46c28fd051d6cb"),
        ("moonshine-base", "encoder_model_quantized.onnx") => Some("1dd9ab0a7f987113d30affcba5a068d11c8f90fa0223caa3e491ade431ad9751"),
        ("moonshine-base", "decoder_model_merged_quantized.onnx") => Some("cc9f3cd6698a369c6008b41aa60aa3fb3322e7f03c9bdf19d8e6b7200afca4f3"),
        ("moonshine-base", "encoder_model.onnx") => Some("153e128e7abd64a74ee47f2c3f585c3171c4d46cbb368b032827934c4e01e779"),
        ("moonshine-base", "decoder_model_merged.onnx") => Some("58778763ca8438963190244d6b26572bdca2cedec56a4b91e828f3f2d69ef3c5"),
        _ => None,
    }
}

fn repo_for(model: &str) -> Option<&'static str> {
    match model {
        "moonshine-tiny" => Some("onnx-community/moonshine-tiny-ONNX"),
        "moonshine-base" => Some("onnx-community/moonshine-base-ONNX"),
        _ => None,
    }
}

fn file_list(quantized: bool) -> &'static [(&'static str, &'static str)] {
    if quantized {
        &[
            (
                "onnx/encoder_model_quantized.onnx",
                "encoder_model_quantized.onnx",
            ),
            (
                "onnx/decoder_model_merged_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ]
    } else {
        &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ]
    }
}

/// CLI download — called by `--download` flag. Prints progress to stderr.
pub fn run(config: &Config) -> Result<()> {
    let Some(repo) = repo_for(&config.model) else {
        bail!(
            "--download supports only moonshine-tiny / moonshine-base; \
             '{}' is a path — place the model files there yourself",
            config.model
        );
    };

    let dest = config.resolved_model_dir().join(&config.model);
    fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    for &(remote, base) in file_list(config.quantized) {
        let url_display = format!("https://huggingface.co/{repo}/resolve/main/{remote}");
        eprintln!("downloading {url_display}");
        download_file(repo, remote, &dest.join(base), &config.model, |done, _total| {
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
pub fn start_background(
    config: Config,
    on_event: impl Fn(DownloadEvent) + Send + 'static,
) {
    std::thread::spawn(move || {
        match run_with_progress(&config, |pct| on_event(DownloadEvent::Progress(pct))) {
            Ok(()) => on_event(DownloadEvent::Complete),
            Err(e) => on_event(DownloadEvent::Failed(format!("{e:#}"))),
        }
    });
}

fn run_with_progress(config: &Config, on_progress: impl Fn(u8)) -> Result<()> {
    let Some(repo) = repo_for(&config.model) else {
        bail!(
            "auto-download: '{}' is not a known model name \
             (only moonshine-tiny / moonshine-base are auto-downloadable)",
            config.model
        );
    };

    let dest = config.resolved_model_dir().join(&config.model);
    fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    let files = file_list(config.quantized);
    let n = files.len() as u8;

    for (i, &(remote, base)) in files.iter().enumerate() {
        // Each file gets an equal share of the 0–99 range; 100 signals Complete.
        let base_pct = (i as u8 * 100) / n;
        let range = (100u8 / n).max(1);
        download_file(repo, remote, &dest.join(base), &config.model, |done, total| {
            let within = if total > 0 {
                ((done * range as u64) / total) as u8
            } else {
                0
            };
            on_progress((base_pct + within).min(99));
        })?;
    }
    Ok(())
}

/// Download one file. Calls `on_chunk(bytes_done, content_length)` after each
/// write. Skips if `dest` already exists. Writes to `{dest}.part`, verifies
/// SHA-256 for pinned ONNX files, then renames on success.
fn download_file(
    repo: &str,
    remote: &str,
    dest: &Path,
    model: &str,
    on_chunk: impl Fn(u64, u64),
) -> Result<()> {
    if dest.exists() {
        info!("have {}", dest.display());
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{remote}");
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

    // Verify SHA-256 for pinned ONNX binaries before promoting the .part file.
    let base = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if let Some(expected) = expected_sha(model, base) {
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
