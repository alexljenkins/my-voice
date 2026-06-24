//! HuggingFace model fetcher — used by `--download` and the first-run auto-download.
//!
//! Streams each file to `{name}.part`, then stages the whole model into a
//! `{name}.partial` sibling dir that's renamed into place only once every file
//! verifies — so a Ctrl-C or crash mid-download never leaves a truncated file
//! or a half-populated model dir masquerading as complete.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::config::Config;
use crate::models::{self, ModelSpec};

/// Give up after this many attempts per file: a flaky network gets a few tries,
/// a dead one fails fast.
const MAX_ATTEMPTS: u32 = 3;
/// Fail a stalled connection instead of hanging the download thread forever.
/// `timeout_read` is per-socket-read, so a slow-but-progressing large file is
/// unaffected — only a truly stuck stream trips it.
const TIMEOUT_CONNECT: Duration = Duration::from_secs(10);
const TIMEOUT_READ: Duration = Duration::from_secs(30);

/// Backoff before the retry following `attempt` (1-based): 1s, 2s, 4s… — short
/// so the tray progress doesn't look frozen.
fn backoff(attempt: u32) -> Duration {
    Duration::from_secs(1 << (attempt - 1))
}

/// Run `f`, retrying failures up to [`MAX_ATTEMPTS`] with [`backoff`] between
/// tries. `sleep` is injected so tests can bound attempts without real delay.
fn with_retry<T>(mut f: impl FnMut() -> Result<T>, mut sleep: impl FnMut(Duration)) -> Result<T> {
    for attempt in 1..MAX_ATTEMPTS {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                warn!("download attempt {attempt} failed: {e:#} — retrying");
                sleep(backoff(attempt));
            }
        }
    }
    f() // final attempt: surface its error
}

fn agent() -> ureq::Agent {
    ureq::builder()
        .timeout_connect(TIMEOUT_CONNECT)
        .timeout_read(TIMEOUT_READ)
        .build()
}

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

    let final_dir = config.resolved_model_dir().join(&config.model);
    install_atomic(config, |dest| {
        let agent = agent();
        for &(remote, base) in files_for(spec, config.quantized) {
            let url_display = format!(
                "https://huggingface.co/{}/resolve/main/{remote}",
                spec.hf_repo
            );
            eprintln!("downloading {url_display}");
            download_file(&agent, spec, remote, &dest.join(base), |done, _total| {
                eprint!("\r  {} KiB", done / 1024);
            })?;
            eprintln!();
        }
        Ok(())
    })?;

    info!("model ready at {}", final_dir.display());
    Ok(())
}

/// Spawn a background thread to download the configured model.
///
/// Fires [`DownloadEvent`]s through `on_event`. Callers should check
/// `config.is_model_downloaded()` first; install is idempotent anyway
/// (`install_atomic` returns early if the model dir exists), so duplicate
/// calls are safe.
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

    install_atomic(config, |dest| {
        let files = files_for(spec, config.quantized);
        let n = files.len() as u8;
        let agent = agent();
        for (i, &(remote, base)) in files.iter().enumerate() {
            let base_pct = (i as u8 * 100) / n;
            let range = (100u8 / n).max(1);
            download_file(&agent, spec, remote, &dest.join(base), |done, total| {
                let within = (done * range as u64).checked_div(total).unwrap_or(0) as u8;
                on_progress((base_pct + within).min(99));
            })?;
        }
        Ok(())
    })
}

/// Install the configured model atomically: `download_into` fetches every file
/// into a sibling `{model}.partial` staging dir, then a single `fs::rename`
/// moves it into place only after they all succeed. Atomicity is per-MODEL, not
/// per-file — the encoder (the sentinel) downloads first, so a crash mid-set
/// would otherwise leave an encoder-present-but-decoder-missing dir that both
/// download gates accept as complete (`is_model_downloaded` checks only the
/// sentinel). After this the final dir only ever exists fully populated.
fn install_atomic(config: &Config, download_into: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let model_dir = config.resolved_model_dir();
    let final_dir = model_dir.join(&config.model);
    let staging = model_dir.join(format!("{}.partial", config.model));

    // Sweep any stale partial from an earlier crash — there is no cross-run resume.
    fs::remove_dir_all(&staging).ok();
    if final_dir.exists() {
        return Ok(()); // already installed; the dir only ever exists complete
    }
    fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;

    download_into(&staging)?;

    // Siblings under model_dir → same filesystem → the rename is atomic.
    fs::rename(&staging, &final_dir)
        .with_context(|| format!("installing model to {}", final_dir.display()))
}

fn files_for(spec: &ModelSpec, quantized: bool) -> &[crate::models::FileEntry] {
    if quantized {
        spec.files_quantized
    } else {
        spec.files_full
    }
}

/// GET `url` and stream the body into `part`, hashing as it goes. Each call
/// truncates `part` fresh, so a retry after a stalled stream starts clean.
fn stream_to(
    agent: &ureq::Agent,
    url: &str,
    part: &Path,
    hasher: &mut Sha256,
    on_chunk: &impl Fn(u64, u64),
) -> Result<()> {
    let resp = agent
        .get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let content_length = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    stream_body(&mut reader, part, content_length, hasher, on_chunk)
}

/// Stream `reader` into `part`, hashing as it goes, then reject a short read:
/// a connection dropped cleanly at EOF yields a truncated file with no error
/// that would otherwise pass straight to the checksum step — or, for
/// `tokenizer.json` (no checksum row in models.rs), install silently and fail
/// cryptically at load. The Err routes through `with_retry` for another attempt.
fn stream_body(
    reader: &mut impl Read,
    part: &Path,
    content_length: u64,
    hasher: &mut Sha256,
    on_chunk: &impl Fn(u64, u64),
) -> Result<()> {
    let mut file =
        fs::File::create(part).with_context(|| format!("creating {}", part.display()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
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
    // `<` not `!=`: a decompressing transport can legitimately yield more than
    // content_length; only a short read signals truncation. Keep the `!= 0`
    // escape — HF may omit the header or use chunked encoding.
    if content_length != 0 && total < content_length {
        let _ = fs::remove_file(part);
        bail!("download incomplete: expected {content_length} bytes, got {total}");
    }
    file.sync_all().ok();
    Ok(())
}

/// Download one file. Calls `on_chunk(bytes_done, content_length)` after each
/// write. Writes to `{dest}.part` (overwritten fresh on each retry — there is no
/// resume), verifies SHA-256 for pinned files, then renames on success. The
/// network fetch is retried on transient failures; the checksum+rename guarantee
/// a retry can't install bad bytes. `dest` lives in a fresh staging dir
/// (see `install_atomic`), so it never pre-exists.
fn download_file(
    agent: &ureq::Agent,
    spec: &ModelSpec,
    remote: &str,
    dest: &Path,
    on_chunk: impl Fn(u64, u64),
) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{remote}",
        spec.hf_repo
    );
    let part = dest.with_extension("part");

    let mut hasher = Sha256::new();
    with_retry(
        || {
            hasher = Sha256::new();
            stream_to(agent, &url, &part, &mut hasher, &on_chunk)
        },
        std::thread::sleep,
    )?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn backoff_is_exponential() {
        assert_eq!(backoff(1), Duration::from_secs(1));
        assert_eq!(backoff(2), Duration::from_secs(2));
        assert_eq!(backoff(3), Duration::from_secs(4));
    }

    #[test]
    fn retry_returns_first_success_without_sleeping() {
        let calls = Cell::new(0);
        let slept = Cell::new(0);
        let r: Result<u32> = with_retry(
            || {
                calls.set(calls.get() + 1);
                Ok(7)
            },
            |_| slept.set(slept.get() + 1),
        );
        assert_eq!(r.unwrap(), 7);
        assert_eq!(calls.get(), 1);
        assert_eq!(slept.get(), 0);
    }

    #[test]
    fn retry_recovers_on_a_later_attempt() {
        let calls = Cell::new(0);
        let slept = Cell::new(0);
        let r: Result<u32> = with_retry(
            || {
                calls.set(calls.get() + 1);
                if calls.get() < 3 {
                    bail!("transient")
                }
                Ok(9)
            },
            |_| slept.set(slept.get() + 1),
        );
        assert_eq!(r.unwrap(), 9);
        assert_eq!(calls.get(), 3);
        assert_eq!(slept.get(), 2); // slept once before each of the two retries
    }

    #[test]
    fn retry_is_bounded_and_surfaces_last_error() {
        let calls = Cell::new(0);
        let slept = Cell::new(0);
        let r: Result<u32> = with_retry(
            || {
                calls.set(calls.get() + 1);
                bail!("always fails")
            },
            |_| slept.set(slept.get() + 1),
        );
        assert!(r.is_err());
        assert_eq!(calls.get(), MAX_ATTEMPTS as i32);
        assert_eq!(slept.get(), (MAX_ATTEMPTS - 1) as i32); // no sleep after the final try
    }

    /// A stream that ends short of content-length (clean EOF, no error) must be
    /// rejected and its partial removed, not passed on as a complete file.
    #[test]
    fn stream_body_rejects_truncated_stream() {
        let part = std::env::temp_dir().join("my-voice-test-truncated.part");
        let _ = fs::remove_file(&part);
        let data: &[u8] = b"only twelve!"; // 12 bytes, but we claim 100
        let mut reader = data;
        let mut hasher = Sha256::new();
        let r = stream_body(&mut reader, &part, 100, &mut hasher, &|_, _| {});
        assert!(r.is_err(), "a short read must error");
        assert!(!part.exists(), "the truncated partial must be removed");
    }

    /// A full stream (total == content-length) succeeds and keeps its partial;
    /// the `<` guard must not false-positive on an exact match.
    #[test]
    fn stream_body_accepts_complete_stream() {
        let part = std::env::temp_dir().join("my-voice-test-complete.part");
        let _ = fs::remove_file(&part);
        let data: &[u8] = b"all twelve!!"; // 12 bytes == claimed length
        let mut reader = data;
        let mut hasher = Sha256::new();
        let r = stream_body(
            &mut reader,
            &part,
            data.len() as u64,
            &mut hasher,
            &|_, _| {},
        );
        assert!(r.is_ok(), "a complete stream must succeed");
        assert!(part.exists(), "the completed partial must remain");
        let _ = fs::remove_file(&part);
    }

    /// A download that fails after the first file must leave NO final model dir,
    /// so the sentinel gate can't mistake a half-set for a complete install.
    #[test]
    fn install_atomic_failure_creates_no_final_dir() {
        let root = std::env::temp_dir().join("my-voice-test-atomic-install");
        let _ = fs::remove_dir_all(&root);
        let config = Config {
            model: "fake-model".into(),
            model_dir: root.to_string_lossy().into_owned(),
            ..Default::default()
        };
        let r = install_atomic(&config, |staging| {
            fs::write(staging.join("encoder.onnx"), b"first file")?;
            bail!("network died after file 1") // crash mid-set
        });
        assert!(r.is_err());
        let final_dir = config.resolved_model_dir().join(&config.model);
        assert!(
            !final_dir.exists(),
            "a half-downloaded model dir must not be installed"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
