mod audio;
mod config;
mod download;
mod hotkey;
mod injector;
mod model_cache;
mod text;
mod transcriber;

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

use audio::AudioRecorder;
use config::Config;
use hotkey::{spawn_listener, HotkeyEvent};
use injector::Injector;
use model_cache::ModelCache;
use text::post_process;

const TEST_WAV: &str = "/tmp/my-voice-test.wav";

#[derive(Parser, Debug)]
#[command(name = "my-voice", version, about = "Hold-to-talk local voice typing.")]
struct Cli {
    /// Fetch the configured model, then exit.
    #[arg(long)]
    download: bool,

    /// Record 3s from the mic, save a wav, print stats, exit (no hotkey).
    #[arg(long)]
    test: bool,

    /// Transcribe a wav file directly (bypasses the mic), print, exit.
    #[arg(long, value_name = "PATH")]
    wav: Option<PathBuf>,

    /// Print audio input device names and exit.
    #[arg(long)]
    list_devices: bool,

    /// Alternate config file.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Increase logging: -v = info, -vv = debug.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    if cli.list_devices {
        return audio::list_devices();
    }

    let config = Config::load(cli.config.as_deref())?;
    debug!(?config, "loaded config");

    if cli.download {
        return download::run(&config);
    }

    if cli.test {
        return run_test(&config);
    }

    if let Some(path) = cli.wav.as_deref() {
        return run_wav(&config, path);
    }

    run_daemon(&config)
}

/// Feed a wav file straight through the transcriber — isolates the inference
/// path from the mic/capture path. Resamples to 16 kHz mono if needed.
fn run_wav(config: &Config, path: &std::path::Path) -> Result<()> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {path:?}"))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    let samples = audio::resample(&mono, spec.sample_rate, 16_000);
    let peak = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    info!(
        "wav: {:.2}s, {} Hz → 16 kHz, {ch} ch → mono, peak {peak:.3}",
        mono.len() as f32 / spec.sample_rate as f32,
        spec.sample_rate
    );
    let mut transcriber = transcriber::create(config)?;
    let text = post_process(&transcriber.transcribe(&samples)?);
    println!("{text}");
    Ok(())
}

/// Record a fixed 3s window, dump a debug wav, transcribe, and print — verifies
/// the full audio→text path without needing hotkey/input permissions.
fn run_test(config: &Config) -> Result<()> {
    let mut transcriber = transcriber::create(config)?;
    let mut recorder = AudioRecorder::new(&config.audio_device)?;
    info!("recording 3s for --test...");
    recorder.start()?;
    thread::sleep(Duration::from_secs(3));
    let samples = recorder.stop();
    let rate = recorder.target_rate();
    info!("captured {:.2}s", samples.len() as f32 / rate as f32);
    if let Err(e) = write_wav(&samples, rate, TEST_WAV) {
        warn!("failed to write {TEST_WAV}: {e}");
    }
    let text = post_process(&transcriber.transcribe(&samples)?);
    println!("{text}");
    Ok(())
}

fn run_daemon(config: &Config) -> Result<()> {
    let _lock = single_instance::acquire().context("single-instance lock")?;

    // Lazy load: the daemon starts cold and holds no model in RAM until the
    // first press. The evict thread reclaims it after idle (Phase 6).
    let cache = ModelCache::new(config);
    cache.start_evict_thread();
    let mut recorder = AudioRecorder::new(&config.audio_device)?;
    let mut typer = injector::detect(config);
    let mut clipper = injector::clipboard();

    #[cfg(unix)]
    install_signal_handlers();

    let (tx, rx) = mpsc::channel::<HotkeyEvent>();
    spawn_listener(config, tx)?;

    info!("ready — hold '{}' to record", config.hotkey);

    let trailing = Duration::from_millis(config.trailing_silence_ms);
    let mut state = State::Idle;

    for event in rx {
        match (&state, event) {
            (State::Idle, HotkeyEvent::Press { clipboard_only }) => {
                if let Err(e) = recorder.start() {
                    warn!("could not start recording: {e}");
                    continue;
                }
                // Kick the cold-start load now so it overlaps with speech;
                // transcribe() later blocks on the same lock if it's not done.
                let preload = Arc::clone(&cache);
                thread::spawn(move || {
                    if let Err(e) = preload.ensure_loaded() {
                        warn!("model preload failed: {e:#}");
                    }
                });
                debug!(clipboard_only, "recording");
                state = State::Recording { clipboard_only };
            }
            (State::Recording { clipboard_only }, HotkeyEvent::Release) => {
                // PTT trailing buffer: catch the tail of the last word.
                thread::sleep(trailing);
                let samples = recorder.stop();
                let inj: &mut dyn Injector = if *clipboard_only {
                    clipper.as_mut()
                } else {
                    typer.as_mut()
                };
                handle_utterance(&cache, &samples, recorder.target_rate(), config, inj);
                state = State::Idle;
            }
            // Recording+Press (autorepeat dupe) and Idle+Release (stale): ignore.
            _ => {}
        }
    }

    hotkey::restore_platform();
    Ok(())
}

#[derive(Debug)]
enum State {
    Idle,
    Recording { clipboard_only: bool },
}

/// Gate, transcribe, post-process, inject. Both gates discard before inference —
/// ASR models hallucinate text on silence, so never feed them empty air.
fn handle_utterance(
    cache: &ModelCache,
    samples: &[f32],
    rate: u32,
    config: &Config,
    injector: &mut dyn Injector,
) {
    let ms = samples.len() as f32 / rate as f32 * 1000.0;
    if ms < config.min_speech_ms as f32 {
        debug!(
            "discarded: {ms:.0}ms < min_speech_ms {}",
            config.min_speech_ms
        );
        return;
    }
    let peak = samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    if peak < 0.01 {
        debug!("discarded: peak {peak:.4} below silence floor");
        return;
    }

    match cache.transcribe(samples) {
        Ok(raw) => {
            let text = post_process(&raw);
            if text.is_empty() {
                debug!("empty transcription");
                return;
            }
            if let Err(e) = injector.inject(&text) {
                warn!("injection failed: {e:#}");
            }
        }
        Err(e) => warn!("transcription failed: {e:#}"),
    }
}

/// Write 16 kHz mono f32 samples as a 16-bit PCM wav.
fn write_wav(samples: &[f32], rate: u32, path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).with_context(|| format!("creating {path}"))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize().context("finalizing wav")?;
    Ok(())
}

fn init_tracing(verbose: u8) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = match verbose {
            0 => "warn",
            1 => "info",
            _ => "debug",
        };
        EnvFilter::new(format!("my_voice={level}"))
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

/// On any clean exit (SIGINT, SIGTERM) restore platform state and let the OS
/// close file descriptors, which releases evdev grabs and the flock.
#[cfg(unix)]
fn install_signal_handlers() {
    extern "C" fn handler(_sig: libc::c_int) {
        hotkey::restore_platform(); // restores hidutil on macOS; no-op on Linux
        std::process::exit(0);
    }
    unsafe {
        libc::signal(libc::SIGTERM, handler as *const () as usize);
        libc::signal(libc::SIGINT, handler as *const () as usize);
    }
}

/// Single-instance enforcement via an exclusive flock. Two daemons grabbing one
/// keyboard is chaos; this is cheap insurance.
mod single_instance {
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;
    use std::path::PathBuf;

    use anyhow::{bail, Result};

    /// Held for the process lifetime — dropping it (or process exit) releases
    /// the lock.
    pub struct Guard {
        _file: File,
    }

    fn lock_path() -> PathBuf {
        if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join("my-voice.lock");
            }
        }
        let uid = unsafe { libc::getuid() };
        PathBuf::from(format!("/tmp/my-voice-{uid}.lock"))
    }

    pub fn acquire() -> Result<Guard> {
        let path = lock_path();
        let mut file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                let mut existing = String::new();
                let _ = file.read_to_string(&mut existing);
                let pid = existing.trim();
                if pid.is_empty() {
                    bail!("my-voice is already running");
                }
                bail!("my-voice is already running (pid {pid})");
            }
            return Err(err.into());
        }

        // We hold the lock: record our pid.
        let _ = file.set_len(0);
        let _ = writeln!(file, "{}", std::process::id());
        Ok(Guard { _file: file })
    }
}
