//! my-voice — hold-to-talk local voice typing.
//!
//! Phase 1: daemon starts, records while the hotkey is held, and writes a
//! playable wav on release. No model, no injection yet.

mod audio;
mod config;
mod hotkey;

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

use audio::AudioRecorder;
use config::Config;
use hotkey::{spawn_listener, HotkeyEvent};

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
        eprintln!("model download arrives in Phase 2 — not available in this build");
        return Ok(());
    }

    if cli.test {
        return run_test(&config);
    }

    run_daemon(&config)
}

/// Record a fixed 3s window and dump a wav — verifies the audio path without
/// needing hotkey/input permissions.
fn run_test(config: &Config) -> Result<()> {
    let mut recorder = AudioRecorder::new(&config.audio_device)?;
    info!("recording 3s for --test...");
    recorder.start()?;
    thread::sleep(Duration::from_secs(3));
    let samples = recorder.stop();
    report_and_write(&samples, recorder.target_rate())?;
    Ok(())
}

fn run_daemon(config: &Config) -> Result<()> {
    let _lock = single_instance::acquire().context("single-instance lock")?;

    let mut recorder = AudioRecorder::new(&config.audio_device)?;

    #[cfg(target_os = "macos")]
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
                debug!(clipboard_only, "recording");
                state = State::Recording;
            }
            (State::Recording, HotkeyEvent::Release) => {
                // PTT trailing buffer: catch the tail of the last word.
                thread::sleep(trailing);
                let samples = recorder.stop();
                handle_utterance(&samples, recorder.target_rate());
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
    Recording,
}

/// Phase 1 utterance handling: gate-log, then dump a wav. (Phase 2 transcribes.)
fn handle_utterance(samples: &[f32], rate: u32) {
    let secs = samples.len() as f32 / rate as f32;
    let peak = samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    info!("utterance: {secs:.2}s, peak {peak:.3}");

    // Gates that Phase 2 will use to discard before inference.
    if (secs * 1000.0) < 300.0 {
        debug!("(gate) below min_speech_ms — Phase 2 would discard");
    }
    if peak < 0.01 {
        debug!("(gate) near-silent — Phase 2 would discard");
    }

    if let Err(e) = write_wav(samples, rate, TEST_WAV) {
        warn!("failed to write {TEST_WAV}: {e}");
    } else {
        info!("wrote {TEST_WAV}");
    }
}

fn report_and_write(samples: &[f32], rate: u32) -> Result<()> {
    let secs = samples.len() as f32 / rate as f32;
    let peak = samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    info!("captured {secs:.2}s, peak {peak:.3}");
    write_wav(samples, rate, TEST_WAV)?;
    info!("wrote {TEST_WAV}");
    println!("{secs:.2}s @ {rate} Hz, peak {peak:.3} → {TEST_WAV}");
    Ok(())
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

#[cfg(target_os = "macos")]
fn install_signal_handlers() {
    extern "C" fn handler(_sig: libc::c_int) {
        hotkey::restore_platform();
        std::process::exit(0);
    }
    unsafe {
        libc::signal(libc::SIGINT, handler as usize);
        libc::signal(libc::SIGTERM, handler as usize);
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
