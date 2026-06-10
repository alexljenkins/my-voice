mod audio;
mod config;
mod download;
mod hotkey;
mod injector;
mod text;
mod transcriber;

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
use injector::Injector;
use text::post_process;
use transcriber::Transcriber;

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
        return download::run(&config);
    }

    if cli.test {
        return run_test(&config);
    }

    run_daemon(&config)
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

    let mut transcriber = transcriber::create(config)?;
    let mut recorder = AudioRecorder::new(&config.audio_device)?;
    let mut typer = injector::detect(config);
    let mut clipper = injector::clipboard();

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
                handle_utterance(transcriber.as_mut(), &samples, recorder.target_rate(), config, inj);
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
    transcriber: &mut dyn Transcriber,
    samples: &[f32],
    rate: u32,
    config: &Config,
    injector: &mut dyn Injector,
) {
    let ms = samples.len() as f32 / rate as f32 * 1000.0;
    if ms < config.min_speech_ms as f32 {
        debug!("discarded: {ms:.0}ms < min_speech_ms {}", config.min_speech_ms);
        return;
    }
    let peak = samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    if peak < 0.01 {
        debug!("discarded: peak {peak:.4} below silence floor");
        return;
    }

    match transcriber.transcribe(samples) {
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
