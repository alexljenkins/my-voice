mod audio;
mod config;
mod download;
mod hotkey;
mod injector;
mod model_cache;
mod text;
mod transcriber;
mod ui;

use std::path::{Path, PathBuf};
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
use ui::{TrayState, UiCommand, UiHandle};

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

    run_daemon(config, cli.config)
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

/// A message into the daemon's single event loop. Both the hotkey listener and
/// the tray UI feed this one channel, so the loop can react to input and to
/// menu actions without a multi-channel select.
enum DaemonMsg {
    Hotkey(HotkeyEvent),
    Ui(UiCommand),
}

fn run_daemon(mut config: Config, config_path: Option<PathBuf>) -> Result<()> {
    let _lock = single_instance::acquire().context("single-instance lock")?;

    // Lazy load: the daemon starts cold and holds no model in RAM until the
    // first press. The evict thread reclaims it after idle.
    let mut cache = ModelCache::new(&config);
    cache.start_evict_thread();
    let mut recorder = AudioRecorder::new(&config.audio_device)?;
    let mut typer = injector::detect(&config);
    let mut clipper = injector::clipboard();

    #[cfg(unix)]
    install_signal_handlers();

    // One channel, two producers. The hotkey listener and tray each get their
    // own typed sender, forwarded into the merged stream the loop drains.
    let (daemon_tx, daemon_rx) = mpsc::channel::<DaemonMsg>();

    let (hk_tx, hk_rx) = mpsc::channel::<HotkeyEvent>();
    spawn_listener(&config, hk_tx)?;
    forward(hk_rx, daemon_tx.clone(), DaemonMsg::Hotkey);

    let (ui_tx, ui_rx) = mpsc::channel::<UiCommand>();
    let ui = ui::spawn(ui_tx);
    forward(ui_rx, daemon_tx, DaemonMsg::Ui);

    info!("ready — hold '{}' to record", config.hotkey);
    ui.set_state(TrayState::Ready);

    let mut trailing = Duration::from_millis(config.trailing_silence_ms);
    let mut state = State::Idle;
    // A reload requested mid-utterance is deferred until we return to Idle, so
    // we never swap the recorder/model out from under an in-flight transcription.
    let mut pending_reload = false;

    for msg in daemon_rx {
        match msg {
            DaemonMsg::Hotkey(event) => match (&state, event) {
                (State::Idle, HotkeyEvent::Press { clipboard_only }) => {
                    if let Err(e) = recorder.start() {
                        warn!("could not start recording: {e}");
                        ui.set_state(TrayState::Error("Couldn't start recording".into()));
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
                    ui.set_state(TrayState::Listening);
                    state = State::Recording { clipboard_only };
                }
                (State::Recording { clipboard_only }, HotkeyEvent::Release) => {
                    // PTT trailing buffer: catch the tail of the last word.
                    thread::sleep(trailing);
                    let samples = recorder.stop();
                    let clipboard_only = *clipboard_only;
                    ui.set_state(TrayState::Transcribing);
                    let inj: &mut dyn Injector = if clipboard_only {
                        clipper.as_mut()
                    } else {
                        typer.as_mut()
                    };
                    match handle_utterance(&cache, &samples, recorder.target_rate(), &config, inj) {
                        Ok(()) => ui.set_state(TrayState::Ready),
                        Err(e) => ui.set_state(TrayState::Error(e)),
                    }
                    state = State::Idle;
                    if pending_reload {
                        pending_reload = false;
                        apply_reload(
                            &mut config,
                            config_path.as_deref(),
                            &mut recorder,
                            &mut typer,
                            &mut cache,
                            &mut trailing,
                            &ui,
                        );
                    }
                }
                // Recording+Press (autorepeat dupe) and Idle+Release (stale): ignore.
                _ => {}
            },
            DaemonMsg::Ui(UiCommand::Quit) => {
                info!("quit requested");
                break;
            }
            DaemonMsg::Ui(UiCommand::ReloadConfig) => match state {
                State::Idle => apply_reload(
                    &mut config,
                    config_path.as_deref(),
                    &mut recorder,
                    &mut typer,
                    &mut cache,
                    &mut trailing,
                    &ui,
                ),
                // Mid-utterance: defer to the Release handler above.
                State::Recording { .. } => pending_reload = true,
            },
        }
    }

    hotkey::restore_platform();
    Ok(())
}

/// Pump one typed channel into the merged daemon channel under `wrap`. Stops
/// when either end closes.
fn forward<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
    tx: mpsc::Sender<DaemonMsg>,
    wrap: fn(T) -> DaemonMsg,
) {
    thread::spawn(move || {
        for item in rx {
            if tx.send(wrap(item)).is_err() {
                break;
            }
        }
    });
}

/// Which live resources a config change forces us to rebuild. Pure function of
/// the old/new config so it can be unit-tested without touching audio/models.
#[derive(Debug, PartialEq, Eq)]
struct ReloadActions {
    recorder: bool,
    injector: bool,
    model: bool,
    /// Hotkey/grab changes can't be applied live (evdev listeners block in
    /// `fetch_events()` and can't be joined yet — §2 epoll rework), so the v1
    /// fallback is a self-restart.
    restart: bool,
}

fn reload_actions(old: &Config, new: &Config) -> ReloadActions {
    ReloadActions {
        recorder: old.audio_device != new.audio_device,
        injector: old.injection != new.injection,
        model: old.model != new.model
            || old.model_dir != new.model_dir
            || old.quantized != new.quantized
            || old.threads != new.threads
            || old.load_timeout_secs != new.load_timeout_secs,
        restart: old.hotkey != new.hotkey || old.grab != new.grab,
    }
}

/// Reload config from disk and re-apply the deltas live. Called only when Idle.
#[allow(clippy::too_many_arguments)]
fn apply_reload(
    config: &mut Config,
    config_path: Option<&Path>,
    recorder: &mut AudioRecorder,
    typer: &mut Box<dyn Injector>,
    cache: &mut Arc<ModelCache>,
    trailing: &mut Duration,
    ui: &UiHandle,
) {
    let new = match Config::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            warn!("config reload failed: {e:#}");
            ui.set_state(TrayState::Error("Bad config file".into()));
            return;
        }
    };
    let actions = reload_actions(config, &new);
    debug!(?actions, "applying config reload");

    if actions.restart {
        info!("hotkey/grab changed — restarting to apply");
        restart_self();
    }
    if actions.recorder {
        match AudioRecorder::new(&new.audio_device) {
            Ok(r) => *recorder = r,
            Err(e) => warn!("could not switch audio device: {e:#}"),
        }
    }
    if actions.injector {
        *typer = injector::detect(&new);
    }
    if actions.model {
        let c = ModelCache::new(&new);
        c.start_evict_thread();
        *cache = c;
    }
    *trailing = Duration::from_millis(new.trailing_silence_ms);
    *config = new;
    ui.set_state(TrayState::Ready);
}

/// Re-exec the current binary in place. The single-instance flock is CLOEXEC,
/// so it releases on exec and the fresh process re-acquires it. Used as the v1
/// fallback for hotkey/grab changes (§1).
#[cfg(unix)]
fn restart_self() -> ! {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        warn!("current_exe failed, cannot restart: {e}");
        std::process::exit(1);
    });
    let args: Vec<String> = std::env::args().skip(1).collect();
    hotkey::restore_platform();
    let err = std::process::Command::new(exe).args(args).exec();
    // exec() only returns on failure.
    warn!("re-exec failed: {err}");
    std::process::exit(1);
}

#[derive(Debug)]
enum State {
    Idle,
    Recording { clipboard_only: bool },
}

/// Gate, transcribe, post-process, inject. Both gates discard before inference —
/// ASR models hallucinate text on silence, so never feed them empty air. Returns
/// a user-facing error string on failure so the caller can surface it on the tray.
fn handle_utterance(
    cache: &ModelCache,
    samples: &[f32],
    rate: u32,
    config: &Config,
    injector: &mut dyn Injector,
) -> Result<(), String> {
    let ms = samples.len() as f32 / rate as f32 * 1000.0;
    if ms < config.min_speech_ms as f32 {
        debug!(
            "discarded: {ms:.0}ms < min_speech_ms {}",
            config.min_speech_ms
        );
        return Ok(());
    }
    let peak = samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    if peak < 0.01 {
        debug!("discarded: peak {peak:.4} below silence floor");
        return Ok(());
    }

    match cache.transcribe(samples) {
        Ok(raw) => {
            let text = post_process(&raw);
            if text.is_empty() {
                debug!("empty transcription");
                return Ok(());
            }
            if let Err(e) = injector.inject(&text) {
                warn!("injection failed: {e:#}");
                return Err("Text didn't appear in the active app".into());
            }
            Ok(())
        }
        Err(e) => {
            warn!("transcription failed: {e:#}");
            Err("Transcription failed".into())
        }
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

        // CLOEXEC so a self-restart (exec) releases the lock for the fresh
        // process; without it the re-exec'd daemon would deadlock on its own lock.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };

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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn no_change_reloads_nothing() {
        let actions = reload_actions(&cfg(), &cfg());
        assert_eq!(
            actions,
            ReloadActions {
                recorder: false,
                injector: false,
                model: false,
                restart: false,
            }
        );
    }

    #[test]
    fn device_change_rebuilds_recorder_only() {
        let new = Config {
            audio_device: "Headset Mic".into(),
            ..cfg()
        };
        let actions = reload_actions(&cfg(), &new);
        assert!(actions.recorder);
        assert!(!actions.injector && !actions.model && !actions.restart);
    }

    #[test]
    fn injection_change_rebuilds_injector_only() {
        let new = Config {
            injection: "clipboard".into(),
            ..cfg()
        };
        let actions = reload_actions(&cfg(), &new);
        assert!(actions.injector);
        assert!(!actions.recorder && !actions.model && !actions.restart);
    }

    #[test]
    fn model_fields_rebuild_cache() {
        for new in [
            Config {
                model: "moonshine-tiny".into(),
                ..cfg()
            },
            Config { threads: 2, ..cfg() },
            Config {
                load_timeout_secs: 60,
                ..cfg()
            },
        ] {
            assert!(reload_actions(&cfg(), &new).model);
        }
    }

    #[test]
    fn hotkey_or_grab_change_forces_restart() {
        let hk = Config {
            hotkey: "F12".into(),
            ..cfg()
        };
        assert!(reload_actions(&cfg(), &hk).restart);
        let grab = Config {
            grab: !cfg().grab,
            ..cfg()
        };
        assert!(reload_actions(&cfg(), &grab).restart);
    }
}
