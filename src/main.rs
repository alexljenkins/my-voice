mod audio;
mod autostart;
mod config;
mod download;
mod hotkey;
mod injector;
#[cfg(target_os = "linux")]
mod keybind_capture;
mod model_cache;
mod models;
mod notify;
mod text;
mod transcriber;
mod ui;

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

use audio::AudioRecorder;
use config::Config;
use hotkey::{spawn_listener, HotkeyEvent};
use injector::Injector;
use model_cache::ModelCache;
use text::post_process;
use ui::{ModelItem, TrayMenuState, TrayState, UiCommand, UiHandle};

#[cfg(feature = "debug-tools")]
const TEST_WAV: &str = "/tmp/my-voice-test.wav";

#[derive(Parser, Debug)]
#[command(name = "my-voice", version, about = "Hold-to-talk local voice typing.")]
struct Cli {
    /// Fetch the configured model, then exit.
    #[arg(long)]
    download: bool,

    /// Record 3s from the mic, save a wav, print stats, exit (no hotkey).
    #[cfg(feature = "debug-tools")]
    #[arg(long)]
    test: bool,

    /// Transcribe a wav file directly (bypasses the mic), print, exit.
    #[cfg(feature = "debug-tools")]
    #[arg(long, value_name = "PATH")]
    wav: Option<PathBuf>,

    /// Save each recording to <DIR>/<timestamp>.wav (and _raw.wav) while running
    /// the normal hold-to-talk flow. Press Ctrl+C when done collecting samples.
    #[arg(long, value_name = "DIR")]
    record: Option<PathBuf>,

    /// Print audio input device names and exit.
    #[arg(long)]
    list_devices: bool,

    /// Open the key-capture popup, write the chosen hotkey to config, exit.
    /// Spawned as a subprocess by the tray's "Set keybind…"; not for direct use.
    #[cfg(target_os = "linux")]
    #[arg(long, hide = true)]
    set_hotkey: bool,

    /// Alternate config file.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Increase logging: -v = info, -vv = debug.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() {
    let cli = Cli::parse();
    #[cfg(feature = "debug-tools")]
    let debug_invocation = cli.test || cli.wav.is_some();
    #[cfg(not(feature = "debug-tools"))]
    let debug_invocation = false;
    #[cfg(target_os = "linux")]
    let set_hotkey = cli.set_hotkey;
    #[cfg(not(target_os = "linux"))]
    let set_hotkey = false;
    let is_daemon = !cli.download && !debug_invocation && !cli.list_devices && !set_hotkey;
    let _log_guard = init_tracing(cli.verbose, is_daemon);

    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    if cli.list_devices {
        return audio::list_devices();
    }

    #[cfg(target_os = "linux")]
    if cli.set_hotkey {
        return run_set_hotkey(cli.config.as_deref());
    }

    let config = Config::load(cli.config.as_deref())?;
    debug!(?config, "loaded config");

    if cli.download {
        return download::run(&config);
    }

    #[cfg(feature = "debug-tools")]
    if cli.test {
        return run_test(&config);
    }

    #[cfg(feature = "debug-tools")]
    if let Some(path) = cli.wav.as_deref() {
        return run_wav(&config, path);
    }

    run_daemon(config, cli.config, cli.record)
}

/// Subprocess entry for the tray's "Set keybind…": open the capture popup, and
/// if the user commits a key, persist it and exit 0; on cancel exit 10 so the
/// parent daemon knows not to restart. Runs its own (winit) event loop, which is
/// why it's a separate process rather than a thread inside the daemon.
#[cfg(target_os = "linux")]
fn run_set_hotkey(config_path: Option<&Path>) -> Result<()> {
    match keybind_capture::capture()? {
        Some(hotkey) => {
            let mut config = Config::load(config_path)?;
            config.hotkey = hotkey.clone();
            config.save(config_path)?;
            info!("hotkey set to '{hotkey}'");
            std::process::exit(0);
        }
        None => {
            info!("hotkey capture cancelled");
            std::process::exit(10);
        }
    }
}

/// Feed a wav file straight through the transcriber — isolates the inference
/// path from the mic/capture path. Resamples to 16 kHz mono if needed.
#[cfg(feature = "debug-tools")]
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
    let resampled = audio::resample(&mono, spec.sample_rate, 16_000);
    let raw_peak = resampled.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let samples = audio::process_capture(&resampled, 16_000);
    info!(
        "wav: {:.2}s, {} Hz → 16 kHz, {ch} ch → mono, raw peak {raw_peak:.3}, processed {:.2}s",
        mono.len() as f32 / spec.sample_rate as f32,
        spec.sample_rate,
        samples.len() as f32 / 16_000.0
    );
    let mut transcriber = transcriber::create(config)?;
    let text = post_process(&transcriber.transcribe(&samples)?);
    println!("{text}");
    Ok(())
}

/// Record a fixed 3s window, dump a debug wav, transcribe, and print — verifies
/// the full audio→text path without needing hotkey/input permissions.
#[cfg(feature = "debug-tools")]
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

/// A message into the daemon's single event loop. Hotkey listener, tray UI,
/// and background download all feed this one channel.
enum DaemonMsg {
    Hotkey(HotkeyEvent),
    Ui(UiCommand),
    DownloadProgress(u8),
    DownloadComplete,
    DownloadFailed(String),
    /// The keybind-capture subprocess committed a new hotkey to disk.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    HotkeyCaptured,
}

fn run_daemon(
    mut config: Config,
    config_path: Option<PathBuf>,
    record_dir: Option<PathBuf>,
) -> Result<()> {
    let _lock = match single_instance::acquire() {
        Ok(l) => l,
        Err(e) => {
            notify::send(
                "Already running",
                "my-voice is already running. Find it in the menu bar.",
            );
            return Err(e);
        }
    };

    notify::init();

    if let Some(ref dir) = record_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating record dir {dir:?}"))?;
        info!("recording mode: saving WAVs to {}", dir.display());
    }

    let mut cache = ModelCache::new(&config);
    cache.start_evict_thread();
    let mut recorder = match AudioRecorder::new(&config.audio_device) {
        Ok(r) => r,
        Err(e) => {
            notify::once(
                notify::ErrorKind::NoMicrophone,
                "No microphone found",
                "my-voice can't find a microphone. Check that one is plugged in.",
            );
            return Err(e.context("no microphone"));
        }
    };
    let mut typer = injector::detect(&config);
    let mut clipper = injector::clipboard();

    // Enumerate input devices once at startup for the tray mic submenu.
    let audio_devices = audio::input_devices();

    #[cfg(unix)]
    install_signal_handlers();

    // One channel, two producers. The hotkey listener and tray each get their
    // own typed sender, forwarded into the merged stream the loop drains.
    let (daemon_tx, daemon_rx) = mpsc::channel::<DaemonMsg>();

    let (hk_tx, hk_rx) = mpsc::channel::<HotkeyEvent>();
    if let Err(e) = spawn_listener(&config, hk_tx) {
        #[cfg(target_os = "linux")]
        notify::once(
            notify::ErrorKind::HotkeySetupNeeded,
            "Hotkey setup needed",
            "Your desktop doesn't support automatic hotkey registration. \
             Run in Terminal: sudo usermod -aG input $USER — then log out and back in.",
        );
        return Err(e.context("hotkey listener failed"));
    }
    forward(hk_rx, daemon_tx.clone(), DaemonMsg::Hotkey);

    let (ui_tx, ui_rx) = mpsc::channel::<UiCommand>();
    let ui = ui::spawn(ui_tx);
    forward(ui_rx, daemon_tx.clone(), DaemonMsg::Ui);

    info!("ready — hold '{}' to record", config.hotkey);
    ui.set_state(TrayState::Ready);
    ui.set_menu(build_tray_menu(&config, &audio_devices));

    // First-run: if the model files aren't present, start a background download
    // immediately. Hotkey presses during download will surface a transcription
    // error — the tray Downloading state makes the reason obvious.
    if !config.is_model_downloaded() {
        info!("model not found — starting background download");
        notify::once(
            notify::ErrorKind::ModelMissing,
            "Speech model not found",
            "Downloading the speech model now (~50 MB). my-voice will be ready in a moment.",
        );
        let tx = daemon_tx.clone();
        download::start_background(config.clone(), move |event| {
            use download::DownloadEvent::*;
            let msg = match event {
                Progress(pct) => DaemonMsg::DownloadProgress(pct),
                Complete => DaemonMsg::DownloadComplete,
                Failed(e) => DaemonMsg::DownloadFailed(e),
            };
            let _ = tx.send(msg);
        });
        ui.set_state(TrayState::Downloading { pct: 0 });
    } else {
        // Model already present — pre-warm the cache so first keydown has no
        // cold-start latency. 2s delay lets the tray settle before the load.
        let preload = Arc::clone(&cache);
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(2));
            if let Err(e) = preload.ensure_loaded() {
                warn!("startup preload failed: {e:#}");
            }
        });
    }

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
                    let samples = if let Some(ref dir) = record_dir {
                        let (raw, raw_rate, processed) = recorder.stop_with_raw();
                        let ts = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let proc_path = dir.join(format!("{ts}.wav"));
                        let raw_path = dir.join(format!("{ts}_raw.wav"));
                        if let Err(e) = write_wav(
                            &processed,
                            recorder.target_rate(),
                            &proc_path.to_string_lossy(),
                        ) {
                            warn!("record save failed: {e}");
                        } else {
                            println!("saved: {}", proc_path.display());
                        }
                        if let Err(e) = write_wav(&raw, raw_rate, &raw_path.to_string_lossy()) {
                            warn!("record raw save failed: {e}");
                        }
                        processed
                    } else {
                        recorder.stop()
                    };
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
                            &audio_devices,
                        );
                    }
                }
                // Recording+Press (autorepeat dupe) and Idle+Release (stale): ignore.
                _ => {}
            },
            DaemonMsg::DownloadProgress(pct) => {
                ui.set_state(TrayState::Downloading { pct });
            }
            DaemonMsg::DownloadComplete => {
                info!("model download complete");
                ui.set_state(TrayState::Ready);
                ui.set_menu(build_tray_menu(&config, &audio_devices));
                notify::send(
                    "Model ready",
                    "Speech model downloaded. my-voice is ready to use.",
                );
            }
            DaemonMsg::DownloadFailed(e) => {
                warn!("model download failed: {e}");
                ui.set_state(TrayState::Error(
                    "Download failed — check internet connection".into(),
                ));
                notify::once(
                    notify::ErrorKind::ModelDownloadFailed,
                    "Download failed",
                    "Couldn't download the speech model. Check your internet connection \
                     and try again from the my-voice menu.",
                );
            }
            DaemonMsg::Ui(UiCommand::Quit) => {
                info!("quit requested");
                break;
            }

            // "Set keybind…": launch the capture popup as a subprocess. It writes
            // the chosen hotkey to disk and exits; HotkeyCaptured then restarts us.
            DaemonMsg::Ui(UiCommand::CaptureHotkey) => {
                #[cfg(target_os = "linux")]
                spawn_keybind_capture(config_path.clone(), daemon_tx.clone());
            }
            DaemonMsg::HotkeyCaptured => {
                info!("hotkey changed via popup — restarting to apply");
                hotkey::restore_platform();
                restart_self();
            }

            // Start at login is a filesystem side effect (XDG autostart entry),
            // not config — apply immediately and refresh the menu's checkmark.
            DaemonMsg::Ui(UiCommand::SetStartAtLogin(on)) => {
                match autostart::set_enabled(on) {
                    Ok(()) => info!("start at login: {on}"),
                    Err(e) => {
                        warn!("start-at-login toggle failed: {e:#}");
                        ui.set_state(TrayState::Error("Couldn't change start-at-login".into()));
                    }
                }
                ui.set_menu(build_tray_menu(&config, &audio_devices));
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
                    &audio_devices,
                ),
                State::Recording { .. } => pending_reload = true,
            },

            // Settings changes that require a self-restart (grab backend threads
            // can't be torn down live). Write to disk before restarting so the
            // fresh process picks up the new value.
            DaemonMsg::Ui(UiCommand::SetGrab(g)) => {
                let mut updated = config.clone();
                updated.grab = g;
                save_config(&updated, config_path.as_deref());
                hotkey::restore_platform();
                restart_self();
            }

            // Settings changes that apply live (no restart needed). Write the
            // new value to disk so apply_reload detects the diff; defer if
            // mid-utterance, apply immediately otherwise.
            DaemonMsg::Ui(
                cmd @ (UiCommand::SetModel(_)
                | UiCommand::SetAudioDevice(_)
                | UiCommand::SetInjection(_)
                | UiCommand::SetClipboardHotkey(_)),
            ) => {
                let mut updated = config.clone();
                match cmd {
                    UiCommand::SetModel(m) => updated.model = m,
                    UiCommand::SetAudioDevice(d) => updated.audio_device = d,
                    UiCommand::SetInjection(inj) => updated.injection = inj,
                    UiCommand::SetClipboardHotkey(b) => updated.clipboard_hotkey = b,
                    _ => unreachable!(),
                }
                save_config(&updated, config_path.as_deref());
                match state {
                    State::Idle => apply_reload(
                        &mut config,
                        config_path.as_deref(),
                        &mut recorder,
                        &mut typer,
                        &mut cache,
                        &mut trailing,
                        &ui,
                        &audio_devices,
                    ),
                    State::Recording { .. } => pending_reload = true,
                }
            }
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
    audio_devices: &[audio::AudioDevice],
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
        let label = match new.model.as_str() {
            "moonshine-base" => "Switched to moonshine-base. Accuracy improved.".to_string(),
            "moonshine-tiny" => "Switched to moonshine-tiny. Speed improved.".to_string(),
            other => format!("Switched to {other}."),
        };
        notify::send("Model ready", &label);
    }
    *trailing = Duration::from_millis(new.trailing_silence_ms);
    *config = new;
    ui.set_state(TrayState::Ready);
    ui.set_menu(build_tray_menu(config, audio_devices));
}

/// Persist config to disk; logs a warning on failure (non-fatal for the daemon).
fn save_config(config: &Config, path: Option<&Path>) {
    if let Err(e) = config.save(path) {
        warn!("failed to save config: {e:#}");
    }
}

/// Build the tray menu state from the current config and available devices.
fn build_tray_menu(config: &Config, audio_devices: &[audio::AudioDevice]) -> TrayMenuState {
    let model_dir = config.resolved_model_dir();
    let models = models::MODELS
        .iter()
        .map(|spec| {
            let dir = model_dir.join(spec.name);
            let sentinel = if config.quantized {
                spec.sentinel_quantized
            } else {
                spec.sentinel_full
            };
            let downloaded = dir.is_dir() && dir.join(sentinel).exists();
            ModelItem {
                name: spec.name.to_string(),
                label: spec.label.to_string(),
                active: config.model == spec.name,
                downloaded,
            }
        })
        .collect();

    let devices = audio_devices
        .iter()
        .map(|d| ui::DeviceItem {
            value: d.value.clone(),
            label: d.label.clone(),
        })
        .collect();
    let (inject_type_available, inject_unlock_hint) = injector::typing_availability();

    TrayMenuState {
        models,
        audio_devices: devices,
        active_device: config.audio_device.clone(),
        hotkey: config.hotkey.clone(),
        injection: config.injection.clone(),
        inject_type_available,
        inject_unlock_hint,
        grab: config.grab,
        clipboard_hotkey: config.clipboard_hotkey,
        start_at_login: autostart::is_enabled(),
    }
}

/// Spawn the keybind-capture popup as a subprocess and, if it commits a new
/// hotkey (exit 0), signal the daemon to restart so the new hotkey takes effect.
#[cfg(target_os = "linux")]
fn spawn_keybind_capture(config_path: Option<PathBuf>, tx: mpsc::Sender<DaemonMsg>) {
    thread::spawn(move || {
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                warn!("current_exe failed, can't open keybind popup: {e}");
                return;
            }
        };
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("--set-hotkey");
        if let Some(p) = config_path {
            cmd.arg("--config").arg(p);
        }
        match cmd.status() {
            Ok(st) if st.success() => {
                let _ = tx.send(DaemonMsg::HotkeyCaptured);
            }
            Ok(_) => debug!("keybind capture cancelled"),
            Err(e) => warn!("keybind popup failed to launch: {e:#}"),
        }
    });
}

/// Restart the daemon to apply a hotkey/grab change (§1). Spawns a *fresh child
/// process* (new PID) then exits, rather than `exec`-ing in place.
///
/// Why a new process and not `exec`: the Linux tray is a ksni StatusNotifierItem
/// whose D-Bus name is `org.kde.StatusNotifierItem-{PID}-{counter}`, and the
/// per-process counter restarts at 1 each launch. `exec` keeps the same PID, so
/// the fresh image re-registered the *identical* bus name the dying one just
/// held — the tray host then pruned it as a stale duplicate and the icon vanished
/// even though the daemon was running. A new PID yields a new SNI name, so the
/// host shows it as a genuinely new item.
///
/// The child is launched with `MY_VOICE_RESTART=1` so its single-instance lock
/// acquire retries briefly: parent and child overlap for the few ms until the
/// parent exits and releases the flock.
#[cfg(unix)]
fn restart_self() -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        warn!("current_exe failed, cannot restart: {e}");
        std::process::exit(1);
    });
    let args: Vec<String> = std::env::args().skip(1).collect();
    hotkey::restore_platform();
    match std::process::Command::new(exe)
        .args(args)
        .env("MY_VOICE_RESTART", "1")
        .spawn()
    {
        // Parent exits immediately so its fds (evdev grab, D-Bus connection, the
        // flock) close before the child finishes its longer startup and grabs.
        Ok(_) => std::process::exit(0),
        Err(e) => {
            warn!("respawn failed: {e}");
            std::process::exit(1);
        }
    }
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
                notify::once(
                    notify::ErrorKind::InjectionFailed,
                    "Text not appearing?",
                    "my-voice couldn't type into the active app. Try switching to \
                     clipboard mode in the my-voice menu, then paste with Ctrl+V.",
                );
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

fn init_tracing(verbose: u8, daemon: bool) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let make_filter = || {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let level = match verbose {
                0 => "warn",
                1 => "info",
                _ => "debug",
            };
            EnvFilter::new(format!("my_voice={level}"))
        })
    };

    if daemon {
        use tracing_subscriber::layer::SubscriberExt as _;
        use tracing_subscriber::util::SubscriberInitExt as _;
        use tracing_subscriber::Layer as _;

        let log_dir = dirs::state_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".local/state"))
            .join("my-voice");
        let _ = std::fs::create_dir_all(&log_dir);
        let file_appender = tracing_appender::rolling::never(&log_dir, "my-voice.log");
        let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_writer(std::io::stderr)
                    .with_filter(make_filter()),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_ansi(false)
                    .with_writer(file_writer)
                    .with_filter(make_filter()),
            )
            .init();

        Some(guard)
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(make_filter())
            .with_target(false)
            .with_writer(std::io::stderr)
            .init();
        None
    }
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

    use anyhow::Result;

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
        // A self-restart (hotkey/grab change) spawns the fresh process *before*
        // the old one exits, so the two briefly overlap. The child is launched
        // with MY_VOICE_RESTART=1; in that case retry the lock for a short window
        // to let the parent exit and release it, rather than failing as a dupe.
        let restarting = std::env::var_os("MY_VOICE_RESTART").is_some();
        std::env::remove_var("MY_VOICE_RESTART");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match try_acquire()? {
                Some(guard) => return Ok(guard),
                None => {
                    if restarting && std::time::Instant::now() < deadline {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    }
                    return Err(already_running());
                }
            }
        }
    }

    /// One non-blocking acquire attempt. `Ok(Some)` = held, `Ok(None)` = the lock
    /// is busy (another instance holds it), `Err` = a hard filesystem error.
    fn try_acquire() -> Result<Option<Guard>> {
        let path = lock_path();
        let mut file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)?;

        // CLOEXEC so a restart's spawned child doesn't inherit this fd (it opens
        // its own), and so any subprocess we launch can't hold the lock open.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };

        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Ok(None);
            }
            return Err(err.into());
        }

        // We hold the lock: record our pid.
        let _ = file.set_len(0);
        let _ = writeln!(file, "{}", std::process::id());
        Ok(Some(Guard { _file: file }))
    }

    /// Build the user-facing "already running" error, naming the holding pid if
    /// the lock file records one.
    fn already_running() -> anyhow::Error {
        let mut existing = String::new();
        if let Ok(mut file) = File::open(lock_path()) {
            let _ = file.read_to_string(&mut existing);
        }
        let pid = existing.trim();
        if pid.is_empty() {
            anyhow::anyhow!("my-voice is already running")
        } else {
            anyhow::anyhow!("my-voice is already running (pid {pid})")
        }
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
                model: "moonshine-base".into(),
                ..cfg()
            },
            Config {
                threads: 2,
                ..cfg()
            },
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
