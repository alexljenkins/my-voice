#[cfg(not(target_os = "linux"))]
compile_error!("my-voice supports Linux only");

mod audio;
mod autostart;
mod config;
mod download;
mod hotkey;
mod indicator;
mod injector;
mod keybind_capture;
mod model_cache;
mod models;
mod notify;
mod text;
mod transcriber;
mod ui;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

use audio::{AudioRecorder, DrainedSegment};
use config::Config;
use hotkey::{spawn_listener, HotkeyEvent};
use injector::{DeliveryMode, Injector};
use model_cache::ModelCache;
use text::{post_process, BoundaryTextJoiner};
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

    /// With --wav: use the daemon's 200 ms segmentation cadence.
    #[cfg(feature = "debug-tools")]
    #[arg(long, requires = "wav")]
    segmented: bool,

    /// With --wav: transcribe N times warm (model loaded+warmed once) and log
    /// per-iteration encode/decode + peak RSS. For the perf bench, not the gate.
    #[cfg(feature = "debug-tools")]
    #[arg(long, value_name = "N", default_value_t = 1)]
    bench_iters: usize,

    /// Save one raw WAV per hold and append its transcript to <DIR>/expected.txt.
    /// Press Ctrl+C when done collecting samples.
    #[arg(long, value_name = "DIR")]
    record: Option<PathBuf>,

    /// Print audio input device names and exit.
    #[arg(long)]
    list_devices: bool,

    /// Report whether a daemon is running (with its pid + configured model), exit.
    #[arg(long)]
    status: bool,

    /// Print a shell completion script for SHELL to stdout, exit.
    #[arg(long, value_name = "SHELL")]
    completions: Option<clap_complete::Shell>,

    /// Print a roff man page to stdout, exit.
    #[arg(long, hide = true)]
    man: bool,

    /// Open the key-capture popup, write the chosen hotkey to config, exit.
    /// Spawned as a subprocess by the tray's "Set keybind…"; not for direct use.
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
    let set_hotkey = cli.set_hotkey;
    let is_daemon = !cli.download
        && !debug_invocation
        && !cli.list_devices
        && !cli.status
        && cli.completions.is_none()
        && !cli.man
        && !set_hotkey;
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

    if let Some(shell) = cli.completions {
        print_completions(shell);
        return Ok(());
    }

    if cli.man {
        return print_man();
    }

    if cli.set_hotkey {
        return run_set_hotkey(cli.config.as_deref());
    }

    let config = Config::load(cli.config.as_deref())?;

    if cli.status {
        print_status(&config);
        return Ok(());
    }

    debug!(?config, "loaded config");

    if cli.download {
        return download::run(&config);
    }

    // Every path below builds ONNX sessions; commit the shared global thread
    // pool first (env is immutable once a session exists).
    transcriber::init_thread_pool(&config);

    #[cfg(feature = "debug-tools")]
    if cli.test {
        return run_test(&config);
    }

    #[cfg(feature = "debug-tools")]
    if let Some(path) = cli.wav.as_deref() {
        return run_wav(&config, path, cli.bench_iters.max(1), cli.segmented);
    }

    run_daemon(config, cli.config, cli.record)
}

/// Emit a shell completion script for the derived `Cli` to stdout. Pure
/// generation — never touches the daemon or the hot path.
fn print_completions(shell: clap_complete::Shell) {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
}

/// Emit a roff man page for the derived `Cli` to stdout (for packaging).
fn print_man() -> Result<()> {
    use clap::CommandFactory;
    clap_mangen::Man::new(Cli::command()).render(&mut std::io::stdout())?;
    Ok(())
}

/// Report whether a daemon is running. The lockfile records the holder's pid on
/// line 1 (single_instance::try_acquire); a `kill(pid, 0)` confirms it's alive.
fn print_status(config: &Config) {
    let pid = single_instance::lock_pid();
    let alive = pid.is_some_and(process_alive);
    println!("{}", status_line(pid.filter(|_| alive), &config.model));
}

/// Format the one-line status report. Split out so it's unit-testable without a
/// live daemon: a present pid means running, `None` means not.
fn status_line(pid: Option<i32>, model: &str) -> String {
    match pid {
        Some(pid) => format!("running (pid {pid}), model {model}"),
        None => "not running".to_string(),
    }
}

/// Signal-0 liveness probe: the process exists (or we lack permission to signal
/// a process that does). ESRCH alone means dead.
fn process_alive(pid: i32) -> bool {
    unsafe {
        libc::kill(pid, 0) == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }
}

/// Subprocess entry for the tray's "Set keybind…": open the capture popup, and
/// if the user commits a key, persist it and exit 0; on cancel exit 10 so the
/// parent daemon knows not to restart. Runs its own (winit) event loop, which is
/// why it's a separate process rather than a thread inside the daemon.
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
fn run_wav(config: &Config, path: &std::path::Path, iters: usize, segmented: bool) -> Result<()> {
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
    let samples = (!segmented).then(|| audio::process_capture(&resampled, 16_000));
    info!(
        "wav: {:.2}s, {} Hz → 16 kHz, {ch} ch → mono, raw peak {raw_peak:.3}, processed {:.2}s",
        mono.len() as f32 / spec.sample_rate as f32,
        spec.sample_rate,
        samples.as_ref().map_or(resampled.len(), Vec::len) as f32 / 16_000.0
    );
    // create() loads + warms the model once, so every pass below is warm. The
    // first pass produces the text we print; extra passes (--bench-iters > 1)
    // only re-time the warm steady state, which is what a daemon user feels.
    let mut transcriber = transcriber::create(config)?;
    let mut text = String::new();
    if segmented {
        let mut marked_text = String::new();
        let segments = audio::segment_samples(
            &mono,
            spec.sample_rate,
            config.segment_pause_ms,
            config.segment_max_ms,
        );
        for (index, timed) in segments.iter().enumerate() {
            info!(
                segment_index = index,
                boundary_seconds = format_args!(
                    "{:.3}",
                    timed.boundary_sample as f64 / spec.sample_rate as f64
                ),
                reason = ?timed.segment.reason,
                "segment boundary"
            );
        }
        let observed_speech_ms = segments
            .iter()
            .map(|timed| timed.segment.observed_speech_ms)
            .sum::<u64>();
        for _ in 0..iters {
            text.clear();
            marked_text.clear();
            let mut audio_processor = audio::CaptureProcessor::new(16_000);
            let mut text_joiner = BoundaryTextJoiner::default();
            let mut marked_text_joiner = BoundaryTextJoiner::with_overlap_markers();
            for timed in &segments {
                let resampled = audio::resample(&timed.segment.raw, timed.segment.raw_rate, 16_000);
                let overlap_samples = audio::resampled_sample_count(
                    timed.segment.overlap_samples,
                    timed.segment.raw_rate,
                    16_000,
                );
                let processed = audio_processor.process_with_overlap(&resampled, overlap_samples);
                let peak = processed.iter().fold(0.0f32, |a, b| a.max(b.abs()));
                if timed.segment.observed_speech_ms == 0 || peak < 0.01 {
                    if let Some(chunk) = text_joiner.break_boundary() {
                        append_joined(&mut text, &chunk);
                    }
                    if let Some(chunk) = marked_text_joiner.break_boundary() {
                        append_joined(&mut marked_text, &chunk);
                    }
                    continue;
                }
                let raw_text = match transcriber.transcribe(&processed) {
                    Ok(raw_text) => raw_text,
                    Err(error) => {
                        if let Some(chunk) = text_joiner.break_boundary() {
                            append_joined(&mut text, &chunk);
                        }
                        if let Some(chunk) = marked_text_joiner.break_boundary() {
                            append_joined(&mut marked_text, &chunk);
                        }
                        warn!("segment transcription failed: {error:#}");
                        continue;
                    }
                };
                let segment_text = post_process(&raw_text, &config.corrections);
                if segment_text.is_empty() {
                    if let Some(chunk) = text_joiner.break_boundary() {
                        append_joined(&mut text, &chunk);
                    }
                    if let Some(chunk) = marked_text_joiner.break_boundary() {
                        append_joined(&mut marked_text, &chunk);
                    }
                    continue;
                }
                if let Some(chunk) = text_joiner.push(
                    &segment_text,
                    timed.segment.reason == audio::DrainReason::Release,
                    timed.segment.overlap_samples > 0,
                ) {
                    append_joined(&mut text, &chunk);
                }
                if let Some(chunk) = marked_text_joiner.push(
                    &segment_text,
                    timed.segment.reason == audio::DrainReason::Release,
                    timed.segment.overlap_samples > 0,
                ) {
                    append_joined(&mut marked_text, &chunk);
                }
            }
        }
        if observed_speech_ms < config.min_speech_ms {
            text.clear();
            marked_text.clear();
        }
        info!(marked_transcript = %marked_text, "segmented transcript");
    } else {
        let samples = samples
            .as_deref()
            .expect("single-pass samples are processed");
        for _ in 0..iters {
            text = post_process(&transcriber.transcribe(samples)?, &config.corrections);
        }
    }
    if let Some(kb) = peak_rss_kb() {
        info!("peak RSS {kb} kB");
    }
    println!("{text}");
    Ok(())
}

/// Process peak resident set size (`VmHWM`) in kB, Linux only. The high-water
/// mark over the whole run = model + ONNX arenas + buffers, i.e. the daemon's
/// real memory footprint. Returns None where /proc isn't available.
#[cfg(feature = "debug-tools")]
fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
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
    let text = post_process(&transcriber.transcribe(&samples)?, &config.corrections);
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
    /// The cpal input stream died mid-capture (mic unplugged, server gone).
    AudioFailed(String),
    /// The keybind-capture subprocess committed a new hotkey to disk.
    HotkeyCaptured,
    SegmentComplete(SegmentResult),
}

struct SegmentRequest {
    hold_id: u64,
    segment_index: u32,
    segment: DrainedSegment,
    cache: Arc<ModelCache>,
    corrections: Vec<(String, String)>,
    record_dir: Option<PathBuf>,
}

struct SegmentResult {
    hold_id: u64,
    segment_index: u32,
    final_segment: bool,
    has_audio_overlap: bool,
    text: Result<Option<String>, String>,
}

struct WorkerAudioState {
    hold_id: Option<u64>,
    processor: Option<audio::CaptureProcessor>,
}

impl WorkerAudioState {
    fn new() -> Self {
        Self {
            hold_id: None,
            processor: None,
        }
    }

    fn begin_hold(&mut self, hold_id: u64) -> bool {
        if self.hold_id == Some(hold_id) {
            return false;
        }
        self.hold_id = Some(hold_id);
        self.processor = Some(audio::CaptureProcessor::new(16_000));
        true
    }

    fn process(&mut self, samples: &[f32], overlap_samples: usize) -> Vec<f32> {
        self.processor
            .as_mut()
            .expect("begin_hold must run before audio processing")
            .process_with_overlap(samples, overlap_samples)
    }
}

fn spawn_transcription_worker(tx: mpsc::Sender<DaemonMsg>) -> SyncSender<SegmentRequest> {
    let (request_tx, request_rx) = mpsc::sync_channel::<SegmentRequest>(3);
    thread::spawn(move || {
        let mut audio_state = WorkerAudioState::new();
        for request in request_rx {
            let SegmentRequest {
                hold_id,
                segment_index,
                segment,
                cache,
                corrections,
                record_dir,
            } = request;
            debug!(?segment.reason, hold_id, segment_index, "processing audio segment");
            audio_state.begin_hold(hold_id);
            let final_segment = segment.reason == audio::DrainReason::Release;
            let has_audio_overlap = segment.overlap_samples > 0;
            if segment.observed_speech_ms == 0 {
                let _ = tx.send(DaemonMsg::SegmentComplete(SegmentResult {
                    hold_id,
                    segment_index,
                    final_segment,
                    has_audio_overlap,
                    text: Ok(None),
                }));
                continue;
            }
            let resampled = audio::resample(&segment.raw, segment.raw_rate, 16_000);
            let overlap_samples =
                audio::resampled_sample_count(segment.overlap_samples, segment.raw_rate, 16_000);
            let processed = audio_state.process(&resampled, overlap_samples);
            let peak = processed.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            let text = if peak < 0.01 {
                Ok(None)
            } else {
                let recording = record_dir
                    .as_deref()
                    .map(|dir| save_raw_recording(dir, hold_id, &segment.raw, segment.raw_rate))
                    .transpose();
                recording
                    .and_then(|filename| {
                        let text = cache
                            .transcribe_for_hold(&processed)
                            .map(|raw| post_process(&raw, &corrections))?;
                        if let (Some(dir), Some(filename)) = (record_dir.as_deref(), filename) {
                            append_expected(dir, &filename, &text)?;
                        }
                        Ok(text)
                    })
                    .map(|text| (!text.is_empty()).then_some(text))
                    .map_err(|e| format!("{e:#}"))
            };
            let _ = tx.send(DaemonMsg::SegmentComplete(SegmentResult {
                hold_id,
                segment_index,
                final_segment,
                has_audio_overlap,
                text,
            }));
        }
    });
    request_tx
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

    install_signal_handlers();

    // One channel, two producers. The hotkey listener and tray each get their
    // own typed sender, forwarded into the merged stream the loop drains.
    let (daemon_tx, daemon_rx) = mpsc::channel::<DaemonMsg>();

    register_audio_error_cb(&mut recorder, &daemon_tx);

    let (hk_tx, hk_rx) = mpsc::channel::<HotkeyEvent>();
    if let Err(e) = spawn_listener(&config, hk_tx) {
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
    let indicator = indicator::spawn(recorder.visual_signal());

    info!("ready — hold '{}' to record", config.hotkey);
    ui.set_state(TrayState::Ready);
    ui.set_menu(build_tray_menu(&config, &audio_devices));

    // First-run: if the model files aren't present, start a background download
    // immediately. Hotkey presses during download will surface a transcription
    // error — the tray Downloading state makes the reason obvious.
    if !config.is_model_downloaded() {
        info!("model not found — starting background download");
        let size = match models::find(&config.model) {
            Some(spec) => format!(" (~{} MB)", spec.approx_mb),
            None => String::new(), // custom model path — size unknown
        };
        notify::once(
            notify::ErrorKind::ModelMissing,
            "Speech model not found",
            &format!("Downloading the speech model now{size}. my-voice will be ready in a moment."),
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
    let segment_tx = spawn_transcription_worker(daemon_tx.clone());
    let mut next_hold_id = 1u64;

    loop {
        let received = match state {
            State::Idle => daemon_rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            State::Recording(_) => daemon_rx.recv_timeout(Duration::from_millis(200)),
        };
        let msg = match received {
            Ok(msg) => msg,
            Err(RecvTimeoutError::Timeout) => {
                if let State::Recording(hold) = &mut state {
                    if let Some(pending) = hold.pending_drain.take() {
                        match queue_segment(
                            &segment_tx,
                            hold,
                            pending,
                            &cache,
                            &config,
                            &record_dir,
                        ) {
                            Ok(()) => {}
                            Err(pending) => {
                                hold.pending_drain = Some(pending);
                                if hold.released {
                                    debug!("waiting for transcription queue space");
                                } else {
                                    warn!("transcription queue remained full; stopping hold");
                                    recorder.cancel();
                                    indicator.hide();
                                    hold.released = true;
                                    ui.set_state(TrayState::Error(
                                        "Transcription can't keep up".into(),
                                    ));
                                }
                            }
                        }
                    } else if !hold.released && record_dir.is_none() {
                        if let Some(segment) = recorder
                            .try_drain_segment(config.segment_pause_ms, config.segment_max_ms)
                        {
                            if let Err(segment) = queue_segment(
                                &segment_tx,
                                hold,
                                segment,
                                &cache,
                                &config,
                                &record_dir,
                            ) {
                                warn!("transcription queue full; retaining one segment");
                                hold.pending_drain = Some(segment);
                            }
                        }
                    }
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match msg {
            DaemonMsg::Hotkey(event) => match (&state, event) {
                (State::Idle, HotkeyEvent::Press { clipboard_only }) => {
                    if let Err(e) = recorder.start() {
                        warn!("could not start recording: {e}");
                        ui.set_state(TrayState::Error("Couldn't start recording".into()));
                        continue;
                    }
                    indicator.show(indicator::choose_style(
                        config.indicator_style,
                        next_hold_id,
                    ));
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
                    state = State::Recording(HoldState {
                        hold_id: next_hold_id,
                        clipboard_only,
                        next_segment: 0,
                        pending_segments: 0,
                        observed_speech_ms: 0,
                        deferred_text: Vec::new(),
                        accumulated_text: String::new(),
                        delivered_any: false,
                        clipboard_deferred: false,
                        delivery_failed: false,
                        released: false,
                        audio_error: false,
                        pending_drain: None,
                        text_joiner: BoundaryTextJoiner::default(),
                    });
                    next_hold_id += 1;
                }
                (State::Recording(_), HotkeyEvent::Release) => {
                    indicator.hide();
                    // PTT trailing buffer: catch the tail of the last word.
                    thread::sleep(trailing);
                    ui.set_state(TrayState::Transcribing);
                    let segment = recorder.stop_raw();
                    if let State::Recording(hold) = &mut state {
                        hold.released = true;
                        let segment = match hold.pending_drain.take() {
                            Some(pending) => merge_release_segment(pending, segment),
                            None => segment,
                        };
                        if let Err(segment) =
                            queue_segment(&segment_tx, hold, segment, &cache, &config, &record_dir)
                        {
                            hold.pending_drain = Some(segment);
                        }
                    }
                }
                // Recording+Press (autorepeat dupe) and Idle+Release (stale): ignore.
                _ => {}
            },
            DaemonMsg::AudioFailed(e) => {
                indicator.hide();
                warn!("audio stream failed: {e}");
                if let State::Recording(hold) = &mut state {
                    recorder.cancel();
                    hold.released = true;
                    hold.audio_error = true;
                    ui.set_state(TrayState::Error("Microphone disconnected".into()));
                    notify::send(
                        "Microphone disconnected",
                        "Recording cancelled — the audio stream reported an error.",
                    );
                    if hold.pending_segments == 0 {
                        cache.finish_hold();
                        state = State::Idle;
                    }
                }
            }
            DaemonMsg::SegmentComplete(result) => {
                let mut finished = false;
                if let State::Recording(hold) = &mut state {
                    if result.hold_id != hold.hold_id {
                        continue;
                    }
                    debug!(segment_index = result.segment_index, "segment complete");
                    hold.pending_segments = hold.pending_segments.saturating_sub(1);
                    match result.text {
                        Ok(Some(text)) if hold.observed_speech_ms >= config.min_speech_ms => {
                            let deferred = std::mem::take(&mut hold.deferred_text);
                            for prior in deferred {
                                deliver_text(hold, &prior, typer.as_mut(), clipper.as_mut());
                            }
                            if let Some(chunk) = hold.text_joiner.push(
                                &text,
                                result.final_segment,
                                result.has_audio_overlap,
                            ) {
                                deliver_text(hold, &chunk, typer.as_mut(), clipper.as_mut());
                            }
                        }
                        Ok(Some(text)) => {
                            if let Some(chunk) = hold.text_joiner.push(
                                &text,
                                result.final_segment,
                                result.has_audio_overlap,
                            ) {
                                hold.deferred_text.push(chunk);
                            }
                        }
                        Ok(None) => flush_pending_boundary(
                            hold,
                            config.min_speech_ms,
                            typer.as_mut(),
                            clipper.as_mut(),
                        ),
                        Err(e) => {
                            flush_pending_boundary(
                                hold,
                                config.min_speech_ms,
                                typer.as_mut(),
                                clipper.as_mut(),
                            );
                            warn!("segment transcription failed: {e}");
                            hold.delivery_failed = true;
                        }
                    }
                    finished =
                        hold.released && hold.pending_segments == 0 && hold.pending_drain.is_none();
                    if finished && hold.clipboard_deferred && !hold.accumulated_text.is_empty() {
                        if let Err(e) = clipper.inject(&hold.accumulated_text) {
                            warn!("final clipboard fallback failed: {e:#}");
                        }
                    }
                    if finished && hold.observed_speech_ms < config.min_speech_ms {
                        hold.deferred_text.clear();
                    }
                }
                if finished {
                    let failed =
                        matches!(&state, State::Recording(h) if h.delivery_failed || h.audio_error);
                    cache.finish_hold();
                    state = State::Idle;
                    if !failed {
                        ui.set_state(TrayState::Ready);
                    }
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
                            &daemon_tx,
                        );
                    }
                }
            }
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
                spawn_keybind_capture(config_path.clone(), daemon_tx.clone());
            }
            DaemonMsg::HotkeyCaptured => {
                info!("hotkey changed via popup — restarting to apply");
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
                    &daemon_tx,
                ),
                State::Recording(_) => pending_reload = true,
            },

            // Settings changes that require a self-restart (grab backend threads
            // can't be torn down live). Write to disk before restarting so the
            // fresh process picks up the new value.
            DaemonMsg::Ui(UiCommand::SetGrab(g)) => {
                let mut updated = config.clone();
                updated.grab = g;
                save_config(&updated, config_path.as_deref());
                restart_self();
            }

            // Settings changes that apply live (no restart needed). Write the
            // new value to disk so apply_reload detects the diff; defer if
            // mid-utterance, apply immediately otherwise.
            DaemonMsg::Ui(
                cmd @ (UiCommand::SetModel(_)
                | UiCommand::SetAudioDevice(_)
                | UiCommand::SetInjection(_)
                | UiCommand::SetClipboardHotkey(_)
                | UiCommand::SetIndicatorStyle(_)),
            ) => {
                let mut updated = config.clone();
                match cmd {
                    UiCommand::SetModel(m) => updated.model = m,
                    UiCommand::SetAudioDevice(d) => updated.audio_device = d,
                    UiCommand::SetInjection(inj) => updated.injection = inj,
                    UiCommand::SetClipboardHotkey(b) => updated.clipboard_hotkey = b,
                    UiCommand::SetIndicatorStyle(style) => updated.indicator_style = style,
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
                        &daemon_tx,
                    ),
                    State::Recording(_) => pending_reload = true,
                }
            }
        }
    }

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
    daemon_tx: &mpsc::Sender<DaemonMsg>,
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
            Ok(mut r) => {
                r.use_visual_signal(recorder.visual_signal());
                *recorder = r;
                register_audio_error_cb(recorder, daemon_tx);
            }
            Err(e) => warn!("could not switch audio device: {e:#}"),
        }
    }
    if actions.injector {
        *typer = injector::detect(&new);
    }
    if actions.model {
        cache.shutdown(); // stop the old evict thread so the old cache + model RAM drop now
        let c = ModelCache::new(&new);
        c.start_evict_thread();
        *cache = c;
        let label: &str = match new.model.as_str() {
            "moonshine-tiny" => "Switched to moonshine-tiny. Fastest, smallest download.",
            "moonshine-base" => "Switched to moonshine-base. Good balance of speed and accuracy.",
            "moonshine-streaming-medium" => {
                "Switched to moonshine-streaming-medium. Highest accuracy."
            }
            _ => "Model switched.",
        };
        notify::send("Model ready", label);
    }
    *trailing = Duration::from_millis(new.trailing_silence_ms);
    *config = new;
    ui.set_state(TrayState::Ready);
    ui.set_menu(build_tray_menu(config, audio_devices));
}

/// Wire the recorder's stream-error callback into the daemon channel so a dying
/// stream cancels the in-flight recording instead of transcribing garbage.
fn register_audio_error_cb(recorder: &mut AudioRecorder, daemon_tx: &mpsc::Sender<DaemonMsg>) {
    let tx = daemon_tx.clone();
    recorder.on_error(move |e| {
        let _ = tx.send(DaemonMsg::AudioFailed(e));
    });
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
        indicator_style: config.indicator_style,
        start_at_login: autostart::is_enabled(),
    }
}

/// Spawn the keybind-capture popup as a subprocess and, if it commits a new
/// hotkey (exit 0), signal the daemon to restart so the new hotkey takes effect.
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
fn restart_self() -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        warn!("current_exe failed, cannot restart: {e}");
        std::process::exit(1);
    });
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    Recording(HoldState),
}

#[derive(Debug)]
struct HoldState {
    hold_id: u64,
    clipboard_only: bool,
    next_segment: u32,
    pending_segments: usize,
    observed_speech_ms: u64,
    deferred_text: Vec<String>,
    accumulated_text: String,
    delivered_any: bool,
    clipboard_deferred: bool,
    delivery_failed: bool,
    released: bool,
    audio_error: bool,
    pending_drain: Option<DrainedSegment>,
    text_joiner: BoundaryTextJoiner,
}

fn append_joined(target: &mut String, text: &str) {
    if !target.is_empty() {
        target.push(' ');
    }
    target.push_str(text);
}

fn flush_pending_boundary(
    hold: &mut HoldState,
    min_speech_ms: u64,
    typer: &mut dyn Injector,
    clipper: &mut dyn Injector,
) {
    let Some(chunk) = hold.text_joiner.break_boundary() else {
        return;
    };
    if hold.observed_speech_ms < min_speech_ms {
        hold.deferred_text.push(chunk);
        return;
    }
    let deferred = std::mem::take(&mut hold.deferred_text);
    for prior in deferred {
        deliver_text(hold, &prior, typer, clipper);
    }
    deliver_text(hold, &chunk, typer, clipper);
}

/// Preserve the final live audio when a previously drained segment is waiting
/// for queue space. Both buffers are contiguous samples from the same stream.
fn merge_release_segment(mut pending: DrainedSegment, tail: DrainedSegment) -> DrainedSegment {
    debug_assert_eq!(pending.raw_rate, tail.raw_rate);
    pending
        .raw
        .extend_from_slice(&tail.raw[tail.overlap_samples.min(tail.raw.len())..]);
    pending.observed_speech_ms = pending
        .observed_speech_ms
        .saturating_add(tail.observed_speech_ms);
    pending.reason = audio::DrainReason::Release;
    pending
}

fn deliver_text(
    hold: &mut HoldState,
    text: &str,
    typer: &mut dyn Injector,
    clipper: &mut dyn Injector,
) {
    append_joined(&mut hold.accumulated_text, text);
    if hold.delivery_failed || hold.clipboard_deferred {
        return;
    }
    if hold.clipboard_only {
        if let Err(e) = clipper.inject(&hold.accumulated_text) {
            warn!("clipboard delivery failed: {e:#}");
            hold.delivery_failed = true;
        }
        return;
    }
    let chunk = if hold.delivered_any {
        format!(" {text}")
    } else {
        text.to_string()
    };
    match typer.inject_effective(&chunk) {
        Ok(DeliveryMode::Typed) => hold.delivered_any = true,
        Ok(DeliveryMode::Clipboard) => hold.clipboard_deferred = true,
        Err(e) => {
            warn!("injection failed: {e:#}");
            notify::once(
                notify::ErrorKind::InjectionFailed,
                "Text not appearing?",
                "my-voice couldn't type into the active app. Switch to clipboard mode and paste with Ctrl+V.",
            );
            hold.delivery_failed = true;
        }
    }
}

fn queue_segment(
    tx: &SyncSender<SegmentRequest>,
    hold: &mut HoldState,
    segment: DrainedSegment,
    cache: &Arc<ModelCache>,
    config: &Config,
    record_dir: &Option<PathBuf>,
) -> std::result::Result<(), DrainedSegment> {
    let index = hold.next_segment;
    let speech = segment.observed_speech_ms;
    let request = SegmentRequest {
        hold_id: hold.hold_id,
        segment_index: index,
        segment,
        cache: Arc::clone(cache),
        corrections: config.corrections.clone(),
        record_dir: record_dir.clone(),
    };
    match tx.try_send(request) {
        Ok(()) => {
            hold.next_segment += 1;
            hold.pending_segments += 1;
            hold.observed_speech_ms += speech;
            Ok(())
        }
        Err(TrySendError::Full(request)) => Err(request.segment),
        Err(TrySendError::Disconnected(request)) => Err(request.segment),
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

fn save_raw_recording(dir: &Path, hold_id: u64, raw: &[f32], raw_rate: u32) -> Result<String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    let filename = format!("{timestamp}_{hold_id}_raw.wav");
    let path = dir.join(&filename);
    write_wav(raw, raw_rate, &path.to_string_lossy())?;
    Ok(filename)
}

fn append_expected(dir: &Path, filename: &str, transcript: &str) -> Result<()> {
    let transcript = transcript.replace(['\t', '\r', '\n'], " ");
    let expected_path = dir.join("expected.txt");
    let mut expected = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&expected_path)
        .with_context(|| format!("opening {}", expected_path.display()))?;
    writeln!(expected, "{filename}\t{transcript}")
        .with_context(|| format!("writing {}", expected_path.display()))?;
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

/// On any clean exit, let the OS close file descriptors.
fn install_signal_handlers() {
    extern "C" fn handler(_sig: libc::c_int) {
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

    /// The pid recorded on line 1 of the lockfile, if the file exists and parses.
    /// Mirrors `already_running()`'s `trim()` parser — line 1 stays a bare pid.
    pub fn lock_pid() -> Option<i32> {
        let mut existing = String::new();
        File::open(lock_path())
            .ok()?
            .read_to_string(&mut existing)
            .ok()?;
        existing.lines().next()?.trim().parse().ok()
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

    struct RecordingInjector {
        injected: Vec<String>,
        mode: DeliveryMode,
    }

    impl RecordingInjector {
        fn typed() -> Self {
            Self {
                injected: Vec::new(),
                mode: DeliveryMode::Typed,
            }
        }

        fn clipboard() -> Self {
            Self {
                injected: Vec::new(),
                mode: DeliveryMode::Clipboard,
            }
        }
    }

    impl Injector for RecordingInjector {
        fn inject(&mut self, text: &str) -> Result<()> {
            self.injected.push(text.to_owned());
            Ok(())
        }

        fn delivery_mode(&self) -> DeliveryMode {
            self.mode
        }

        fn name(&self) -> &'static str {
            "recording"
        }
    }

    fn hold_state() -> HoldState {
        HoldState {
            hold_id: 1,
            clipboard_only: false,
            next_segment: 0,
            pending_segments: 0,
            observed_speech_ms: 0,
            deferred_text: Vec::new(),
            accumulated_text: String::new(),
            delivered_any: false,
            clipboard_deferred: false,
            delivery_failed: false,
            released: false,
            audio_error: false,
            pending_drain: None,
            text_joiner: BoundaryTextJoiner::default(),
        }
    }

    #[test]
    fn worker_audio_state_resets_only_between_holds() {
        let mut state = WorkerAudioState::new();

        assert!(state.begin_hold(1));
        assert!(!state.begin_hold(1));
        assert!(state.begin_hold(2));
        assert_eq!(state.hold_id, Some(2));
    }

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
    fn typed_segments_are_incremental_and_joined_once() {
        let mut hold = hold_state();
        let mut typer = RecordingInjector::typed();
        let mut clipper = RecordingInjector::clipboard();

        deliver_text(&mut hold, "first phrase", &mut typer, &mut clipper);
        deliver_text(&mut hold, "second phrase", &mut typer, &mut clipper);

        assert_eq!(typer.injected, ["first phrase", " second phrase"]);
        assert!(clipper.injected.is_empty());
        assert_eq!(hold.accumulated_text, "first phrase second phrase");
    }

    #[test]
    fn silent_release_segment_is_not_processed_or_saved() {
        let dir =
            std::env::temp_dir().join(format!("my-voice-silent-segment-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = Config::default();
        let cache = ModelCache::new(&config);
        let (daemon_tx, daemon_rx) = mpsc::channel();
        let segment_tx = spawn_transcription_worker(daemon_tx);

        segment_tx
            .send(SegmentRequest {
                hold_id: 7,
                segment_index: 1,
                segment: DrainedSegment {
                    raw: vec![0.0; 3_200],
                    raw_rate: 16_000,
                    observed_speech_ms: 0,
                    reason: audio::DrainReason::Release,
                    overlap_samples: 0,
                },
                cache,
                corrections: Vec::new(),
                record_dir: Some(dir.clone()),
            })
            .unwrap();

        let DaemonMsg::SegmentComplete(result) = daemon_rx.recv().unwrap() else {
            panic!("expected segment completion");
        };
        assert_eq!(result.hold_id, 7);
        assert_eq!(result.segment_index, 1);
        assert!(matches!(result.text, Ok(None)));
        assert!(!dir.join("7_0001_raw.wav").exists());
        assert!(!dir.join("7_0001.wav").exists());

        drop(segment_tx);
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn recording_writes_one_raw_wav_and_one_expected_line() {
        let dir = std::env::temp_dir().join(format!("my-voice-recording-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let filename = save_raw_recording(&dir, 42, &[0.25, -0.25], 16_000).unwrap();
        let wav = dir.join(&filename);
        assert!(!dir.join("expected.txt").exists());
        append_expected(&dir, &filename, "text with spaces").unwrap();

        let wav_files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| {
                let path = entry.unwrap().path();
                (path.extension().and_then(|value| value.to_str()) == Some("wav")).then_some(path)
            })
            .collect();
        assert_eq!(wav_files, std::slice::from_ref(&wav));
        assert!(wav
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("_42_raw.wav"));
        let expected = std::fs::read_to_string(dir.join("expected.txt")).unwrap();
        assert_eq!(
            expected,
            format!(
                "{}\ttext with spaces\n",
                wav.file_name().unwrap().to_string_lossy()
            )
        );

        std::fs::remove_file(wav).unwrap();
        std::fs::remove_file(dir.join("expected.txt")).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn release_merge_keeps_all_captured_audio() {
        let pending = DrainedSegment {
            raw: vec![1.0, 2.0],
            raw_rate: 16_000,
            observed_speech_ms: 200,
            reason: audio::DrainReason::Pause,
            overlap_samples: 0,
        };
        let tail = DrainedSegment {
            raw: vec![2.0, 3.0, 4.0],
            raw_rate: 16_000,
            observed_speech_ms: 100,
            reason: audio::DrainReason::Release,
            overlap_samples: 1,
        };

        let merged = merge_release_segment(pending, tail);

        assert_eq!(merged.raw, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(merged.observed_speech_ms, 300);
        assert_eq!(merged.reason, audio::DrainReason::Release);
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
                model: "moonshine-streaming-medium".into(),
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

    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn status_line_reflects_liveness() {
        assert_eq!(
            status_line(Some(123), "moonshine-base"),
            "running (pid 123), model moonshine-base"
        );
        assert_eq!(status_line(None, "moonshine-base"), "not running");
    }

    #[test]
    fn completions_and_man_generate_nonempty() {
        use clap::CommandFactory;
        let mut bash = Vec::new();
        clap_complete::generate(
            clap_complete::Shell::Bash,
            &mut Cli::command(),
            "my-voice",
            &mut bash,
        );
        assert!(bash.windows(8).any(|w| w == b"my-voice"));

        let mut man = Vec::new();
        clap_mangen::Man::new(Cli::command())
            .render(&mut man)
            .unwrap();
        assert!(man.windows(8).any(|w| w == b"my-voice"));
    }
}
