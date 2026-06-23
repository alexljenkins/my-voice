//! AudioRecorder: cpal input stream, mono downmix, sinc/FFT resample to 16 kHz.
//! Pipeline: native-rate capture → rubato FFT resample → WebRTC APM → peak normalize → VAD silence trim.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat};
use sonora::config::{AdaptiveDigital, GainController2, NoiseSuppression, NoiseSuppressionLevel};
use sonora::{AudioProcessing, Config, StreamConfig};
use tracing::{debug, error, info};

const TARGET_RATE: u32 = 16_000;
const MAX_SECONDS: usize = 60;

pub struct AudioRecorder {
    device: cpal::Device,
    sample_format: SampleFormat,
    channels: usize,
    sample_rate: u32,
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    /// Invoked (from cpal's error callback thread) when the stream dies
    /// mid-capture — e.g. the microphone is unplugged.
    error_cb: Option<Arc<dyn Fn(String) + Send + Sync>>,
}

impl AudioRecorder {
    /// Pick the input device (substring match on `audio_device`, else default)
    /// and cache its native config. Does not open the stream yet.
    pub fn new(audio_device: &str) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_device(&host, audio_device)?;
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());

        let (sample_format, channels, sample_rate) = select_stream_config(&device)?;
        info!("audio device: {name} ({sample_rate} Hz, {channels} ch, {sample_format:?})");

        let cap = sample_rate as usize * MAX_SECONDS;
        Ok(Self {
            device,
            sample_format,
            channels,
            sample_rate,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(cap.min(16_000 * 60)))),
            stream: None,
            error_cb: None,
        })
    }

    /// Register a callback fired when the input stream reports a fatal error
    /// (device unplugged, server died). Called from cpal's callback thread.
    pub fn on_error(&mut self, cb: impl Fn(String) + Send + Sync + 'static) {
        self.error_cb = Some(Arc::new(cb));
    }

    /// Open the input stream and begin appending mono samples to the buffer.
    pub fn start(&mut self) -> Result<()> {
        lock_buf(&self.buffer).clear();

        let config = cpal::StreamConfig {
            channels: self.channels as u16,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let channels = self.channels;
        let cap = self.sample_rate as usize * MAX_SECONDS;
        let buf = self.buffer.clone();
        let cb = self.error_cb.clone();
        let err_fn = move |e: cpal::StreamError| {
            error!("audio stream error: {e}");
            if let Some(cb) = &cb {
                cb(e.to_string());
            }
        };

        let stream = match self.sample_format {
            SampleFormat::F32 => self.device.build_input_stream(
                &config,
                move |data: &[f32], _| append_mono(&buf, data, channels, cap),
                err_fn,
                None,
            ),
            SampleFormat::I16 => self.device.build_input_stream(
                &config,
                move |data: &[i16], _| append_mono(&buf, data, channels, cap),
                err_fn,
                None,
            ),
            SampleFormat::U16 => self.device.build_input_stream(
                &config,
                move |data: &[u16], _| append_mono(&buf, data, channels, cap),
                err_fn,
                None,
            ),
            SampleFormat::U8 => self.device.build_input_stream(
                &config,
                move |data: &[u8], _| append_mono(&buf, data, channels, cap),
                err_fn,
                None,
            ),
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        }
        .context("building input stream")?;

        stream.play().context("starting input stream")?;
        self.stream = Some(stream);
        debug!("recording started");
        Ok(())
    }

    /// Stop the stream and discard whatever was captured (no processing).
    pub fn cancel(&mut self) {
        self.stream = None;
        lock_buf(&self.buffer).clear();
    }

    /// Stop the stream and return 16 kHz mono f32 samples in [-1, 1].
    pub fn stop(&mut self) -> Vec<f32> {
        let (_, _, processed) = self.stop_with_raw();
        processed
    }

    /// Stop the stream and return both the raw native-rate mono samples and the
    /// fully processed 16 kHz samples. The raw buffer is at `self.sample_rate`
    /// and is useful for writing a before/after comparison WAV.
    pub fn stop_with_raw(&mut self) -> (Vec<f32>, u32, Vec<f32>) {
        self.stream = None; // drop → stop the stream
        let raw = std::mem::take(&mut *lock_buf(&self.buffer));
        let raw_peak = raw.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        debug!(
            "captured {} samples at {} Hz ({:.2}s), raw peak {raw_peak:.3}",
            raw.len(),
            self.sample_rate,
            raw.len() as f32 / self.sample_rate as f32
        );
        // APM only supports 8/16/32/48 kHz; resample first to avoid pitch-shifted garbage.
        let resampled = resample(&raw, self.sample_rate, TARGET_RATE);
        let processed = process_capture(&resampled, TARGET_RATE);
        (raw, self.sample_rate, processed)
    }

    pub fn target_rate(&self) -> u32 {
        TARGET_RATE
    }
}

/// §8b: prefer 16 kHz native capture; falls back to the device default if unsupported.
/// Eliminates the resample step entirely on compatible hardware.
fn select_stream_config(device: &cpal::Device) -> Result<(SampleFormat, usize, u32)> {
    if let Ok(mut configs) = device.supported_input_configs() {
        if let Some(cfg) = configs
            .find(|c| c.min_sample_rate().0 <= TARGET_RATE && c.max_sample_rate().0 >= TARGET_RATE)
        {
            debug!("device supports 16 kHz natively — resample step skipped");
            return Ok((cfg.sample_format(), cfg.channels() as usize, TARGET_RATE));
        }
    }
    let default = device
        .default_input_config()
        .context("querying default input config")?;
    Ok((
        default.sample_format(),
        default.channels() as usize,
        default.sample_rate().0,
    ))
}

fn select_device(host: &cpal::Host, wanted: &str) -> Result<cpal::Device> {
    if !wanted.is_empty() {
        let want = wanted.to_lowercase();
        if let Ok(devices) = host.input_devices() {
            for d in devices {
                if let Ok(name) = d.name() {
                    if name.to_lowercase().contains(&want) {
                        return Ok(d);
                    }
                }
            }
        }
        anyhow::bail!("no input device matching '{wanted}' (see --list-devices)");
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("no default input device (see --list-devices)"))
}

/// Lock the sample buffer, recovering from poison. The append closure runs on
/// cpal's realtime callback thread, which swallows panics silently — a single
/// poison would otherwise wedge every later lock and brick capture until
/// restart, the hardest failure for a user to diagnose. The buffer is plain
/// owned data, so recovering it and carrying on is safe.
fn lock_buf(buf: &Mutex<Vec<f32>>) -> MutexGuard<'_, Vec<f32>> {
    buf.lock().unwrap_or_else(|p| p.into_inner())
}

/// Append interleaved samples to the shared buffer, converting to f32 and
/// downmixing to mono by averaging across channels. Caps at `cap` samples.
fn append_mono<T>(buf: &Arc<Mutex<Vec<f32>>>, data: &[T], channels: usize, cap: usize)
where
    T: Sample,
    f32: FromSample<T>,
{
    let mut b = lock_buf(buf);
    let denom = channels.max(1) as f32;
    for frame in data.chunks(channels.max(1)) {
        if b.len() >= cap {
            break;
        }
        let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
        b.push(sum / denom);
    }
}

/// Full capture post-processing chain on 16 kHz mono samples:
/// WebRTC APM (NS + AGC2) → peak normalize → silence trim. The single entry
/// point for both live capture (`stop_with_raw`) and the `--wav` debug path,
/// so offline runs exercise exactly what the mic path produces.
pub fn process_capture(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let mut processed = apply_audio_processing(samples, sample_rate);
    normalize_peak(&mut processed);
    trim_silence(&processed, sample_rate)
}

/// Loudest sample lands here after normalization — leaves ~0.5 dB headroom so
/// nothing clips on the 16-bit wav write / model input.
const NORM_TARGET_PEAK: f32 = 0.95;
/// Cap upward gain so a near-silent capture doesn't amplify the noise floor.
const NORM_MAX_GAIN: f32 = 8.0;

/// Peak-normalize in place: scale the whole buffer so its loudest sample sits at
/// `NORM_TARGET_PEAK`. Pulls APM overshoot (>1.0) back under the clip ceiling and
/// lifts quiet captures to a consistent level. Upward gain is capped.
pub fn normalize_peak(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if peak <= 0.0 {
        return;
    }
    let gain = (NORM_TARGET_PEAK / peak).min(NORM_MAX_GAIN);
    debug!("normalize: peak {peak:.3} → gain {gain:.2}");
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// §8a: polyphase FFT resample via rubato. Falls back to linear on init error.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    match resample_fft(samples, from_rate, to_rate) {
        Ok(v) => v,
        Err(e) => {
            error!("rubato init failed ({e}); falling back to linear resample");
            resample_linear(samples, from_rate, to_rate)
        }
    }
}

fn resample_fft(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedOut, Resampler};

    const OUT_CHUNK: usize = 1600; // 100 ms at 16 kHz
    let mut resampler =
        FftFixedOut::<f32>::new(from_rate as usize, to_rate as usize, OUT_CHUNK, 2, 1)
            .map_err(|e| anyhow!("rubato: {e}"))?;

    let expected = (samples.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
    let mut out = Vec::with_capacity(expected);
    let mut pos = 0;

    while pos < samples.len() {
        let needed = resampler.input_frames_next();
        if pos + needed <= samples.len() {
            let chunk = resampler
                .process(&[&samples[pos..pos + needed]], None)
                .map_err(|e| anyhow!("rubato process: {e}"))?;
            out.extend_from_slice(&chunk[0]);
            pos += needed;
        } else {
            // Tail: zero-pad to full chunk, keep only proportional output frames.
            let remaining = samples.len() - pos;
            let mut padded = vec![0.0f32; needed];
            padded[..remaining].copy_from_slice(&samples[pos..]);
            let chunk = resampler
                .process(&[&padded], None)
                .map_err(|e| anyhow!("rubato tail: {e}"))?;
            let keep = (remaining as f64 * to_rate as f64 / from_rate as f64).ceil() as usize;
            out.extend_from_slice(&chunk[0][..chunk[0].len().min(keep)]);
            break;
        }
    }

    Ok(out)
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        out.push(if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else {
            samples.get(idx).copied().unwrap_or(0.0)
        });
    }
    out
}

/// Run WebRTC audio processing (HPF + NS + AGC2) on mono PCM at `sample_rate`.
///
/// Processes in 10ms frames. Batch post-processing — no latency added during recording.
/// Returns a new buffer of equal length; if the input isn't a multiple of the frame size
/// the tail samples pass through unprocessed (they're silence from the trailing gap).
pub fn apply_audio_processing(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let cfg = Config {
        noise_suppression: Some(NoiseSuppression {
            level: NoiseSuppressionLevel::Moderate,
            ..Default::default()
        }),
        gain_controller2: Some(GainController2 {
            adaptive_digital: Some(AdaptiveDigital {
                headroom_db: 1.0,  // target -1 dBFS; normalize_peak is the real ceiling
                max_gain_db: 12.0, // quiet mics need headroom; NS before AGC limits noise amp
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let stream_cfg = StreamConfig::new(sample_rate, 1);
    let frame_size = stream_cfg.num_frames(); // samples per 10ms
    let mut apm = AudioProcessing::builder()
        .config(cfg)
        .capture_config(stream_cfg)
        .render_config(StreamConfig::new(sample_rate, 1))
        .build();
    let mut out = Vec::with_capacity(samples.len());

    for chunk in samples.chunks(frame_size) {
        if chunk.len() < frame_size {
            // Partial tail: pad to a full frame, process, then take only the real samples.
            let mut padded = chunk.to_vec();
            padded.resize(frame_size, 0.0);
            let mut dest = vec![0.0f32; frame_size];
            let _ = apm.process_capture_f32(&[&padded], &mut [&mut dest]);
            out.extend_from_slice(&dest[..chunk.len()]);
        } else {
            let mut dest = vec![0.0f32; frame_size];
            let _ = apm.process_capture_f32(&[chunk], &mut [&mut dest]);
            out.extend_from_slice(&dest);
        }
    }

    out
}

/// §8d: trim leading/trailing silence using windowed RMS energy.
/// After NS+AGC+normalize, the noise floor is well below SPEECH_RMS so speech
/// frames stand out clearly. Falls back to the full buffer if nothing crosses the
/// threshold (all-silence recordings are handled by the min-speech gate downstream).
fn trim_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    const WINDOW_MS: u32 = 10;
    const SPEECH_RMS: f32 = 0.02;
    const PAD_MS: u32 = 80;
    const MIN_KEEP_MS: u32 = 100;

    if samples.is_empty() {
        return Vec::new();
    }

    let window = (sample_rate * WINDOW_MS / 1000) as usize;
    let pad = (sample_rate * PAD_MS / 1000) as usize;
    let min_keep = (sample_rate * MIN_KEEP_MS / 1000) as usize;

    let speech: Vec<bool> = samples
        .chunks(window.max(1))
        .map(|w| (w.iter().map(|&s| s * s).sum::<f32>() / w.len() as f32).sqrt() > SPEECH_RMS)
        .collect();

    match (
        speech.iter().position(|&s| s),
        speech.iter().rposition(|&s| s),
    ) {
        (Some(f), Some(l)) => {
            let start = (f * window).saturating_sub(pad);
            let end = ((l + 1) * window + pad).min(samples.len());
            if end - start < min_keep {
                return samples.to_vec();
            }
            debug!(
                "silence trim: {:.0}ms → {:.0}ms",
                samples.len() as f32 / sample_rate as f32 * 1000.0,
                (end - start) as f32 / sample_rate as f32 * 1000.0,
            );
            samples[start..end].to_vec()
        }
        _ => samples.to_vec(),
    }
}

/// List input device names to stdout (for `--list-devices`).
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();
    println!("Input devices:");
    for d in host.input_devices().context("enumerating input devices")? {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        let marker = if name == default { " (default)" } else { "" };
        println!("  {name}{marker}");
    }
    Ok(())
}

/// A selectable input device for the tray menu: `value` is matched (substring)
/// against `cpal` device names in `select_device`; `label` is what the user sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDevice {
    pub value: String,
    pub label: String,
}

/// Curated list of input devices for the tray menu. Raw ALSA enumeration is full
/// of plumbing PCMs (`hw:`, `front:`, `surround*`, `dsnoop:`, `sysdefault:`, …);
/// we keep only entries a human would recognise:
///   • one entry per physical sound card, labelled from `/proc/asound/cards`,
///     routed through `plughw:CARD=<id>` (format-converting, most compatible);
///   • the high-level server PCMs (`pipewire`, `pulse`) shown by friendly name.
/// Everything else is dropped. Empty on enumeration failure.
pub fn input_devices() -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let Ok(devices) = host.input_devices() else {
        return Vec::new();
    };
    let names: Vec<String> = devices.filter_map(|d| d.name().ok()).collect();
    let card_names = card_friendly_names();

    let mut out: Vec<AudioDevice> = Vec::new();
    let mut seen_cards: Vec<String> = Vec::new();

    for name in &names {
        // High-level server PCMs: keep, prettify, dedupe.
        if let Some(label) = high_level_label(name) {
            if !out.iter().any(|d| d.label == label) {
                out.push(AudioDevice {
                    value: name.clone(),
                    label,
                });
            }
            continue;
        }
        // ALSA hardware PCM: collapse to one entry per card.
        if let Some(card) = card_id(name) {
            if seen_cards.iter().any(|c| c == card) {
                continue;
            }
            seen_cards.push(card.to_string());
            let label = card_names
                .get(card)
                .cloned()
                .unwrap_or_else(|| card.to_string());
            out.push(AudioDevice {
                value: format!("plughw:CARD={card}"),
                label,
            });
        }
        // Anything else (raw `hw:`, `front:`, `surround*`, `dsnoop:`, …): dropped.
    }
    out
}

/// Friendly server-PCM label for the well-known high-level device names, else None.
fn high_level_label(name: &str) -> Option<String> {
    match name {
        "pipewire" => Some("PipeWire".into()),
        "pulse" => Some("PulseAudio".into()),
        "jack" => Some("JACK".into()),
        _ => None,
    }
}

/// Extract the `CARD=<id>` token from an ALSA PCM name (e.g. `plughw:CARD=PCH,DEV=0`).
fn card_id(name: &str) -> Option<&str> {
    let rest = name.split("CARD=").nth(1)?;
    Some(rest.split([',', ' ']).next().unwrap_or(rest))
}

/// Parse `/proc/asound/cards` into a card-id → friendly-name map. The friendly
/// name is the descriptive tail of each card's first line. Empty off Linux or on
/// read failure (callers fall back to the raw card id).
fn card_friendly_names() -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string("/proc/asound/cards") else {
        return map;
    };
    // Lines look like: ` 1 [Snowball       ]: USB-Audio - Blue Snowball`
    for line in text.lines() {
        let Some(open) = line.find('[') else { continue };
        let Some(close) = line.find(']') else {
            continue;
        };
        if close < open {
            continue;
        }
        let id = line[open + 1..close].trim().to_string();
        if id.is_empty() {
            continue;
        }
        let friendly = line[close + 1..]
            .split(" - ")
            .nth(1)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(&id)
            .to_string();
        map.entry(id).or_insert(friendly);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::{
        append_mono, apply_audio_processing, card_id, high_level_label, lock_buf, resample,
    };
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Mutex};

    /// Poison the buffer mutex (panic while holding it), then assert lock_buf still
    /// hands back the data instead of unwrapping Err — the bug that would otherwise
    /// brick capture forever after one swallowed callback-thread panic.
    #[test]
    fn lock_buf_recovers_from_poison() {
        let buf = Arc::new(Mutex::new(vec![1.0f32, 2.0]));
        let b = Arc::clone(&buf);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _g = b.lock().unwrap();
            panic!("poison the lock");
        }));
        assert!(buf.is_poisoned());
        assert_eq!(*lock_buf(&buf), vec![1.0, 2.0]);
    }

    /// The realtime append path must keep capturing after a poison, not panic.
    #[test]
    fn append_mono_recovers_from_poison() {
        let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
        let b = Arc::clone(&buf);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _g = b.lock().unwrap();
            panic!("poison the lock");
        }));
        assert!(buf.is_poisoned());
        append_mono(&buf, &[0.5f32, 0.5], 1, 16);
        assert_eq!(*lock_buf(&buf), vec![0.5, 0.5]);
    }

    #[test]
    fn card_id_extracts_token() {
        assert_eq!(card_id("plughw:CARD=PCH,DEV=0"), Some("PCH"));
        assert_eq!(card_id("hw:CARD=Snowball,DEV=0"), Some("Snowball"));
        assert_eq!(card_id("sysdefault:CARD=PCH"), Some("PCH"));
        assert_eq!(card_id("pipewire"), None);
    }

    #[test]
    fn high_level_labels() {
        assert_eq!(high_level_label("pipewire").as_deref(), Some("PipeWire"));
        assert_eq!(high_level_label("pulse").as_deref(), Some("PulseAudio"));
        assert_eq!(high_level_label("hw:CARD=PCH,DEV=0"), None);
    }

    #[test]
    fn resample_identity() {
        let s = vec![0.1, 0.2, 0.3];
        assert_eq!(resample(&s, 16_000, 16_000), s);
    }

    #[test]
    fn resample_empty() {
        assert!(resample(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn resample_downsample_length() {
        // 48k → 16k is a 1/3 ratio: ceil(300 * 1/3) = 100.
        let s = vec![0.0f32; 300];
        let out = resample(&s, 48_000, 16_000);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn apm_preserves_length() {
        let rate = 48_000u32;
        let samples: Vec<f32> = (0..rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        let out = apply_audio_processing(&samples, rate);
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn apm_empty_passthrough() {
        assert!(apply_audio_processing(&[], 48_000).is_empty());
    }

    #[test]
    fn apm_bounds() {
        // Processed samples must stay within a reasonable range (NS+AGC can go slightly above 1.0).
        let rate = 16_000u32;
        let samples: Vec<f32> = (0..rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / rate as f32).sin())
            .collect();
        let out = apply_audio_processing(&samples, rate);
        assert!(out.iter().all(|v| v.abs() < 2.0));
    }

    #[test]
    fn resample_sine_continuity() {
        // A 440 Hz sine resampled 48k→16k stays bounded and non-trivial.
        let from = 48_000u32;
        let n = from as usize; // 1 second
        let sine: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin())
            .collect();
        let out = resample(&sine, from, 16_000);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|v| v.abs() <= 1.05)); // FFT resampler can have minor overshoot
        let peak = out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.5, "resampled sine lost amplitude: {peak}");
    }
}
