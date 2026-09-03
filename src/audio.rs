//! AudioRecorder: cpal input stream, mono downmix, sinc/FFT resample to 16 kHz.
//! Pipeline: native-rate capture → rubato FFT resample → WebRTC APM → peak normalize → VAD silence trim.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat};
use sonora::config::{AdaptiveDigital, GainController2, NoiseSuppression, NoiseSuppressionLevel};
use sonora::{AudioProcessing, Config, StreamConfig};
use tracing::{debug, error, info};

const TARGET_RATE: u32 = 16_000;
const CAPTURE_PREALLOC_SECONDS: usize = 60;
const RAW_WINDOW_MS: u64 = 20;
const RAW_SPEECH_RMS: f32 = 0.008;
const FORCE_SPLIT_AFTER_SOFT_MS: u64 = 18_000;
const MIN_SEGMENT_PAUSE_MS: u64 = 120;

fn force_split_ms(soft_max_ms: u64) -> u64 {
    soft_max_ms.saturating_add(FORCE_SPLIT_AFTER_SOFT_MS)
}

/// Reduce the pause needed to close a segment as it approaches the soft cap.
/// A fresh segment waits for a natural pause. At the soft cap, even a short
/// hesitation closes it. Continuous speech gets another 18 seconds before a
/// forced split.
fn adaptive_pause_ms(initial_pause_ms: u64, duration_ms: u64, soft_max_ms: u64) -> u64 {
    let initial = initial_pause_ms.max(MIN_SEGMENT_PAUSE_MS);
    if soft_max_ms == 0 || duration_ms >= soft_max_ms {
        return MIN_SEGMENT_PAUSE_MS;
    }
    let reduction =
        (initial - MIN_SEGMENT_PAUSE_MS) as u128 * duration_ms as u128 / soft_max_ms as u128;
    initial - reduction as u64
}

fn segment_drain_reason(
    duration_ms: u64,
    observed_speech_ms: u64,
    trailing_silence_ms: u64,
    initial_pause_ms: u64,
    soft_max_ms: u64,
    emergency: bool,
) -> Option<DrainReason> {
    if emergency || duration_ms >= force_split_ms(soft_max_ms) {
        return Some(DrainReason::MaxDuration);
    }
    let required_pause_ms = adaptive_pause_ms(initial_pause_ms, duration_ms, soft_max_ms);
    if observed_speech_ms == 0 || trailing_silence_ms < required_pause_ms {
        return None;
    }
    Some(if duration_ms >= soft_max_ms {
        DrainReason::MaxDuration
    } else {
        DrainReason::Pause
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainReason {
    Pause,
    MaxDuration,
    Release,
}

#[derive(Debug)]
pub struct DrainedSegment {
    pub raw: Vec<f32>,
    pub raw_rate: u32,
    pub observed_speech_ms: u64,
    pub reason: DrainReason,
}

#[cfg(feature = "debug-tools")]
pub struct TimedSegment {
    pub segment: DrainedSegment,
    pub boundary_sample: usize,
}

pub struct AudioRecorder {
    device: cpal::Device,
    sample_format: SampleFormat,
    channels: usize,
    sample_rate: u32,
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    detector_pos: usize,
    observed_speech_ms: u64,
    trailing_silence_ms: u64,
    overrun: Arc<AtomicBool>,
    visual_signal: Arc<AtomicU32>,
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

        let cap = sample_rate as usize * CAPTURE_PREALLOC_SECONDS;
        Ok(Self {
            device,
            sample_format,
            channels,
            sample_rate,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(cap))),
            stream: None,
            detector_pos: 0,
            observed_speech_ms: 0,
            trailing_silence_ms: 0,
            overrun: Arc::new(AtomicBool::new(false)),
            visual_signal: Arc::new(AtomicU32::new(0)),
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
        self.reset_detector();

        let config = cpal::StreamConfig {
            channels: self.channels as u16,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let channels = self.channels;
        let cap = self.sample_rate as usize * CAPTURE_PREALLOC_SECONDS;
        self.overrun.store(false, Ordering::Relaxed);
        self.visual_signal.store(0, Ordering::Relaxed);
        let overrun = self.overrun.clone();
        let buf = self.buffer.clone();
        let visual_signal = self.visual_signal.clone();
        let cb = self.error_cb.clone();
        let err_fn = move |e: cpal::StreamError| {
            error!("audio stream error: {e}");
            if let Some(cb) = &cb {
                cb(e.to_string());
            }
        };

        // Every cpal sample format `append_mono` can convert to f32 — a USB/pro
        // interface that only offers i32/i8 must not hard-error. `FromSample`
        // handles the per-format scaling (and 8/16/32/64-bit unsigned origins).
        let stream = match self.sample_format {
            SampleFormat::F32 => self.device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::F64 => self.device.build_input_stream(
                &config,
                move |data: &[f64], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::I8 => self.device.build_input_stream(
                &config,
                move |data: &[i8], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => self.device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::I32 => self.device.build_input_stream(
                &config,
                move |data: &[i32], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::I64 => self.device.build_input_stream(
                &config,
                move |data: &[i64], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::U8 => self.device.build_input_stream(
                &config,
                move |data: &[u8], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::U16 => self.device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::U32 => self.device.build_input_stream(
                &config,
                move |data: &[u32], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
                err_fn,
                None,
            ),
            SampleFormat::U64 => self.device.build_input_stream(
                &config,
                move |data: &[u64], _| {
                    append_mono(&buf, data, channels, cap, &overrun, &visual_signal)
                },
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
        self.visual_signal.store(0, Ordering::Relaxed);
        lock_buf(&self.buffer).clear();
        self.reset_detector();
    }

    /// Stop the stream and return 16 kHz mono f32 samples in [-1, 1].
    #[cfg_attr(not(feature = "debug-tools"), allow(dead_code))]
    pub fn stop(&mut self) -> Vec<f32> {
        let (_, _, processed) = self.stop_with_raw();
        processed
    }

    /// Stop the stream and return both the raw native-rate mono samples and the
    /// fully processed 16 kHz samples. The raw buffer is at `self.sample_rate`
    /// and is useful for writing a before/after comparison WAV.
    #[cfg_attr(not(feature = "debug-tools"), allow(dead_code))]
    pub fn stop_with_raw(&mut self) -> (Vec<f32>, u32, Vec<f32>) {
        let segment = self.stop_raw();
        let raw = segment.raw;
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

    pub fn stop_raw(&mut self) -> DrainedSegment {
        self.stream = None;
        self.visual_signal.store(0, Ordering::Relaxed);
        self.update_detector();
        let raw = std::mem::take(&mut *lock_buf(&self.buffer));
        let segment = DrainedSegment {
            raw,
            raw_rate: self.sample_rate,
            observed_speech_ms: self.observed_speech_ms,
            reason: DrainReason::Release,
        };
        self.reset_detector();
        segment
    }

    pub fn try_drain_segment(&mut self, pause_ms: u64, max_ms: u64) -> Option<DrainedSegment> {
        self.update_detector();
        let len = lock_buf(&self.buffer).len();
        let duration_ms = len as u64 * 1000 / self.sample_rate as u64;
        let emergency = self.overrun.swap(false, Ordering::Relaxed);
        let reason = segment_drain_reason(
            duration_ms,
            self.observed_speech_ms,
            self.trailing_silence_ms,
            pause_ms,
            max_ms,
            emergency,
        )?;
        if emergency {
            error!(
                duration_ms,
                "capture poll stalled beyond emergency capacity; forcing drain"
            );
        }
        let raw = std::mem::take(&mut *lock_buf(&self.buffer));
        let speech = self.observed_speech_ms;
        self.reset_detector();
        if reason == DrainReason::MaxDuration && speech == 0 {
            debug!(duration_ms, "discarded silent maximum-duration segment");
            return None;
        }
        Some(DrainedSegment {
            raw,
            raw_rate: self.sample_rate,
            observed_speech_ms: speech,
            reason,
        })
    }

    fn update_detector(&mut self) {
        let b = lock_buf(&self.buffer);
        update_segment_detector(
            &b,
            self.sample_rate,
            &mut self.detector_pos,
            &mut self.observed_speech_ms,
            &mut self.trailing_silence_ms,
        );
    }

    fn reset_detector(&mut self) {
        self.detector_pos = 0;
        self.observed_speech_ms = 0;
        self.trailing_silence_ms = 0;
    }

    #[cfg_attr(not(feature = "debug-tools"), allow(dead_code))]
    pub fn target_rate(&self) -> u32 {
        TARGET_RATE
    }

    /// A lossy loudness feed for visual UI. It never owns captured samples.
    pub fn visual_signal(&self) -> Arc<AtomicU32> {
        self.visual_signal.clone()
    }

    /// Keep UI listeners attached when a config reload replaces the device.
    pub fn use_visual_signal(&mut self, signal: Arc<AtomicU32>) {
        self.visual_signal = signal;
    }
}

fn update_segment_detector(
    samples: &[f32],
    sample_rate: u32,
    detector_pos: &mut usize,
    observed_speech_ms: &mut u64,
    trailing_silence_ms: &mut u64,
) {
    let window = (sample_rate as u64 * RAW_WINDOW_MS / 1000) as usize;
    while *detector_pos + window <= samples.len() {
        let w = &samples[*detector_pos..*detector_pos + window];
        let rms = (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt();
        if rms >= RAW_SPEECH_RMS {
            *observed_speech_ms += RAW_WINDOW_MS;
            *trailing_silence_ms = 0;
        } else if *observed_speech_ms > 0 {
            *trailing_silence_ms += RAW_WINDOW_MS;
        }
        *detector_pos += window;
    }
}

/// Split fixed audio at the same 200 ms poll points as the daemon.
/// The detector uses sample counts because a WAV file has no wall clock.
#[cfg(feature = "debug-tools")]
pub fn segment_samples(
    samples: &[f32],
    sample_rate: u32,
    pause_ms: u64,
    max_ms: u64,
) -> Vec<TimedSegment> {
    const POLL_MS: usize = 200;

    let poll_samples = sample_rate as usize * POLL_MS / 1000;
    let mut segments = Vec::new();
    let mut buffer = Vec::new();
    let mut detector_pos = 0;
    let mut observed_speech_ms = 0;
    let mut trailing_silence_ms = 0;
    let mut consumed = 0;

    for chunk in samples.chunks(poll_samples.max(1)) {
        buffer.extend_from_slice(chunk);
        consumed += chunk.len();
        update_segment_detector(
            &buffer,
            sample_rate,
            &mut detector_pos,
            &mut observed_speech_ms,
            &mut trailing_silence_ms,
        );
        let duration_ms = buffer.len() as u64 * 1000 / sample_rate as u64;
        let Some(reason) = segment_drain_reason(
            duration_ms,
            observed_speech_ms,
            trailing_silence_ms,
            pause_ms,
            max_ms,
            false,
        ) else {
            continue;
        };

        let segment = DrainedSegment {
            raw: std::mem::take(&mut buffer),
            raw_rate: sample_rate,
            observed_speech_ms,
            reason,
        };
        detector_pos = 0;
        observed_speech_ms = 0;
        trailing_silence_ms = 0;
        segments.push(TimedSegment {
            segment,
            boundary_sample: consumed,
        });
    }

    segments.push(TimedSegment {
        segment: DrainedSegment {
            raw: buffer,
            raw_rate: sample_rate,
            observed_speech_ms,
            reason: DrainReason::Release,
        },
        boundary_sample: samples.len(),
    });
    segments
}

/// §8b: prefer 16 kHz native capture; falls back to the device default if unsupported.
/// Eliminates the resample step entirely on compatible hardware. A device may
/// expose several 16 kHz-capable configs in different sample formats; pick the
/// one we consume most cleanly (`format_rank`) rather than whatever it lists first.
fn select_stream_config(device: &cpal::Device) -> Result<(SampleFormat, usize, u32)> {
    if let Ok(configs) = device.supported_input_configs() {
        if let Some(cfg) = configs
            .filter(|c| {
                c.min_sample_rate().0 <= TARGET_RATE && c.max_sample_rate().0 >= TARGET_RATE
            })
            .min_by_key(|c| format_rank(c.sample_format()))
        {
            let fmt = cfg.sample_format();
            debug!("device supports 16 kHz natively ({fmt:?}) — resample step skipped");
            return Ok((fmt, cfg.channels() as usize, TARGET_RATE));
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

/// Rank input sample formats by how cleanly we consume them, lower = preferred.
/// Float needs no scaling; 16-bit is the universal mic format; wider ints work
/// but cost a conversion; 8-bit carries the most quantization noise. Every
/// variant is handled by `start`, so this only chooses *which* config to open.
fn format_rank(f: SampleFormat) -> u8 {
    match f {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::I32 => 2,
        SampleFormat::F64 => 3,
        SampleFormat::U16 => 4,
        SampleFormat::I64 => 5,
        SampleFormat::U32 => 6,
        SampleFormat::U64 => 7,
        SampleFormat::I8 => 8,
        SampleFormat::U8 => 9,
        _ => 10,
    }
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
/// downmixing to mono by averaging across channels. Crossing `cap` flags the
/// daemon to force-drain; samples continue growing so a delayed poll loses none.
fn append_mono<T>(
    buf: &Arc<Mutex<Vec<f32>>>,
    data: &[T],
    channels: usize,
    cap: usize,
    overrun: &AtomicBool,
    visual_signal: &AtomicU32,
) where
    T: Sample,
    f32: FromSample<T>,
{
    let mut b = lock_buf(buf);
    let denom = channels.max(1) as f32;
    let mut squares = 0.0;
    let mut count = 0usize;
    for frame in data.chunks(channels.max(1)) {
        if b.len() >= cap {
            overrun.store(true, Ordering::Relaxed);
        }
        let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
        let mono = sum / denom;
        b.push(mono);
        squares += mono * mono;
        count += 1;
    }
    let rms = (squares / count.max(1) as f32).sqrt();
    let level = ((rms - 0.012) * 6.2).clamp(0.0, 1.0);
    visual_signal.store(level.to_bits(), Ordering::Relaxed);
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
    #[cfg(feature = "debug-tools")]
    use super::segment_samples;
    use super::{
        adaptive_pause_ms, append_mono, apply_audio_processing, card_id, force_split_ms,
        format_rank, high_level_label, lock_buf, resample, segment_drain_reason, DrainReason,
    };
    use cpal::SampleFormat;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
        append_mono(
            &buf,
            &[0.5f32, 0.5],
            1,
            16,
            &AtomicBool::new(false),
            &AtomicU32::new(0),
        );
        assert_eq!(*lock_buf(&buf), vec![0.5, 0.5]);
    }

    #[test]
    fn append_mono_flags_capacity_without_dropping_samples() {
        let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
        let overrun = AtomicBool::new(false);
        append_mono(
            &buf,
            &[0.1f32, 0.2, 0.3],
            1,
            2,
            &overrun,
            &AtomicU32::new(0),
        );
        assert_eq!(*lock_buf(&buf), vec![0.1, 0.2, 0.3]);
        assert!(overrun.load(Ordering::Relaxed));
    }

    #[test]
    fn append_publishes_visual_level_without_consuming_audio() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let signal = AtomicU32::new(0);
        append_mono(
            &buf,
            &[0.5f32, -0.5],
            1,
            16,
            &AtomicBool::new(false),
            &signal,
        );
        assert_eq!(lock_buf(&buf).as_slice(), &[0.5, -0.5]);
        assert!(f32::from_bits(signal.load(Ordering::Relaxed)) > 0.9);
    }

    #[test]
    fn pause_requirement_shrinks_toward_soft_cap() {
        assert_eq!(adaptive_pause_ms(300, 0, 30_000), 300);
        assert_eq!(adaptive_pause_ms(300, 10_000, 30_000), 240);
        assert_eq!(adaptive_pause_ms(300, 20_000, 30_000), 180);
        assert_eq!(adaptive_pause_ms(300, 30_000, 30_000), 120);
        assert_eq!(adaptive_pause_ms(300, 40_000, 30_000), 120);
    }

    #[test]
    fn pause_requirement_never_drops_below_detector_window_floor() {
        assert_eq!(adaptive_pause_ms(50, 0, 30_000), 120);
        assert_eq!(adaptive_pause_ms(300, 1, 0), 120);
    }

    #[test]
    fn thirty_second_soft_boundary_forces_at_forty_eight_seconds() {
        assert_eq!(force_split_ms(30_000), 48_000);
    }

    #[test]
    fn segment_drain_tracks_the_shrinking_pause() {
        assert_eq!(
            segment_drain_reason(1_000, 700, 293, 300, 30_000, false),
            None
        );
        assert_eq!(
            segment_drain_reason(1_000, 700, 294, 300, 30_000, false),
            Some(DrainReason::Pause)
        );
        assert_eq!(
            segment_drain_reason(20_000, 18_000, 180, 300, 30_000, false),
            Some(DrainReason::Pause)
        );
        assert_eq!(
            segment_drain_reason(30_000, 28_000, 120, 300, 30_000, false),
            Some(DrainReason::MaxDuration)
        );
    }

    #[test]
    fn segment_drain_requires_speech_until_the_hard_cap() {
        assert_eq!(
            segment_drain_reason(30_000, 0, 30_000, 300, 30_000, false),
            None
        );
        assert_eq!(
            segment_drain_reason(48_000, 0, 48_000, 300, 30_000, false),
            Some(DrainReason::MaxDuration)
        );
    }

    #[cfg(feature = "debug-tools")]
    #[test]
    fn wav_segmentation_polls_at_two_hundred_millisecond_boundaries() {
        let mut samples = vec![0.02; 6_400];
        samples.extend(vec![0.0; 6_400]);
        samples.extend(vec![0.02; 3_200]);

        let segments = segment_samples(&samples, 16_000, 300, 30_000);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].boundary_sample, 12_800);
        assert_eq!(segments[0].segment.reason, DrainReason::Pause);
        assert_eq!(segments[1].boundary_sample, 16_000);
        assert_eq!(segments[1].segment.reason, DrainReason::Release);
    }

    #[test]
    fn card_id_extracts_token() {
        assert_eq!(card_id("plughw:CARD=PCH,DEV=0"), Some("PCH"));
        assert_eq!(card_id("hw:CARD=Snowball,DEV=0"), Some("Snowball"));
        assert_eq!(card_id("sysdefault:CARD=PCH"), Some("PCH"));
        assert_eq!(card_id("pipewire"), None);
    }

    #[test]
    fn format_rank_prefers_cleanest() {
        // Float first, then 16-bit, then wider ints; 8-bit last.
        assert!(format_rank(SampleFormat::F32) < format_rank(SampleFormat::I16));
        assert!(format_rank(SampleFormat::I16) < format_rank(SampleFormat::I32));
        assert!(format_rank(SampleFormat::I32) < format_rank(SampleFormat::U8));
        assert!(format_rank(SampleFormat::I8) < format_rank(SampleFormat::U8));
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
