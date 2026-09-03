//! AudioRecorder: cpal input stream, mono downmix, sinc/FFT resample to 16 kHz.
//! Pipeline: native-rate capture → rubato FFT resample → WebRTC APM → peak normalize → VAD silence trim.

use std::sync::atomic::{AtomicBool, Ordering};
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
const EARLY_SEGMENT_MS: u64 = 3_000;
const EARLY_PAUSE_MS: u64 = 800;
const OVERLAP_SEARCH_MS: u64 = 2_000;
const OVERLAP_PAUSE_MS: u64 = 40;

fn force_split_ms(soft_max_ms: u64) -> u64 {
    soft_max_ms.saturating_add(FORCE_SPLIT_AFTER_SOFT_MS)
}

/// Keep the early pause threshold through the first 3 seconds, then reduce it
/// linearly until the forced split. This avoids both tiny model inputs and a
/// sudden threshold change at 3 seconds.
fn adaptive_pause_ms(initial_pause_ms: u64, duration_ms: u64, soft_max_ms: u64) -> u64 {
    if soft_max_ms == 0 {
        return MIN_SEGMENT_PAUSE_MS;
    }
    let initial = initial_pause_ms.max(EARLY_PAUSE_MS);
    if duration_ms <= EARLY_SEGMENT_MS {
        return initial;
    }
    let forced_split_ms = force_split_ms(soft_max_ms);
    if duration_ms >= forced_split_ms {
        return MIN_SEGMENT_PAUSE_MS;
    }
    let ramp_ms = forced_split_ms - EARLY_SEGMENT_MS;
    let elapsed_ms = duration_ms - EARLY_SEGMENT_MS;
    let reduction = (initial - MIN_SEGMENT_PAUSE_MS) as u128 * elapsed_ms as u128 / ramp_ms as u128;
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
    /// Raw prefix copied from the previous segment for boundary context.
    pub overlap_samples: usize,
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
    overlap_samples: usize,
    observed_speech_ms: u64,
    trailing_silence_ms: u64,
    overrun: Arc<AtomicBool>,
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
            overlap_samples: 0,
            observed_speech_ms: 0,
            trailing_silence_ms: 0,
            overrun: Arc::new(AtomicBool::new(false)),
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
        self.reset_detector(0);

        let config = cpal::StreamConfig {
            channels: self.channels as u16,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let channels = self.channels;
        let cap = self.sample_rate as usize * CAPTURE_PREALLOC_SECONDS;
        self.overrun.store(false, Ordering::Relaxed);
        let overrun = self.overrun.clone();
        let buf = self.buffer.clone();
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
                move |data: &[f32], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::F64 => self.device.build_input_stream(
                &config,
                move |data: &[f64], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::I8 => self.device.build_input_stream(
                &config,
                move |data: &[i8], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::I16 => self.device.build_input_stream(
                &config,
                move |data: &[i16], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::I32 => self.device.build_input_stream(
                &config,
                move |data: &[i32], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::I64 => self.device.build_input_stream(
                &config,
                move |data: &[i64], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::U8 => self.device.build_input_stream(
                &config,
                move |data: &[u8], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::U16 => self.device.build_input_stream(
                &config,
                move |data: &[u16], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::U32 => self.device.build_input_stream(
                &config,
                move |data: &[u32], _| append_mono(&buf, data, channels, cap, &overrun),
                err_fn,
                None,
            ),
            SampleFormat::U64 => self.device.build_input_stream(
                &config,
                move |data: &[u64], _| append_mono(&buf, data, channels, cap, &overrun),
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
        self.reset_detector(0);
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
        self.update_detector();
        let raw = std::mem::take(&mut *lock_buf(&self.buffer));
        let segment = DrainedSegment {
            raw,
            raw_rate: self.sample_rate,
            observed_speech_ms: self.observed_speech_ms,
            reason: DrainReason::Release,
            overlap_samples: self.overlap_samples,
        };
        self.reset_detector(0);
        segment
    }

    pub fn try_drain_segment(&mut self, pause_ms: u64, max_ms: u64) -> Option<DrainedSegment> {
        self.update_detector();
        let len = lock_buf(&self.buffer).len();
        let new_samples = len.saturating_sub(self.overlap_samples);
        let duration_ms = new_samples as u64 * 1000 / self.sample_rate as u64;
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
        let prior_overlap_samples = self.overlap_samples;
        let raw = std::mem::take(&mut *lock_buf(&self.buffer));
        let mut next_buffer = boundary_overlap_start(&raw, self.sample_rate)
            .map(|start| raw[start..].to_vec())
            .unwrap_or_default();
        let next_overlap_samples = next_buffer.len();
        let mut buffer = lock_buf(&self.buffer);
        next_buffer.append(&mut *buffer);
        *buffer = next_buffer;
        drop(buffer);
        let speech = self.observed_speech_ms;
        self.reset_detector(next_overlap_samples);
        if reason == DrainReason::MaxDuration && speech == 0 {
            debug!(duration_ms, "discarded silent maximum-duration segment");
            return None;
        }
        Some(DrainedSegment {
            raw,
            raw_rate: self.sample_rate,
            observed_speech_ms: speech,
            reason,
            overlap_samples: prior_overlap_samples,
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

    fn reset_detector(&mut self, overlap_samples: usize) {
        self.detector_pos = overlap_samples;
        self.overlap_samples = overlap_samples;
        self.observed_speech_ms = 0;
        self.trailing_silence_ms = 0;
    }

    #[cfg_attr(not(feature = "debug-tools"), allow(dead_code))]
    pub fn target_rate(&self) -> u32 {
        TARGET_RATE
    }
}

/// Find the most recent raw-audio pause before the final spoken word.
///
/// The search only examines a bounded tail. If continuous speech contains no
/// suitable pause, the next segment starts without overlap.
fn boundary_overlap_start(samples: &[f32], sample_rate: u32) -> Option<usize> {
    let window = (sample_rate as u64 * RAW_WINDOW_MS / 1000) as usize;
    let pause_windows = OVERLAP_PAUSE_MS.div_ceil(RAW_WINDOW_MS) as usize;
    let search_samples = (sample_rate as u64 * OVERLAP_SEARCH_MS / 1000) as usize;
    if window == 0 || samples.len() < window {
        return None;
    }

    let search_start = samples.len().saturating_sub(search_samples);
    let windows: Vec<bool> = samples[search_start..]
        .chunks_exact(window)
        .map(|chunk| {
            let rms = (chunk.iter().map(|sample| sample * sample).sum::<f32>()
                / chunk.len() as f32)
                .sqrt();
            rms >= RAW_SPEECH_RMS
        })
        .collect();
    let final_speech = windows.iter().rposition(|&speech| speech)?;
    if final_speech == 0 {
        return None;
    }

    let mut cursor = final_speech;
    while cursor > 0 {
        cursor -= 1;
        if windows[cursor] {
            continue;
        }
        let pause_end = cursor + 1;
        while cursor > 0 && !windows[cursor - 1] {
            cursor -= 1;
        }
        if pause_end - cursor >= pause_windows {
            return Some(search_start + cursor * window);
        }
    }
    None
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
    let mut overlap_samples = 0;
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
        let new_samples = buffer.len().saturating_sub(overlap_samples);
        let duration_ms = new_samples as u64 * 1000 / sample_rate as u64;
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

        let raw = std::mem::take(&mut buffer);
        let next_overlap = boundary_overlap_start(&raw, sample_rate)
            .map(|start| raw[start..].to_vec())
            .unwrap_or_default();
        let next_overlap_samples = next_overlap.len();
        let segment = DrainedSegment {
            raw,
            raw_rate: sample_rate,
            observed_speech_ms,
            reason,
            overlap_samples,
        };
        buffer = next_overlap;
        detector_pos = next_overlap_samples;
        overlap_samples = next_overlap_samples;
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
            overlap_samples,
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
) where
    T: Sample,
    f32: FromSample<T>,
{
    let mut b = lock_buf(buf);
    let denom = channels.max(1) as f32;
    for frame in data.chunks(channels.max(1)) {
        if b.len() >= cap {
            overrun.store(true, Ordering::Relaxed);
        }
        let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
        b.push(sum / denom);
    }
}

/// Process one complete capture with a fresh APM instance.
///
/// Live segmented capture uses [`CaptureProcessor`] directly so APM state lasts
/// for the full hold. One-shot callers keep isolated state between captures.
pub fn process_capture(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    CaptureProcessor::new(sample_rate).process(samples)
}

/// Loudest sample lands here after normalization — leaves ~0.5 dB headroom so
/// nothing clips on the 16-bit wav write / model input.
const NORM_TARGET_PEAK: f32 = 0.95;
const NORM_MAX_GAIN: f32 = 8.0;

/// Scale a capture toward the model's expected amplitude without amplifying a
/// near-silent buffer beyond the measured noise-safe ceiling.
fn normalize_peak(samples: &mut [f32]) {
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

pub fn resampled_sample_count(sample_count: usize, from_rate: u32, to_rate: u32) -> usize {
    if from_rate == 0 {
        return 0;
    }
    (sample_count as u128 * to_rate as u128)
        .div_ceil(from_rate as u128)
        .min(usize::MAX as u128) as usize
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

/// Stateful WebRTC processing for one key hold.
#[derive(Debug)]
pub struct CaptureProcessor {
    apm: AudioProcessing,
    frame_size: usize,
    sample_rate: u32,
}

impl CaptureProcessor {
    pub fn new(sample_rate: u32) -> Self {
        let cfg = Config {
            noise_suppression: Some(NoiseSuppression {
                level: NoiseSuppressionLevel::Moderate,
                ..Default::default()
            }),
            gain_controller2: Some(GainController2 {
                adaptive_digital: Some(AdaptiveDigital {
                    headroom_db: 1.0,
                    max_gain_db: 12.0,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let stream_cfg = StreamConfig::new(sample_rate, 1);
        let frame_size = stream_cfg.num_frames();
        let apm = AudioProcessing::builder()
            .config(cfg)
            .capture_config(stream_cfg)
            .render_config(StreamConfig::new(sample_rate, 1))
            .build();

        Self {
            apm,
            frame_size,
            sample_rate,
        }
    }

    /// Run APM, normalize peak level, then trim silence from this segment.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        let processed = self.apply_audio_processing(samples);
        finalize_processed(processed, self.sample_rate)
    }

    /// Process repeated boundary context without replaying it through this
    /// hold's persistent APM state.
    pub fn process_with_overlap(&mut self, samples: &[f32], overlap_samples: usize) -> Vec<f32> {
        let split = overlap_samples.min(samples.len());
        if split == 0 {
            return self.process(samples);
        }

        let mut processed =
            CaptureProcessor::new(self.sample_rate).apply_audio_processing(&samples[..split]);
        processed.extend(self.apply_audio_processing(&samples[split..]));
        finalize_processed(processed, self.sample_rate)
    }

    /// Run WebRTC APM in 10 ms frames while retaining its adaptive state.
    fn apply_audio_processing(&mut self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(samples.len());

        for chunk in samples.chunks(self.frame_size) {
            if chunk.len() < self.frame_size {
                // Partial tail: pad to a full frame, process, then take only the real samples.
                let mut padded = chunk.to_vec();
                padded.resize(self.frame_size, 0.0);
                let mut dest = vec![0.0f32; self.frame_size];
                let _ = self.apm.process_capture_f32(&[&padded], &mut [&mut dest]);
                out.extend_from_slice(&dest[..chunk.len()]);
            } else {
                let mut dest = vec![0.0f32; self.frame_size];
                let _ = self.apm.process_capture_f32(&[chunk], &mut [&mut dest]);
                out.extend_from_slice(&dest);
            }
        }

        out
    }
}

fn finalize_processed(mut samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    normalize_peak(&mut samples);
    trim_silence(&samples, sample_rate)
}

/// One-shot APM entry point for focused diagnostics and tests.
#[cfg(test)]
pub fn apply_audio_processing(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    CaptureProcessor::new(sample_rate).apply_audio_processing(samples)
}

/// §8d: trim leading/trailing silence using windowed RMS energy.
/// After NS, AGC, and limiting, the noise floor is well below SPEECH_RMS so speech
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
        adaptive_pause_ms, append_mono, apply_audio_processing, boundary_overlap_start, card_id,
        finalize_processed, force_split_ms, format_rank, high_level_label, lock_buf,
        normalize_peak, resample, resampled_sample_count, segment_drain_reason, CaptureProcessor,
        DrainReason,
    };
    use cpal::SampleFormat;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};
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
        append_mono(&buf, &[0.5f32, 0.5], 1, 16, &AtomicBool::new(false));
        assert_eq!(*lock_buf(&buf), vec![0.5, 0.5]);
    }

    #[test]
    fn append_mono_flags_capacity_without_dropping_samples() {
        let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
        let overrun = AtomicBool::new(false);
        append_mono(&buf, &[0.1f32, 0.2, 0.3], 1, 2, &overrun);
        assert_eq!(*lock_buf(&buf), vec![0.1, 0.2, 0.3]);
        assert!(overrun.load(Ordering::Relaxed));
    }

    #[test]
    fn pause_requirement_shrinks_toward_forced_split() {
        assert_eq!(adaptive_pause_ms(300, 0, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 1_000, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 2_000, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 2_999, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 3_000, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 5_000, 30_000), 770);
        assert_eq!(adaptive_pause_ms(300, 10_000, 30_000), 695);
        assert_eq!(adaptive_pause_ms(300, 20_000, 30_000), 544);
        assert_eq!(adaptive_pause_ms(300, 30_000, 30_000), 392);
        assert_eq!(adaptive_pause_ms(300, 40_000, 30_000), 241);
        assert_eq!(adaptive_pause_ms(300, 47_000, 30_000), 136);
        assert_eq!(adaptive_pause_ms(300, 48_000, 30_000), 120);
    }

    #[test]
    fn pause_requirement_never_drops_below_detector_window_floor() {
        assert_eq!(adaptive_pause_ms(50, 0, 30_000), 800);
        assert_eq!(adaptive_pause_ms(300, 1, 0), 120);
    }

    #[test]
    fn thirty_second_soft_boundary_forces_at_forty_eight_seconds() {
        assert_eq!(force_split_ms(30_000), 48_000);
    }

    #[test]
    fn segment_drain_tracks_the_shrinking_pause() {
        assert_eq!(
            segment_drain_reason(1_200, 400, 799, 300, 30_000, false),
            None
        );
        assert_eq!(
            segment_drain_reason(1_200, 400, 800, 300, 30_000, false),
            Some(DrainReason::Pause)
        );
        assert_eq!(
            segment_drain_reason(20_000, 18_000, 544, 300, 30_000, false),
            Some(DrainReason::Pause)
        );
        assert_eq!(
            segment_drain_reason(30_000, 28_000, 392, 300, 30_000, false),
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

    #[test]
    fn overlap_starts_at_pause_before_final_word() {
        let rate = 1_000;
        let mut samples = vec![0.02; 100];
        samples.extend(vec![0.0; 60]);
        samples.extend(vec![0.02; 200]);
        samples.extend(vec![0.0; 100]);

        assert_eq!(boundary_overlap_start(&samples, rate), Some(100));
    }

    #[test]
    fn continuous_speech_has_no_overlap_fallback() {
        let samples = vec![0.02; 4_000];
        assert_eq!(boundary_overlap_start(&samples, 1_000), None);
    }

    #[cfg(feature = "debug-tools")]
    #[test]
    fn wav_segmentation_polls_at_two_hundred_millisecond_boundaries() {
        let mut samples = vec![0.02; 6_400];
        samples.extend(vec![0.0; 16_000]);
        samples.extend(vec![0.02; 3_200]);

        let segments = segment_samples(&samples, 16_000, 300, 30_000);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].boundary_sample, 19_200);
        assert_eq!(segments[0].segment.reason, DrainReason::Pause);
        assert_eq!(segments[1].boundary_sample, 25_600);
        assert_eq!(segments[1].segment.reason, DrainReason::Release);
    }

    #[cfg(feature = "debug-tools")]
    #[test]
    fn wav_release_segment_includes_pause_selected_overlap() {
        let mut samples = vec![0.02; 200];
        samples.extend(vec![0.0; 60]);
        samples.extend(vec![0.02; 200]);
        samples.extend(vec![0.0; 1_000]);
        samples.extend(vec![0.02; 200]);

        let segments = segment_samples(&samples, 1_000, 300, 30_000);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].boundary_sample, 1_400);
        assert_eq!(segments[0].segment.overlap_samples, 0);
        assert_eq!(segments[1].segment.reason, DrainReason::Release);
        assert_eq!(segments[1].segment.overlap_samples, 1_200);
        assert_eq!(&segments[1].segment.raw[..1_200], &samples[200..1_400]);
        assert_eq!(&segments[1].segment.raw[1_200..], &samples[1_400..]);
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
    fn resampled_count_maps_overlap_prefix() {
        assert_eq!(resampled_sample_count(960, 48_000, 16_000), 320);
        assert_eq!(resampled_sample_count(961, 48_000, 16_000), 321);
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
    fn persistent_apm_matches_contiguous_frame_processing() {
        let rate = 16_000u32;
        let samples: Vec<f32> = (0..rate as usize * 2)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();

        let contiguous = CaptureProcessor::new(rate).apply_audio_processing(&samples);
        let mut segmented_processor = CaptureProcessor::new(rate);
        let split = rate as usize;
        let mut segmented = segmented_processor.apply_audio_processing(&samples[..split]);
        segmented.extend(segmented_processor.apply_audio_processing(&samples[split..]));

        assert_eq!(segmented, contiguous);
    }

    #[test]
    fn overlap_does_not_advance_persistent_apm_twice() {
        let rate = 16_000u32;
        let samples: Vec<f32> = (0..rate as usize * 3)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
            .collect();
        let first = &samples[..rate as usize];
        let novel = &samples[rate as usize..rate as usize * 2];
        let next = &samples[rate as usize * 2..];
        let overlap = &first[first.len() - 3_200..];
        let mut repeated_segment = overlap.to_vec();
        repeated_segment.extend_from_slice(novel);

        let mut actual = CaptureProcessor::new(rate);
        actual.process(first);
        actual.process_with_overlap(&repeated_segment, overlap.len());

        let mut expected = CaptureProcessor::new(rate);
        expected.process(first);
        expected.process(novel);

        assert_eq!(actual.process(next), expected.process(next));
    }

    #[test]
    fn overlap_segment_gets_one_combined_trim() {
        let rate = 1_000;
        let mut prefix = vec![0.1; 200];
        prefix.extend(vec![0.0; 200]);
        let mut novel = vec![0.0; 200];
        novel.extend(vec![0.1; 200]);
        let mut combined = prefix.clone();
        combined.extend_from_slice(&novel);

        let combined = finalize_processed(combined, rate);
        let mut separately_processed = finalize_processed(prefix, rate);
        separately_processed.extend(finalize_processed(novel, rate));

        assert_eq!(combined.len(), 800);
        assert_eq!(separately_processed.len(), 560);
    }

    #[test]
    fn normalization_raises_quiet_audio_with_a_cap() {
        let mut samples = vec![0.1, -0.2, 0.3];
        normalize_peak(&mut samples);
        let peak = samples.iter().fold(0.0f32, |a, value| a.max(value.abs()));
        assert!((peak - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn normalization_reduces_overshoot() {
        let mut samples = vec![0.5, -2.0, 1.0];
        normalize_peak(&mut samples);
        let peak = samples.iter().fold(0.0f32, |a, value| a.max(value.abs()));
        assert!((peak - 0.95).abs() < f32::EPSILON);
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
