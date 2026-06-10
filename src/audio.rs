//! AudioRecorder: cpal input stream, mono downmix, linear resample to 16 kHz.
//!
//! The stream is created on `start()` and dropped on `stop()`. We accept the
//! device's native rate/format (mics are 44.1/48 kHz) and resample to 16 kHz
//! ourselves — requesting 16 kHz directly makes cpal error on most hardware.

use std::sync::{Arc, Mutex};

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
}

impl AudioRecorder {
    /// Pick the input device (substring match on `audio_device`, else default)
    /// and cache its native config. Does not open the stream yet.
    pub fn new(audio_device: &str) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_device(&host, audio_device)?;
        let name = device.name().unwrap_or_else(|_| "<unknown>".into());

        let supported = device
            .default_input_config()
            .context("querying default input config")?;
        let sample_format = supported.sample_format();
        let channels = supported.channels() as usize;
        let sample_rate = supported.sample_rate().0;
        info!("audio device: {name} ({sample_rate} Hz, {channels} ch, {sample_format:?})");

        // Pre-allocate ~60s of mono audio at the device rate.
        let cap = sample_rate as usize * MAX_SECONDS;
        Ok(Self {
            device,
            sample_format,
            channels,
            sample_rate,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(cap.min(16_000 * 60)))),
            stream: None,
        })
    }

    /// Open the input stream and begin appending mono samples to the buffer.
    pub fn start(&mut self) -> Result<()> {
        self.buffer.lock().unwrap().clear();

        let config = cpal::StreamConfig {
            channels: self.channels as u16,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let channels = self.channels;
        let cap = self.sample_rate as usize * MAX_SECONDS;
        let buf = self.buffer.clone();
        let err_fn = |e| error!("audio stream error: {e}");

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
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        }
        .context("building input stream")?;

        stream.play().context("starting input stream")?;
        self.stream = Some(stream);
        debug!("recording started");
        Ok(())
    }

    /// Stop the stream and return 16 kHz mono f32 samples in [-1, 1].
    pub fn stop(&mut self) -> Vec<f32> {
        self.stream = None; // drop → stop the stream
        let raw = std::mem::take(&mut *self.buffer.lock().unwrap());
        let raw_peak = raw.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        debug!(
            "captured {} samples at {} Hz ({:.2}s), raw peak {raw_peak:.3}",
            raw.len(),
            self.sample_rate,
            raw.len() as f32 / self.sample_rate as f32
        );
        // Resample to 16 kHz *first*, then run the WebRTC APM at 16 kHz. The APM
        // only supports 8/16/32/48 kHz; feeding it the device's native rate (e.g.
        // 44.1 kHz) produces pitch-shifted, noisy garbage.
        let resampled = resample(&raw, self.sample_rate, TARGET_RATE);
        let mut processed = apply_audio_processing(&resampled, TARGET_RATE);
        normalize_peak(&mut processed);
        processed
    }

    pub fn target_rate(&self) -> u32 {
        TARGET_RATE
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

/// Append interleaved samples to the shared buffer, converting to f32 and
/// downmixing to mono by averaging across channels. Caps at `cap` samples.
fn append_mono<T>(buf: &Arc<Mutex<Vec<f32>>>, data: &[T], channels: usize, cap: usize)
where
    T: Sample,
    f32: FromSample<T>,
{
    let mut b = buf.lock().unwrap();
    let denom = channels.max(1) as f32;
    for frame in data.chunks(channels.max(1)) {
        if b.len() >= cap {
            break;
        }
        let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
        b.push(sum / denom);
    }
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

/// Linear-interpolation resample. [voxtype — verbatim algorithm]
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
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
                headroom_db: 3.0, // target -3 dBFS
                max_gain_db: 5.0, // cap boost at 5 dB
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

#[cfg(test)]
mod tests {
    use super::{apply_audio_processing, resample};

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
        assert!(out.iter().all(|v| v.abs() <= 1.0001));
        let peak = out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.5, "resampled sine lost amplitude: {peak}");
    }
}
