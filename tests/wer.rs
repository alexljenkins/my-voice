//! WER + latency regression harness over `samples/*.wav` + `samples/expected.txt`.
//!
//! Reports three things per run: the gated normalized WER (lowercased,
//! punctuation-stripped), a *strict* WER that preserves case + punctuation so
//! text-quality changes are visible, and an RTF / x-realtime speed figure.
//! Only the normalized WER asserts; strict + RTF are visibility, not a gate.
//!
//! Ignored by default: needs a downloaded model and ~seconds of inference.
//! Run with:
//!
//! ```sh
//! cargo test --features debug-tools --test wer -- --ignored --nocapture
//! ```
//!
//! Env knobs:
//! * `MY_VOICE_WER_MODEL` — model name to test (default: moonshine-base)
//! * `MY_VOICE_WER_MAX`   — max aggregate WER before failure (default: 0.25)
#![cfg(feature = "debug-tools")]

use std::path::Path;
use std::process::Command;

/// Word-level Levenshtein distance.
fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let (n, m) = (reference.len(), hypothesis.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let sub = prev[j - 1] + usize::from(reference[i - 1] != hypothesis[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Lowercase, strip everything but letters/digits/apostrophes, split on whitespace.
fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Whitespace-split only: case + punctuation preserved, so the strict score
/// *sees* the text-quality dimensions `normalize` is blind to by design.
fn normalize_strict(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Audio duration in seconds, read straight off the WAV header.
fn audio_seconds(wav: &Path) -> f64 {
    let reader = hound::WavReader::open(wav).expect("open wav");
    let spec = reader.spec();
    reader.len() as f64 / spec.channels as f64 / spec.sample_rate as f64
}

/// Pull `encode 123ms, decode 456ms` out of the binary's stderr log line.
fn parse_timings(stderr: &str) -> Option<(u64, u64)> {
    let grab = |key: &str| -> Option<u64> {
        let idx = stderr.find(key)?;
        let rest = &stderr[idx + key.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    };
    Some((grab("encode ")?, grab("decode ")?))
}

#[test]
#[ignore = "needs a downloaded model; run with --ignored"]
fn samples_wer() {
    let model =
        std::env::var("MY_VOICE_WER_MODEL").unwrap_or_else(|_| "moonshine-base".to_string());
    let max_wer: f64 = std::env::var("MY_VOICE_WER_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25);

    let samples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("samples");
    let expected = std::fs::read_to_string(samples_dir.join("expected.txt"))
        .expect("samples/expected.txt missing");

    let config_path =
        std::env::temp_dir().join(format!("my-voice-wer-{}.toml", std::process::id()));
    std::fs::write(&config_path, format!("model = \"{model}\"\n")).unwrap();

    let mut total_words = 0usize;
    let mut total_errors = 0usize;
    let mut strict_words = 0usize;
    let mut strict_errors = 0usize;
    let mut total_encode_ms = 0u64;
    let mut total_decode_ms = 0u64;
    let mut total_audio_s = 0f64;
    let mut rows = Vec::new();

    for line in expected.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (file, ref_text) = line
            .split_once(char::is_whitespace)
            .expect("bad expected.txt line");
        let wav = samples_dir.join(file);
        assert!(wav.exists(), "missing sample {}", wav.display());

        let out = Command::new(env!("CARGO_BIN_EXE_my-voice"))
            .args(["-v", "--config"])
            .arg(&config_path)
            .arg("--wav")
            .arg(&wav)
            .output()
            .expect("failed to run my-voice");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "--wav failed for {file}:\n{stderr}");
        let hyp_text = String::from_utf8_lossy(&out.stdout).trim().to_string();

        let reference = normalize(ref_text);
        let hypothesis = normalize(&hyp_text);
        let errors = edit_distance(&reference, &hypothesis);
        total_words += reference.len();
        total_errors += errors;

        let s_ref = normalize_strict(ref_text);
        let s_hyp = normalize_strict(&hyp_text);
        let s_errors = edit_distance(&s_ref, &s_hyp);
        strict_words += s_ref.len();
        strict_errors += s_errors;

        let (enc, dec) = parse_timings(&stderr).unwrap_or((0, 0));
        total_encode_ms += enc;
        total_decode_ms += dec;
        let audio_s = audio_seconds(&wav);
        total_audio_s += audio_s;
        let rtf = (enc + dec) as f64 / 1000.0 / audio_s.max(0.001);

        let wer = errors as f64 / reference.len().max(1) as f64;
        let strict = s_errors as f64 / s_ref.len().max(1) as f64;
        rows.push(format!(
            "{file}: WER {wer:.2} strict {strict:.2} ({errors}/{} words)  encode {enc}ms decode {dec}ms RTF {rtf:.2}x\n    ref: {ref_text}\n    hyp: {hyp_text}",
            reference.len()
        ));
    }
    let _ = std::fs::remove_file(&config_path);

    assert!(total_words > 0, "expected.txt had no usable lines");
    let aggregate = total_errors as f64 / total_words as f64;
    let quiet = std::env::var_os("MY_VOICE_WER_QUIET").is_some();
    let strict_agg = strict_errors as f64 / strict_words.max(1) as f64;
    let proc_s = (total_encode_ms + total_decode_ms) as f64 / 1000.0;
    let rtf = proc_s / total_audio_s.max(0.001);
    if !quiet {
        println!("\n== WER harness ({model}) ==");
        for row in &rows {
            println!("{row}");
        }
    }
    println!(
        "aggregate WER {aggregate:.3} ({total_errors}/{total_words} words), strict WER {strict_agg:.3} ({strict_errors}/{strict_words} words)"
    );
    println!(
        "total encode {total_encode_ms}ms, decode {total_decode_ms}ms over {total_audio_s:.1}s audio — RTF {rtf:.3} ({:.1}x realtime)",
        1.0 / rtf.max(0.001)
    );
    // Only the normalized WER gates; strict + RTF are reported visibility, not a hard limit (yet).
    assert!(
        aggregate <= max_wer,
        "aggregate WER {aggregate:.3} exceeds limit {max_wer}"
    );
}

#[test]
fn edit_distance_basics() {
    let r = normalize("hello hello this is a test");
    assert_eq!(edit_distance(&r, &r), 0);
    assert_eq!(edit_distance(&r, &normalize("hello this is a test")), 1);
    assert_eq!(edit_distance(&normalize("a b c"), &normalize("")), 3);
    assert_eq!(
        normalize("This is perfect, amazing job!"),
        normalize("this is perfect amazing job")
    );
}

#[test]
fn strict_sees_case_and_punctuation() {
    // The exact case normalize() is blind to: lenient collapses these, strict catches them.
    let a = "This is perfect, amazing job!";
    let b = "this is perfect amazing job";
    assert_eq!(normalize(a), normalize(b), "lenient is blind by design");
    assert_eq!(edit_distance(&normalize_strict(a), &normalize_strict(b)), 3);
    let s = "Set timer for 25 minutes";
    assert_eq!(edit_distance(&normalize_strict(s), &normalize_strict(s)), 0);
}
