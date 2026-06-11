//! WER + latency regression harness over `samples/*.wav` + `samples/expected.txt`.
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
    let mut total_encode_ms = 0u64;
    let mut total_decode_ms = 0u64;
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

        let (enc, dec) = parse_timings(&stderr).unwrap_or((0, 0));
        total_encode_ms += enc;
        total_decode_ms += dec;

        let wer = errors as f64 / reference.len().max(1) as f64;
        rows.push(format!(
            "{file}: WER {wer:.2} ({errors}/{} words)  encode {enc}ms decode {dec}ms\n    ref: {ref_text}\n    hyp: {hyp_text}",
            reference.len()
        ));
    }
    let _ = std::fs::remove_file(&config_path);

    assert!(total_words > 0, "expected.txt had no usable lines");
    let aggregate = total_errors as f64 / total_words as f64;
    println!("\n== WER harness ({model}) ==");
    for row in &rows {
        println!("{row}");
    }
    println!(
        "aggregate WER {aggregate:.3} ({total_errors}/{total_words} words), total encode {total_encode_ms}ms, total decode {total_decode_ms}ms"
    );
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
