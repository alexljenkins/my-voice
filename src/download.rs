//! HuggingFace model fetcher for `--download`.
//!
//! English-only Moonshine ONNX, MIT. Streams each file to `{name}.part` and
//! renames on completion so a Ctrl-C mid-download never leaves a truncated file
//! masquerading as complete. Called by `--download` and auto-triggered when the
//! model directory is absent.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use tracing::info;

use crate::config::Config;

fn repo_for(model: &str) -> Option<&'static str> {
    match model {
        "moonshine-tiny" => Some("onnx-community/moonshine-tiny-ONNX"),
        "moonshine-base" => Some("onnx-community/moonshine-base-ONNX"),
        _ => None,
    }
}

/// Fetch the configured model's three files into `{model_dir}/{model}/`.
pub fn run(config: &Config) -> Result<()> {
    let Some(repo) = repo_for(&config.model) else {
        bail!(
            "--download supports only moonshine-tiny / moonshine-base; \
             '{}' is a path — place the model files there yourself",
            config.model
        );
    };

    let dest = config.resolved_model_dir().join(&config.model);
    fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    // When quantized, fetch only the quantized pair (smaller, faster, negligible
    // WER cost); the full-precision pair otherwise.
    let (encoder, decoder) = if config.quantized {
        (
            "onnx/encoder_model_quantized.onnx",
            "onnx/decoder_model_merged_quantized.onnx",
        )
    } else {
        ("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx")
    };

    for remote in [encoder, decoder, "tokenizer.json"] {
        let basename = Path::new(remote).file_name().unwrap();
        download_file(repo, remote, &dest.join(basename))?;
    }

    info!("model ready at {}", dest.display());
    Ok(())
}

fn download_file(repo: &str, remote: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        info!("have {}", dest.display());
        return Ok(());
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{remote}");
    eprintln!("downloading {url}");

    let resp = ureq::get(&url)
        .call()
        .with_context(|| format!("GET {url}"))?;

    let part = dest.with_extension("part");
    let mut reader = resp.into_reader();
    let mut file =
        fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;

    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf).context("reading response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        total += n as u64;
        eprint!("\r  {} KiB", total / 1024);
    }
    eprintln!();
    file.sync_all().ok();

    fs::rename(&part, dest).with_context(|| format!("renaming into {}", dest.display()))?;
    Ok(())
}
