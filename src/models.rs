/// Static model registry — the single source of truth for every named model.
///
/// Adding a new auto-downloadable model means adding one entry here; no other
/// files need touching.

pub type FileEntry = (&'static str, &'static str);

/// Which transcription backend a model requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    Moonshine,
    Whisper,
    /// Not yet implemented — reserved for a future Parakeet transcriber.
    #[allow(dead_code)]
    Parakeet,
}

pub struct ModelSpec {
    /// The string users put in `config.model`, e.g. `"moonshine-tiny"`.
    pub name: &'static str,
    /// Display label shown in the tray menu.
    pub label: &'static str,
    pub backend: Backend,
    /// HuggingFace repo slug, e.g. `"onnx-community/moonshine-tiny-ONNX"`.
    pub hf_repo: &'static str,
    /// Files to download when `config.quantized = true`. Each entry is
    /// `(remote_path_in_repo, local_filename)`.
    pub files_quantized: &'static [FileEntry],
    /// Files to download when `config.quantized = false`.
    pub files_full: &'static [FileEntry],
    /// Local filename (inside the model subdirectory) checked to determine
    /// whether the quantized variant is downloaded.
    pub sentinel_quantized: &'static str,
    /// Local filename checked when `config.quantized = false`.
    pub sentinel_full: &'static str,
    /// SHA-256 checksums for pinned files: `(local_filename, hex_digest)`.
    /// Files not listed here skip integrity verification.
    pub checksums: &'static [(&'static str, &'static str)],
    /// True = only show/download when compiled with `--features whisper`.
    pub whisper_feature: bool,
}

pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "moonshine-tiny",
        label: "Faster  •  moonshine-tiny",
        backend: Backend::Moonshine,
        hf_repo: "onnx-community/moonshine-tiny-ONNX",
        files_quantized: &[
            ("onnx/encoder_model_quantized.onnx", "encoder_model_quantized.onnx"),
            ("onnx/decoder_model_merged_quantized.onnx", "decoder_model_merged_quantized.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
        files_full: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            ("onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
        sentinel_quantized: "encoder_model_quantized.onnx",
        sentinel_full: "encoder_model.onnx",
        checksums: &[
            ("encoder_model_quantized.onnx", "c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a"),
            ("decoder_model_merged_quantized.onnx", "eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367"),
            ("encoder_model.onnx", "cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5"),
            ("decoder_model_merged.onnx", "4131cef00b62942e9cdef691101f2cc7dbbcd828d71eee8c6c46c28fd051d6cb"),
        ],
        whisper_feature: false,
    },
    ModelSpec {
        name: "moonshine-base",
        label: "Accurate  •  moonshine-base",
        backend: Backend::Moonshine,
        hf_repo: "onnx-community/moonshine-base-ONNX",
        files_quantized: &[
            ("onnx/encoder_model_quantized.onnx", "encoder_model_quantized.onnx"),
            ("onnx/decoder_model_merged_quantized.onnx", "decoder_model_merged_quantized.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
        files_full: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            ("onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
        sentinel_quantized: "encoder_model_quantized.onnx",
        sentinel_full: "encoder_model.onnx",
        checksums: &[
            ("encoder_model_quantized.onnx", "1dd9ab0a7f987113d30affcba5a068d11c8f90fa0223caa3e491ade431ad9751"),
            ("decoder_model_merged_quantized.onnx", "cc9f3cd6698a369c6008b41aa60aa3fb3322e7f03c9bdf19d8e6b7200afca4f3"),
            ("encoder_model.onnx", "153e128e7abd64a74ee47f2c3f585c3171c4d46cbb368b032827934c4e01e779"),
            ("decoder_model_merged.onnx", "58778763ca8438963190244d6b26572bdca2cedec56a4b91e828f3f2d69ef3c5"),
        ],
        whisper_feature: false,
    },
    ModelSpec {
        name: "distil-whisper-large-v3",
        label: "Robust  •  distil-whisper-large-v3",
        backend: Backend::Whisper,
        hf_repo: "distil-whisper/distil-large-v3-ggml",
        files_quantized: &[("ggml-distil-large-v3.bin", "ggml-distil-large-v3.bin")],
        files_full: &[("ggml-distil-large-v3.bin", "ggml-distil-large-v3.bin")],
        sentinel_quantized: "ggml-distil-large-v3.bin",
        sentinel_full: "ggml-distil-large-v3.bin",
        checksums: &[], // TODO: pin sha256 after first verified download
        whisper_feature: true,
    },
];

/// Look up a named model in the registry.
pub fn find(name: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|s| s.name == name)
}
