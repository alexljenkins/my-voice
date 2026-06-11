/// Static model registry — the single source of truth for every named model.
///
/// Every model is Moonshine (ONNX). Adding a new auto-downloadable model means
/// adding one entry here; no other files need touching.
pub type FileEntry = (&'static str, &'static str);

pub struct ModelSpec {
    /// The string users put in `config.model`, e.g. `"moonshine-tiny"`.
    pub name: &'static str,
    /// Display label shown in the tray menu.
    pub label: &'static str,
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
    /// Approximate download size of the quantized (default) file set, in MB.
    /// User-facing text only — never used for allocation or verification.
    pub approx_mb: u32,
}

pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "moonshine-tiny",
        label: "Faster  •  moonshine-tiny",
        hf_repo: "onnx-community/moonshine-tiny-ONNX",
        files_quantized: &[
            (
                "onnx/encoder_model_quantized.onnx",
                "encoder_model_quantized.onnx",
            ),
            (
                "onnx/decoder_model_merged_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        files_full: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        sentinel_quantized: "encoder_model_quantized.onnx",
        sentinel_full: "encoder_model.onnx",
        checksums: &[
            (
                "encoder_model_quantized.onnx",
                "c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a",
            ),
            (
                "decoder_model_merged_quantized.onnx",
                "eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367",
            ),
            (
                "encoder_model.onnx",
                "cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5",
            ),
            (
                "decoder_model_merged.onnx",
                "4131cef00b62942e9cdef691101f2cc7dbbcd828d71eee8c6c46c28fd051d6cb",
            ),
        ],
        approx_mb: 31,
    },
    ModelSpec {
        name: "moonshine-base",
        label: "Balanced  •  moonshine-base",
        hf_repo: "onnx-community/moonshine-base-ONNX",
        files_quantized: &[
            (
                "onnx/encoder_model_quantized.onnx",
                "encoder_model_quantized.onnx",
            ),
            (
                "onnx/decoder_model_merged_quantized.onnx",
                "decoder_model_merged_quantized.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        files_full: &[
            ("onnx/encoder_model.onnx", "encoder_model.onnx"),
            (
                "onnx/decoder_model_merged.onnx",
                "decoder_model_merged.onnx",
            ),
            ("tokenizer.json", "tokenizer.json"),
        ],
        sentinel_quantized: "encoder_model_quantized.onnx",
        sentinel_full: "encoder_model.onnx",
        checksums: &[
            (
                "encoder_model_quantized.onnx",
                "1dd9ab0a7f987113d30affcba5a068d11c8f90fa0223caa3e491ade431ad9751",
            ),
            (
                "decoder_model_merged_quantized.onnx",
                "cc9f3cd6698a369c6008b41aa60aa3fb3322e7f03c9bdf19d8e6b7200afca4f3",
            ),
            (
                "encoder_model.onnx",
                "153e128e7abd64a74ee47f2c3f585c3171c4d46cbb368b032827934c4e01e779",
            ),
            (
                "decoder_model_merged.onnx",
                "58778763ca8438963190244d6b26572bdca2cedec56a4b91e828f3f2d69ef3c5",
            ),
        ],
        approx_mb: 64,
    },
    // Streaming Moonshine (split-decoder ONNX). These ship int8-quantized only,
    // so the "full" file set is identical to the quantized one. We run them as a
    // single-pass push-to-talk transcription (full audio at once), not chunked.
    ModelSpec {
        name: "moonshine-streaming-small",
        label: "Accurate  •  moonshine-small",
        hf_repo: "Mazino0/moonshine-streaming-small-onnx",
        files_quantized: STREAMING_FILES,
        files_full: STREAMING_FILES,
        sentinel_quantized: "encoder_model_int8.onnx",
        sentinel_full: "encoder_model_int8.onnx",
        checksums: &[
            (
                "encoder_model_int8.onnx",
                "9bb6562667da35c8b6994bd76139528610738a33c1c3fa234024c75a6affa509",
            ),
            (
                "decoder_model_int8.onnx",
                "8c1a86e1b3059950d8285a47f3dae1fb6166f0337046e115965498e7957be158",
            ),
            (
                "decoder_with_past_model_int8.onnx",
                "e9bfbc4f2b34ea82ff5b562cc20d3eafcf87a8a25ea9bcaabd8513078dbc0565",
            ),
            (
                "tokenizer.json",
                "7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f",
            ),
        ],
        approx_mb: 345,
    },
    ModelSpec {
        name: "moonshine-streaming-medium",
        label: "Most accurate  •  moonshine-medium",
        hf_repo: "Mazino0/moonshine-streaming-medium-onnx",
        files_quantized: STREAMING_FILES,
        files_full: STREAMING_FILES,
        sentinel_quantized: "encoder_model_int8.onnx",
        sentinel_full: "encoder_model_int8.onnx",
        checksums: &[
            (
                "encoder_model_int8.onnx",
                "4f6c491eb4018a06f2e9ecf5b6bab5c6fa4e679c9ed5dde02a0a27969649be90",
            ),
            (
                "decoder_model_int8.onnx",
                "38dfe5829fcb814e33634c00baedceaa877acaac7b731203e88eb956d4419875",
            ),
            (
                "decoder_with_past_model_int8.onnx",
                "36d7ea3cf4feb6e37fe784ba3ac7cee0bb5f4d757ab05433e2550b8eae035a7e",
            ),
            (
                "tokenizer.json",
                "7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f",
            ),
        ],
        approx_mb: 566,
    },
];

/// Streaming repos lay files at the repo root with `_int8` suffixes and ship no
/// full-precision variant, so quantized and full share this list.
const STREAMING_FILES: &[FileEntry] = &[
    ("encoder_model_int8.onnx", "encoder_model_int8.onnx"),
    ("decoder_model_int8.onnx", "decoder_model_int8.onnx"),
    (
        "decoder_with_past_model_int8.onnx",
        "decoder_with_past_model_int8.onnx",
    ),
    ("tokenizer.json", "tokenizer.json"),
];

/// Look up a named model in the registry.
pub fn find(name: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|s| s.name == name)
}
