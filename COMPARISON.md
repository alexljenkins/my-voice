# my-voice vs FluidVoice — user-facing comparison

Both are push-to-talk / hotkey dictation tools that type transcribed speech into whatever app is focused, running fully on-device. Below is what a user would actually notice day-to-day — not the tech stack.

| | **my-voice** | **[FluidVoice](https://github.com/altic-dev/FluidVoice)** |
|---|---|---|
| **Languages** | English only (Moonshine models) | ~99 languages (Whisper), ~40 (Nemotron), 25 (Parakeed TDT v3), 14 (Cohere) — pick per language/latency need |
| **Feels-like latency** | Push-to-talk: hold key, speak, release, then wait for one transcription pass. No feedback while speaking. | Live preview overlay — words appear on screen *as you speak* ("Parakeet Flash" aims for near-zero delay). Push-to-talk model also feels near-instant on Apple Silicon. |
| **Model sizes / footprint** | 31 MB–566 MB (default 345 MB). Smallest tier (~31 MB) targets weak CPUs. | 250 MB–2.9 GB per speech model, +3.5 GB optional for the "Fluid Intelligence" local AI enhancer. Materially heavier disk/RAM budget. |
| **Post-processing / AI enhancement** | None — raw transcript only, plus a static find/replace `corrections` list in config. | Optional AI rewrite/formatting layer (cloud: OpenAI/Groq, or local "Fluid Intelligence"), "Write Mode" to rewrite selected text in place, "Command Mode" to trigger app actions by voice. |
| **Extra features** | Clipboard-paste mode, multiple text-injection backends for Wayland/X11 compatibility, corrections list. | Audio history w/ export, per-app prompt profiles, usage stats, notch-aware overlay sizing, auto-update channel. |
| **Openness** | Fully open source. | Core app is GPLv3 open source; the AI-enhancement layer ("Fluid Intelligence") is closed-source. |

## The 3 differences a user would actually feel

1. **English-only vs. near-universal language support.** my-voice is hard-limited to English (Moonshine has no other language models). FluidVoice covers ~99 languages via Whisper and offers several other engines tuned for different language/latency trade-offs. Any non-English speaker notices this immediately — it's not a matter of degree.

2. **Silent push-to-talk vs. live streaming preview.** my-voice gives no feedback until you release the key — you speak into silence, then text appears (or doesn't, if something went wrong). FluidVoice shows words appearing in real time while you're still talking, which changes the felt experience from "record and hope" to "see it working."

3. **Bare transcript vs. AI-enhanced dictation assistant.** my-voice types exactly what was said (barring a manual find/replace list) — nothing more. FluidVoice layers in optional AI cleanup (punctuation/formatting/context-aware rewriting), a rewrite-selected-text mode, and voice-triggered system commands. That turns FluidVoice into a broader productivity tool rather than a pure speech-to-text pipe — at the cost of it no longer being purely local by default if a cloud provider is chosen, and a much bigger disk/RAM footprint if the local "Fluid Intelligence" model is enabled.

**Where my-voice wins on feel:** lower resource footprint (a 31–64 MB model vs. hundreds of MB to gigabytes) and a fully open, no-hidden-layer transcript — nothing is ever sent anywhere, and there's no closed-source component in the pipeline.
