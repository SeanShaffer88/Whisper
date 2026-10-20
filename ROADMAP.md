# Whisper Roadmap

## v0.1 (current)
Vendored whisper.cpp engine (pinned, hash-verified, statically linked), ggml model contexts behind u64 handles, WAV→JSON transcription with segments, greedy decoding, language auto-detect or force, translate flag, CPU-only. Proven end-to-end locally (JFK sample, tiny.en) and in Windows CI.

## Candidates
- **In-memory transcription**: `ws_transcribe_pcm(ctx, samples, n, ...)` so hosts that already decoded audio (or capture it live) skip the temp-WAV round-trip.
- **More input formats**: a small, permissively-licensed decoder story for MP3/FLAC/Ogg — evaluated against the collection's dependency rules before anything is promised.
- **Better resampling**: a windowed-sinc resampler behind the same front-end when the linear interpolator's quality is measured to matter.
- **Decoding controls**: beam search / temperature fallback parameters, `no_speech` thresholds, word-level timestamps (`token_timestamps` + DTW) as opt-in flags.
- **Streaming**: chunked transcription with partial results via callback — needs a careful FFI callback contract (threading + lifetime) before it is designed, let alone shipped.
- **GPU backends**: opt-in build feature (Vulkan/CUDA) without changing the CPU default or the ABI.

## Explicitly not planned
- Bundling or downloading models at runtime. The model is the caller's input, chosen by the app and its user; the DLL never phones home for one.
- Any network component of any kind. Transcription is local by design.
