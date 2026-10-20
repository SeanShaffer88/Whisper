# Whisper

Offline speech-to-text behind one clean C ABI — DLL #5 in the Windows DLL Collection.

Load a ggml Whisper model once, then transcribe WAV files with a single call that returns JSON. Everything runs locally against the model file you supply: no network, no accounts, no audio ever leaving the machine.

```c
uint64_t ctx = ws_context_create("ggml-tiny.en.bin");
char *json = ws_transcribe_wav(ctx, "jfk.wav", NULL, 0);
/* Actual output for the JFK clip with tiny.en:
 * {"text": "And so my fellow Americans ask not what your country can
 *   do for you ask what you can do for your country.",
 *  "language": "en", "duration_ms": 11000,
 *  "segments": [{"start_ms": 0, "end_ms": 8000,
 *    "text": " And so my fellow Americans ask not what your country can do for you"},
 *   {"start_ms": 8000, "end_ms": 11000,
 *    "text": " ask what you can do for your country."}]} */
ws_free_string(json);
ws_context_free(ctx);
```

## How it is built

The Rust crate is zero-dependency. The recognition engine is [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT), treated the way this collection treats every upstream: **never committed, always pinned**.

- `pins.txt` records the exact version, URL, and SHA-256 of the whisper.cpp source tarball (and of the test-fixture model below).
- `fetch.sh` / `fetch.ps1` download into `vendor/` (gitignored) and refuse to proceed on a hash mismatch.
- `build.rs` configures and builds whisper.cpp + ggml as **static libraries** with CMake (Release; shared libs, examples, tests, and server off; `GGML_NATIVE=OFF` so the DLL runs on machines other than the builder; `GGML_OPENMP=OFF` so there is no OpenMP runtime dependency) and links them into the DLL.

```sh
./fetch.sh        # or .\fetch.ps1 on Windows
cargo build --release
```

Without the fetch step the crate still builds and unit-tests, and the engine entry points report `WS_ERR_NOT_SUPPORTED` — so linting and logic tests never need the ~90 MB download. (If you built *before* fetching and the engine seems missing afterwards, `cargo clean -p whisper` clears cargo's cached build probe; the fetch scripts stamp the vendored header to make this a non-issue in the normal flow.)

## Models

Bring your own ggml-format Whisper model (`ggml-*.bin`, from the whisper.cpp project). The model is a runtime input, never bundled. Rough guide:

| Model | Size | Character |
|---|---|---|
| tiny / tiny.en | ~75 MB | Fastest; rough edges on names and punctuation |
| base / base.en | ~140 MB | Solid default for clear speech |
| small / small.en | ~460 MB | Noticeably better accuracy, still real-time on modest CPUs |
| medium / large | ~1.5–3 GB | Best accuracy; wants a decent CPU (or patience) |

`.en` models are English-only and a bit more accurate on English; the multilingual builds detect the language and support `WS_FLAG_TRANSLATE`.

## C ABI (`include/whisper.h`)

| Function | Purpose |
|---|---|
| `ws_version`, `ws_version_major/minor/patch` | Version ("0.1.0") |
| `ws_context_create(model_path)` | Load a model → context handle, or 0 + `ws_last_error` |
| `ws_context_free(ctx)` | Release a context (`WS_ERR_NOT_FOUND` for a bad handle) |
| `ws_transcribe_wav(ctx, wav_path, lang, flags)` | Transcribe → caller-owned JSON string, or NULL + `ws_last_error` |
| `ws_free_string(s)` | Free a returned string |
| `ws_last_error`, `ws_last_error_code` | Thread-local error detail |

Error codes are the collection's shared 0–6: `WS_OK`, `WS_ERR_INVALID_ARG`, `WS_ERR_NOT_SUPPORTED`, `WS_ERR_NOT_FOUND`, `WS_ERR_ACCESS_DENIED`, `WS_ERR_PLATFORM`, `WS_ERR_INTERNAL`. A missing model file is `NOT_FOUND` on every build; a file that exists but is not a loadable ggml model is `PLATFORM`.

Transcription parameters: `lang` is a code like `"en"` / `"de"`, or NULL / `"auto"` / `""` to auto-detect. Flags: `WS_FLAG_TRANSLATE` (decode to English), `WS_FLAG_NO_SEGMENTS` (omit the `segments` array). Decoding is greedy (whisper.cpp's `WHISPER_SAMPLING_GREEDY` defaults otherwise).

**Threading:** distinct contexts may transcribe in parallel. One context is used by one thread at a time — concurrent calls on the same handle are serialized by the library, never corrupting state, and freeing a handle mid-call on another thread is safe (the context dies when the call finishes; the handle is invalid immediately).

## Audio input

WAV files only in v0.1: PCM integer (8/16/24/32-bit) or IEEE float (32/64-bit), any channel count, any sample rate. Channels are mixed to mono and the audio is resampled to 16 kHz internally.

## v0.1 limits (stated plainly)

- **WAV input only** — no MP3/FLAC/Ogg decoding, no microphone or streaming input.
- **Linear-interpolation resampling** — fine for speech; not a high-fidelity SRC.
- **Greedy decoding only** — no beam search, temperature fallback, VAD, or word-level timestamps (segment-level times only).
- **No diarization** — segments carry no speaker labels.
- **CPU inference only** — GPU backends are compiled out (`use_gpu` is forced off for determinism).
- Synchronous calls: a long file blocks the calling thread for the duration.

## Verification

- Unit tests: WAV decode (PCM16 stereo, PCM8, float), resampler math, JSON shape/escaping, FFI lifecycle errors, the registry contract.
- **End-to-end proof** (`tests/e2e.rs`, env-gated): transcribes the JFK sample that ships in the whisper.cpp tree with the pinned `ggml-tiny.en` test model and asserts the transcript contains "fellow Americans". It runs locally and as a labeled step in Windows CI — which is also where the MSVC link of the engine is proven.
- One engine behavior worth knowing: language auto-detection is whisper.cpp's own, and on an English-only (`.en`) model it can misfire (the JFK clip auto-detects as `"sq"` with raw whisper.cpp — verified with a pure-C call against the pinned build). The DLL therefore reports `"en"` for non-multilingual models (via `whisper_is_multilingual`) and only surfaces detection for multilingual ones.

## Roadmap

See ROADMAP.md.

## License

MIT (see LICENSE). whisper.cpp is MIT-licensed upstream; this repo does not redistribute its sources — the fetch step downloads them from the pinned upstream URL and verifies the hash.
