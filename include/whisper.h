/* Whisper.dll — offline speech-to-text behind one clean C ABI.
 *
 * Load a ggml Whisper model once, transcribe WAV files many times:
 *
 *     uint64_t ctx = ws_context_create("ggml-base.en.bin");
 *     char *json = ws_transcribe_wav(ctx, "speech.wav", NULL, 0);
 *     // {"text": "...", "language": "en", "duration_ms": 11000,
 *     //  "segments": [{"start_ms": 0, "end_ms": 11000, "text": "..."}]}
 *     ws_free_string(json);
 *     ws_context_free(ctx);
 *
 * Everything runs locally against the model file you supply; no audio
 * or text ever leaves the machine.
 *
 * Strings are UTF-8. Returned char* are caller-owned: free with
 * ws_free_string(). Errors: thread-local ws_last_error() /
 * ws_last_error_code().
 *
 * Threading: distinct contexts may transcribe in parallel. One context
 * must be used by one thread at a time — concurrent calls on the same
 * handle are serialized by the library, and freeing a handle while a
 * transcription is in flight on another thread is safe (the context
 * dies when the call finishes; the handle is invalid immediately).
 *
 * NOTE: this header declares the DLL's own ABI. It is not the
 * whisper.cpp engine header, despite the file name — if you vendor
 * both, include them from different directories. */
#ifndef WHISPER_DLL_H
#define WHISPER_DLL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(WS_BUILD)
#define WS_API __declspec(dllexport)
#elif defined(_WIN32)
#define WS_API __declspec(dllimport)
#else
#define WS_API
#endif

enum {
    WS_OK = 0,
    WS_ERR_INVALID_ARG = 1,
    WS_ERR_NOT_SUPPORTED = 2,
    WS_ERR_NOT_FOUND = 3,
    WS_ERR_ACCESS_DENIED = 4,
    WS_ERR_PLATFORM = 5,
    WS_ERR_INTERNAL = 6
};

/* ws_transcribe_wav flags. Unknown bits are ignored. */
#define WS_FLAG_TRANSLATE    (1u << 0) /* decode to English */
#define WS_FLAG_NO_SEGMENTS  (1u << 1) /* omit the segments array */

/* "0.1.0" — static string, do not free. */
WS_API const char *ws_version(void);
WS_API uint32_t ws_version_major(void);
WS_API uint32_t ws_version_minor(void);
WS_API uint32_t ws_version_patch(void);

/* Load a ggml-format Whisper model; returns a context handle (never
 * 0), or 0 on failure. NOT_FOUND: the file does not exist.
 * NOT_SUPPORTED: this build has no engine compiled in. PLATFORM: the
 * file exists but is not a loadable ggml Whisper model. */
WS_API uint64_t ws_context_create(const char *model_path);

/* Free a context. WS_OK, or WS_ERR_NOT_FOUND for an unknown handle. */
WS_API int ws_context_free(uint64_t ctx);

/* Transcribe a WAV file (PCM 8/16/24/32-bit or IEEE float, any
 * channels, any sample rate; resampled internally to 16 kHz mono).
 * lang: "en", "de", ... or NULL / "auto" / "" to auto-detect.
 * Returns a caller-owned JSON string, or NULL on failure. */
WS_API char *ws_transcribe_wav(uint64_t ctx, const char *wav_path,
                               const char *lang, uint32_t flags);

WS_API void ws_free_string(char *s);
WS_API const char *ws_last_error(void);
WS_API int ws_last_error_code(void);

#ifdef __cplusplus
}
#endif

#endif /* WHISPER_DLL_H */
