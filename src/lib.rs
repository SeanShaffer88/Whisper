//! Whisper — pure logic core: error convention, JSON, transcript model.
//! The audio pipeline lives in [`audio`], the whisper.cpp engine in
//! [`engine`], and the C ABI in [`ffi`].
//!
//! Design: load a ggml Whisper model once into a context handle, then
//! transcribe WAV files with it. Transcription is fully offline — the
//! model file is the only input besides the audio. Nothing is uploaded
//! anywhere.
//!
//! The engine (whisper.cpp) is a vendored component: pinned in
//! pins.txt, fetched + hash-verified by fetch scripts, never committed.
//! When it is absent at build time the crate still compiles and its
//! engine entry points report [`ERR_NOT_SUPPORTED`].

pub mod audio;
pub mod engine;
pub mod ffi;

pub const VERSION: &str = "0.1.0";
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

pub const WS_OK: i32 = 0;
pub const ERR_INVALID_ARG: i32 = 1;
pub const ERR_NOT_SUPPORTED: i32 = 2;
pub const ERR_NOT_FOUND: i32 = 3;
pub const ERR_ACCESS_DENIED: i32 = 4;
pub const ERR_PLATFORM: i32 = 5;
pub const ERR_INTERNAL: i32 = 6;

pub type CResult<T> = Result<T, (i32, String)>;

/// `ws_transcribe_wav` flag: translate the audio to English.
pub const FLAG_TRANSLATE: u32 = 1 << 0;
/// `ws_transcribe_wav` flag: omit the per-segment list from the JSON.
pub const FLAG_NO_SEGMENTS: u32 = 1 << 1;

/// One transcript segment (times in milliseconds from the audio start).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// A completed transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    /// Concatenation of the segment texts, trimmed.
    pub text: String,
    /// Detected (or requested) language code, e.g. "en".
    pub language: String,
    /// Duration of the transcribed audio.
    pub duration_ms: u64,
    pub segments: Vec<Segment>,
}

/// Render a transcript in the JSON shape `ws_transcribe_wav` returns:
/// `{"text": ..., "language": ..., "duration_ms": ..., "segments": [...]}`
/// with the `segments` key omitted entirely when `include_segments` is
/// false (the FLAG_NO_SEGMENTS behavior).
pub fn transcript_json(t: &Transcript, include_segments: bool) -> String {
    let mut out = String::with_capacity(t.text.len() + 128);
    out.push_str("{\"text\": \"");
    out.push_str(&json_escape(&t.text));
    out.push_str("\", \"language\": \"");
    out.push_str(&json_escape(&t.language));
    out.push_str("\", \"duration_ms\": ");
    out.push_str(&t.duration_ms.to_string());
    if include_segments {
        out.push_str(", \"segments\": [");
        for (i, s) in t.segments.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str("{\"start_ms\": ");
            out.push_str(&s.start_ms.to_string());
            out.push_str(", \"end_ms\": ");
            out.push_str(&s.end_ms.to_string());
            out.push_str(", \"text\": \"");
            out.push_str(&json_escape(&s.text));
            out.push_str("\"}");
        }
        out.push(']');
    }
    out.push('}');
    out
}

pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Transcript {
        Transcript {
            text: "Ask not what your country can do for you.".into(),
            language: "en".into(),
            duration_ms: 11_000,
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 5_400,
                    text: " Ask not what your country".into(),
                },
                Segment {
                    start_ms: 5_400,
                    end_ms: 11_000,
                    text: " can do for you.".into(),
                },
            ],
        }
    }

    #[test]
    fn json_shape_with_segments() {
        let j = transcript_json(&sample(), true);
        assert_eq!(
            j,
            "{\"text\": \"Ask not what your country can do for you.\", \
             \"language\": \"en\", \"duration_ms\": 11000, \"segments\": [\
             {\"start_ms\": 0, \"end_ms\": 5400, \"text\": \" Ask not what your country\"}, \
             {\"start_ms\": 5400, \"end_ms\": 11000, \"text\": \" can do for you.\"}]}"
        );
    }

    #[test]
    fn json_shape_without_segments_omits_key() {
        let j = transcript_json(&sample(), false);
        assert!(!j.contains("segments"), "{j}");
        assert!(j.contains("\"duration_ms\": 11000"), "{j}");
    }

    #[test]
    fn json_escapes_quotes_backslash_and_controls() {
        let mut t = sample();
        t.text = "say \"hi\" \\ \u{1}\n".into();
        let j = transcript_json(&t, false);
        assert!(j.contains("say \\\"hi\\\" \\\\ \\u0001\\n"), "{j}");
    }

    #[test]
    fn json_empty_segments_is_empty_array() {
        let mut t = sample();
        t.segments.clear();
        let j = transcript_json(&t, true);
        assert!(j.contains("\"segments\": []"), "{j}");
    }
}
