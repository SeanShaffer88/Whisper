//! End-to-end transcription proof — the real engine, the real model,
//! real speech. Ignored by default; run explicitly:
//!
//!   WS_E2E=1 WS_MODEL=vendor/ggml-tiny.en.bin \
//!   WS_SAMPLE=vendor/src-whisper/samples/jfk.wav \
//!   cargo test --test e2e -- --ignored --nocapture
//!
//! The model is the pinned test fixture from pins.txt (downloaded by
//! fetch.sh / fetch.ps1); the sample is the JFK clip that ships in the
//! whisper.cpp tree. The assertion is the famous line every Whisper
//! user knows: "And so, my fellow Americans: ask not what your country
//! can do for you — ask what you can do for your country."

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use whisper::ffi::{ws_context_create, ws_context_free, ws_free_string, ws_transcribe_wav};

fn env_path(name: &str) -> String {
    let v =
        std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set to run the e2e test"));
    assert!(
        Path::new(&v).is_file(),
        "{name}={v} is not a file (run fetch.sh / fetch.ps1 first)"
    );
    v
}

fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn take(p: *mut c_char) -> String {
    assert!(!p.is_null(), "transcription returned null");
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
    ws_free_string(p);
    s
}

#[test]
#[ignore = "needs the whisper engine + pinned model/sample (WS_E2E, WS_MODEL, WS_SAMPLE)"]
fn jfk_transcription() {
    assert_eq!(
        std::env::var("WS_E2E").as_deref(),
        Ok("1"),
        "set WS_E2E=1 to run the e2e transcription proof"
    );
    let model = env_path("WS_MODEL");
    let sample = env_path("WS_SAMPLE");

    let ctx = ws_context_create(cs(&model).as_ptr());
    assert!(ctx != 0, "context creation failed for {model}");

    // Full JSON, auto-detected language. Note: the tiny.en model is
    // English-only, and raw whisper.cpp auto-detection on this clip
    // misreports it ("sq" — verified with a pure-C call against the
    // pinned engine). The DLL reports "en" for non-multilingual
    // models via whisper_is_multilingual, which is what we assert.
    let json = take(ws_transcribe_wav(
        ctx,
        cs(&sample).as_ptr(),
        std::ptr::null(),
        0,
    ));
    println!("transcript JSON: {json}");
    let lower = json.to_lowercase();
    assert!(
        lower.contains("fellow americans"),
        "expected the JFK line in the transcript, got: {json}"
    );
    assert!(json.contains("\"language\": \"en\""), "{json}");
    assert!(json.contains("\"segments\": ["), "{json}");
    assert!(json.contains("\"duration_ms\": 11000"), "{json}");

    // NO_SEGMENTS flag drops the array but keeps the text.
    let json2 = take(ws_transcribe_wav(
        ctx,
        cs(&sample).as_ptr(),
        cs("en").as_ptr(),
        whisper::FLAG_NO_SEGMENTS,
    ));
    assert!(!json2.contains("segments"), "{json2}");
    assert!(json2.to_lowercase().contains("fellow americans"), "{json2}");
    assert!(json2.contains("\"language\": \"en\""), "{json2}");

    assert_eq!(ws_context_free(ctx), whisper::WS_OK);
    // The handle is gone now.
    assert_eq!(ws_context_free(ctx), whisper::ERR_NOT_FOUND);
}
