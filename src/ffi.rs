//! C ABI surface — everything declared in include/whisper.h.
//! Conventions (house standard): string inputs are const char* UTF-8;
//! `*mut c_char` results are caller-owned (free with ws_free_string);
//! failures return an ERR_* code (or null / 0) + thread-local last
//! error; nothing unwinds across the boundary.

// The exported functions ARE the raw-pointer C ABI declared in
// include/whisper.h; pointer validity is the documented contract there.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

use crate::{
    engine, transcript_json, ERR_INTERNAL, ERR_INVALID_ARG, FLAG_NO_SEGMENTS, FLAG_TRANSLATE, WS_OK,
};

thread_local! {
    static LAST_ERROR: RefCell<(i32, CString)> =
        RefCell::new((WS_OK, CString::new("").unwrap()));
}

fn set_error(code: i32, msg: &str) {
    let clean: String = msg.chars().filter(|&c| c != '\0').collect();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = (
            code,
            CString::new(clean).unwrap_or_else(|_| CString::new("error").unwrap()),
        );
    });
}

fn clear_error() {
    set_error(WS_OK, "");
}

fn guard<T, F: FnOnce() -> T>(f: F) -> std::thread::Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
}

fn input(ptr: *const c_char, what: &str) -> Option<String> {
    if ptr.is_null() {
        set_error(ERR_INVALID_ARG, &format!("{what} must not be null"));
        return None;
    }
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => Some(s.to_string()),
        Err(_) => {
            set_error(ERR_INVALID_ARG, &format!("{what} is not valid UTF-8"));
            None
        }
    }
}

/// Move a Rust String across the boundary as a caller-owned C string.
fn string_out(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => {
            clear_error();
            c.into_raw()
        }
        Err(_) => {
            set_error(ERR_INTERNAL, "result contained an interior NUL byte");
            std::ptr::null_mut()
        }
    }
}

fn fail<T>(r: crate::CResult<T>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err((c, m)) => {
            set_error(c, &m);
            None
        }
    }
}

#[no_mangle]
pub extern "C" fn ws_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn ws_version_major() -> u32 {
    crate::VERSION_MAJOR
}

#[no_mangle]
pub extern "C" fn ws_version_minor() -> u32 {
    crate::VERSION_MINOR
}

#[no_mangle]
pub extern "C" fn ws_version_patch() -> u32 {
    crate::VERSION_PATCH
}

/// Load a ggml Whisper model and return a context handle (never 0),
/// or 0 on failure (see ws_last_error). Load once, transcribe many
/// times. Free with ws_context_free.
#[no_mangle]
pub extern "C" fn ws_context_create(model_path: *const c_char) -> u64 {
    match guard(|| {
        let Some(path) = input(model_path, "model path") else {
            return 0;
        };
        match fail(engine::create_context(&path)) {
            Some(h) => {
                clear_error();
                h
            }
            None => 0,
        }
    }) {
        Ok(h) => h,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            0
        }
    }
}

/// Free a context. Returns WS_OK, or WS_ERR_NOT_FOUND for a bad handle.
/// If another thread is mid-transcription on this handle, the context
/// is destroyed when that call finishes; the handle is invalid from
/// this call on either way.
#[no_mangle]
pub extern "C" fn ws_context_free(handle: u64) -> c_int {
    match guard(|| match engine::free_context(handle) {
        Ok(()) => {
            clear_error();
            WS_OK
        }
        Err((c, m)) => {
            set_error(c, &m);
            c
        }
    }) {
        Ok(c) => c,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            ERR_INTERNAL
        }
    }
}

/// Transcribe a WAV file; returns a caller-owned JSON string (free
/// with ws_free_string), or null on failure:
///
///   {"text": "...", "language": "en", "duration_ms": 11000,
///    "segments": [{"start_ms": 0, "end_ms": 11000, "text": "..."}]}
///
/// `lang`: a language code like "en" / "de", or null / "auto" / "" to
/// auto-detect. flags: WS_FLAG_TRANSLATE (decode to English),
/// WS_FLAG_NO_SEGMENTS (omit the segments array). Unknown flag bits
/// are ignored.
#[no_mangle]
pub extern "C" fn ws_transcribe_wav(
    handle: u64,
    wav_path: *const c_char,
    lang: *const c_char,
    flags: u32,
) -> *mut c_char {
    match guard(|| {
        let Some(wav_path) = input(wav_path, "wav path") else {
            return std::ptr::null_mut();
        };
        // lang is the one nullable input: null means auto-detect.
        let lang = if lang.is_null() {
            None
        } else {
            match input(lang, "language") {
                Some(l) => Some(l),
                None => return std::ptr::null_mut(),
            }
        };
        let Some((samples, duration_ms)) = fail(crate::audio::load_for_whisper(&wav_path)) else {
            return std::ptr::null_mut();
        };
        let translate = flags & FLAG_TRANSLATE != 0;
        let Some(t) = fail(engine::transcribe(
            handle,
            &samples,
            duration_ms,
            lang.as_deref(),
            translate,
        )) else {
            return std::ptr::null_mut();
        };
        string_out(transcript_json(&t, flags & FLAG_NO_SEGMENTS == 0))
    }) {
        Ok(p) => p,
        Err(_) => {
            set_error(ERR_INTERNAL, "internal panic caught at the FFI boundary");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn ws_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = guard(|| unsafe {
            drop(CString::from_raw(ptr));
        });
    }
}

#[no_mangle]
pub extern "C" fn ws_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().1.as_ptr())
}

#[no_mangle]
pub extern "C" fn ws_last_error_code() -> c_int {
    LAST_ERROR.with(|e| e.borrow().0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn cs(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    #[test]
    fn ffi_contract() {
        assert!(!ws_version().is_null());
        assert_eq!(ws_version_major(), 0);
        assert_eq!(ws_version_minor(), 1);
        assert_eq!(ws_version_patch(), 0);

        // Null model path -> INVALID_ARG, no handle.
        assert_eq!(ws_context_create(ptr::null()), 0);
        assert_eq!(ws_last_error_code(), ERR_INVALID_ARG);

        // Missing model file -> NOT_FOUND on every build flavor.
        assert_eq!(ws_context_create(cs("no/such/model.bin").as_ptr()), 0);
        assert_eq!(ws_last_error_code(), crate::ERR_NOT_FOUND);

        // Freeing an unknown handle -> NOT_FOUND.
        assert_eq!(ws_context_free(424_242), crate::ERR_NOT_FOUND as c_int);

        // Transcribe on an unknown handle -> null + NOT_FOUND.
        // (A real temp WAV is needed to get past audio loading.)
        let wav = std::env::temp_dir().join(format!("ws-ffi-test-{}.wav", std::process::id()));
        std::fs::write(&wav, tiny_wav_bytes()).unwrap();
        let p = ws_transcribe_wav(424_242, cs(wav.to_str().unwrap()).as_ptr(), ptr::null(), 0);
        assert!(p.is_null());
        assert_eq!(ws_last_error_code(), crate::ERR_NOT_FOUND);
        let _ = std::fs::remove_file(&wav);

        // Null wav path -> INVALID_ARG.
        assert!(ws_transcribe_wav(1, ptr::null(), ptr::null(), 0).is_null());
        assert_eq!(ws_last_error_code(), ERR_INVALID_ARG);

        ws_free_string(ptr::null_mut());
    }

    /// Half a second of 16 kHz mono PCM silence as a WAVE file.
    fn tiny_wav_bytes() -> Vec<u8> {
        let frames = 8_000usize;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&((36 + frames * 2) as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&16_000u32.to_le_bytes());
        v.extend_from_slice(&32_000u32.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&((frames * 2) as u32).to_le_bytes());
        v.resize(v.len() + frames * 2, 0);
        v
    }

    #[cfg(not(ws_engine))]
    #[test]
    fn engine_absent_reports_not_supported() {
        // An existing (dummy) model file passes the NOT_FOUND check and
        // reaches the engine stub.
        let f = std::env::temp_dir().join(format!("ws-dummy-model-{}.bin", std::process::id()));
        std::fs::write(&f, b"not a real model").unwrap();
        assert_eq!(ws_context_create(cs(f.to_str().unwrap()).as_ptr()), 0);
        assert_eq!(ws_last_error_code(), crate::ERR_NOT_SUPPORTED);
        let _ = std::fs::remove_file(&f);
    }
}
