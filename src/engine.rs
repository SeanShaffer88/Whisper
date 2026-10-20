//! Engine: context registry + the whisper.cpp integration.
//!
//! Contexts are opaque u64 handles into a process-wide registry (the
//! FastSearch pattern): the registry lock is held only to clone the
//! per-context `Arc<Mutex<Context>>` out, never during transcription,
//! so distinct contexts transcribe in parallel while one context is
//! used by at most one thread at a time (a caller that shares a handle
//! across threads gets serialized calls, never corruption).
//!
//! When whisper.cpp was not vendored at build time (`cfg(ws_engine)`
//! off), creation reports ERR_NOT_SUPPORTED after the caller's inputs
//! have been validated — argument and file-existence errors keep their
//! precise codes on every build.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::{CResult, Transcript, ERR_INTERNAL, ERR_NOT_FOUND};

pub struct Context {
    /// Path the model was loaded from (kept for error messages).
    model_path: String,
    #[cfg(ws_engine)]
    raw: *mut sys::WhisperContext,
}

// The raw whisper context is owned by exactly this struct and only ever
// touched while holding the per-context Mutex in the registry below.
#[cfg(ws_engine)]
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(ws_engine)]
        unsafe {
            sys::whisper_free(self.raw);
        }
    }
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Mutex<Context>>>> {
    static REG: OnceLock<Mutex<HashMap<u64, Arc<Mutex<Context>>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn lookup(handle: u64) -> CResult<Arc<Mutex<Context>>> {
    registry()
        .lock()
        .map_err(|_| (ERR_INTERNAL, "context registry lock poisoned".into()))?
        .get(&handle)
        .cloned()
        .ok_or((ERR_NOT_FOUND, format!("no such context handle: {handle}")))
}

/// Load a model and register a context; returns its handle (never 0).
/// The caller (ffi) has already verified the path argument itself; the
/// file must exist on every build so a typo'd path is ERR_NOT_FOUND,
/// not a misleading NOT_SUPPORTED.
pub fn create_context(model_path: &str) -> CResult<u64> {
    if !std::path::Path::new(model_path).is_file() {
        return Err((
            ERR_NOT_FOUND,
            format!("model file not found: '{model_path}'"),
        ));
    }
    let ctx = load_context(model_path)?;
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
    registry()
        .lock()
        .map_err(|_| (ERR_INTERNAL, "context registry lock poisoned".into()))?
        .insert(handle, Arc::new(Mutex::new(ctx)));
    Ok(handle)
}

/// Free a context. Unknown handles are ERR_NOT_FOUND.
pub fn free_context(handle: u64) -> CResult<()> {
    let removed = registry()
        .lock()
        .map_err(|_| (ERR_INTERNAL, "context registry lock poisoned".into()))?
        .remove(&handle);
    match removed {
        // The Arc may still be alive inside an in-flight call on
        // another thread; the Context (and its whisper context) drops
        // when that call finishes. From here on the handle is invalid.
        Some(_) => Ok(()),
        None => Err((ERR_NOT_FOUND, format!("no such context handle: {handle}"))),
    }
}

/// Transcribe 16 kHz mono f32 samples with the given context.
/// `lang`: None (or "auto"/"") = auto-detect. `translate`: decode to
/// English regardless of the source language.
pub fn transcribe(
    handle: u64,
    samples: &[f32],
    duration_ms: u64,
    lang: Option<&str>,
    translate: bool,
) -> CResult<Transcript> {
    let arc = lookup(handle)?;
    let ctx = arc
        .lock()
        .map_err(|_| (ERR_INTERNAL, "context lock poisoned".into()))?;
    run(&ctx, samples, duration_ms, lang, translate)
}

#[cfg(not(ws_engine))]
fn load_context(model_path: &str) -> CResult<Context> {
    let _ = model_path;
    Err((
        crate::ERR_NOT_SUPPORTED,
        "this build has no whisper engine: the vendored whisper.cpp sources were absent at \
         build time (run fetch.sh / fetch.ps1 and rebuild)"
            .into(),
    ))
}

#[cfg(not(ws_engine))]
fn run(
    ctx: &Context,
    _samples: &[f32],
    _duration_ms: u64,
    _lang: Option<&str>,
    _translate: bool,
) -> CResult<Transcript> {
    let _ = &ctx.model_path;
    Err((
        crate::ERR_NOT_SUPPORTED,
        "this build has no whisper engine (vendored sources absent at build time)".into(),
    ))
}

#[cfg(ws_engine)]
fn load_context(model_path: &str) -> CResult<Context> {
    use std::ffi::CString;
    let c_path = CString::new(model_path).map_err(|_| {
        (
            crate::ERR_INVALID_ARG,
            "model path contains a NUL byte".into(),
        )
    })?;
    let mut cparams = unsafe { sys::whisper_context_default_params() };
    // CPU-only build: say so explicitly so behavior never depends on
    // whichever GPU backends happen to be compiled in.
    cparams.use_gpu = false;
    cparams.flash_attn = false;
    let raw = unsafe { sys::whisper_init_from_file_with_params(c_path.as_ptr(), cparams) };
    if raw.is_null() {
        return Err((
            crate::ERR_PLATFORM,
            format!("failed to load model '{model_path}' (not a ggml Whisper model?)"),
        ));
    }
    Ok(Context {
        model_path: model_path.to_string(),
        raw,
    })
}

#[cfg(ws_engine)]
fn run(
    ctx: &Context,
    samples: &[f32],
    duration_ms: u64,
    lang: Option<&str>,
    translate: bool,
) -> CResult<Transcript> {
    use std::ffi::{CStr, CString};
    if samples.len() > i32::MAX as usize {
        return Err((
            crate::ERR_INVALID_ARG,
            "audio is too long for one transcription call".into(),
        ));
    }
    let _ = &ctx.model_path;
    let mut params = unsafe { sys::whisper_full_default_params(sys::WHISPER_SAMPLING_GREEDY) };
    params.translate = translate;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
        .clamp(1, 8);
    // Keep the CString alive across the call if a language is forced.
    let lang_c = match lang {
        Some(l) if !l.is_empty() && l != "auto" => Some(CString::new(l).map_err(|_| {
            (
                crate::ERR_INVALID_ARG,
                "language contains a NUL byte".into(),
            )
        })?),
        _ => None,
    };
    params.language = lang_c
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let rc = unsafe { sys::whisper_full(ctx.raw, params, samples.as_ptr(), samples.len() as i32) };
    if rc != 0 {
        return Err((
            crate::ERR_PLATFORM,
            format!("whisper_full failed with code {rc} (bad language code? corrupt model?)"),
        ));
    }
    let n = unsafe { sys::whisper_full_n_segments(ctx.raw) };
    let mut segments = Vec::with_capacity(n.max(0) as usize);
    let mut text = String::new();
    for i in 0..n {
        // whisper timestamps are in 10 ms ticks.
        let t0 = unsafe { sys::whisper_full_get_segment_t0(ctx.raw, i) };
        let t1 = unsafe { sys::whisper_full_get_segment_t1(ctx.raw, i) };
        let seg_text = unsafe {
            let p = sys::whisper_full_get_segment_text(ctx.raw, i);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        text.push_str(&seg_text);
        segments.push(crate::Segment {
            start_ms: (t0.max(0) as u64) * 10,
            end_ms: (t1.max(0) as u64) * 10,
            text: seg_text,
        });
    }
    let lang_id = unsafe { sys::whisper_full_lang_id(ctx.raw) };
    // An English-only (.en) model has exactly one language by
    // definition, but whisper_full's auto-detector still runs and can
    // report a nonsense language for it (verified against the pinned
    // engine with a pure-C call: tiny.en + the JFK clip auto-detects
    // as "sq"). Ask the engine whether the model is multilingual and
    // only trust detection when it is.
    let multilingual = unsafe { sys::whisper_is_multilingual(ctx.raw) } != 0;
    let language = if !multilingual {
        "en".to_string()
    } else {
        unsafe {
            let p = sys::whisper_lang_str(lang_id);
            if p.is_null() {
                "en".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    };
    Ok(Transcript {
        text: text.trim().to_string(),
        language,
        duration_ms,
        segments,
    })
}

/// Hand-declared whisper.cpp C API (pinned version: see pins.txt).
///
/// The two params structs are mirrored from the vendored whisper.h and
/// locked with compile-time size/offset assertions taken from a C probe
/// compiled against that exact header — if a version bump changes the
/// layout, the build fails here instead of corrupting memory. Values
/// always come from whisper's own *_default_params() constructors; this
/// code only mutates named fields.
/// whisper_full()/whisper_full_get_segment_* are documented "not thread
/// safe for the same context" — the registry's per-context Mutex is
/// what upholds that.
#[cfg(ws_engine)]
mod sys {
    #![allow(non_camel_case_types, dead_code)]
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    pub enum WhisperContext {}

    pub const WHISPER_SAMPLING_GREEDY: c_int = 0;

    #[repr(C)]
    pub struct WhisperAhead {
        pub n_text_layer: c_int,
        pub n_head: c_int,
    }

    #[repr(C)]
    pub struct WhisperAheads {
        pub n_heads: usize,
        pub heads: *const WhisperAhead,
    }

    #[repr(C)]
    pub struct WhisperContextParams {
        pub use_gpu: bool,
        pub flash_attn: bool,
        pub gpu_device: c_int,
        pub dtw_token_timestamps: bool,
        pub dtw_aheads_preset: c_int, // enum whisper_alignment_heads_preset
        pub dtw_n_top: c_int,
        pub dtw_aheads: WhisperAheads,
        pub dtw_mem_size: usize,
    }

    #[repr(C)]
    pub struct WhisperVadParams {
        pub threshold: f32,
        pub min_speech_duration_ms: c_int,
        pub min_silence_duration_ms: c_int,
        pub max_speech_duration_s: f32,
        pub speech_pad_ms: c_int,
        pub samples_overlap: f32,
    }

    #[repr(C)]
    pub struct WhisperFullParams {
        pub strategy: c_int, // enum whisper_sampling_strategy
        pub n_threads: c_int,
        pub n_max_text_ctx: c_int,
        pub offset_ms: c_int,
        pub duration_ms: c_int,
        pub translate: bool,
        pub no_context: bool,
        pub no_timestamps: bool,
        pub single_segment: bool,
        pub print_special: bool,
        pub print_progress: bool,
        pub print_realtime: bool,
        pub print_timestamps: bool,
        pub token_timestamps: bool,
        pub thold_pt: f32,
        pub thold_ptsum: f32,
        pub max_len: c_int,
        pub split_on_word: bool,
        pub max_tokens: c_int,
        pub debug_mode: bool,
        pub audio_ctx: c_int,
        pub tdrz_enable: bool,
        pub suppress_regex: *const c_char,
        pub initial_prompt: *const c_char,
        pub carry_initial_prompt: bool,
        pub prompt_tokens: *const c_int, // const whisper_token *
        pub prompt_n_tokens: c_int,
        pub language: *const c_char,
        pub detect_language: bool,
        pub suppress_blank: bool,
        pub suppress_nst: bool,
        pub temperature: f32,
        pub max_initial_ts: f32,
        pub length_penalty: f32,
        pub temperature_inc: f32,
        pub entropy_thold: f32,
        pub logprob_thold: f32,
        pub no_speech_thold: f32,
        pub greedy: Greedy,
        pub beam_search: BeamSearch,
        pub new_segment_callback: *const c_void,
        pub new_segment_callback_user_data: *mut c_void,
        pub progress_callback: *const c_void,
        pub progress_callback_user_data: *mut c_void,
        pub encoder_begin_callback: *const c_void,
        pub encoder_begin_callback_user_data: *mut c_void,
        pub abort_callback: *const c_void, // ggml_abort_callback
        pub abort_callback_user_data: *mut c_void,
        pub logits_filter_callback: *const c_void,
        pub logits_filter_callback_user_data: *mut c_void,
        pub grammar_rules: *const c_void, // const whisper_grammar_element **
        pub n_grammar_rules: usize,
        pub i_start_rule: usize,
        pub grammar_penalty: f32,
        pub vad: bool,
        pub vad_model_path: *const c_char,
        pub vad_params: WhisperVadParams,
    }

    #[repr(C)]
    pub struct Greedy {
        pub best_of: c_int,
    }

    #[repr(C)]
    pub struct BeamSearch {
        pub beam_size: c_int,
        pub patience: f32,
    }

    // Layout locks for whisper.cpp 1.9.5 (x86_64 C ABI; probed with a C
    // program against the vendored whisper.h). A mismatch fails the
    // build — re-probe and update deliberately on a version bump.
    const _: () = assert!(std::mem::size_of::<WhisperContextParams>() == 48);
    const _: () = assert!(std::mem::offset_of!(WhisperContextParams, dtw_aheads) == 24);
    const _: () = assert!(std::mem::offset_of!(WhisperContextParams, dtw_mem_size) == 40);
    const _: () = assert!(std::mem::size_of::<WhisperVadParams>() == 24);
    const _: () = assert!(std::mem::size_of::<WhisperFullParams>() == 304);
    const _: () = assert!(std::mem::offset_of!(WhisperFullParams, language) == 104);
    const _: () = assert!(std::mem::offset_of!(WhisperFullParams, temperature) == 116);
    const _: () = assert!(std::mem::offset_of!(WhisperFullParams, new_segment_callback) == 160);
    const _: () = assert!(std::mem::offset_of!(WhisperFullParams, vad_params) == 280);

    extern "C" {
        pub fn whisper_context_default_params() -> WhisperContextParams;
        pub fn whisper_init_from_file_with_params(
            path_model: *const c_char,
            params: WhisperContextParams,
        ) -> *mut WhisperContext;
        pub fn whisper_free(ctx: *mut WhisperContext);
        pub fn whisper_full_default_params(strategy: c_int) -> WhisperFullParams;
        pub fn whisper_full(
            ctx: *mut WhisperContext,
            params: WhisperFullParams,
            samples: *const f32,
            n_samples: c_int,
        ) -> c_int;
        pub fn whisper_full_n_segments(ctx: *mut WhisperContext) -> c_int;
        pub fn whisper_is_multilingual(ctx: *mut WhisperContext) -> c_int;
        pub fn whisper_full_lang_id(ctx: *mut WhisperContext) -> c_int;
        pub fn whisper_lang_str(id: c_int) -> *const c_char;
        pub fn whisper_full_get_segment_text(
            ctx: *mut WhisperContext,
            i_segment: c_int,
        ) -> *const c_char;
        pub fn whisper_full_get_segment_t0(ctx: *mut WhisperContext, i_segment: c_int) -> i64;
        pub fn whisper_full_get_segment_t1(ctx: *mut WhisperContext, i_segment: c_int) -> i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_unknown_handle_is_not_found() {
        assert_eq!(free_context(987_654_321).unwrap_err().0, ERR_NOT_FOUND);
        assert!(matches!(lookup(987_654_321), Err((ERR_NOT_FOUND, _))));
    }

    #[test]
    fn missing_model_file_is_not_found_on_every_build() {
        let err = create_context("definitely/not/a/model.bin").unwrap_err();
        assert_eq!(err.0, ERR_NOT_FOUND);
    }
}
