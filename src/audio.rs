//! WAV decoding + resampling — the Rust audio front-end.
//!
//! Input contract for v0.1: RIFF/WAVE files with PCM (8/16/24/32-bit
//! integer) or IEEE float samples, any channel count, any sample rate.
//! Everything is mixed down to mono f32 in [-1, 1] and resampled to the
//! 16 kHz the Whisper model expects. Resampling is linear interpolation
//! — adequate for speech, and its simplicity is deliberate (see the
//! README's v0.1 limits).

use crate::{CResult, ERR_INVALID_ARG, ERR_NOT_FOUND, ERR_NOT_SUPPORTED, ERR_PLATFORM};

/// The sample rate whisper.cpp requires.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decoded audio: mono f32 samples at `sample_rate`.
#[derive(Clone, Debug, PartialEq)]
pub struct WavData {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

/// Read + decode a WAV file from disk.
pub fn read_wav_file(path: &str) -> CResult<WavData> {
    let bytes = std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => (ERR_NOT_FOUND, format!("cannot read '{path}': {e}")),
        std::io::ErrorKind::PermissionDenied => (
            crate::ERR_ACCESS_DENIED,
            format!("cannot read '{path}': {e}"),
        ),
        _ => (ERR_PLATFORM, format!("cannot read '{path}': {e}")),
    })?;
    parse_wav(&bytes)
}

/// Decode RIFF/WAVE bytes to mono f32.
pub fn parse_wav(bytes: &[u8]) -> CResult<WavData> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err((
            ERR_INVALID_ARG,
            "not a WAVE file (missing RIFF/WAVE header)".into(),
        ));
    }
    let mut fmt: Option<Fmt> = None;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let start = pos + 8;
        let end = start
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or((
                ERR_INVALID_ARG,
                "WAVE chunk extends past end of file".into(),
            ))?;
        match id {
            b"fmt " => fmt = Some(parse_fmt(&bytes[start..end])?),
            b"data" => data = Some(&bytes[start..end]),
            _ => {} // fact/LIST/etc. chunks are skipped
        }
        // Chunks are word-aligned: odd sizes carry one pad byte.
        pos = end + (size % 2);
    }
    let fmt = fmt.ok_or((ERR_INVALID_ARG, "WAVE file has no fmt chunk".into()))?;
    let data = data.ok_or((ERR_INVALID_ARG, "WAVE file has no data chunk".into()))?;
    if fmt.channels == 0 {
        return Err((ERR_INVALID_ARG, "WAVE file declares zero channels".into()));
    }
    if fmt.sample_rate == 0 {
        return Err((
            ERR_INVALID_ARG,
            "WAVE file declares a zero sample rate".into(),
        ));
    }
    let samples = decode(&fmt, data)?;
    if samples.is_empty() {
        return Err((ERR_INVALID_ARG, "WAVE file contains no audio frames".into()));
    }
    Ok(WavData {
        sample_rate: fmt.sample_rate,
        samples,
    })
}

struct Fmt {
    format: u16, // 1 = PCM integer, 3 = IEEE float
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

fn parse_fmt(b: &[u8]) -> CResult<Fmt> {
    if b.len() < 16 {
        return Err((ERR_INVALID_ARG, "WAVE fmt chunk is truncated".into()));
    }
    let rd16 = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
    let rd32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    Ok(Fmt {
        format: rd16(0),
        channels: rd16(2),
        sample_rate: rd32(4),
        bits: rd16(14),
    })
}

fn decode(fmt: &Fmt, data: &[u8]) -> CResult<Vec<f32>> {
    let ch = fmt.channels as usize;
    let bytes_per_sample = (fmt.bits / 8) as usize;
    if bytes_per_sample == 0 {
        return Err((
            ERR_INVALID_ARG,
            "WAVE file declares zero bits per sample".into(),
        ));
    }
    let frame_bytes = bytes_per_sample * ch;
    let frames = data.len() / frame_bytes;
    let mut out = Vec::with_capacity(frames);
    let read_one = |raw: &[u8]| -> CResult<f32> {
        match (fmt.format, fmt.bits) {
            // 8-bit PCM is unsigned, centered on 128.
            (1, 8) => Ok((raw[0] as f32 - 128.0) / 128.0),
            (1, 16) => Ok(i16::from_le_bytes(raw.try_into().unwrap()) as f32 / 32_768.0),
            (1, 24) => {
                let v = i32::from_le_bytes([raw[0], raw[1], raw[2], (raw[2] >> 7) * 0xFF]);
                Ok(v as f32 / 8_388_608.0)
            }
            (1, 32) => Ok(i32::from_le_bytes(raw.try_into().unwrap()) as f32 / 2_147_483_648.0),
            (3, 32) => Ok(f32::from_le_bytes(raw.try_into().unwrap()).clamp(-1.0, 1.0)),
            (3, 64) => Ok((f64::from_le_bytes(raw.try_into().unwrap()) as f32).clamp(-1.0, 1.0)),
            (1, b) => Err((
                ERR_NOT_SUPPORTED,
                format!("unsupported PCM bit depth: {b} (supported: 8/16/24/32)"),
            )),
            (3, b) => Err((
                ERR_NOT_SUPPORTED,
                format!("unsupported float bit depth: {b} (supported: 32/64)"),
            )),
            (f, _) => Err((
                ERR_NOT_SUPPORTED,
                format!("unsupported WAVE format tag: {f} (supported: 1 = PCM, 3 = IEEE float)"),
            )),
        }
    };
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            let off = (f * ch + c) * bytes_per_sample;
            acc += read_one(&data[off..off + bytes_per_sample])?;
        }
        out.push(acc / ch as f32);
    }
    Ok(out)
}

/// Linear-interpolation resample. Output length is
/// `round(len * to / from)`; a constant signal stays constant.
pub fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if samples.is_empty() || from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to {
        return samples.to_vec();
    }
    let n = samples.len();
    let out_len = ((n as u64 * to as u64) / from as u64).max(1) as usize;
    let step = from as f64 / to as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * step;
        let idx = pos.floor() as usize;
        if idx + 1 >= n {
            out.push(samples[n - 1]);
        } else {
            let frac = (pos - idx as f64) as f32;
            out.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        }
    }
    out
}

/// Full front-end: read a WAV file and return 16 kHz mono f32 samples
/// plus the audio duration in milliseconds.
pub fn load_for_whisper(path: &str) -> CResult<(Vec<f32>, u64)> {
    let wav = read_wav_file(path)?;
    let samples = resample_linear(&wav.samples, wav.sample_rate, WHISPER_SAMPLE_RATE);
    let duration_ms = samples.len() as u64 * 1000 / WHISPER_SAMPLE_RATE as u64;
    Ok((samples, duration_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a WAVE byte vector by hand (fmt sizes stay textbook: the
    /// fmt chunk is exactly 16 bytes).
    fn make_wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&format.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        let byte_rate = rate * channels as u32 * (bits / 8) as u32;
        v.extend_from_slice(&byte_rate.to_le_bytes());
        v.extend_from_slice(&(channels * (bits / 8)).to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
        if data.len() % 2 == 1 {
            v.push(0);
        }
        v
    }

    fn sine_pcm16_stereo(rate: u32, frames: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(frames * 4);
        for i in 0..frames {
            let t = i as f32 / rate as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let v = (s * 16_384.0) as i16;
            data.extend_from_slice(&v.to_le_bytes()); // L
            data.extend_from_slice(&(-v).to_le_bytes()); // R: inverted — mixdown of a frame is (v + -v)/2
        }
        data
    }

    #[test]
    fn pcm16_stereo_44100_decodes_to_mono() {
        let data = sine_pcm16_stereo(44_100, 4_410); // 0.1 s
        let wav = parse_wav(&make_wav(1, 2, 44_100, 16, &data)).unwrap();
        assert_eq!(wav.sample_rate, 44_100);
        assert_eq!(wav.samples.len(), 4_410);
        // L and R are inverted copies, so every mixed frame is ~0.
        assert!(wav.samples.iter().all(|s| s.abs() < 0.001));
    }

    #[test]
    fn pcm16_mono_values_scale() {
        let mut data = Vec::new();
        for v in [0i16, 16_384, -16_384, 32_767] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wav = parse_wav(&make_wav(1, 1, 16_000, 16, &data)).unwrap();
        assert_eq!(wav.samples[0], 0.0);
        assert_eq!(wav.samples[1], 0.5);
        assert_eq!(wav.samples[2], -0.5);
        assert!(wav.samples[3] > 0.9999);
    }

    #[test]
    fn pcm8_mono_8000_unsigned_centered() {
        let data = [128u8, 255, 0, 192];
        let wav = parse_wav(&make_wav(1, 1, 8_000, 8, &data)).unwrap();
        assert_eq!(wav.sample_rate, 8_000);
        assert_eq!(wav.samples[0], 0.0);
        assert!(wav.samples[1] > 0.99);
        assert_eq!(wav.samples[2], -1.0);
        assert_eq!(wav.samples[3], 0.5);
    }

    #[test]
    fn float32_mono_passes_through_and_clamps() {
        let mut data = Vec::new();
        for v in [0.25f32, -0.25, 2.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wav = parse_wav(&make_wav(3, 1, 16_000, 32, &data)).unwrap();
        assert_eq!(wav.samples, vec![0.25, -0.25, 1.0]);
    }

    #[test]
    fn rejects_non_wav_and_empty_audio() {
        let err = parse_wav(b"this is not a wav file at all, nope").unwrap_err();
        assert_eq!(err.0, ERR_INVALID_ARG);
        let err = parse_wav(&make_wav(1, 1, 16_000, 16, &[])).unwrap_err();
        assert_eq!(err.0, ERR_INVALID_ARG);
        let err = parse_wav(&make_wav(85, 1, 16_000, 16, &[0, 0])).unwrap_err();
        assert_eq!(err.0, ERR_NOT_SUPPORTED); // e.g. MP3-in-WAV
    }

    #[test]
    fn resample_lengths_follow_ratio() {
        let src: Vec<f32> = (0..8_000).map(|i| i as f32).collect();
        let up = resample_linear(&src, 8_000, 16_000);
        assert_eq!(up.len(), 16_000);
        let down = resample_linear(&src, 44_100, 16_000);
        assert_eq!(down.len(), 8_000 * 16_000 / 44_100);
        let same = resample_linear(&src, 16_000, 16_000);
        assert_eq!(same, src);
        assert!(resample_linear(&[], 8_000, 16_000).is_empty());
    }

    #[test]
    fn resample_constant_signal_stays_constant() {
        let src = vec![0.5f32; 1_000];
        let out = resample_linear(&src, 8_000, 16_000);
        assert!(out.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn resample_ramps_interpolate() {
        let src = vec![0.0f32, 1.0];
        let out = resample_linear(&src, 2, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert_eq!(out[2], 1.0);
    }
}
