//! praatfan-core-rs: Rust reimplementation of Praat's core acoustic analysis algorithms
//!
//! This library provides bit-accurate implementations of Praat's acoustic analysis
//! functions, designed to produce identical output to Praat/parselmouth.
//!
//! # Core Types
//!
//! - [`Sound`] - Audio samples with sample rate
//! - [`Pitch`] - F0 contour from autocorrelation analysis
//! - [`Intensity`] - RMS energy contour in dB
//! - [`Formant`] - LPC-based formant tracks
//! - [`FormantPath`] - Multi-ceiling Burg formant analysis with Viterbi-optimal ceiling selection
//! - [`Harmonicity`] - HNR (harmonics-to-noise ratio) contour
//! - [`Spectrum`] - Single-frame FFT magnitude spectrum
//! - [`Spectrogram`] - Time-frequency representation
//!
//! # Parallelism
//!
//! With the `parallel` feature (enabled by the Python bindings and the `pipe`
//! feature), per-frame work in pitch, harmonicity, formant, FormantPath,
//! intensity, spectrogram and resampling runs on a crate-owned rayon pool.
//! Results are bit-identical to the serial build at any thread count. The
//! pool honours `RAYON_NUM_THREADS`, is rebuilt after `fork()`, and is
//! bypassed in favour of the caller's pool when called from inside
//! `rayon::ThreadPool::install`. On `wasm32` the feature is ignored and
//! everything runs serially.
//!
//! # Citing
//!
//! If you use this library in published work, please cite:
//!
//! - Boersma, P. & Weenink, D. (2024). *Praat: doing phonetics by computer.*
//!   <https://www.fon.hum.uva.nl/praat/>
//! - Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
//!   A Python interface to Praat. *Journal of Phonetics, 71*, 1–15.
//! - Weenink, D. (2015). Improved formant frequency measurements of short
//!   segments. *Proceedings of ICPhS 2015*, Glasgow. (For `FormantPath`.)

pub mod sound;
pub mod window;
pub mod interpolation;
pub mod intensity;
pub mod spectrum;
pub mod pitch;
pub mod formant;
pub mod formant_modeler;
pub mod formant_path;
pub mod harmonicity;
pub mod point_process;
pub mod spectrogram;
pub mod speech_reference;

#[cfg(feature = "pipe-core")]
pub mod pipe;

pub mod utils;
mod par;

// Re-export main types at crate root
pub use sound::Sound;
pub use window::{praat_formant_window, WindowShape};
pub use interpolation::{Interpolation, PeakInterpolation};
pub use intensity::{Intensity, IntensityAveraging, intensity_from_channels};
pub use spectrum::Spectrum;
pub use pitch::{Pitch, PitchMethod, StrengthUnit, pitch_from_channels, sinc_interpolate, improve_maximum, BRENT_TOTAL_ITERS, BRENT_TOTAL_CALLS};
pub use formant::Formant;
pub use formant_path::{FormantPath, PathFinderCorrections};
pub use point_process::{voice_report, PointProcess, Shimmer, VoiceReport};
pub use harmonicity::{Harmonicity, harmonicity_from_channels_ac, harmonicity_from_channels_cc};
pub use spectrogram::{Spectrogram, spectrogram_from_channels};
pub use speech_reference::{SpeechReference, estimate_speech_reference, estimate_speech_reference_default};

use thiserror::Error;

/// Errors that can occur in praatfan-core operations
#[derive(Error, Debug)]
pub enum PraatError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "decode")]
    #[error("WAV decoding error: {0}")]
    WavDecode(#[from] hound::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("Value undefined at time {time}: {reason}")]
    UndefinedValue { time: f64, reason: String },
}

pub type Result<T> = std::result::Result<T, PraatError>;

/// Units for pitch values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchUnit {
    Hertz,
    Mel,
    Semitones,
    SemitonesRe100Hz,
    SemitonesRe200Hz,
    SemitonesRe440Hz,
    Erb,
}

/// Units for frequency values (formants, spectrum)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyUnit {
    Hertz,
    Bark,
    Mel,
    Erb,
}

impl PitchUnit {
    /// Convert a frequency in Hz to this unit
    pub fn from_hertz(self, hz: f64) -> f64 {
        match self {
            PitchUnit::Hertz => hz,
            PitchUnit::Mel => hz_to_mel(hz),
            PitchUnit::Semitones => hz_to_semitones(hz, 1.0),
            PitchUnit::SemitonesRe100Hz => hz_to_semitones(hz, 100.0),
            PitchUnit::SemitonesRe200Hz => hz_to_semitones(hz, 200.0),
            PitchUnit::SemitonesRe440Hz => hz_to_semitones(hz, 440.0),
            PitchUnit::Erb => hz_to_erb(hz),
        }
    }

    /// Convert from this unit to Hz
    pub fn to_hertz(self, value: f64) -> f64 {
        match self {
            PitchUnit::Hertz => value,
            PitchUnit::Mel => mel_to_hz(value),
            PitchUnit::Semitones => semitones_to_hz(value, 1.0),
            PitchUnit::SemitonesRe100Hz => semitones_to_hz(value, 100.0),
            PitchUnit::SemitonesRe200Hz => semitones_to_hz(value, 200.0),
            PitchUnit::SemitonesRe440Hz => semitones_to_hz(value, 440.0),
            PitchUnit::Erb => erb_to_hz(value),
        }
    }
}

impl FrequencyUnit {
    /// Convert a frequency in Hz to this unit
    pub fn from_hertz(self, hz: f64) -> f64 {
        match self {
            FrequencyUnit::Hertz => hz,
            FrequencyUnit::Bark => hz_to_bark(hz),
            FrequencyUnit::Mel => hz_to_mel(hz),
            FrequencyUnit::Erb => hz_to_erb(hz),
        }
    }

    /// Convert from this unit to Hz
    pub fn to_hertz(self, value: f64) -> f64 {
        match self {
            FrequencyUnit::Hertz => value,
            FrequencyUnit::Bark => bark_to_hz(value),
            FrequencyUnit::Mel => mel_to_hz(value),
            FrequencyUnit::Erb => erb_to_hz(value),
        }
    }
}

// Frequency scale conversions: Praat 6.1.38's exact formulas
// (melder/NUMear.cpp; semitones from fon/Pitch.cpp). Negative inputs, which
// Praat maps to `undefined`, give NaN.

const SEMITONE_FACTOR: f64 = 12.0 / std::f64::consts::LN_2;

/// `NUMhertzToMel`: 550 ln(1 + f/550)
fn hz_to_mel(hz: f64) -> f64 {
    if hz < 0.0 { f64::NAN } else { 550.0 * libm::log(1.0 + hz / 550.0) }
}

/// `NUMmelToHertz`
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 0.0 { f64::NAN } else { 550.0 * (libm::exp(mel / 550.0) - 1.0) }
}

/// Praat's Pitch semitones: (12 / ln 2) ln(f / ref); undefined for f <= 0
fn hz_to_semitones(hz: f64, reference: f64) -> f64 {
    if hz <= 0.0 { f64::NAN } else { SEMITONE_FACTOR * libm::log(hz / reference) }
}

/// ref * exp(st * ln 2 / 12)
fn semitones_to_hz(semitones: f64, reference: f64) -> f64 {
    reference * libm::exp(semitones * (std::f64::consts::LN_2 / 12.0))
}

/// `NUMhertzToErb`: 11.17 ln((f + 312) / (f + 14680)) + 43
fn hz_to_erb(hz: f64) -> f64 {
    if hz < 0.0 { f64::NAN } else { 11.17 * libm::log((hz + 312.0) / (hz + 14680.0)) + 43.0 }
}

/// `NUMerbToHertz`
fn erb_to_hz(erb: f64) -> f64 {
    if erb < 0.0 {
        return f64::NAN;
    }
    let dum = libm::exp((erb - 43.0) / 11.17);
    (14680.0 * dum - 312.0) / (1.0 - dum)
}

/// `NUMhertzToBark`: 7 asinh(f / 650), written as Praat writes it
fn hz_to_bark(hz: f64) -> f64 {
    if hz < 0.0 {
        return f64::NAN;
    }
    let r = hz / 650.0;
    7.0 * libm::log(r + libm::sqrt(1.0 + r * r))
}

/// `NUMbarkToHertz`
fn bark_to_hz(bark: f64) -> f64 {
    if bark < 0.0 { f64::NAN } else { 650.0 * libm::sinh(bark / 7.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hz_mel_roundtrip() {
        for hz in [100.0, 440.0, 1000.0, 5000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert_relative_eq!(hz, back, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hz_semitones_roundtrip() {
        for hz in [100.0, 200.0, 440.0] {
            let st = hz_to_semitones(hz, 100.0);
            let back = semitones_to_hz(st, 100.0);
            assert_relative_eq!(hz, back, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hz_erb_roundtrip() {
        for hz in [100.0, 440.0, 1000.0, 5000.0] {
            let erb = hz_to_erb(hz);
            let back = erb_to_hz(erb);
            assert_relative_eq!(hz, back, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hz_bark_roundtrip() {
        for hz in [100.0, 440.0, 1000.0, 5000.0] {
            let bark = hz_to_bark(hz);
            let back = bark_to_hz(bark);
            assert_relative_eq!(hz, back, epsilon = 1e-10);
        }
    }
}
