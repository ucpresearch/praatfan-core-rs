//! praat-core-rs: Rust reimplementation of Praat's core acoustic analysis algorithms
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
//! - [`Harmonicity`] - HNR (harmonics-to-noise ratio) contour
//! - [`Spectrum`] - Single-frame FFT magnitude spectrum
//! - [`Spectrogram`] - Time-frequency representation

pub mod sound;
pub mod window;
pub mod interpolation;
pub mod intensity;
pub mod spectrum;
pub mod pitch;
pub mod formant;
pub mod harmonicity;
pub mod spectrogram;

pub mod utils;

// Re-export main types at crate root
pub use sound::Sound;
pub use window::{praat_formant_window, WindowShape};
pub use interpolation::Interpolation;
pub use intensity::Intensity;
pub use spectrum::Spectrum;
pub use pitch::Pitch;
pub use formant::Formant;
pub use harmonicity::Harmonicity;
pub use spectrogram::Spectrogram;

use thiserror::Error;

/// Errors that can occur in praat-core operations
#[derive(Error, Debug)]
pub enum PraatError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

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

// Frequency scale conversions (matching Praat's formulas)

/// Convert Hz to Mel scale (Praat formula)
fn hz_to_mel(hz: f64) -> f64 {
    550.0 * (1.0 + hz / 550.0).ln()
}

/// Convert Mel to Hz
fn mel_to_hz(mel: f64) -> f64 {
    550.0 * ((mel / 550.0).exp() - 1.0)
}

/// Convert Hz to semitones relative to a reference frequency
fn hz_to_semitones(hz: f64, reference: f64) -> f64 {
    12.0 * (hz / reference).log2()
}

/// Convert semitones to Hz
fn semitones_to_hz(semitones: f64, reference: f64) -> f64 {
    reference * 2.0_f64.powf(semitones / 12.0)
}

/// Convert Hz to ERB (Equivalent Rectangular Bandwidth) rate
/// Using Glasberg & Moore (1990) formula as used in Praat
fn hz_to_erb(hz: f64) -> f64 {
    21.4 * (0.00437 * hz + 1.0).log10()
}

/// Convert ERB rate to Hz
fn erb_to_hz(erb: f64) -> f64 {
    (10.0_f64.powf(erb / 21.4) - 1.0) / 0.00437
}

/// Convert Hz to Bark scale (Traunmüller formula as in Praat)
fn hz_to_bark(hz: f64) -> f64 {
    let hz_over_600 = hz / 600.0;
    7.0 * (hz_over_600 + (hz_over_600 * hz_over_600 + 1.0).sqrt()).ln()
}

/// Convert Bark to Hz
fn bark_to_hz(bark: f64) -> f64 {
    600.0 * (bark / 7.0).sinh()
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
