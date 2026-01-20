//! WebAssembly bindings for praatfan-core-rs
//!
//! This module provides JavaScript-friendly bindings for the praatfan-core-rs
//! acoustic analysis library using wasm-bindgen.

use wasm_bindgen::prelude::*;
use js_sys::Float64Array;

// Re-export from praatfan-core-rs
use praatfan_core::{
    FrequencyUnit, Interpolation, PitchUnit, WindowShape,
};

/// Initialize panic hook for better error messages in WASM
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Sound type for audio data - WASM wrapper
#[wasm_bindgen]
pub struct Sound {
    inner: praatfan_core::Sound,
}

#[wasm_bindgen]
impl Sound {
    /// Create a Sound from raw samples
    ///
    /// @param samples - Float64Array of audio samples
    /// @param sample_rate - Sample rate in Hz
    #[wasm_bindgen(constructor)]
    pub fn new(samples: &Float64Array, sample_rate: f64) -> Sound {
        let samples: Vec<f64> = samples.to_vec();
        Sound {
            inner: praatfan_core::Sound::from_samples_owned(samples, sample_rate),
        }
    }

    /// Get the sample rate in Hz
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    /// Get the total duration in seconds
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Get the number of samples
    #[wasm_bindgen(getter)]
    pub fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Get the start time
    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    /// Get the end time
    #[wasm_bindgen(getter)]
    pub fn end_time(&self) -> f64 {
        self.inner.end_time()
    }

    /// Get the audio samples as a Float64Array
    pub fn samples(&self) -> Float64Array {
        let samples = self.inner.samples();
        Float64Array::from(samples)
    }

    /// Get the sample value at a specific time using linear interpolation
    pub fn get_value_at_time(&self, time: f64) -> Option<f64> {
        self.inner.get_value_at_time(time)
    }

    /// Apply pre-emphasis filter
    pub fn pre_emphasis(&self, from_frequency: f64) -> Sound {
        Sound {
            inner: self.inner.pre_emphasis(from_frequency),
        }
    }

    /// Apply de-emphasis filter
    pub fn de_emphasis(&self, from_frequency: f64) -> Sound {
        Sound {
            inner: self.inner.de_emphasis(from_frequency),
        }
    }

    /// Get the root-mean-square amplitude
    pub fn rms(&self) -> f64 {
        self.inner.rms()
    }

    /// Get the peak amplitude
    pub fn peak(&self) -> f64 {
        self.inner.peak()
    }

    /// Compute pitch contour
    ///
    /// @param time_step - Time between analysis frames (0.0 for automatic)
    /// @param pitch_floor - Minimum pitch (Hz), typically 75
    /// @param pitch_ceiling - Maximum pitch (Hz), typically 600
    pub fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        Pitch {
            inner: praatfan_core::Pitch::from_sound(&self.inner, time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Compute formants using Burg's LPC method
    ///
    /// @param time_step - Time between analysis frames (0.0 for automatic)
    /// @param max_num_formants - Maximum number of formants (typically 5)
    /// @param max_formant_hz - Maximum formant frequency (Hz)
    /// @param window_length - Analysis window duration (typically 0.025)
    /// @param pre_emphasis_from - Pre-emphasis frequency (Hz), typically 50
    pub fn to_formant_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Formant {
        Formant {
            inner: praatfan_core::Formant::from_sound_burg(
                &self.inner,
                time_step,
                max_num_formants,
                max_formant_hz,
                window_length,
                pre_emphasis_from,
            ),
        }
    }

    /// Compute intensity contour
    ///
    /// @param min_pitch - Minimum expected pitch (Hz)
    /// @param time_step - Time between frames (0.0 for automatic)
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity {
        Intensity {
            inner: praatfan_core::Intensity::from_sound(&self.inner, min_pitch, time_step, true),
        }
    }

    /// Compute spectrum (single-frame FFT)
    ///
    /// @param fast - If true, use power-of-2 FFT size
    pub fn to_spectrum(&self, fast: bool) -> Spectrum {
        Spectrum {
            inner: praatfan_core::Spectrum::from_sound(&self.inner, fast),
        }
    }

    /// Compute spectrogram
    ///
    /// @param effective_analysis_width - Effective window duration (seconds)
    /// @param max_frequency - Maximum frequency (Hz), 0 for Nyquist
    /// @param time_step - Time between frames (0 for automatic)
    /// @param frequency_step - Frequency resolution (Hz), 0 for automatic
    /// @param window_shape - Window function: "gaussian", "hanning", "hamming", "rectangular"
    pub fn to_spectrogram(
        &self,
        effective_analysis_width: f64,
        max_frequency: f64,
        time_step: f64,
        frequency_step: f64,
        window_shape: &str,
    ) -> Result<Spectrogram, JsError> {
        let ws = parse_window_shape(window_shape)?;
        Ok(Spectrogram {
            inner: praatfan_core::Spectrogram::from_sound(
                &self.inner,
                effective_analysis_width,
                max_frequency,
                time_step,
                frequency_step,
                ws,
            ),
        })
    }

    /// Compute harmonicity using autocorrelation method
    pub fn to_harmonicity_ac(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity {
            inner: praatfan_core::Harmonicity::from_sound_ac(
                &self.inner,
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Compute harmonicity using cross-correlation method
    pub fn to_harmonicity_cc(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity {
            inner: praatfan_core::Harmonicity::from_sound_cc(
                &self.inner,
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Create a pure tone (sine wave)
    pub fn create_tone(
        frequency: f64,
        duration: f64,
        sample_rate: f64,
        amplitude: f64,
        phase: f64,
    ) -> Sound {
        Sound {
            inner: praatfan_core::Sound::create_tone(frequency, duration, sample_rate, amplitude, phase),
        }
    }

    /// Create silence
    pub fn create_silence(duration: f64, sample_rate: f64) -> Sound {
        Sound {
            inner: praatfan_core::Sound::create_silence(duration, sample_rate),
        }
    }
}

/// Pitch contour - WASM wrapper
#[wasm_bindgen]
pub struct Pitch {
    inner: praatfan_core::Pitch,
}

#[wasm_bindgen]
impl Pitch {
    /// Get pitch value at a specific time
    ///
    /// @param time - Time to query
    /// @param unit - Unit: "hertz", "mel", "semitones", "erb"
    /// @param interpolation - Interpolation: "nearest", "linear", "cubic"
    /// @returns Pitch value or undefined if unvoiced
    pub fn get_value_at_time(
        &self,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> Result<Option<f64>, JsError> {
        let unit = parse_pitch_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, unit, interp))
    }

    /// Get the time of a specific frame
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    /// Get all pitch values as Float64Array (NaN for unvoiced)
    pub fn values(&self) -> Float64Array {
        let values: Vec<f64> = self
            .inner
            .frames()
            .iter()
            .map(|f| {
                if f.candidates.is_empty() || f.candidates[0].frequency <= 0.0 {
                    f64::NAN
                } else {
                    f.candidates[0].frequency
                }
            })
            .collect();
        Float64Array::from(&values[..])
    }

    /// Get all frame times as Float64Array
    pub fn times(&self) -> Float64Array {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        Float64Array::from(&times[..])
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[wasm_bindgen(getter)]
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[wasm_bindgen(getter)]
    pub fn pitch_floor(&self) -> f64 {
        self.inner.pitch_floor()
    }

    #[wasm_bindgen(getter)]
    pub fn pitch_ceiling(&self) -> f64 {
        self.inner.pitch_ceiling()
    }
}

/// Formant contour - WASM wrapper
#[wasm_bindgen]
pub struct Formant {
    inner: praatfan_core::Formant,
}

#[wasm_bindgen]
impl Formant {
    /// Get formant value at a specific time
    ///
    /// @param formant_number - Formant number (1 for F1, 2 for F2, etc.)
    /// @param time - Time to query
    /// @param unit - Unit: "hertz", "bark", "mel", "erb"
    /// @param interpolation - Interpolation: "nearest", "linear", "cubic"
    pub fn get_value_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> Result<Option<f64>, JsError> {
        let unit = parse_frequency_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(formant_number, time, unit, interp))
    }

    /// Get bandwidth at a specific time
    pub fn get_bandwidth_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> Result<Option<f64>, JsError> {
        let unit = parse_frequency_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_bandwidth_at_time(formant_number, time, unit, interp))
    }

    /// Get the time of a specific frame
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    /// Get all frame times as Float64Array
    pub fn times(&self) -> Float64Array {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        Float64Array::from(&times[..])
    }

    /// Get all values for a specific formant as Float64Array
    pub fn formant_values(&self, formant_number: usize) -> Float64Array {
        let values: Vec<f64> = (0..self.inner.num_frames())
            .map(|frame| {
                self.inner.get_value_at_frame(formant_number, frame)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        Float64Array::from(&values[..])
    }

    /// Get all bandwidths for a specific formant as Float64Array
    pub fn bandwidth_values(&self, formant_number: usize) -> Float64Array {
        let values: Vec<f64> = (0..self.inner.num_frames())
            .map(|frame| {
                self.inner.get_bandwidth_at_frame(formant_number, frame)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        Float64Array::from(&values[..])
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[wasm_bindgen(getter)]
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[wasm_bindgen(getter)]
    pub fn max_num_formants(&self) -> usize {
        self.inner.max_num_formants()
    }
}

/// Intensity contour - WASM wrapper
#[wasm_bindgen]
pub struct Intensity {
    inner: praatfan_core::Intensity,
}

#[wasm_bindgen]
impl Intensity {
    /// Get intensity value at a specific time
    pub fn get_value_at_time(&self, time: f64, interpolation: &str) -> Result<Option<f64>, JsError> {
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, interp))
    }

    /// Get all intensity values as Float64Array
    pub fn values(&self) -> Float64Array {
        Float64Array::from(self.inner.values())
    }

    /// Get all frame times as Float64Array
    pub fn times(&self) -> Float64Array {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        Float64Array::from(&times[..])
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[wasm_bindgen(getter)]
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    pub fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    pub fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    pub fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
}

/// Spectrum - WASM wrapper
#[wasm_bindgen]
pub struct Spectrum {
    inner: praatfan_core::Spectrum,
}

#[wasm_bindgen]
impl Spectrum {
    /// Get band energy between two frequencies
    pub fn get_band_energy(&self, freq_min: f64, freq_max: f64) -> f64 {
        self.inner.get_band_energy(freq_min, freq_max)
    }

    /// Get spectral center of gravity
    pub fn get_center_of_gravity(&self, power: f64) -> f64 {
        self.inner.get_center_of_gravity(power)
    }

    /// Get spectral standard deviation
    pub fn get_standard_deviation(&self, power: f64) -> f64 {
        self.inner.get_standard_deviation(power)
    }

    /// Get spectral skewness
    pub fn get_skewness(&self, power: f64) -> f64 {
        self.inner.get_skewness(power)
    }

    /// Get spectral kurtosis
    pub fn get_kurtosis(&self, power: f64) -> f64 {
        self.inner.get_kurtosis(power)
    }

    /// Get total energy
    pub fn get_total_energy(&self) -> f64 {
        self.inner.get_total_energy()
    }

    #[wasm_bindgen(getter)]
    pub fn num_bins(&self) -> usize {
        self.inner.num_bins()
    }

    #[wasm_bindgen(getter)]
    pub fn df(&self) -> f64 {
        self.inner.df()
    }

    #[wasm_bindgen(getter)]
    pub fn max_frequency(&self) -> f64 {
        self.inner.max_frequency()
    }
}

/// Spectrogram - WASM wrapper
#[wasm_bindgen]
pub struct Spectrogram {
    inner: praatfan_core::Spectrogram,
}

#[wasm_bindgen]
impl Spectrogram {
    /// Get the spectrogram data as a flat Float64Array
    /// Data is in row-major order [freq_0_time_0, freq_0_time_1, ..., freq_n_time_m]
    pub fn values(&self) -> Float64Array {
        // values() returns &Vec<Vec<f64>> where inner[freq_bin][time_frame]
        let inner_data = self.inner.values();

        // Flatten to row-major order
        let mut data = Vec::new();
        for freq_row in inner_data {
            data.extend_from_slice(freq_row);
        }
        Float64Array::from(&data[..])
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[wasm_bindgen(getter)]
    pub fn num_freq_bins(&self) -> usize {
        self.inner.num_freq_bins()
    }

    #[wasm_bindgen(getter)]
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[wasm_bindgen(getter)]
    pub fn freq_step(&self) -> f64 {
        self.inner.freq_step()
    }

    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[wasm_bindgen(getter)]
    pub fn freq_min(&self) -> f64 {
        self.inner.freq_min()
    }

    #[wasm_bindgen(getter)]
    pub fn freq_max(&self) -> f64 {
        self.inner.freq_max()
    }

    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    pub fn get_frequency_from_bin(&self, bin: usize) -> f64 {
        self.inner.get_frequency_from_bin(bin)
    }
}

/// Harmonicity (HNR) - WASM wrapper
#[wasm_bindgen]
pub struct Harmonicity {
    inner: praatfan_core::Harmonicity,
}

#[wasm_bindgen]
impl Harmonicity {
    /// Get HNR value at a specific time
    pub fn get_value_at_time(&self, time: f64, interpolation: &str) -> Result<Option<f64>, JsError> {
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, interp))
    }

    /// Get all HNR values as Float64Array
    pub fn values(&self) -> Float64Array {
        Float64Array::from(self.inner.values())
    }

    /// Get all frame times as Float64Array
    pub fn times(&self) -> Float64Array {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        Float64Array::from(&times[..])
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[wasm_bindgen(getter)]
    pub fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[wasm_bindgen(getter)]
    pub fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    pub fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    pub fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    pub fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
}

// Helper functions for parsing string arguments

fn parse_window_shape(s: &str) -> Result<WindowShape, JsError> {
    match s.to_lowercase().as_str() {
        "rectangular" | "rect" => Ok(WindowShape::Rectangular),
        "triangular" | "tri" => Ok(WindowShape::Triangular),
        "parabolic" | "para" => Ok(WindowShape::Parabolic),
        "hanning" | "hann" => Ok(WindowShape::Hanning),
        "hamming" => Ok(WindowShape::Hamming),
        "gaussian" | "gauss" => Ok(WindowShape::Gaussian),
        "kaiser" => Ok(WindowShape::Kaiser),
        _ => Err(JsError::new(&format!(
            "Unknown window shape: '{}'. Valid options: rectangular, triangular, parabolic, hanning, hamming, gaussian, kaiser",
            s
        ))),
    }
}

fn parse_interpolation(s: &str) -> Result<Interpolation, JsError> {
    match s.to_lowercase().as_str() {
        "nearest" | "none" => Ok(Interpolation::Nearest),
        "linear" => Ok(Interpolation::Linear),
        "cubic" => Ok(Interpolation::Cubic),
        _ => Err(JsError::new(&format!(
            "Unknown interpolation: '{}'. Valid options: nearest, linear, cubic",
            s
        ))),
    }
}

fn parse_pitch_unit(s: &str) -> Result<PitchUnit, JsError> {
    match s.to_lowercase().as_str() {
        "hertz" | "hz" => Ok(PitchUnit::Hertz),
        "mel" => Ok(PitchUnit::Mel),
        "semitones" | "st" => Ok(PitchUnit::Semitones),
        "semitones_re_100hz" | "st100" => Ok(PitchUnit::SemitonesRe100Hz),
        "semitones_re_200hz" | "st200" => Ok(PitchUnit::SemitonesRe200Hz),
        "semitones_re_440hz" | "st440" => Ok(PitchUnit::SemitonesRe440Hz),
        "erb" => Ok(PitchUnit::Erb),
        _ => Err(JsError::new(&format!(
            "Unknown pitch unit: '{}'. Valid options: hertz, mel, semitones, erb",
            s
        ))),
    }
}

fn parse_frequency_unit(s: &str) -> Result<FrequencyUnit, JsError> {
    match s.to_lowercase().as_str() {
        "hertz" | "hz" => Ok(FrequencyUnit::Hertz),
        "bark" => Ok(FrequencyUnit::Bark),
        "mel" => Ok(FrequencyUnit::Mel),
        "erb" => Ok(FrequencyUnit::Erb),
        _ => Err(JsError::new(&format!(
            "Unknown frequency unit: '{}'. Valid options: hertz, bark, mel, erb",
            s
        ))),
    }
}
