//! Python bindings for praatfan-core-rs
//!
//! This module provides Python bindings using PyO3 for the praatfan-core-rs
//! acoustic analysis library.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Import from praatfan-core-rs (use :: to disambiguate from pymodule name)
use ::praatfan_core::{
    FrequencyUnit, Interpolation, PitchUnit, WindowShape,
    Sound as RustSound,
    Pitch as RustPitch,
    Formant as RustFormant,
    Intensity as RustIntensity,
    Spectrum as RustSpectrum,
    Spectrogram as RustSpectrogram,
    Harmonicity as RustHarmonicity,
};

/// Python wrapper for Sound
#[pyclass(name = "Sound")]
pub struct PySound {
    inner: RustSound,
}

#[pymethods]
impl PySound {
    /// Create a Sound from raw samples
    ///
    /// Parameters
    /// ----------
    /// samples : numpy.ndarray
    ///     Audio samples as a 1D float64 array
    /// sample_rate : float
    ///     Sample rate in Hz
    #[new]
    fn new(samples: PyReadonlyArray1<f64>, sample_rate: f64) -> Self {
        let samples = samples.as_slice().unwrap().to_vec();
        PySound {
            inner: RustSound::from_samples_owned(samples, sample_rate),
        }
    }

    /// Load a Sound from an audio file
    ///
    /// Supports WAV, MP3, FLAC, and OGG formats.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the audio file
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        RustSound::from_file(path)
            .map(|s| PySound { inner: s })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get the sample rate in Hz
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    /// Get the total duration in seconds
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// Get the number of samples
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Get the start time (time of first sample)
    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    /// Get the end time
    #[getter]
    fn end_time(&self) -> f64 {
        self.inner.end_time()
    }

    /// Get the audio samples as a numpy array
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.samples().to_vec().into_pyarray_bound(py)
    }

    /// Get the sample value at a specific time using linear interpolation
    fn get_value_at_time(&self, time: f64) -> Option<f64> {
        self.inner.get_value_at_time(time)
    }

    /// Extract a portion of the sound
    fn extract_part(
        &self,
        start_time: f64,
        end_time: f64,
        window_shape: &str,
        relative_width: f64,
        preserve_times: bool,
    ) -> PyResult<Self> {
        let ws = parse_window_shape(window_shape)?;
        self.inner
            .extract_part(start_time, end_time, ws, relative_width, preserve_times)
            .map(|s| PySound { inner: s })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply pre-emphasis filter
    fn pre_emphasis(&self, from_frequency: f64) -> Self {
        PySound {
            inner: self.inner.pre_emphasis(from_frequency),
        }
    }

    /// Apply de-emphasis filter (inverse of pre-emphasis)
    fn de_emphasis(&self, from_frequency: f64) -> Self {
        PySound {
            inner: self.inner.de_emphasis(from_frequency),
        }
    }

    /// Get the root-mean-square amplitude
    fn rms(&self) -> f64 {
        self.inner.rms()
    }

    /// Get the peak amplitude
    fn peak(&self) -> f64 {
        self.inner.peak()
    }

    /// Compute pitch contour
    ///
    /// Parameters
    /// ----------
    /// time_step : float
    ///     Time between analysis frames (0.0 for automatic)
    /// pitch_floor : float
    ///     Minimum pitch (Hz), typically 75
    /// pitch_ceiling : float
    ///     Maximum pitch (Hz), typically 600
    fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> PyPitch {
        PyPitch {
            inner: RustPitch::from_sound(&self.inner, time_step, pitch_floor, pitch_ceiling),
        }
    }

    /// Compute formants using Burg's LPC method
    ///
    /// Parameters
    /// ----------
    /// time_step : float
    ///     Time between analysis frames (0.0 for automatic)
    /// max_num_formants : int
    ///     Maximum number of formants (typically 5)
    /// max_formant_hz : float
    ///     Maximum formant frequency (Hz). Use ~5500 for male, ~5000 for female
    /// window_length : float
    ///     Analysis window duration (typically 0.025)
    /// pre_emphasis_from : float
    ///     Pre-emphasis frequency (Hz), typically 50
    fn to_formant_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> PyFormant {
        PyFormant {
            inner: RustFormant::from_sound_burg(
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
    /// Parameters
    /// ----------
    /// min_pitch : float
    ///     Minimum expected pitch (Hz), determines window size
    /// time_step : float
    ///     Time between frames (0.0 for automatic)
    fn to_intensity(&self, min_pitch: f64, time_step: f64) -> PyIntensity {
        PyIntensity {
            inner: RustIntensity::from_sound(&self.inner, min_pitch, time_step, true),
        }
    }

    /// Compute spectrum (single-frame FFT)
    ///
    /// Parameters
    /// ----------
    /// fast : bool
    ///     If True, use power-of-2 FFT size for faster computation
    fn to_spectrum(&self, fast: bool) -> PySpectrum {
        PySpectrum {
            inner: RustSpectrum::from_sound(&self.inner, fast),
        }
    }

    /// Compute spectrogram (time-frequency representation)
    ///
    /// Parameters
    /// ----------
    /// effective_analysis_width : float
    ///     Effective window duration (seconds)
    /// max_frequency : float
    ///     Maximum frequency (Hz), 0 for Nyquist
    /// time_step : float
    ///     Time between frames (0 for automatic)
    /// frequency_step : float
    ///     Frequency resolution (Hz), 0 for automatic
    /// window_shape : str
    ///     Window function: "gaussian", "hanning", "hamming", "rectangular"
    fn to_spectrogram(
        &self,
        effective_analysis_width: f64,
        max_frequency: f64,
        time_step: f64,
        frequency_step: f64,
        window_shape: &str,
    ) -> PyResult<PySpectrogram> {
        let ws = parse_window_shape(window_shape)?;
        Ok(PySpectrogram {
            inner: RustSpectrogram::from_sound(
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
    ///
    /// Parameters
    /// ----------
    /// time_step : float
    ///     Time between analysis frames (0.0 for automatic)
    /// min_pitch : float
    ///     Minimum expected pitch (Hz)
    /// silence_threshold : float
    ///     Threshold for silence detection (typically 0.1)
    /// periods_per_window : float
    ///     Number of periods per analysis window (typically 1.0)
    fn to_harmonicity_ac(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> PyHarmonicity {
        PyHarmonicity {
            inner: RustHarmonicity::from_sound_ac(
                &self.inner,
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Compute harmonicity using cross-correlation method
    ///
    /// Parameters
    /// ----------
    /// time_step : float
    ///     Time between analysis frames (0.0 for automatic)
    /// min_pitch : float
    ///     Minimum expected pitch (Hz)
    /// silence_threshold : float
    ///     Threshold for silence detection (typically 0.1)
    /// periods_per_window : float
    ///     Number of periods per analysis window (typically 1.0)
    fn to_harmonicity_cc(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> PyHarmonicity {
        PyHarmonicity {
            inner: RustHarmonicity::from_sound_cc(
                &self.inner,
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            ),
        }
    }

    /// Create a pure tone (sine wave)
    #[staticmethod]
    fn create_tone(
        frequency: f64,
        duration: f64,
        sample_rate: f64,
        amplitude: f64,
        phase: f64,
    ) -> Self {
        PySound {
            inner: RustSound::create_tone(frequency, duration, sample_rate, amplitude, phase),
        }
    }

    /// Create silence
    #[staticmethod]
    fn create_silence(duration: f64, sample_rate: f64) -> Self {
        PySound {
            inner: RustSound::create_silence(duration, sample_rate),
        }
    }
}

/// Python wrapper for Pitch
#[pyclass(name = "Pitch")]
pub struct PyPitch {
    inner: RustPitch,
}

#[pymethods]
impl PyPitch {
    /// Get pitch value at a specific time
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///     Time to query
    /// unit : str
    ///     Unit for the result: "hertz", "mel", "semitones", "erb"
    /// interpolation : str
    ///     Interpolation method: "nearest", "linear", "cubic"
    ///
    /// Returns
    /// -------
    /// float or None
    ///     Pitch value, or None if unvoiced
    fn get_value_at_time(
        &self,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> PyResult<Option<f64>> {
        let unit = parse_pitch_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, unit, interp))
    }

    /// Get the time of a specific frame
    fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    /// Get all pitch values as a numpy array (NaN for unvoiced)
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
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
        values.into_pyarray_bound(py)
    }

    /// Get all frame times as a numpy array
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        times.into_pyarray_bound(py)
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[getter]
    fn pitch_floor(&self) -> f64 {
        self.inner.pitch_floor()
    }

    #[getter]
    fn pitch_ceiling(&self) -> f64 {
        self.inner.pitch_ceiling()
    }
}

/// Python wrapper for Formant
#[pyclass(name = "Formant")]
pub struct PyFormant {
    inner: RustFormant,
}

#[pymethods]
impl PyFormant {
    /// Get formant value at a specific time
    ///
    /// Parameters
    /// ----------
    /// formant_number : int
    ///     Formant number (1 for F1, 2 for F2, etc.)
    /// time : float
    ///     Time to query
    /// unit : str
    ///     Unit: "hertz", "bark", "mel", "erb"
    /// interpolation : str
    ///     Interpolation: "nearest", "linear", "cubic"
    fn get_value_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> PyResult<Option<f64>> {
        let unit = parse_frequency_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(formant_number, time, unit, interp))
    }

    /// Get bandwidth at a specific time
    fn get_bandwidth_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: &str,
        interpolation: &str,
    ) -> PyResult<Option<f64>> {
        let unit = parse_frequency_unit(unit)?;
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_bandwidth_at_time(formant_number, time, unit, interp))
    }

    /// Get the time of a specific frame
    fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    /// Get all frame times as a numpy array
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        times.into_pyarray_bound(py)
    }

    /// Get all values for a specific formant as a numpy array
    fn formant_values<'py>(&self, py: Python<'py>, formant_number: usize) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = (0..self.inner.num_frames())
            .map(|frame| {
                self.inner.get_value_at_frame(formant_number, frame)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        values.into_pyarray_bound(py)
    }

    /// Get all bandwidths for a specific formant as a numpy array
    fn bandwidth_values<'py>(&self, py: Python<'py>, formant_number: usize) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = (0..self.inner.num_frames())
            .map(|frame| {
                self.inner.get_bandwidth_at_frame(formant_number, frame)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        values.into_pyarray_bound(py)
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[getter]
    fn max_num_formants(&self) -> usize {
        self.inner.max_num_formants()
    }
}

/// Python wrapper for Intensity
#[pyclass(name = "Intensity")]
pub struct PyIntensity {
    inner: RustIntensity,
}

#[pymethods]
impl PyIntensity {
    /// Get intensity value at a specific time
    fn get_value_at_time(&self, time: f64, interpolation: &str) -> PyResult<Option<f64>> {
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, interp))
    }

    /// Get all intensity values as a numpy array
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values().to_vec().into_pyarray_bound(py)
    }

    /// Get all frame times as a numpy array
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        times.into_pyarray_bound(py)
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
}

/// Python wrapper for Spectrum
#[pyclass(name = "Spectrum")]
pub struct PySpectrum {
    inner: RustSpectrum,
}

#[pymethods]
impl PySpectrum {
    /// Get band energy between two frequencies
    fn get_band_energy(&self, freq_min: f64, freq_max: f64) -> f64 {
        self.inner.get_band_energy(freq_min, freq_max)
    }

    /// Get spectral center of gravity
    fn get_center_of_gravity(&self, power: f64) -> f64 {
        self.inner.get_center_of_gravity(power)
    }

    /// Get spectral standard deviation
    fn get_standard_deviation(&self, power: f64) -> f64 {
        self.inner.get_standard_deviation(power)
    }

    /// Get spectral skewness
    fn get_skewness(&self, power: f64) -> f64 {
        self.inner.get_skewness(power)
    }

    /// Get spectral kurtosis
    fn get_kurtosis(&self, power: f64) -> f64 {
        self.inner.get_kurtosis(power)
    }

    /// Get total energy
    fn get_total_energy(&self) -> f64 {
        self.inner.get_total_energy()
    }

    #[getter]
    fn num_bins(&self) -> usize {
        self.inner.num_bins()
    }

    #[getter]
    fn df(&self) -> f64 {
        self.inner.df()
    }

    #[getter]
    fn max_frequency(&self) -> f64 {
        self.inner.max_frequency()
    }
}

/// Python wrapper for Spectrogram
#[pyclass(name = "Spectrogram")]
pub struct PySpectrogram {
    inner: RustSpectrogram,
}

#[pymethods]
impl PySpectrogram {
    /// Get the spectrogram data as a 2D numpy array [freq, time]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // values() returns &Vec<Vec<f64>> where inner[freq_bin][time_frame]
        let inner_data = self.inner.values();
        let n_freqs = inner_data.len();
        let n_times = if n_freqs > 0 { inner_data[0].len() } else { 0 };

        // Flatten to row-major order [freq_0, time_0..n], [freq_1, time_0..n], ...
        let mut data = Vec::with_capacity(n_freqs * n_times);
        for freq_row in inner_data {
            data.extend_from_slice(freq_row);
        }

        // Create 2D array using from_vec2_bound
        let nested: Vec<Vec<f64>> = inner_data.clone();
        PyArray2::from_vec2_bound(py, &nested).expect("Failed to create 2D array")
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn num_freq_bins(&self) -> usize {
        self.inner.num_freq_bins()
    }

    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[getter]
    fn freq_step(&self) -> f64 {
        self.inner.freq_step()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    #[getter]
    fn freq_min(&self) -> f64 {
        self.inner.freq_min()
    }

    #[getter]
    fn freq_max(&self) -> f64 {
        self.inner.freq_max()
    }

    fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.inner.get_time_from_frame(frame)
    }

    fn get_frequency_from_bin(&self, bin: usize) -> f64 {
        self.inner.get_frequency_from_bin(bin)
    }
}

/// Python wrapper for Harmonicity
#[pyclass(name = "Harmonicity")]
pub struct PyHarmonicity {
    inner: RustHarmonicity,
}

#[pymethods]
impl PyHarmonicity {
    /// Get HNR value at a specific time
    fn get_value_at_time(&self, time: f64, interpolation: &str) -> PyResult<Option<f64>> {
        let interp = parse_interpolation(interpolation)?;
        Ok(self.inner.get_value_at_time(time, interp))
    }

    /// Get all HNR values as a numpy array
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.values().to_vec().into_pyarray_bound(py)
    }

    /// Get all frame times as a numpy array
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let times: Vec<f64> = (0..self.inner.num_frames())
            .map(|i| self.inner.get_time_from_frame(i))
            .collect();
        times.into_pyarray_bound(py)
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn time_step(&self) -> f64 {
        self.inner.time_step()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
}

// Helper functions for parsing string arguments

fn parse_window_shape(s: &str) -> PyResult<WindowShape> {
    match s.to_lowercase().as_str() {
        "rectangular" | "rect" => Ok(WindowShape::Rectangular),
        "triangular" | "tri" => Ok(WindowShape::Triangular),
        "parabolic" | "para" => Ok(WindowShape::Parabolic),
        "hanning" | "hann" => Ok(WindowShape::Hanning),
        "hamming" => Ok(WindowShape::Hamming),
        "gaussian" | "gauss" => Ok(WindowShape::Gaussian),
        "kaiser" => Ok(WindowShape::Kaiser),
        _ => Err(PyValueError::new_err(format!(
            "Unknown window shape: '{}'. Valid options: rectangular, triangular, parabolic, hanning, hamming, gaussian, kaiser",
            s
        ))),
    }
}

fn parse_interpolation(s: &str) -> PyResult<Interpolation> {
    match s.to_lowercase().as_str() {
        "nearest" | "none" => Ok(Interpolation::Nearest),
        "linear" => Ok(Interpolation::Linear),
        "cubic" => Ok(Interpolation::Cubic),
        _ => Err(PyValueError::new_err(format!(
            "Unknown interpolation: '{}'. Valid options: nearest, linear, cubic",
            s
        ))),
    }
}

fn parse_pitch_unit(s: &str) -> PyResult<PitchUnit> {
    match s.to_lowercase().as_str() {
        "hertz" | "hz" => Ok(PitchUnit::Hertz),
        "mel" => Ok(PitchUnit::Mel),
        "semitones" | "st" => Ok(PitchUnit::Semitones),
        "semitones_re_100hz" | "st100" => Ok(PitchUnit::SemitonesRe100Hz),
        "semitones_re_200hz" | "st200" => Ok(PitchUnit::SemitonesRe200Hz),
        "semitones_re_440hz" | "st440" => Ok(PitchUnit::SemitonesRe440Hz),
        "erb" => Ok(PitchUnit::Erb),
        _ => Err(PyValueError::new_err(format!(
            "Unknown pitch unit: '{}'. Valid options: hertz, mel, semitones, erb",
            s
        ))),
    }
}

fn parse_frequency_unit(s: &str) -> PyResult<FrequencyUnit> {
    match s.to_lowercase().as_str() {
        "hertz" | "hz" => Ok(FrequencyUnit::Hertz),
        "bark" => Ok(FrequencyUnit::Bark),
        "mel" => Ok(FrequencyUnit::Mel),
        "erb" => Ok(FrequencyUnit::Erb),
        _ => Err(PyValueError::new_err(format!(
            "Unknown frequency unit: '{}'. Valid options: hertz, bark, mel, erb",
            s
        ))),
    }
}

/// praatfan_gpl - Praat-compatible acoustic analysis in Python
///
/// This module provides exact reimplementations of Praat's acoustic analysis
/// algorithms, designed to produce bit-accurate output matching Praat/parselmouth.
///
/// Main classes:
/// - Sound: Audio data with loading, filtering, and analysis methods
/// - Pitch: F0 contour from autocorrelation analysis
/// - Formant: LPC-based formant tracks (F1-F4 + bandwidths)
/// - Intensity: RMS energy contour in dB
/// - Spectrum: Single-frame FFT magnitude spectrum
/// - Spectrogram: Time-frequency representation
/// - Harmonicity: HNR (harmonics-to-noise ratio) contour
#[pymodule]
fn praatfan_gpl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySound>()?;
    m.add_class::<PyPitch>()?;
    m.add_class::<PyFormant>()?;
    m.add_class::<PyIntensity>()?;
    m.add_class::<PySpectrum>()?;
    m.add_class::<PySpectrogram>()?;
    m.add_class::<PyHarmonicity>()?;
    Ok(())
}
