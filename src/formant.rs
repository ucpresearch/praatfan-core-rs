//! Formant analysis using Linear Predictive Coding (LPC)
//!
//! This module computes formant frequencies and bandwidths from audio signals
//! using Burg's method for LPC analysis. Formants are resonance frequencies
//! of the vocal tract, typically labeled F1, F2, F3, etc.
//!
//! The algorithm:
//! 1. Pre-emphasize the signal to flatten spectral slope
//! 2. Extract overlapping frames with Gaussian window
//! 3. Compute LPC coefficients using Burg's method
//! 4. Find formants as roots of the LPC polynomial
//! 5. Track formants across frames

use crate::interpolation::Interpolation;
use crate::utils::lpc::{lpc_burg, lpc_to_formants, FormantCandidate};
use crate::window::praat_formant_window;
use crate::{FrequencyUnit, Sound};

/// A single formant measurement (frequency and bandwidth)
#[derive(Debug, Clone, Copy, Default)]
pub struct FormantPoint {
    /// Formant frequency in Hz (NaN if undefined)
    pub frequency: f64,
    /// Formant bandwidth in Hz (NaN if undefined)
    pub bandwidth: f64,
}

/// A single frame of formant analysis containing multiple formants
#[derive(Debug, Clone)]
pub struct FormantFrame {
    /// Formants for this frame (F1, F2, F3, ...)
    formants: Vec<FormantPoint>,
    /// Intensity of the frame (for weighting in future tracking algorithms)
    #[allow(dead_code)]
    intensity: f64,
}

impl FormantFrame {
    /// Create a new formant frame
    pub fn new(formants: Vec<FormantPoint>, intensity: f64) -> Self {
        Self { formants, intensity }
    }

    /// Get a specific formant (1-indexed: F1, F2, F3, ...)
    pub fn get_formant(&self, formant_number: usize) -> Option<&FormantPoint> {
        if formant_number == 0 {
            return None;
        }
        self.formants.get(formant_number - 1)
    }

    /// Get the number of formants in this frame
    pub fn num_formants(&self) -> usize {
        self.formants.len()
    }
}

/// Formant contour representing resonance frequencies over time
#[derive(Debug, Clone)]
pub struct Formant {
    /// Analysis frames
    frames: Vec<FormantFrame>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Maximum number of formants per frame
    max_num_formants: usize,
    /// Maximum formant frequency used for analysis
    max_formant_hz: f64,
}

impl Formant {
    /// Compute formants from a Sound using Burg's LPC method
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic: 0.25 × window_length)
    /// * `max_num_formants` - Maximum number of formants to track (typically 5)
    /// * `max_formant_hz` - Maximum formant frequency (Hz). Use ~5500 for male, ~5000 for female voices
    /// * `window_length` - Analysis window duration in seconds (typically 0.025)
    /// * `pre_emphasis_from` - Pre-emphasis frequency (Hz), typically 50
    ///
    /// # Algorithm
    /// For each frame:
    /// 1. Apply pre-emphasis filter
    /// 2. Extract frame with Gaussian window
    /// 3. Compute LPC coefficients (order = 2 × max_num_formants + 2)
    /// 4. Find roots of LPC polynomial
    /// 5. Convert poles to formant frequencies and bandwidths
    pub fn from_sound_burg(
        sound: &Sound,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Self {
        // Validate parameters
        let max_num_formants = max_num_formants.max(1).min(10);
        let window_length = window_length.max(0.01).min(0.1);
        let max_formant_hz = max_formant_hz.max(1000.0).min(sound.sample_rate() / 2.0);

        // Time step: default is 0.25 × window_length (Praat convention)
        let time_step = if time_step <= 0.0 {
            window_length / 4.0
        } else {
            time_step
        };

        // Praat resamples to 2 × max_formant before LPC analysis
        // This is critical for accurate formant estimation
        let target_sample_rate = 2.0 * max_formant_hz;
        let resampled = if sound.sample_rate() > target_sample_rate {
            sound.resample(target_sample_rate)
        } else {
            sound.clone()
        };

        // Pre-emphasize AFTER resampling (Praat does it this way)
        // The pre-emphasis filter coefficient depends on sample rate,
        // so applying it after resampling gives the correct effect
        let emphasized = if pre_emphasis_from > 0.0 {
            resampled.pre_emphasis(pre_emphasis_from)
        } else {
            resampled
        };

        let sample_rate = emphasized.sample_rate();
        let samples = emphasized.samples();
        let dx = 1.0 / sample_rate;
        // Praat's x1 is the time of the first sample: xmin + 0.5*dx
        let x1_sound = emphasized.start_time() + 0.5 * dx;

        // LPC order: 2 × max_formants (Praat convention for Sound -> Formant)
        // Each formant needs 2 poles (complex conjugate pair)
        let lpc_order = 2 * max_num_formants;

        // CRITICAL: window_length parameter is actually halfdt_window in Praat!
        // The actual analysis window is TWICE this value.
        // See Sound_to_Formant.cpp: dt_window = 2.0 * halfdt_window
        let halfdt_window = window_length;
        let dt_window = 2.0 * halfdt_window;
        let nsamp_window = (dt_window / dx).floor() as usize;
        let halfnsamp_window = nsamp_window / 2;

        // Generate Praat's exact Gaussian window for formant analysis
        let window = praat_formant_window(nsamp_window);

        // Frame timing - exactly as Praat does it (verified with source code)
        // physicalDuration = nx * dx
        let physical_duration = samples.len() as f64 * dx;
        let num_frames = 1 + ((physical_duration - dt_window) / time_step).floor() as usize;

        if num_frames == 0 || samples.is_empty() {
            return Self {
                frames: Vec::new(),
                start_time: sound.start_time(),
                time_step,
                max_num_formants,
                max_formant_hz,
            };
        }

        // t1 = x1 + 0.5 * (physicalDuration - dx - (numberOfFrames - 1) * dt)
        // This is Praat's exact formula from Sound_to_Formant.cpp
        let first_frame_time =
            x1_sound + 0.5 * (physical_duration - dx - (num_frames - 1) as f64 * time_step);

        let mut frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;

            // Sample extraction - exactly as Praat does it (Sound_to_Formant.cpp)
            // leftSample = floor((t - x1) / dx)
            // rightSample = leftSample + 1
            // startSample = rightSample - halfnsamp_window
            // endSample = leftSample + halfnsamp_window
            let left_sample = ((frame_time - x1_sound) / dx).floor() as isize;
            let right_sample = left_sample + 1;
            let start_sample_raw = right_sample - halfnsamp_window as isize;
            let end_sample_raw = left_sample + halfnsamp_window as isize;

            // Clamp to valid range
            let start_sample = start_sample_raw.max(0) as usize;
            let end_sample = (end_sample_raw as usize).min(samples.len().saturating_sub(1));
            let actual_frame_length = if end_sample >= start_sample {
                end_sample - start_sample + 1
            } else {
                0
            };

            // Extract windowed frame - apply Gaussian window to samples
            let mut windowed = Vec::with_capacity(actual_frame_length);
            let mut energy = 0.0;

            for i in 0..actual_frame_length {
                let sample_idx = start_sample + i;
                let w = if i < window.len() { window[i] } else { 0.0 };
                let s = samples[sample_idx] * w;
                windowed.push(s);
                energy += s * s;
            }

            // Add tiny dither to prevent exactly zero samples.
            // This matches Praat's behavior where sinc resampling introduces
            // tiny numerical artifacts (~1e-7) that allow LPC to compute
            // meaningful coefficients even in near-silent regions.
            // Include frame position in dither so silent frames don't all get identical values.
            let dither_amplitude = 1e-10;
            let frame_offset = frame_idx as f64 * 17.3; // Prime-ish offset per frame
            for (i, s) in windowed.iter_mut().enumerate() {
                let global_pos = i as f64 + frame_offset;
                let dither = dither_amplitude * ((global_pos * 0.7).sin() + (global_pos * 1.3).cos());
                *s += dither;
            }

            // Compute LPC coefficients using Burg's method
            let formant_points = if let Some(lpc_result) = lpc_burg(&windowed, lpc_order) {
                // Find formants from LPC polynomial roots
                let candidates = lpc_to_formants(&lpc_result.coefficients, sample_rate);

                // Filter candidates - reject clearly invalid formants
                // Valid formants must have:
                // - Frequency in range [50, max_formant_hz]
                // - Positive bandwidth (pole inside unit circle)
                // - Bandwidth less than max_formant_hz (Praat's criterion)
                // - Bandwidth shouldn't be excessively large relative to frequency
                //   (a formant with bandwidth >> frequency is poorly defined)
                let mut valid_formants: Vec<FormantCandidate> = candidates
                    .into_iter()
                    .filter(|f| {
                        f.frequency > 50.0
                            && f.frequency < max_formant_hz
                            && f.bandwidth > 0.0
                            && f.bandwidth < max_formant_hz
                            && f.bandwidth < f.frequency * 2.0 // Bandwidth < 2x frequency
                    })
                    .collect();

                // Sort by frequency
                valid_formants.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());

                // Take up to max_num_formants
                valid_formants
                    .into_iter()
                    .take(max_num_formants)
                    .map(|c| FormantPoint {
                        frequency: c.frequency,
                        bandwidth: c.bandwidth,
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // Pad with undefined formants if we found fewer than expected
            let mut formants = formant_points;
            while formants.len() < max_num_formants {
                formants.push(FormantPoint {
                    frequency: f64::NAN,
                    bandwidth: f64::NAN,
                });
            }

            frames.push(FormantFrame::new(formants, energy));
        }

        Self {
            frames,
            start_time: first_frame_time,
            time_step,
            max_num_formants,
            max_formant_hz,
        }
    }

    /// Get the formant frequency at a specific time
    ///
    /// # Arguments
    /// * `formant_number` - Which formant (1 = F1, 2 = F2, etc.)
    /// * `time` - Time at which to query
    /// * `unit` - Frequency unit for the result
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    /// Formant frequency in the specified unit, or None if undefined
    pub fn get_value_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: FrequencyUnit,
        interpolation: Interpolation,
    ) -> Option<f64> {
        if formant_number == 0 || formant_number > self.max_num_formants {
            return None;
        }

        if self.frames.is_empty() {
            return None;
        }

        // Get frame position
        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.frames.len() as f64 - 0.5 {
            return None;
        }

        // Extract formant frequencies for interpolation
        let frequencies: Vec<f64> = self
            .frames
            .iter()
            .map(|f| {
                f.get_formant(formant_number)
                    .map(|p| p.frequency)
                    .unwrap_or(f64::NAN)
            })
            .collect();

        // Interpolate
        let hz = interpolation.interpolate_with_undefined(&frequencies, position.max(0.0))?;

        // Convert to requested unit
        Some(unit.from_hertz(hz))
    }

    /// Get the formant bandwidth at a specific time
    ///
    /// # Arguments
    /// * `formant_number` - Which formant (1 = F1, 2 = F2, etc.)
    /// * `time` - Time at which to query
    /// * `unit` - Frequency unit for the result
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    /// Formant bandwidth in the specified unit, or None if undefined
    pub fn get_bandwidth_at_time(
        &self,
        formant_number: usize,
        time: f64,
        unit: FrequencyUnit,
        interpolation: Interpolation,
    ) -> Option<f64> {
        if formant_number == 0 || formant_number > self.max_num_formants {
            return None;
        }

        if self.frames.is_empty() {
            return None;
        }

        // Get frame position
        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.frames.len() as f64 - 0.5 {
            return None;
        }

        // Extract bandwidths for interpolation
        let bandwidths: Vec<f64> = self
            .frames
            .iter()
            .map(|f| {
                f.get_formant(formant_number)
                    .map(|p| p.bandwidth)
                    .unwrap_or(f64::NAN)
            })
            .collect();

        // Interpolate
        let hz = interpolation.interpolate_with_undefined(&bandwidths, position.max(0.0))?;

        // Convert to requested unit
        Some(unit.from_hertz(hz))
    }

    /// Get formant values at a specific frame
    pub fn get_value_at_frame(&self, formant_number: usize, frame: usize) -> Option<f64> {
        self.frames
            .get(frame)
            .and_then(|f| f.get_formant(formant_number))
            .map(|p| p.frequency)
            .filter(|&f| !f.is_nan())
    }

    /// Get bandwidth at a specific frame
    pub fn get_bandwidth_at_frame(&self, formant_number: usize, frame: usize) -> Option<f64> {
        self.frames
            .get(frame)
            .and_then(|f| f.get_formant(formant_number))
            .map(|p| p.bandwidth)
            .filter(|&b| !b.is_nan())
    }

    /// Get the time of a specific frame
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.start_time + frame as f64 * self.time_step
    }

    /// Get the frame index nearest to a specific time
    pub fn get_frame_from_time(&self, time: f64) -> usize {
        let position = (time - self.start_time) / self.time_step;
        let frame = position.round() as isize;
        frame.max(0).min(self.frames.len() as isize - 1) as usize
    }

    /// Get the number of frames
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the maximum number of formants per frame
    pub fn max_num_formants(&self) -> usize {
        self.max_num_formants
    }

    /// Get the time step between frames
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the start time
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Get the end time
    pub fn end_time(&self) -> f64 {
        if self.frames.is_empty() {
            self.start_time
        } else {
            self.start_time + (self.frames.len() - 1) as f64 * self.time_step
        }
    }

    /// Get mean value for a formant across all frames
    pub fn get_mean(&self, formant_number: usize) -> Option<f64> {
        if formant_number == 0 || formant_number > self.max_num_formants {
            return None;
        }

        let values: Vec<f64> = self
            .frames
            .iter()
            .filter_map(|f| {
                f.get_formant(formant_number)
                    .map(|p| p.frequency)
                    .filter(|&freq| freq.is_finite())
            })
            .collect();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    /// Get minimum value for a formant
    pub fn get_min(&self, formant_number: usize) -> Option<f64> {
        if formant_number == 0 || formant_number > self.max_num_formants {
            return None;
        }

        self.frames
            .iter()
            .filter_map(|f| {
                f.get_formant(formant_number)
                    .map(|p| p.frequency)
                    .filter(|&freq| freq.is_finite())
            })
            .reduce(f64::min)
    }

    /// Get maximum value for a formant
    pub fn get_max(&self, formant_number: usize) -> Option<f64> {
        if formant_number == 0 || formant_number > self.max_num_formants {
            return None;
        }

        self.frames
            .iter()
            .filter_map(|f| {
                f.get_formant(formant_number)
                    .map(|p| p.frequency)
                    .filter(|&freq| freq.is_finite())
            })
            .reduce(f64::max)
    }

    /// Get the maximum formant frequency setting used for analysis
    pub fn max_formant_hz(&self) -> f64 {
        self.max_formant_hz
    }
}

// Add to_formant_burg method to Sound
impl Sound {
    /// Compute formant tracks from this sound using Burg's LPC method
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `max_num_formants` - Maximum formants to find (typically 5)
    /// * `max_formant_hz` - Maximum formant frequency (~5500 Hz for adult male, ~5000 Hz for female)
    /// * `window_length` - Analysis window (typically 0.025 s)
    /// * `pre_emphasis_from` - Pre-emphasis frequency (typically 50 Hz)
    ///
    /// # Returns
    /// Formant object with F1, F2, F3, ... tracks
    pub fn to_formant_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Formant {
        Formant::from_sound_burg(
            self,
            time_step,
            max_num_formants,
            max_formant_hz,
            window_length,
            pre_emphasis_from,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_basic() {
        // Create a simple test signal (vowel-like with known formants is hard to synthesize,
        // so we just test that the algorithm runs without crashing)
        let sound = Sound::create_tone(200.0, 0.5, 16000.0, 0.5, 0.0);

        let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);

        // Should have frames
        assert!(formant.num_frames() > 0);
        assert_eq!(formant.max_num_formants(), 5);
    }

    #[test]
    fn test_formant_query() {
        // Create test signal
        let sound = Sound::create_tone(150.0, 0.3, 16000.0, 0.5, 0.0);

        let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

        // Query at middle of sound
        let t = sound.duration() / 2.0;

        // F1 query (may or may not find a formant for a pure tone)
        let _f1 = formant.get_value_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear);

        // Frame-based query
        if formant.num_frames() > 5 {
            let frame_time = formant.get_time_from_frame(5);
            assert!(frame_time > formant.start_time());
        }
    }

    #[test]
    fn test_formant_statistics() {
        // Create a longer test signal for statistics
        let sound = Sound::create_tone(200.0, 0.5, 16000.0, 0.5, 0.0);

        let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);

        // Statistics functions should not crash
        let _mean_f1 = formant.get_mean(1);
        let _min_f1 = formant.get_min(1);
        let _max_f1 = formant.get_max(1);

        // Invalid formant number should return None
        assert!(formant.get_mean(0).is_none());
        assert!(formant.get_mean(10).is_none());
    }

    #[test]
    fn test_formant_with_complex_signal() {
        // Create a signal with multiple frequency components
        // This should produce more interesting formant results
        let sample_rate = 16000.0;
        let duration = 0.5;
        let n_samples = (duration * sample_rate) as usize;

        let mut samples = vec![0.0; n_samples];
        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            // Fundamental + harmonics (simulating a voiced sound)
            samples[i] = 0.5 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 300.0 * t).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * 450.0 * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 600.0 * t).sin();
        }

        let sound = Sound::from_samples(&samples, sample_rate);
        let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

        assert!(formant.num_frames() > 0);

        // Should be able to query formants
        let t = duration / 2.0;
        let f1 = formant.get_value_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear);
        // For this synthetic signal, we may or may not get valid formants
        // The important thing is that the code runs without panicking
        let _ = f1;
    }

    #[test]
    fn test_formant_bandwidth() {
        let sound = Sound::create_tone(200.0, 0.3, 16000.0, 0.5, 0.0);
        let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

        if formant.num_frames() > 0 {
            // Try to get bandwidth at first frame
            let _bw = formant.get_bandwidth_at_frame(1, 0);

            // Time-based query
            let t = formant.get_time_from_frame(0);
            let _bw_time = formant.get_bandwidth_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear);
        }
    }
}
