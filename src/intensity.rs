//! Intensity (loudness) analysis
//!
//! This module computes intensity contours from audio signals, representing
//! the perceived loudness over time in decibels (dB).
//!
//! The algorithm computes RMS energy in overlapping windows and converts
//! to dB relative to a reference pressure (air reference: 2×10⁻⁵ Pa).

use crate::interpolation::Interpolation;
use crate::window::WindowShape;
use crate::Sound;

/// Reference pressure for dB SPL calculation (2×10⁻⁵ Pa)
/// In Praat, intensity values are relative to this air reference.
const REFERENCE_PRESSURE: f64 = 2e-5;

/// Intensity contour representing energy over time
#[derive(Debug, Clone)]
pub struct Intensity {
    /// Intensity values in dB
    values: Vec<f64>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Minimum pitch used for analysis (determines window size)
    min_pitch: f64,
}

impl Intensity {
    /// Create an Intensity from the given values and timing parameters
    pub fn new(values: Vec<f64>, start_time: f64, time_step: f64, min_pitch: f64) -> Self {
        Self {
            values,
            start_time,
            time_step,
            min_pitch,
        }
    }

    /// Compute intensity from a Sound
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `min_pitch` - Minimum expected fundamental frequency (Hz).
    ///                 This determines the window length: period = 3.2 / min_pitch
    /// * `time_step` - Time between analysis frames (seconds).
    ///                 If 0.0, uses min_pitch / 4 (Praat default).
    /// * `subtract_mean` - If true, subtract the mean from each window before computing RMS
    ///
    /// # Algorithm
    /// For each frame:
    /// 1. Extract a window centered at the frame time
    /// 2. Apply a Hanning window
    /// 3. Optionally subtract the mean (removes DC offset)
    /// 4. Compute RMS energy
    /// 5. Convert to dB: 10 * log10(rms² / reference²)
    pub fn from_sound(
        sound: &Sound,
        min_pitch: f64,
        time_step: f64,
        subtract_mean: bool,
    ) -> Self {
        // Validate parameters
        let min_pitch = min_pitch.max(1.0); // Avoid division by zero

        // Window duration: Praat uses 3.2 periods of the minimum pitch
        // This ensures good frequency resolution for low frequencies
        let window_duration = 3.2 / min_pitch;

        // Time step: default is 0.8 periods of minimum pitch (gives ~4x overlap)
        let time_step = if time_step <= 0.0 {
            0.8 / min_pitch
        } else {
            time_step
        };

        let sample_rate = sound.sample_rate();
        let window_samples = (window_duration * sample_rate).round() as usize;
        let half_window_samples = window_samples / 2;

        // Generate Hanning window
        let window = WindowShape::Hanning.generate(window_samples, None);
        let window_sum_sq: f64 = window.iter().map(|&w| w * w).sum();

        // Determine frame positions
        // First frame is centered at half_window_duration from the start
        let half_window_duration = window_duration / 2.0;
        let first_frame_time = sound.start_time() + half_window_duration;
        let last_frame_time = sound.end_time() - half_window_duration;

        if first_frame_time >= last_frame_time {
            // Sound too short for even one frame
            return Self::new(Vec::new(), sound.start_time(), time_step, min_pitch);
        }

        let num_frames = ((last_frame_time - first_frame_time) / time_step).floor() as usize + 1;

        let mut values = Vec::with_capacity(num_frames);
        let samples = sound.samples();

        for frame_idx in 0..num_frames {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;
            let center_sample = sound.time_to_index_clamped(frame_time);

            // Extract window of samples
            let start_sample = center_sample.saturating_sub(half_window_samples);
            let end_sample = (center_sample + half_window_samples + 1).min(samples.len());

            if end_sample <= start_sample {
                values.push(f64::NEG_INFINITY);
                continue;
            }

            // Compute windowed energy
            let mut sum_sq = 0.0;
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &sample) in samples[start_sample..end_sample].iter().enumerate() {
                let window_idx = i + (half_window_samples.saturating_sub(center_sample - start_sample));
                let w = if window_idx < window.len() {
                    window[window_idx]
                } else {
                    0.0
                };

                let w_sq = w * w;
                sum += sample * w_sq;
                weight_sum += w_sq;
            }

            // Compute mean if needed
            let mean = if subtract_mean && weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };

            // Compute weighted RMS
            for (i, &sample) in samples[start_sample..end_sample].iter().enumerate() {
                let window_idx = i + (half_window_samples.saturating_sub(center_sample - start_sample));
                let w = if window_idx < window.len() {
                    window[window_idx]
                } else {
                    0.0
                };

                let centered = sample - mean;
                sum_sq += centered * centered * w * w;
            }

            // Normalize by window energy
            let mean_sq = if window_sum_sq > 0.0 {
                sum_sq / window_sum_sq
            } else {
                0.0
            };

            // Convert to dB SPL
            // Assuming samples are in the range [-1, 1] representing normalized amplitude
            // Praat assumes a pressure of 1 Pa for amplitude 1
            let intensity_db = if mean_sq > 0.0 {
                10.0 * (mean_sq / (REFERENCE_PRESSURE * REFERENCE_PRESSURE)).log10()
            } else {
                f64::NEG_INFINITY
            };

            values.push(intensity_db);
        }

        Self::new(values, first_frame_time, time_step, min_pitch)
    }

    /// Get intensity value at a specific time
    ///
    /// # Arguments
    /// * `time` - The time at which to query intensity
    /// * `interpolation` - Interpolation method to use
    ///
    /// # Returns
    /// Intensity in dB, or None if time is outside the valid range
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }

        // Convert time to frame position
        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.values.len() as f64 - 0.5 {
            return None;
        }

        interpolation.interpolate(&self.values, position.max(0.0))
    }

    /// Get the time of a specific frame
    pub fn get_time_from_frame(&self, frame: usize) -> f64 {
        self.start_time + frame as f64 * self.time_step
    }

    /// Get the frame index nearest to a specific time
    pub fn get_frame_from_time(&self, time: f64) -> usize {
        let position = (time - self.start_time) / self.time_step;
        let frame = position.round() as isize;
        frame.max(0).min(self.values.len() as isize - 1) as usize
    }

    /// Get all intensity values
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get the number of frames
    pub fn num_frames(&self) -> usize {
        self.values.len()
    }

    /// Get the time step between frames
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    /// Get the start time (time of first frame center)
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Get the end time (time of last frame center)
    pub fn end_time(&self) -> f64 {
        if self.values.is_empty() {
            self.start_time
        } else {
            self.start_time + (self.values.len() - 1) as f64 * self.time_step
        }
    }

    /// Get the minimum intensity value
    pub fn min(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v.is_finite())
            .cloned()
            .reduce(f64::min)
    }

    /// Get the maximum intensity value
    pub fn max(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v.is_finite())
            .cloned()
            .reduce(f64::max)
    }

    /// Get the mean intensity value
    pub fn mean(&self) -> Option<f64> {
        let finite_values: Vec<f64> = self.values.iter().filter(|&&v| v.is_finite()).cloned().collect();
        if finite_values.is_empty() {
            return None;
        }
        Some(finite_values.iter().sum::<f64>() / finite_values.len() as f64)
    }

    /// Get the minimum pitch used for analysis
    pub fn min_pitch(&self) -> f64 {
        self.min_pitch
    }
}

// Add to_intensity method to Sound
impl Sound {
    /// Compute intensity contour from this sound
    ///
    /// # Arguments
    /// * `min_pitch` - Minimum expected pitch (Hz), determines window size
    /// * `time_step` - Time between frames (0.0 for automatic)
    ///
    /// # Returns
    /// Intensity contour in dB SPL
    pub fn to_intensity(&self, min_pitch: f64, time_step: f64) -> Intensity {
        Intensity::from_sound(self, min_pitch, time_step, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_intensity_pure_tone() {
        // Create a pure tone at 440 Hz
        let sound = Sound::create_tone(440.0, 0.5, 44100.0, 0.1, 0.0);

        let intensity = sound.to_intensity(100.0, 0.0);

        // Should have some frames
        assert!(intensity.num_frames() > 0);

        // All frames should have similar intensity (steady tone)
        let values: Vec<f64> = intensity.values().iter().filter(|&&v| v.is_finite()).cloned().collect();
        if values.len() > 1 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            for &v in &values[1..values.len() - 1] {
                // Interior frames should be within 1 dB of mean
                assert!((v - mean).abs() < 1.0, "Intensity variation too large: {} vs mean {}", v, mean);
            }
        }
    }

    #[test]
    fn test_intensity_silence() {
        let sound = Sound::create_silence(0.5, 44100.0);
        let intensity = sound.to_intensity(100.0, 0.0);

        // All values should be -infinity for silence
        for &v in intensity.values() {
            assert!(v == f64::NEG_INFINITY || v < -100.0);
        }
    }

    #[test]
    fn test_intensity_amplitude_relationship() {
        // Doubling amplitude should increase intensity by ~6 dB
        let sound1 = Sound::create_tone(440.0, 0.5, 44100.0, 0.1, 0.0);
        let sound2 = Sound::create_tone(440.0, 0.5, 44100.0, 0.2, 0.0);

        let intensity1 = sound1.to_intensity(100.0, 0.01);
        let intensity2 = sound2.to_intensity(100.0, 0.01);

        // Get mean intensity (excluding edge effects)
        let mean1 = intensity1.mean().unwrap();
        let mean2 = intensity2.mean().unwrap();

        // Doubling amplitude = +6.02 dB
        let db_diff = mean2 - mean1;
        assert_relative_eq!(db_diff, 6.02, epsilon = 0.5);
    }

    #[test]
    fn test_intensity_interpolation() {
        let sound = Sound::create_tone(440.0, 0.5, 44100.0, 0.1, 0.0);
        let intensity = sound.to_intensity(100.0, 0.01);

        // Query at frame centers should work
        let t = intensity.get_time_from_frame(5);
        let v = intensity.get_value_at_time(t, Interpolation::Linear);
        assert!(v.is_some());

        // Query between frames should also work
        let t_between = (intensity.get_time_from_frame(5) + intensity.get_time_from_frame(6)) / 2.0;
        let v_between = intensity.get_value_at_time(t_between, Interpolation::Linear);
        assert!(v_between.is_some());
    }
}
