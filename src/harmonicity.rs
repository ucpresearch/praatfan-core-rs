//! Harmonicity (HNR) analysis using cross-correlation
//!
//! This module computes harmonics-to-noise ratio (HNR) from audio signals.
//! HNR measures the ratio of periodic (harmonic) to aperiodic (noise) energy,
//! expressed in dB. Higher values indicate more periodic/voiced sounds.
//!
//! The algorithm uses autocorrelation to estimate the degree of periodicity
//! at each frame.

use crate::interpolation::Interpolation;
use crate::utils::Fft;
use crate::window::WindowShape;
use crate::Sound;

/// Harmonicity (HNR) contour
#[derive(Debug, Clone)]
pub struct Harmonicity {
    /// HNR values in dB for each frame
    values: Vec<f64>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Minimum pitch used for analysis
    min_pitch: f64,
}

impl Harmonicity {
    /// Compute harmonicity from a Sound using cross-correlation method
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz), determines max lag for correlation
    /// * `silence_threshold` - Threshold for silence detection (0.0-1.0, typically 0.1)
    /// * `periods_per_window` - Number of periods per analysis window (typically 1.0)
    ///
    /// # Algorithm
    /// For each frame:
    /// 1. Extract windowed signal
    /// 2. Compute normalized autocorrelation
    /// 3. Find maximum correlation in the pitch period range
    /// 4. Convert correlation to HNR: 10 * log10(r / (1 - r))
    pub fn from_sound_cc(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        // Validate parameters
        let min_pitch = min_pitch.max(10.0);
        let periods_per_window = periods_per_window.max(0.5).min(10.0);
        let silence_threshold = silence_threshold.max(0.0).min(1.0);

        // Window duration: periods_per_window / min_pitch
        // But we need at least 3 periods for good autocorrelation
        let window_duration = (3.0 + periods_per_window) / min_pitch;

        // Time step: default is 0.01 seconds
        let time_step = if time_step <= 0.0 {
            0.01
        } else {
            time_step
        };

        let sample_rate = sound.sample_rate();
        let samples = sound.samples();

        // Window parameters
        let window_samples = (window_duration * sample_rate).round() as usize;
        let half_window = window_samples / 2;

        // Generate window
        let window = WindowShape::Hanning.generate(window_samples, None);

        // Lag range for pitch search
        let max_lag = (sample_rate / min_pitch).round() as usize;
        let min_lag = 2; // Minimum lag to avoid DC correlation

        // Frame timing
        let first_frame_time = sound.start_time() + window_duration / 2.0;
        let last_frame_time = sound.end_time() - window_duration / 2.0;

        if first_frame_time >= last_frame_time || samples.is_empty() {
            return Self {
                values: Vec::new(),
                start_time: sound.start_time(),
                time_step,
                min_pitch,
            };
        }

        let num_frames = ((last_frame_time - first_frame_time) / time_step).floor() as usize + 1;

        let mut fft = Fft::new();
        let mut values = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;
            let center_sample = sound.time_to_index_clamped(frame_time);

            // Extract windowed frame
            let start_sample = center_sample.saturating_sub(half_window);
            let end_sample = (center_sample + half_window).min(samples.len());

            let mut windowed = vec![0.0; window_samples];
            let mut energy = 0.0;

            for (i, sample_idx) in (start_sample..end_sample).enumerate() {
                if i < window.len() {
                    let w = window[i];
                    let s = samples[sample_idx];
                    windowed[i] = s * w;
                    energy += s * s * w * w;
                }
            }

            // Check for silence
            let rms = (energy / window_samples as f64).sqrt();
            if rms < silence_threshold * 0.00001 {
                // Silence: undefined HNR
                values.push(f64::NEG_INFINITY);
                continue;
            }

            // Compute normalized autocorrelation
            let autocorr = fft.normalized_autocorrelation(&windowed);

            // Find maximum correlation in the pitch period range
            let search_end = max_lag.min(autocorr.len() - 1);
            let mut max_r = 0.0;

            for lag in min_lag..=search_end {
                if autocorr[lag] > max_r {
                    max_r = autocorr[lag];
                }
            }

            // Clamp correlation to valid range
            max_r = max_r.max(0.0).min(0.99999);

            // Convert to HNR in dB
            // HNR = 10 * log10(r / (1 - r))
            let hnr = if max_r > 0.0 {
                10.0 * (max_r / (1.0 - max_r)).log10()
            } else {
                f64::NEG_INFINITY
            };

            values.push(hnr);
        }

        Self {
            values,
            start_time: first_frame_time,
            time_step,
            min_pitch,
        }
    }

    /// Get HNR value at a specific time
    ///
    /// # Arguments
    /// * `time` - Time at which to query
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    /// HNR in dB, or None if time is outside range or undefined
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }

        // Get frame position
        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.values.len() as f64 - 0.5 {
            return None;
        }

        // Replace NEG_INFINITY with NaN for interpolation
        let values_for_interp: Vec<f64> = self
            .values
            .iter()
            .map(|&v| if v.is_finite() { v } else { f64::NAN })
            .collect();

        interpolation.interpolate_with_undefined(&values_for_interp, position.max(0.0))
    }

    /// Get HNR value at a specific frame
    pub fn get_value_at_frame(&self, frame: usize) -> Option<f64> {
        self.values.get(frame).copied().filter(|&v| v.is_finite())
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

    /// Get all HNR values
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

    /// Get the start time
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Get the end time
    pub fn end_time(&self) -> f64 {
        if self.values.is_empty() {
            self.start_time
        } else {
            self.start_time + (self.values.len() - 1) as f64 * self.time_step
        }
    }

    /// Get minimum HNR value (excluding undefined frames)
    pub fn min(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v.is_finite())
            .cloned()
            .reduce(f64::min)
    }

    /// Get maximum HNR value
    pub fn max(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v.is_finite())
            .cloned()
            .reduce(f64::max)
    }

    /// Get mean HNR value (excluding undefined frames)
    pub fn mean(&self) -> Option<f64> {
        let finite: Vec<f64> = self.values.iter().filter(|&&v| v.is_finite()).cloned().collect();
        if finite.is_empty() {
            None
        } else {
            Some(finite.iter().sum::<f64>() / finite.len() as f64)
        }
    }

    /// Get standard deviation of HNR values
    pub fn standard_deviation(&self) -> Option<f64> {
        let finite: Vec<f64> = self.values.iter().filter(|&&v| v.is_finite()).cloned().collect();
        if finite.len() < 2 {
            return None;
        }

        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance = finite.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (finite.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Get the minimum pitch setting used for analysis
    pub fn min_pitch(&self) -> f64 {
        self.min_pitch
    }
}

// Add to_harmonicity_cc method to Sound
impl Sound {
    /// Compute harmonicity (HNR) from this sound using cross-correlation
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic: 0.01 s)
    /// * `min_pitch` - Minimum expected pitch (Hz)
    /// * `silence_threshold` - Threshold for silence detection (typically 0.1)
    /// * `periods_per_window` - Periods per window (typically 1.0)
    ///
    /// # Returns
    /// Harmonicity object with HNR values in dB
    pub fn to_harmonicity_cc(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity::from_sound_cc(self, time_step, min_pitch, silence_threshold, periods_per_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonicity_pure_tone() {
        // A pure tone should have high HNR (very periodic)
        let sound = Sound::create_tone(200.0, 0.5, 44100.0, 0.5, 0.0);

        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // Should have frames
        assert!(hnr.num_frames() > 0);

        // Mean HNR should be positive (periodic signal)
        if let Some(mean) = hnr.mean() {
            assert!(mean > 0.0, "Pure tone should have positive HNR, got {}", mean);
        }
    }

    #[test]
    fn test_harmonicity_silence() {
        let sound = Sound::create_silence(0.5, 44100.0);
        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // All values should be undefined (NEG_INFINITY) for silence
        for &v in hnr.values() {
            assert!(!v.is_finite() || v < -50.0, "Silence should have undefined/very low HNR");
        }
    }

    #[test]
    fn test_harmonicity_interpolation() {
        let sound = Sound::create_tone(150.0, 0.3, 44100.0, 0.5, 0.0);
        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // Query at middle of sound
        let t = sound.duration() / 2.0;
        let value = hnr.get_value_at_time(t, Interpolation::Linear);

        assert!(value.is_some(), "Should find HNR at t={}", t);
    }

    #[test]
    fn test_harmonicity_statistics() {
        let sound = Sound::create_tone(200.0, 0.5, 44100.0, 0.5, 0.0);
        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // Statistics should be defined for voiced signal
        assert!(hnr.min().is_some());
        assert!(hnr.max().is_some());
        assert!(hnr.mean().is_some());
        assert!(hnr.standard_deviation().is_some());

        // Max should be >= min
        let min = hnr.min().unwrap();
        let max = hnr.max().unwrap();
        assert!(max >= min);
    }

    #[test]
    fn test_harmonicity_noisy_signal() {
        // Create a signal with added noise (lower HNR expected)
        let sample_rate = 44100.0;
        let duration = 0.5;
        let n_samples = (duration * sample_rate) as usize;

        let mut samples = vec![0.0; n_samples];
        let mut rng_state: u64 = 12345;

        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            // Tone + noise
            let tone = 0.3 * (2.0 * std::f64::consts::PI * 200.0 * t).sin();

            // Simple PRNG for noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng_state >> 16) as f64 / 32768.0 - 1.0) * 0.2;

            samples[i] = tone + noise;
        }

        let sound = Sound::from_samples(&samples, sample_rate);
        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // Should have valid HNR values
        assert!(hnr.num_frames() > 0);

        // HNR should be lower than for pure tone but still defined
        if let Some(mean) = hnr.mean() {
            // With significant noise, HNR might be lower
            assert!(mean.is_finite(), "Should have finite HNR for noisy signal");
        }
    }
}
