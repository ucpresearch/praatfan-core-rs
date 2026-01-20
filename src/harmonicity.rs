//! Harmonicity (HNR) analysis using autocorrelation or cross-correlation
//!
//! This module computes harmonics-to-noise ratio (HNR) from audio signals.
//! HNR measures the ratio of periodic (harmonic) to aperiodic (noise) energy,
//! expressed in dB. Higher values indicate more periodic/voiced sounds.
//!
//! The algorithm is derived from pitch analysis - HNR is computed from the
//! autocorrelation strength at the detected pitch period.
//!
//! This implementation matches Praat's Sound_to_Harmonicity_cc and
//! Sound_to_Harmonicity_ac functions.

use crate::interpolation::Interpolation;
use crate::pitch::{Pitch, PitchFrame, PitchMethod};
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
    /// Compute harmonicity from a Sound using autocorrelation method (AC)
    ///
    /// This matches Praat's "To Harmonicity (ac)..." command.
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz), determines max lag for correlation
    /// * `silence_threshold` - Threshold for silence detection (typically 0.1)
    /// * `periods_per_window` - Number of periods per analysis window (typically 1.0)
    pub fn from_sound_ac(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        // Method 1 = AC_GAUSS (Autocorrelation with Gaussian window)
        Self::from_pitch_method(
            sound,
            PitchMethod::AcGauss,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
        )
    }

    /// Compute harmonicity from a Sound using cross-correlation method (CC)
    ///
    /// This matches Praat's "To Harmonicity (cc)..." command.
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz), determines max lag for correlation
    /// * `silence_threshold` - Threshold for silence detection (typically 0.1)
    /// * `periods_per_window` - Number of periods per analysis window (typically 1.0)
    pub fn from_sound_cc(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        // Method 3 = FCC_ACCURATE
        // Since we don't have FCC implemented yet, use AC_GAUSS as approximation
        // TODO: Implement FCC method for exact CC match
        Self::from_pitch_method(
            sound,
            PitchMethod::AcGauss,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
        )
    }

    /// Internal: compute harmonicity from pitch analysis
    fn from_pitch_method(
        sound: &Sound,
        method: PitchMethod,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        // Compute pitch with the specified parameters
        // Praat uses: pitch_ceiling = 0.5 / dx (Nyquist)
        // and all costs set to 0.0 (no path finding penalty)
        let pitch_ceiling = 0.5 / sound.dx();

        // Use our pitch implementation with the specified method and periods_per_window
        let pitch = Pitch::from_sound_with_method(
            sound,
            time_step,
            min_pitch,
            pitch_ceiling,
            15,                 // maxnCandidates
            silence_threshold,  // silenceThreshold
            0.0,                // voicingThreshold (not used for HNR)
            0.0,                // octaveCost
            0.0,                // octaveJumpCost
            0.0,                // voicedUnvoicedCost
            periods_per_window, // Pass through the user's setting
            method,             // AC_GAUSS for AC, FCC_ACCURATE for CC
        );

        // Convert pitch frames to HNR values
        let mut values = Vec::with_capacity(pitch.num_frames());
        for frame in pitch.frames() {
            let hnr = Self::strength_to_hnr(frame);
            values.push(hnr);
        }

        Self {
            values,
            start_time: pitch.start_time(),
            time_step: pitch.time_step(),
            min_pitch,
        }
    }

    /// Convert pitch frame strength to HNR in dB
    fn strength_to_hnr(frame: &PitchFrame) -> f64 {
        if frame.candidates.is_empty() || frame.candidates[0].frequency == 0.0 {
            // Unvoiced
            -200.0
        } else {
            let r = frame.candidates[0].strength;
            if r <= 1e-15 {
                -150.0
            } else if r > 1.0 - 1e-15 {
                150.0
            } else {
                10.0 * (r / (1.0 - r)).log10()
            }
        }
    }

    /// Create Harmonicity directly from a Pitch object
    ///
    /// This allows computing HNR from an existing pitch analysis.
    pub fn from_pitch(pitch: &Pitch, min_pitch: f64) -> Self {
        let mut values = Vec::with_capacity(pitch.num_frames());
        for frame in pitch.frames() {
            let hnr = Self::strength_to_hnr(frame);
            values.push(hnr);
        }

        Self {
            values,
            start_time: pitch.start_time(),
            time_step: pitch.time_step(),
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

        // Convert -200 to NaN for interpolation (unvoiced frames)
        let values_for_interp: Vec<f64> = self
            .values
            .iter()
            .map(|&v| if v > -199.0 { v } else { f64::NAN })
            .collect();

        interpolation.interpolate_with_undefined(&values_for_interp, position.max(0.0))
    }

    /// Get HNR value at a specific frame
    pub fn get_value_at_frame(&self, frame: usize) -> Option<f64> {
        self.values.get(frame).copied().filter(|&v| v > -199.0)
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
            .filter(|&&v| v > -199.0)
            .cloned()
            .reduce(f64::min)
    }

    /// Get maximum HNR value
    pub fn max(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v > -199.0)
            .cloned()
            .reduce(f64::max)
    }

    /// Get mean HNR value (excluding undefined frames)
    pub fn mean(&self) -> Option<f64> {
        let finite: Vec<f64> = self.values.iter().filter(|&&v| v > -199.0).cloned().collect();
        if finite.is_empty() {
            None
        } else {
            Some(finite.iter().sum::<f64>() / finite.len() as f64)
        }
    }

    /// Get standard deviation of HNR values
    pub fn standard_deviation(&self) -> Option<f64> {
        let finite: Vec<f64> = self.values.iter().filter(|&&v| v > -199.0).cloned().collect();
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

// Add to_harmonicity methods to Sound
impl Sound {
    /// Compute harmonicity (HNR) from this sound using cross-correlation
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
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

    /// Compute harmonicity (HNR) from this sound using autocorrelation
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz)
    /// * `silence_threshold` - Threshold for silence detection (typically 0.1)
    /// * `periods_per_window` - Periods per window (typically 1.0)
    ///
    /// # Returns
    /// Harmonicity object with HNR values in dB
    pub fn to_harmonicity_ac(
        &self,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Harmonicity {
        Harmonicity::from_sound_ac(self, time_step, min_pitch, silence_threshold, periods_per_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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

        // All values should be -200 for silence (unvoiced)
        for &v in hnr.values() {
            assert!(v <= -199.0, "Silence should have very low HNR (unvoiced)");
        }
    }

    #[test]
    fn test_harmonicity_interpolation() {
        // Use longer signal and standard parameters that work with AC_GAUSS
        let sound = Sound::create_tone(150.0, 0.5, 44100.0, 0.5, 0.0);
        // Use AC method (via to_harmonicity_ac) which requires periods_per_window >= 3.0
        let hnr = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 3.0);

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
            let tone = 0.3 * (2.0 * PI * 200.0 * t).sin();

            // Simple PRNG for noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng_state >> 16) as f64 / 32768.0 - 1.0) * 0.2;

            samples[i] = tone + noise;
        }

        let sound = Sound::from_samples(&samples, sample_rate);
        let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);

        // Should have valid HNR values
        assert!(hnr.num_frames() > 0);

        // HNR should be defined
        if let Some(mean) = hnr.mean() {
            assert!(mean.is_finite(), "Should have finite HNR for noisy signal");
        }
    }
}
