//! Intensity (loudness) analysis
//!
//! This module computes intensity contours from audio signals, representing
//! the perceived loudness over time in decibels (dB).
//!
//! The algorithm matches Praat's `Sound -> To Intensity` exactly:
//! - Uses Kaiser-Bessel window with modified Bessel I0
//! - Physical window duration is 6.4 / min_pitch
//! - Frame timing uses Sampled_shortTermAnalysis formula

use crate::interpolation::Interpolation;
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

/// Modified Bessel function I0 (first kind, order zero)
/// Matches Praat's NUMbessel_i0_f from melder/NUMspecfunc.cpp
fn bessel_i0(x: f64) -> f64 {
    let x = x.abs();
    if x < 3.75 {
        // Formula 9.8.1 from Abramowitz & Stegun. Accuracy 1.6e-7.
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
            + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    } else {
        // Formula 9.8.2 from Abramowitz & Stegun. Accuracy 1.9e-7.
        let t = 3.75 / x;
        (x.exp() / x.sqrt()) * (0.39894228 + t * (0.01328592
            + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281
            + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633
            + t * 0.00392377))))))))
    }
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

    /// Compute intensity from a Sound (matches Praat exactly)
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `min_pitch` - Minimum expected fundamental frequency (Hz).
    ///                 This determines the window length.
    /// * `time_step` - Time between analysis frames (seconds).
    ///                 If 0.0, uses automatic default: 0.8 / min_pitch
    /// * `subtract_mean` - If true, subtract the mean from each window (removes DC offset)
    ///
    /// # Algorithm (from Praat's Sound_to_Intensity.cpp)
    /// - logical_window_duration = 3.2 / min_pitch
    /// - physical_window_duration = 2 * logical = 6.4 / min_pitch
    /// - Window: Kaiser-Bessel using bessel_i0((2π² + 0.5) * sqrt(1 - x²))
    /// - Energy: weighted mean of amplitude², weights = window (not window²)
    pub fn from_sound(
        sound: &Sound,
        min_pitch: f64,
        time_step: f64,
        subtract_mean: bool,
    ) -> Self {
        // Validate parameters
        let min_pitch = min_pitch.max(1.0);

        // Praat uses these exact formulas:
        // logicalWindowDuration = 3.2 / pitchFloor
        // physicalWindowDuration = 2.0 * logicalWindowDuration = 6.4 / pitchFloor
        let logical_window_duration = 3.2 / min_pitch;
        let physical_window_duration = 2.0 * logical_window_duration;

        // Time step: default is logicalWindowDuration / 4 = 0.8 / min_pitch
        let time_step = if time_step <= 0.0 {
            logical_window_duration / 4.0
        } else {
            time_step
        };

        let dx = 1.0 / sound.sample_rate();
        let half_window_duration = 0.5 * physical_window_duration;
        let half_window_samples = (half_window_duration / dx).floor() as i64;
        let window_num_samples = (2 * half_window_samples + 1) as usize;
        let window_centre = half_window_samples + 1; // 1-based center

        // Generate Kaiser-Bessel window (Praat's formula)
        // window[i] = bessel_i0((2π² + 0.5) * sqrt(1 - x²))
        // where x = (i - center) * dx / half_window_duration
        let mut window = vec![0.0; window_num_samples];
        let bessel_arg = 2.0 * std::f64::consts::PI * std::f64::consts::PI + 0.5;

        for i in 0..window_num_samples {
            let i_1based = (i + 1) as f64;
            let x = (i_1based - window_centre as f64) * dx / half_window_duration;
            let root = (1.0 - x * x).max(0.0).sqrt();
            window[i] = bessel_i0(bessel_arg * root);
        }

        // Frame timing using Sampled_shortTermAnalysis formula
        // myDuration = nx * dx
        // numberOfFrames = floor((myDuration - windowDuration) / timeStep) + 1
        // ourMidTime = x1 - 0.5*dx + 0.5*myDuration
        // thyDuration = numberOfFrames * timeStep
        // firstTime = ourMidTime - 0.5*thyDuration + 0.5*timeStep
        let nx = sound.num_samples();
        let my_duration = nx as f64 * dx;

        if physical_window_duration > my_duration {
            return Self::new(Vec::new(), sound.start_time(), time_step, min_pitch);
        }

        let num_frames = ((my_duration - physical_window_duration) / time_step).floor() as usize + 1;

        let x1 = sound.start_time() + 0.5 * dx; // Time of first sample (Praat convention)
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = num_frames as f64 * time_step;
        let first_time = our_mid_time - 0.5 * thy_duration + 0.5 * time_step;

        let samples = sound.samples();
        let mut values = Vec::with_capacity(num_frames);

        for iframe in 0..num_frames {
            // midTime = Sampled_indexToX(thee, iframe) = firstTime + iframe * timeStep
            let mid_time = first_time + iframe as f64 * time_step;

            // soundCentreSampleNumber = round((midTime - x1) / dx) + 1 (1-based)
            let sound_centre_sample = ((mid_time - x1) / dx).round() as i64 + 1;

            let left_sample = sound_centre_sample - half_window_samples;
            let right_sample = sound_centre_sample + half_window_samples;

            // Clamp to valid range (1-based to 0-based conversion)
            let left_sample_clamped = left_sample.max(1) as usize;
            let right_sample_clamped = (right_sample as usize).min(nx);

            if right_sample_clamped < left_sample_clamped {
                values.push(-300.0);
                continue;
            }

            // Calculate window offset
            let window_from_sound_offset = window_centre - sound_centre_sample;

            // Compute weighted sum
            let mut sumxw: f64 = 0.0;
            let mut sumw: f64 = 0.0;

            // For stereo, Praat sums energies across channels
            // Our Sound is mono, but we handle it the same way

            // First pass: compute UNWEIGHTED mean if needed (Praat's centre_VEC_inout uses NUMmean)
            let mean = if subtract_mean {
                let mut sum = 0.0;
                let mut count = 0;
                for isamp in left_sample_clamped..=right_sample_clamped {
                    let window_idx = (window_from_sound_offset + isamp as i64) as usize;
                    if window_idx > 0 && window_idx <= window_num_samples {
                        let s = samples[isamp - 1]; // Convert to 0-based
                        sum += s;
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f64 } else { 0.0 }
            } else {
                0.0
            };

            // Second pass: compute weighted energy
            for isamp in left_sample_clamped..=right_sample_clamped {
                let window_idx = (window_from_sound_offset + isamp as i64) as usize;
                if window_idx > 0 && window_idx <= window_num_samples {
                    let w = window[window_idx - 1]; // Convert to 0-based
                    let s = samples[isamp - 1] - mean; // Convert to 0-based
                    sumxw += s * s * w;
                    sumw += w;
                }
            }

            // intensity_in_Pa2 = sumxw / sumw
            let intensity_in_pa2 = if sumw > 0.0 { sumxw / sumw } else { 0.0 };

            // Convert to dB re hearing threshold
            let hearing_threshold_pa2 = REFERENCE_PRESSURE * REFERENCE_PRESSURE;
            let intensity_re_threshold = intensity_in_pa2 / hearing_threshold_pa2;

            let intensity_db = if intensity_re_threshold < 1.0e-30 {
                -300.0
            } else {
                10.0 * intensity_re_threshold.log10()
            };

            values.push(intensity_db);
        }

        Self::new(values, first_time, time_step, min_pitch)
    }

    /// Compute intensity from multiple channels (matches Praat for stereo/multichannel)
    ///
    /// Praat's Sound_to_Intensity sums energy across channels in the inner loop:
    /// `intensity = Σ_chan Σ_samp (s²·w) / Σ_chan Σ_samp w`
    /// This is equivalent to averaging per-channel energy, which differs from
    /// computing energy on sample-averaged mono.
    pub fn from_channels(
        sounds: &[Sound],
        min_pitch: f64,
        time_step: f64,
        subtract_mean: bool,
    ) -> Self {
        if sounds.is_empty() {
            return Self::new(Vec::new(), 0.0, if time_step > 0.0 { time_step } else { 0.01 }, min_pitch);
        }
        if sounds.len() == 1 {
            return Self::from_sound(&sounds[0], min_pitch, time_step, subtract_mean);
        }

        let nchan = sounds.len();
        let min_pitch = min_pitch.max(1.0);
        let logical_window_duration = 3.2 / min_pitch;
        let physical_window_duration = 2.0 * logical_window_duration;
        let time_step = if time_step <= 0.0 {
            logical_window_duration / 4.0
        } else {
            time_step
        };

        // Use first channel for timing (all channels must have same sample rate/length)
        let sound = &sounds[0];
        let dx = 1.0 / sound.sample_rate();
        let half_window_duration = 0.5 * physical_window_duration;
        let half_window_samples = (half_window_duration / dx).floor() as i64;
        let window_num_samples = (2 * half_window_samples + 1) as usize;
        let window_centre = half_window_samples + 1;

        let bessel_arg = 2.0 * std::f64::consts::PI * std::f64::consts::PI + 0.5;
        let mut window = vec![0.0; window_num_samples];
        for i in 0..window_num_samples {
            let i_1based = (i + 1) as f64;
            let x = (i_1based - window_centre as f64) * dx / half_window_duration;
            let root = (1.0 - x * x).max(0.0).sqrt();
            window[i] = bessel_i0(bessel_arg * root);
        }

        let nx = sound.num_samples();
        let my_duration = nx as f64 * dx;

        if physical_window_duration > my_duration {
            return Self::new(Vec::new(), sound.start_time(), time_step, min_pitch);
        }

        let num_frames = ((my_duration - physical_window_duration) / time_step).floor() as usize + 1;
        let x1 = sound.start_time() + 0.5 * dx;
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = num_frames as f64 * time_step;
        let first_time = our_mid_time - 0.5 * thy_duration + 0.5 * time_step;

        // Collect all channel samples
        let all_samples: Vec<&[f64]> = sounds.iter().map(|s| s.samples()).collect();

        let mut values = Vec::with_capacity(num_frames);

        for iframe in 0..num_frames {
            let mid_time = first_time + iframe as f64 * time_step;
            let sound_centre_sample = ((mid_time - x1) / dx).round() as i64 + 1;
            let left_sample = sound_centre_sample - half_window_samples;
            let right_sample = sound_centre_sample + half_window_samples;
            let left_sample_clamped = left_sample.max(1) as usize;
            let right_sample_clamped = (right_sample as usize).min(nx);

            if right_sample_clamped < left_sample_clamped {
                values.push(-300.0);
                continue;
            }

            let window_from_sound_offset = window_centre - sound_centre_sample;

            // Compute per-channel means if needed (Praat's centre_VEC_inout is per-channel)
            let mut chan_means = vec![0.0; nchan];
            if subtract_mean {
                for ch in 0..nchan {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for isamp in left_sample_clamped..=right_sample_clamped {
                        let window_idx = (window_from_sound_offset + isamp as i64) as usize;
                        if window_idx > 0 && window_idx <= window_num_samples {
                            sum += all_samples[ch][isamp - 1];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        chan_means[ch] = sum / count as f64;
                    }
                }
            }

            // Sum energy across all channels (Praat: inner loop over ichan)
            let mut sumxw: f64 = 0.0;
            let mut sumw: f64 = 0.0;

            for ch in 0..nchan {
                let samples = all_samples[ch];
                let mean = chan_means[ch];
                for isamp in left_sample_clamped..=right_sample_clamped {
                    let window_idx = (window_from_sound_offset + isamp as i64) as usize;
                    if window_idx > 0 && window_idx <= window_num_samples {
                        let w = window[window_idx - 1];
                        let s = samples[isamp - 1] - mean;
                        sumxw += s * s * w;
                        sumw += w;
                    }
                }
            }

            let intensity_in_pa2 = if sumw > 0.0 { sumxw / sumw } else { 0.0 };
            let hearing_threshold_pa2 = REFERENCE_PRESSURE * REFERENCE_PRESSURE;
            let intensity_re_threshold = intensity_in_pa2 / hearing_threshold_pa2;

            let intensity_db = if intensity_re_threshold < 1.0e-30 {
                -300.0
            } else {
                10.0 * intensity_re_threshold.log10()
            };

            values.push(intensity_db);
        }

        Self::new(values, first_time, time_step, min_pitch)
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
            .filter(|&&v| v.is_finite() && v > -300.0)
            .cloned()
            .reduce(f64::min)
    }

    /// Get the maximum intensity value
    pub fn max(&self) -> Option<f64> {
        self.values
            .iter()
            .filter(|&&v| v.is_finite() && v > -300.0)
            .cloned()
            .reduce(f64::max)
    }

    /// Get the mean intensity value
    pub fn mean(&self) -> Option<f64> {
        let finite_values: Vec<f64> = self.values.iter()
            .filter(|&&v| v.is_finite() && v > -300.0)
            .cloned()
            .collect();
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

/// Compute intensity from multiple Sound channels (Praat-compatible stereo handling)
///
/// Praat sums energy across channels in the inner loop, then normalizes by
/// total weight (including channel count). This gives the average energy
/// across channels, which differs from computing energy on sample-averaged mono.
pub fn intensity_from_channels(sounds: &[Sound], min_pitch: f64, time_step: f64) -> Intensity {
    Intensity::from_channels(sounds, min_pitch, time_step, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bessel_i0() {
        // Test against known values
        assert_relative_eq!(bessel_i0(0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(bessel_i0(1.0), 1.2660658, epsilon = 1e-6);
        assert_relative_eq!(bessel_i0(2.0), 2.2795853, epsilon = 1e-5);
        assert_relative_eq!(bessel_i0(3.75), 9.1192, epsilon = 1e-3);
        assert_relative_eq!(bessel_i0(5.0), 27.2399, epsilon = 1e-3);
    }

    #[test]
    fn test_intensity_pure_tone() {
        // Create a pure tone at 440 Hz
        let sound = Sound::create_tone(440.0, 0.5, 44100.0, 0.1, 0.0);

        let intensity = sound.to_intensity(100.0, 0.0);

        // Should have some frames
        assert!(intensity.num_frames() > 0);

        // All frames should have similar intensity (steady tone)
        let values: Vec<f64> = intensity.values().iter()
            .filter(|&&v| v.is_finite() && v > -300.0)
            .cloned()
            .collect();
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

        // All values should be -300 (Praat's minimum) for silence
        for &v in intensity.values() {
            assert!(v <= -300.0 || !v.is_finite());
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
