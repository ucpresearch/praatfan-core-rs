//! Pitch (F0) analysis using autocorrelation method
//!
//! This module implements Praat's autocorrelation-based pitch tracking algorithm
//! based on Boersma (1993): "Accurate short-term analysis of the fundamental
//! frequency and the harmonics-to-noise ratio of a sampled sound."
//!
//! The algorithm computes autocorrelation in overlapping frames, finds peaks
//! corresponding to pitch candidates, and uses dynamic programming to select
//! the optimal pitch path.

use crate::interpolation::{parabolic_peak, Interpolation};
use crate::utils::Fft;
use crate::window::WindowShape;
use crate::{PitchUnit, Sound};

/// Maximum number of pitch candidates per frame
const MAX_CANDIDATES: usize = 15;

/// A pitch candidate for a single frame
#[derive(Debug, Clone, Copy)]
struct PitchCandidate {
    /// Frequency in Hz (0.0 for unvoiced)
    frequency: f64,
    /// Strength (autocorrelation value or voicelessness measure)
    strength: f64,
}

/// A single frame of pitch analysis
#[derive(Debug, Clone)]
struct PitchFrame {
    /// Pitch candidates for this frame
    candidates: Vec<PitchCandidate>,
    /// Index of the selected candidate
    selected: usize,
}

/// Pitch contour representing fundamental frequency over time
#[derive(Debug, Clone)]
pub struct Pitch {
    /// Analysis frames
    frames: Vec<PitchFrame>,
    /// Time of first frame center
    start_time: f64,
    /// Time step between frames
    time_step: f64,
    /// Pitch floor used for analysis (Hz)
    pitch_floor: f64,
    /// Pitch ceiling used for analysis (Hz)
    pitch_ceiling: f64,
}

impl Pitch {
    /// Compute pitch from a Sound using autocorrelation method
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic: 0.75/pitch_floor)
    /// * `pitch_floor` - Minimum expected pitch (Hz), typically 75 Hz for male, 100 Hz for female
    /// * `pitch_ceiling` - Maximum expected pitch (Hz), typically 500-600 Hz
    ///
    /// # Algorithm
    /// 1. Divide signal into overlapping frames (window = 3/pitch_floor)
    /// 2. For each frame:
    ///    - Apply Hanning window
    ///    - Compute normalized autocorrelation via FFT
    ///    - Find peaks in lag range [1/pitch_ceiling, 1/pitch_floor]
    ///    - Create pitch candidates from peaks
    /// 3. Use Viterbi algorithm to select optimal path through candidates
    pub fn from_sound(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
    ) -> Self {
        // Validate and clamp parameters
        let pitch_floor = pitch_floor.max(10.0).min(pitch_ceiling - 1.0);
        let pitch_ceiling = pitch_ceiling.max(pitch_floor + 1.0);

        // Time step: default is 0.75 / pitch_floor (gives good time resolution)
        let time_step = if time_step <= 0.0 {
            0.75 / pitch_floor
        } else {
            time_step
        };

        // Window duration: 3 periods of minimum pitch
        let window_duration = 3.0 / pitch_floor;
        let sample_rate = sound.sample_rate();
        let samples = sound.samples();

        // Calculate frame parameters
        let window_samples = (window_duration * sample_rate).round() as usize;
        let half_window_samples = window_samples / 2;

        // Frame timing
        let first_frame_time = sound.start_time() + window_duration / 2.0;
        let last_frame_time = sound.end_time() - window_duration / 2.0;

        if first_frame_time >= last_frame_time || samples.is_empty() {
            return Self {
                frames: Vec::new(),
                start_time: sound.start_time(),
                time_step,
                pitch_floor,
                pitch_ceiling,
            };
        }

        let num_frames = ((last_frame_time - first_frame_time) / time_step).floor() as usize + 1;

        // Generate window
        let window = WindowShape::Hanning.generate(window_samples, None);

        // Lag range in samples
        let min_lag = (sample_rate / pitch_ceiling).round() as usize;
        let max_lag = (sample_rate / pitch_floor).round() as usize;

        // FFT for autocorrelation (need power of 2 for efficiency)
        let fft_size = (2 * window_samples).next_power_of_two();
        let mut fft = Fft::new();

        // Voiceless strength threshold
        let voiceless_threshold = 0.45; // Praat default

        let mut frames = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;
            let center_sample = sound.time_to_index_clamped(frame_time);

            // Extract and window the frame
            let start_sample = center_sample.saturating_sub(half_window_samples);
            let end_sample = (center_sample + half_window_samples).min(samples.len());

            let mut windowed = vec![0.0; window_samples];
            for (i, &s) in samples[start_sample..end_sample].iter().enumerate() {
                if i < window.len() {
                    windowed[i] = s * window[i];
                }
            }

            // Compute normalized autocorrelation
            let autocorr = compute_normalized_autocorrelation(&mut fft, &windowed, fft_size);

            // Find pitch candidates from autocorrelation peaks
            let candidates = find_pitch_candidates(
                &autocorr,
                sample_rate,
                min_lag,
                max_lag,
                voiceless_threshold,
            );

            frames.push(PitchFrame {
                candidates,
                selected: 0, // Will be set by Viterbi
            });
        }

        // Run Viterbi algorithm to select optimal pitch path
        viterbi_path(&mut frames, pitch_floor, pitch_ceiling);

        Self {
            frames,
            start_time: first_frame_time,
            time_step,
            pitch_floor,
            pitch_ceiling,
        }
    }

    /// Get pitch value at a specific time
    ///
    /// # Arguments
    /// * `time` - Time at which to query pitch
    /// * `unit` - Unit for the returned value
    /// * `interpolation` - Interpolation method
    ///
    /// # Returns
    /// Pitch value in the specified unit, or None if unvoiced or outside range
    pub fn get_value_at_time(
        &self,
        time: f64,
        unit: PitchUnit,
        interpolation: Interpolation,
    ) -> Option<f64> {
        if self.frames.is_empty() {
            return None;
        }

        // Get frame position
        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.frames.len() as f64 - 0.5 {
            return None;
        }

        // Extract pitch values for interpolation
        let pitch_values: Vec<f64> = self
            .frames
            .iter()
            .map(|f| {
                let candidate = &f.candidates[f.selected];
                if candidate.frequency > 0.0 {
                    candidate.frequency
                } else {
                    f64::NAN // Unvoiced
                }
            })
            .collect();

        // Interpolate
        let hz = interpolation.interpolate_with_undefined(&pitch_values, position.max(0.0))?;

        // Convert to requested unit
        Some(unit.from_hertz(hz))
    }

    /// Get the pitch value at a specific frame
    pub fn get_value_at_frame(&self, frame: usize) -> Option<f64> {
        self.frames.get(frame).and_then(|f| {
            let candidate = &f.candidates[f.selected];
            if candidate.frequency > 0.0 {
                Some(candidate.frequency)
            } else {
                None
            }
        })
    }

    /// Get the strength (autocorrelation) at a specific frame
    pub fn get_strength_at_frame(&self, frame: usize) -> Option<f64> {
        self.frames.get(frame).map(|f| f.candidates[f.selected].strength)
    }

    /// Check if a frame is voiced
    pub fn is_voiced(&self, frame: usize) -> bool {
        self.frames
            .get(frame)
            .map(|f| f.candidates[f.selected].frequency > 0.0)
            .unwrap_or(false)
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

    /// Get minimum pitch value (over voiced frames)
    pub fn min(&self) -> Option<f64> {
        self.frames
            .iter()
            .filter_map(|f| {
                let freq = f.candidates[f.selected].frequency;
                if freq > 0.0 { Some(freq) } else { None }
            })
            .reduce(f64::min)
    }

    /// Get maximum pitch value (over voiced frames)
    pub fn max(&self) -> Option<f64> {
        self.frames
            .iter()
            .filter_map(|f| {
                let freq = f.candidates[f.selected].frequency;
                if freq > 0.0 { Some(freq) } else { None }
            })
            .reduce(f64::max)
    }

    /// Get mean pitch value (over voiced frames)
    pub fn mean(&self) -> Option<f64> {
        let voiced: Vec<f64> = self
            .frames
            .iter()
            .filter_map(|f| {
                let freq = f.candidates[f.selected].frequency;
                if freq > 0.0 { Some(freq) } else { None }
            })
            .collect();

        if voiced.is_empty() {
            None
        } else {
            Some(voiced.iter().sum::<f64>() / voiced.len() as f64)
        }
    }

    /// Count voiced frames
    pub fn count_voiced(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.candidates[f.selected].frequency > 0.0)
            .count()
    }

    /// Get the pitch floor used for analysis
    pub fn pitch_floor(&self) -> f64 {
        self.pitch_floor
    }

    /// Get the pitch ceiling used for analysis
    pub fn pitch_ceiling(&self) -> f64 {
        self.pitch_ceiling
    }
}

/// Compute normalized autocorrelation of a windowed signal
fn compute_normalized_autocorrelation(fft: &mut Fft, windowed: &[f64], _fft_size: usize) -> Vec<f64> {
    // Compute autocorrelation via FFT
    let raw_autocorr = fft.autocorrelation(windowed);

    if raw_autocorr.is_empty() || raw_autocorr[0] == 0.0 {
        return vec![0.0; windowed.len()];
    }

    // Normalize by the lag-0 value
    let norm = 1.0 / raw_autocorr[0];

    // Apply window correction for normalized autocorrelation
    // This corrects for the fact that we're computing windowed autocorrelation
    let n = windowed.len();
    let mut normalized = Vec::with_capacity(n);

    for lag in 0..n {
        // Window correction factor (triangular tapering)
        let correction = (n - lag) as f64 / n as f64;
        if correction > 0.0 {
            normalized.push((raw_autocorr[lag] * norm) / correction);
        } else {
            normalized.push(0.0);
        }
    }

    normalized
}

/// Find pitch candidates from autocorrelation peaks
fn find_pitch_candidates(
    autocorr: &[f64],
    sample_rate: f64,
    min_lag: usize,
    max_lag: usize,
    voiceless_threshold: f64,
) -> Vec<PitchCandidate> {
    let mut candidates = Vec::with_capacity(MAX_CANDIDATES);

    // First candidate is always "unvoiced"
    candidates.push(PitchCandidate {
        frequency: 0.0,
        strength: voiceless_threshold,
    });

    if autocorr.len() < max_lag + 2 {
        return candidates;
    }

    // Find local maxima in the autocorrelation within the lag range
    let search_start = min_lag.max(1);
    let search_end = max_lag.min(autocorr.len() - 2);

    for lag in search_start..search_end {
        // Check if this is a local maximum
        if autocorr[lag] > autocorr[lag - 1] && autocorr[lag] > autocorr[lag + 1] {
            // Use parabolic interpolation to refine the peak
            let (offset, peak_value) = parabolic_peak(
                autocorr[lag - 1],
                autocorr[lag],
                autocorr[lag + 1],
            );

            let refined_lag = lag as f64 + offset;
            let frequency = sample_rate / refined_lag;

            // Only keep positive correlation peaks
            if peak_value > 0.0 {
                candidates.push(PitchCandidate {
                    frequency,
                    strength: peak_value.min(1.0), // Clamp to valid range
                });

                if candidates.len() >= MAX_CANDIDATES {
                    break;
                }
            }
        }
    }

    // Sort by strength (descending)
    candidates[1..].sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

    candidates
}

/// Viterbi algorithm for optimal pitch path selection
fn viterbi_path(frames: &mut [PitchFrame], pitch_floor: f64, pitch_ceiling: f64) {
    if frames.is_empty() {
        return;
    }

    let num_frames = frames.len();

    // Cost parameters (Praat defaults)
    let octave_cost = 0.01;
    let voiced_unvoiced_cost = 0.14;
    let octave_jump_cost = 0.35;

    // Viterbi scores and backpointers
    let mut scores: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| vec![f64::NEG_INFINITY; f.candidates.len()])
        .collect();
    let mut backptr: Vec<Vec<usize>> = frames
        .iter()
        .map(|f| vec![0; f.candidates.len()])
        .collect();

    // Initialize first frame
    for (j, cand) in frames[0].candidates.iter().enumerate() {
        let mut score = cand.strength;

        // Penalize candidates far from center of pitch range (in octaves)
        if cand.frequency > 0.0 {
            let center = (pitch_floor * pitch_ceiling).sqrt();
            let octaves_from_center = (cand.frequency / center).log2().abs();
            score -= octave_cost * octaves_from_center;
        }

        scores[0][j] = score;
    }

    // Forward pass
    for i in 1..num_frames {
        for (j, cand_j) in frames[i].candidates.iter().enumerate() {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_prev = 0;

            for (k, cand_k) in frames[i - 1].candidates.iter().enumerate() {
                let mut transition_cost = 0.0;

                // Voiced-unvoiced transition cost
                let j_voiced = cand_j.frequency > 0.0;
                let k_voiced = cand_k.frequency > 0.0;

                if j_voiced != k_voiced {
                    transition_cost += voiced_unvoiced_cost;
                } else if j_voiced && k_voiced {
                    // Octave jump cost (penalize large frequency changes)
                    let octave_jump = (cand_j.frequency / cand_k.frequency).log2().abs();
                    transition_cost += octave_jump_cost * octave_jump;
                }

                let score = scores[i - 1][k] - transition_cost;
                if score > best_score {
                    best_score = score;
                    best_prev = k;
                }
            }

            // Add local score
            let mut local_score = cand_j.strength;
            if cand_j.frequency > 0.0 {
                let center = (pitch_floor * pitch_ceiling).sqrt();
                let octaves_from_center = (cand_j.frequency / center).log2().abs();
                local_score -= octave_cost * octaves_from_center;
            }

            scores[i][j] = best_score + local_score;
            backptr[i][j] = best_prev;
        }
    }

    // Backward pass - find best final state
    let mut best_final = 0;
    let mut best_final_score = f64::NEG_INFINITY;
    for (j, &score) in scores[num_frames - 1].iter().enumerate() {
        if score > best_final_score {
            best_final_score = score;
            best_final = j;
        }
    }

    // Trace back
    frames[num_frames - 1].selected = best_final;
    for i in (0..num_frames - 1).rev() {
        let selected = frames[i + 1].selected;
        frames[i].selected = backptr[i + 1][selected];
    }
}

// Add to_pitch method to Sound
impl Sound {
    /// Compute pitch contour from this sound
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `pitch_floor` - Minimum expected pitch (Hz)
    /// * `pitch_ceiling` - Maximum expected pitch (Hz)
    ///
    /// # Returns
    /// Pitch contour with F0 values
    pub fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        Pitch::from_sound(self, time_step, pitch_floor, pitch_ceiling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_pure_tone() {
        // Create a pure tone at 200 Hz
        let freq = 200.0;
        let sound = Sound::create_tone(freq, 0.5, 44100.0, 0.5, 0.0);

        let pitch = sound.to_pitch(0.0, 75.0, 600.0);

        // Should have frames
        assert!(pitch.num_frames() > 0);

        // Most frames should be voiced with pitch near 200 Hz
        let mut voiced_count = 0;
        let mut sum = 0.0;

        for i in 0..pitch.num_frames() {
            if let Some(f0) = pitch.get_value_at_frame(i) {
                voiced_count += 1;
                sum += f0;
            }
        }

        assert!(voiced_count > pitch.num_frames() / 2, "Most frames should be voiced");

        if voiced_count > 0 {
            let mean_f0 = sum / voiced_count as f64;
            assert!(
                (mean_f0 - freq).abs() < 10.0,
                "Mean pitch {} Hz should be close to {} Hz",
                mean_f0,
                freq
            );
        }
    }

    #[test]
    fn test_pitch_silence() {
        let sound = Sound::create_silence(0.5, 44100.0);
        let pitch = sound.to_pitch(0.0, 75.0, 600.0);

        // All frames should be unvoiced
        for i in 0..pitch.num_frames() {
            assert!(!pitch.is_voiced(i), "Frame {} should be unvoiced", i);
        }
    }

    #[test]
    fn test_pitch_frequency_tracking() {
        // Create a tone at 150 Hz
        let freq = 150.0;
        let sound = Sound::create_tone(freq, 0.3, 44100.0, 0.5, 0.0);

        let pitch = sound.to_pitch(0.01, 75.0, 500.0);

        // Query at middle of the sound
        let t = sound.duration() / 2.0;
        let f0 = pitch.get_value_at_time(t, PitchUnit::Hertz, Interpolation::Linear);

        assert!(f0.is_some(), "Should find pitch at t={}", t);
        let f0 = f0.unwrap();
        assert!(
            (f0 - freq).abs() < 5.0,
            "Pitch {} Hz should be close to {} Hz",
            f0,
            freq
        );
    }

    #[test]
    fn test_pitch_statistics() {
        let freq = 180.0;
        let sound = Sound::create_tone(freq, 0.5, 44100.0, 0.5, 0.0);
        let pitch = sound.to_pitch(0.0, 75.0, 600.0);

        // Statistics should be defined
        assert!(pitch.min().is_some());
        assert!(pitch.max().is_some());
        assert!(pitch.mean().is_some());

        // All stats should be close to the true frequency
        let mean = pitch.mean().unwrap();
        assert!(
            (mean - freq).abs() < 10.0,
            "Mean {} Hz should be close to {} Hz",
            mean,
            freq
        );
    }
}
