//! Pitch (F0) analysis using autocorrelation method
//!
//! This module implements Praat's autocorrelation-based pitch tracking algorithm
//! based on Boersma (1993): "Accurate short-term analysis of the fundamental
//! frequency and the harmonics-to-noise ratio of a sampled sound."
//!
//! The algorithm computes autocorrelation in overlapping frames, finds peaks
//! corresponding to pitch candidates, and uses dynamic programming to select
//! the optimal pitch path.

use crate::interpolation::Interpolation;
use crate::utils::Fft;
use crate::{PitchUnit, Sound};
use std::f64::consts::PI;

/// Maximum number of pitch candidates per frame
const MAX_CANDIDATES: usize = 15;

/// Pitch extraction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    /// Autocorrelation with Hanning window (Praat's AC_HANNING, method 0)
    AcHanning,
    /// Autocorrelation with Gaussian window (Praat's AC_GAUSS, method 1)
    AcGauss,
    /// Forward cross-correlation, accurate (Praat's FCC_ACCURATE, method 3)
    /// Used by Harmonicity CC - computes correlation in time domain without windowing
    FccAccurate,
}

/// A pitch candidate for a single frame
#[derive(Debug, Clone, Copy)]
pub struct PitchCandidate {
    /// Frequency in Hz (0.0 for unvoiced)
    pub frequency: f64,
    /// Strength (autocorrelation value)
    pub strength: f64,
}

/// A single frame of pitch analysis
#[derive(Debug, Clone)]
pub struct PitchFrame {
    /// Pitch candidates for this frame (first is always the "winner" after path finding)
    pub candidates: Vec<PitchCandidate>,
    /// Intensity of this frame (normalized local peak)
    pub intensity: f64,
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
    /// Start time of the original sound
    #[allow(dead_code)]
    xmin: f64,
    /// End time of the original sound
    #[allow(dead_code)]
    xmax: f64,
}

impl Pitch {
    /// Compute pitch from a Sound using autocorrelation method (Praat-compatible)
    ///
    /// This implements Praat's `Sound_to_Pitch` with default parameters:
    /// - method: AC_HANNING (autocorrelation with Hanning window)
    /// - periodsPerWindow: 3.0
    /// - maxnCandidates: 15
    /// - silenceThreshold: 0.03
    /// - voicingThreshold: 0.45
    /// - octaveCost: 0.01
    /// - octaveJumpCost: 0.35
    /// - voicedUnvoicedCost: 0.14
    pub fn from_sound(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
    ) -> Self {
        Self::from_sound_full(
            sound,
            time_step,
            pitch_floor,
            pitch_ceiling,
            MAX_CANDIDATES,
            0.03,  // silenceThreshold
            0.45,  // voicingThreshold
            0.01,  // octaveCost
            0.35,  // octaveJumpCost
            0.14,  // voicedUnvoicedCost
        )
    }

    /// Compute pitch with full parameter control (matches Sound_to_Pitch_rawAc)
    pub fn from_sound_full(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
    ) -> Self {
        // Default periodsPerWindow = 3.0 for AC_HANNING
        Self::from_sound_full_ex(
            sound,
            time_step,
            pitch_floor,
            pitch_ceiling,
            max_candidates,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            3.0, // periods_per_window
        )
    }

    /// Compute pitch with full parameter control including periods_per_window
    pub fn from_sound_full_ex(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
        periods_per_window: f64,
    ) -> Self {
        Self::from_sound_with_method(
            sound,
            time_step,
            pitch_floor,
            pitch_ceiling,
            max_candidates,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            periods_per_window,
            PitchMethod::AcHanning,
        )
    }

    /// Compute pitch with full parameter control including method selection
    pub fn from_sound_with_method(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
        periods_per_window: f64,
        method: PitchMethod,
    ) -> Self {
        // Match Praat's parameter validation
        let pitch_floor = pitch_floor.max(10.0);
        let pitch_ceiling = pitch_ceiling.min(0.5 / sound.dx());

        // Method-specific parameters
        // - AC_GAUSS doubles periods_per_window
        // - FCC uses interpolation_depth = 1.0
        let (periods_per_window, interpolation_depth) = match method {
            PitchMethod::AcHanning => (periods_per_window, 0.5),
            PitchMethod::AcGauss => (periods_per_window * 2.0, 0.25),
            PitchMethod::FccAccurate => (periods_per_window, 1.0),
        };

        // For FCC, use the FCC-specific implementation
        if method == PitchMethod::FccAccurate {
            return Self::from_sound_fcc(
                sound,
                time_step,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                silence_threshold,
                voicing_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
                periods_per_window,
            );
        }

        // Time step: default is periods_per_window / pitch_floor / 4
        let dt = if time_step <= 0.0 {
            periods_per_window / pitch_floor / 4.0
        } else {
            time_step
        };

        let dx = sound.dx();
        let nx = sound.num_samples();
        let xmin = sound.start_time();
        let xmax = sound.end_time();
        let x1 = sound.x1();

        // Number of samples in the longest period
        let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
        let halfnsamp_period = nsamp_period / 2 + 1;

        // Window duration and samples (matching Praat exactly)
        let dt_window = periods_per_window / pitch_floor;
        let nsamp_window_raw = (dt_window / dx).floor() as usize;
        let halfnsamp_window = (nsamp_window_raw / 2).saturating_sub(1);
        if halfnsamp_window < 2 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let nsamp_window = halfnsamp_window * 2;

        // Minimum and maximum lags
        let _minimum_lag = (1.0 / dx / pitch_ceiling).floor() as usize;
        let _minimum_lag = _minimum_lag.max(2);
        let maximum_lag = ((nsamp_window as f64 / periods_per_window).floor() as usize + 2)
            .min(nsamp_window);

        // Calculate number of frames using Praat's Sampled_shortTermAnalysis
        // From Sampled.cpp:
        // myDuration = my dx * my nx
        // *numberOfFrames = floor((myDuration - windowDuration) / timeStep) + 1
        // ourMidTime = my x1 - 0.5 * my dx + 0.5 * myDuration
        // thyDuration = *numberOfFrames * timeStep
        // *firstTime = ourMidTime - 0.5 * thyDuration + 0.5 * timeStep
        let my_duration = dx * nx as f64;
        if my_duration < dt_window {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let number_of_frames = ((my_duration - dt_window) / dt).floor() as usize + 1;
        if number_of_frames < 1 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = number_of_frames as f64 * dt;
        let t1 = our_mid_time - 0.5 * thy_duration + 0.5 * dt;

        // FFT size
        let mut nsamp_fft = 1;
        while nsamp_fft < (nsamp_window as f64 * (1.0 + interpolation_depth)) as usize {
            nsamp_fft *= 2;
        }

        // Create window based on method
        // Note: FCC is handled by early return above, so this only handles AC methods
        let mut window = vec![0.0; nsamp_window];
        match method {
            PitchMethod::AcGauss => {
                // Gaussian window (Praat's formula from Sound_to_Pitch.cpp line 407-411)
                // exp(-48.0 * (i - imid)^2 / (nsamp_window + 1)^2) with edge subtraction
                let imid = 0.5 * (nsamp_window + 1) as f64;
                let edge = (-12.0_f64).exp();
                for i in 0..nsamp_window {
                    let i1 = (i + 1) as f64; // 1-based
                    let exponent = -48.0 * (i1 - imid).powi(2)
                        / ((nsamp_window + 1) as f64).powi(2);
                    window[i] = (exponent.exp() - edge) / (1.0 - edge);
                }
            }
            PitchMethod::AcHanning => {
                // Hanning window (Praat's formula)
                for i in 0..nsamp_window {
                    let i1 = i + 1; // 1-based
                    window[i] = 0.5 - 0.5 * (2.0 * PI * i1 as f64 / (nsamp_window + 1) as f64).cos();
                }
            }
            PitchMethod::FccAccurate => {
                // FCC is handled by early return above; this should be unreachable
                unreachable!("FCC method should have been handled earlier")
            }
        }

        // Compute normalized autocorrelation of the window
        let mut fft = Fft::new();
        let mut window_r = vec![0.0; nsamp_fft];
        for i in 0..nsamp_window {
            window_r[i] = window[i];
        }
        // FFT forward
        let mut window_fft = vec![0.0; nsamp_fft];
        for i in 0..nsamp_fft {
            window_fft[i] = window_r[i];
        }
        let window_autocorr = fft.autocorrelation(&window_fft);
        // Normalize
        let mut window_r_norm = vec![0.0; nsamp_window + 1];
        if window_autocorr[0] > 0.0 {
            window_r_norm[0] = 1.0;
            for i in 1..=nsamp_window.min(window_autocorr.len() - 1) {
                window_r_norm[i] = window_autocorr[i] / window_autocorr[0];
            }
        } else {
            window_r_norm[0] = 1.0;
        }

        let brent_ixmax = (nsamp_window as f64 * interpolation_depth).floor() as usize;

        // Compute global peak for intensity normalization
        let samples = sound.samples();
        let mean: f64 = samples.iter().sum::<f64>() / nx as f64;
        let global_peak = samples
            .iter()
            .map(|&s| (s - mean).abs())
            .fold(0.0, f64::max);

        if global_peak == 0.0 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }

        // Adjust max_candidates if needed
        let max_candidates = max_candidates.max((pitch_ceiling / pitch_floor).floor() as usize);

        // Process each frame
        let mut frames = Vec::with_capacity(number_of_frames);
        for iframe in 0..number_of_frames {
            let time = t1 + iframe as f64 * dt;
            let frame = compute_pitch_frame(
                samples,
                dx,
                x1,
                time,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                voicing_threshold,
                octave_cost,
                nsamp_window,
                halfnsamp_window,
                nsamp_period,
                halfnsamp_period,
                maximum_lag,
                brent_ixmax,
                global_peak,
                &window,
                &window_r_norm,
                &mut fft,
                nsamp_fft,
            );
            frames.push(frame);
        }

        // Run path finder (Viterbi)
        pitch_path_finder(
            &mut frames,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            pitch_ceiling,
            dt,
        );

        Self {
            frames,
            start_time: t1,
            time_step: dt,
            pitch_floor,
            pitch_ceiling,
            xmin,
            xmax,
        }
    }

    fn empty(xmin: f64, xmax: f64, dt: f64, pitch_floor: f64, pitch_ceiling: f64) -> Self {
        Self {
            frames: Vec::new(),
            start_time: xmin,
            time_step: dt,
            pitch_floor,
            pitch_ceiling,
            xmin,
            xmax,
        }
    }

    /// Compute pitch from multiple Sound channels, summing autocorrelations
    ///
    /// This matches Praat's behavior for multi-channel audio: each channel's
    /// autocorrelation (power spectrum) is computed separately, then summed.
    /// This is different from mixing channels to mono first.
    ///
    /// For stereo: `AC(ch1) + AC(ch2)` (correct for Praat compatibility)
    /// vs: `AC((ch1+ch2)/2)` (what you get with sample averaging)
    #[allow(clippy::too_many_arguments)]
    pub fn from_channels_with_method(
        sounds: &[Sound],
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
        periods_per_window: f64,
        method: PitchMethod,
    ) -> Self {
        if sounds.is_empty() {
            return Self::empty(0.0, 0.0, time_step, pitch_floor, pitch_ceiling);
        }

        // For single channel, just use the standard method
        if sounds.len() == 1 {
            return Self::from_sound_with_method(
                &sounds[0],
                time_step,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                silence_threshold,
                voicing_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
                periods_per_window,
                method,
            );
        }

        // Verify all sounds have same sample rate
        let sample_rate = sounds[0].sample_rate();
        for sound in sounds.iter().skip(1) {
            assert!(
                (sound.sample_rate() - sample_rate).abs() < 0.01,
                "All sounds must have the same sample rate"
            );
        }

        // For FCC (CC method), use the FCC-specific multi-channel implementation
        if method == PitchMethod::FccAccurate {
            return Self::from_channels_fcc(
                sounds,
                time_step,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                silence_threshold,
                voicing_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
                periods_per_window,
            );
        }

        // Match Praat's parameter validation (use first sound for timing)
        let sound = &sounds[0];
        let pitch_floor = pitch_floor.max(10.0);
        let pitch_ceiling = pitch_ceiling.min(0.5 / sound.dx());

        // Method-specific parameters
        let (periods_per_window, interpolation_depth) = match method {
            PitchMethod::AcHanning => (periods_per_window, 0.5),
            PitchMethod::AcGauss => (periods_per_window * 2.0, 0.25),
            PitchMethod::FccAccurate => unreachable!(),
        };

        // Time step: default is periods_per_window / pitch_floor / 4
        let dt = if time_step <= 0.0 {
            periods_per_window / pitch_floor / 4.0
        } else {
            time_step
        };

        let dx = sound.dx();
        let nx = sound.num_samples();
        let xmin = sound.start_time();
        let xmax = sound.end_time();
        let x1 = sound.x1();

        // Number of samples in the longest period
        let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
        let halfnsamp_period = nsamp_period / 2 + 1;

        // Window duration and samples
        let dt_window = periods_per_window / pitch_floor;
        let nsamp_window_raw = (dt_window / dx).floor() as usize;
        let halfnsamp_window = (nsamp_window_raw / 2).saturating_sub(1);
        if halfnsamp_window < 2 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let nsamp_window = halfnsamp_window * 2;

        // Maximum lag
        let maximum_lag = ((nsamp_window as f64 / periods_per_window).floor() as usize + 2)
            .min(nsamp_window);

        // Calculate number of frames
        let my_duration = dx * nx as f64;
        if my_duration < dt_window {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let number_of_frames = ((my_duration - dt_window) / dt).floor() as usize + 1;
        if number_of_frames < 1 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = number_of_frames as f64 * dt;
        let t1 = our_mid_time - 0.5 * thy_duration + 0.5 * dt;

        // FFT size
        let mut nsamp_fft = 1;
        while nsamp_fft < (nsamp_window as f64 * (1.0 + interpolation_depth)) as usize {
            nsamp_fft *= 2;
        }

        // Create window based on method
        let mut window = vec![0.0; nsamp_window];
        match method {
            PitchMethod::AcGauss => {
                let imid = 0.5 * (nsamp_window + 1) as f64;
                let edge = (-12.0_f64).exp();
                for i in 0..nsamp_window {
                    let i1 = (i + 1) as f64;
                    let exponent = -48.0 * (i1 - imid).powi(2) / ((nsamp_window + 1) as f64).powi(2);
                    window[i] = (exponent.exp() - edge) / (1.0 - edge);
                }
            }
            PitchMethod::AcHanning => {
                for i in 0..nsamp_window {
                    let i1 = i + 1;
                    window[i] = 0.5 - 0.5 * (2.0 * PI * i1 as f64 / (nsamp_window + 1) as f64).cos();
                }
            }
            PitchMethod::FccAccurate => unreachable!(),
        }

        // Compute normalized autocorrelation of the window
        let mut fft = Fft::new();
        let mut window_fft = vec![0.0; nsamp_fft];
        for i in 0..nsamp_window {
            window_fft[i] = window[i];
        }
        let window_autocorr = fft.autocorrelation(&window_fft);
        let mut window_r_norm = vec![0.0; nsamp_window + 1];
        if window_autocorr[0] > 0.0 {
            window_r_norm[0] = 1.0;
            for i in 1..=nsamp_window.min(window_autocorr.len() - 1) {
                window_r_norm[i] = window_autocorr[i] / window_autocorr[0];
            }
        } else {
            window_r_norm[0] = 1.0;
        }

        let brent_ixmax = (nsamp_window as f64 * interpolation_depth).floor() as usize;

        // Compute global peak across all channels
        let mut global_peak = 0.0;
        for sound in sounds {
            let samples = sound.samples();
            let mean: f64 = samples.iter().sum::<f64>() / nx as f64;
            let channel_peak = samples.iter().map(|&s| (s - mean).abs()).fold(0.0, f64::max);
            if channel_peak > global_peak {
                global_peak = channel_peak;
            }
        }

        if global_peak == 0.0 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }

        // Adjust max_candidates if needed
        let max_candidates = max_candidates.max((pitch_ceiling / pitch_floor).floor() as usize);

        // Collect samples from all channels
        let channel_samples: Vec<&[f64]> = sounds.iter().map(|s| s.samples()).collect();

        // Process each frame
        let mut frames = Vec::with_capacity(number_of_frames);
        for iframe in 0..number_of_frames {
            let time = t1 + iframe as f64 * dt;
            let frame = compute_pitch_frame_multichannel(
                &channel_samples,
                dx,
                x1,
                time,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                voicing_threshold,
                octave_cost,
                nsamp_window,
                halfnsamp_window,
                nsamp_period,
                halfnsamp_period,
                maximum_lag,
                brent_ixmax,
                global_peak,
                &window,
                &window_r_norm,
                &mut fft,
                nsamp_fft,
            );
            frames.push(frame);
        }

        // Run path finder (Viterbi)
        pitch_path_finder(
            &mut frames,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            pitch_ceiling,
            dt,
        );

        Self {
            frames,
            start_time: t1,
            time_step: dt,
            pitch_floor,
            pitch_ceiling,
            xmin,
            xmax,
        }
    }

    /// FCC (Forward Cross-Correlation) multi-channel implementation
    #[allow(clippy::too_many_arguments)]
    fn from_channels_fcc(
        sounds: &[Sound],
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
        periods_per_window: f64,
    ) -> Self {
        let sound = &sounds[0];
        let pitch_floor = pitch_floor.max(10.0);
        let pitch_ceiling = pitch_ceiling.min(0.5 / sound.dx());

        let dx = sound.dx();
        let nx = sound.num_samples();
        let xmin = sound.start_time();
        let xmax = sound.end_time();
        let x1 = sound.x1();

        let interpolation_depth = 1.0;
        let dt_window = periods_per_window / pitch_floor;
        let nsamp_window = (dt_window / dx).floor() as usize;
        if nsamp_window < 4 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }

        let maximum_lag = (1.0 / pitch_floor / dx).floor() as usize;
        let dt = if time_step <= 0.0 {
            periods_per_window / pitch_floor / 4.0
        } else {
            time_step
        };

        let dt_window_for_timing = 1.0 / pitch_floor + dt_window;
        let my_duration = dx * nx as f64;
        if my_duration < dt_window_for_timing {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let number_of_frames = ((my_duration - dt_window_for_timing) / dt).floor() as usize + 1;
        if number_of_frames < 1 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = number_of_frames as f64 * dt;
        let t1 = our_mid_time - 0.5 * thy_duration + 0.5 * dt;

        let brent_ixmax = (nsamp_window as f64 * interpolation_depth).floor() as usize;

        // Compute global peak across all channels
        let mut global_peak = 0.0;
        for sound in sounds {
            let samples = sound.samples();
            let mean: f64 = samples.iter().sum::<f64>() / nx as f64;
            let channel_peak = samples.iter().map(|&s| (s - mean).abs()).fold(0.0, f64::max);
            if channel_peak > global_peak {
                global_peak = channel_peak;
            }
        }

        if global_peak == 0.0 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }

        let max_candidates = max_candidates.max((pitch_ceiling / pitch_floor).floor() as usize);
        let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
        let halfnsamp_period = nsamp_period / 2 + 1;
        let halfnsamp_window = nsamp_window / 2;

        // Collect samples from all channels
        let channel_samples: Vec<&[f64]> = sounds.iter().map(|s| s.samples()).collect();

        let mut frames = Vec::with_capacity(number_of_frames);
        for iframe in 0..number_of_frames {
            let time = t1 + iframe as f64 * dt;
            let frame = compute_pitch_frame_fcc_multichannel(
                &channel_samples,
                dx,
                x1,
                nx,
                time,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                voicing_threshold,
                octave_cost,
                nsamp_window,
                halfnsamp_window,
                halfnsamp_period,
                maximum_lag,
                brent_ixmax,
                global_peak,
            );
            frames.push(frame);
        }

        pitch_path_finder(
            &mut frames,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            pitch_ceiling,
            dt,
        );

        Self {
            frames,
            start_time: t1,
            time_step: dt,
            pitch_floor,
            pitch_ceiling,
            xmin,
            xmax,
        }
    }

    /// FCC (Forward Cross-Correlation) implementation
    /// This matches Praat's FCC_ACCURATE method used by Harmonicity CC
    #[allow(clippy::too_many_arguments)]
    fn from_sound_fcc(
        sound: &Sound,
        time_step: f64,
        pitch_floor: f64,
        pitch_ceiling: f64,
        max_candidates: usize,
        silence_threshold: f64,
        voicing_threshold: f64,
        octave_cost: f64,
        octave_jump_cost: f64,
        voiced_unvoiced_cost: f64,
        periods_per_window: f64,
    ) -> Self {
        let pitch_floor = pitch_floor.max(10.0);
        let pitch_ceiling = pitch_ceiling.min(0.5 / sound.dx());

        let dx = sound.dx();
        let nx = sound.num_samples();
        let xmin = sound.start_time();
        let xmax = sound.end_time();
        let x1 = sound.x1();

        // FCC uses interpolation_depth = 1.0
        let interpolation_depth = 1.0;

        // Window duration for FCC
        let dt_window = periods_per_window / pitch_floor;

        // Number of samples in analysis window
        let nsamp_window = (dt_window / dx).floor() as usize;
        if nsamp_window < 4 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }

        // Maximum lag (longest period)
        let maximum_lag = (1.0 / pitch_floor / dx).floor() as usize;

        // Time step
        let dt = if time_step <= 0.0 {
            periods_per_window / pitch_floor / 4.0
        } else {
            time_step
        };

        // For FCC, the effective window for frame timing is longer:
        // dt_window_for_timing = 1.0 / pitch_floor + dt_window
        let dt_window_for_timing = 1.0 / pitch_floor + dt_window;

        // Calculate number of frames using Praat's Sampled_shortTermAnalysis
        let my_duration = dx * nx as f64;
        if my_duration < dt_window_for_timing {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let number_of_frames = ((my_duration - dt_window_for_timing) / dt).floor() as usize + 1;
        if number_of_frames < 1 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }
        let our_mid_time = x1 - 0.5 * dx + 0.5 * my_duration;
        let thy_duration = number_of_frames as f64 * dt;
        let t1 = our_mid_time - 0.5 * thy_duration + 0.5 * dt;

        // brent_ixmax for FCC
        let brent_ixmax = (nsamp_window as f64 * interpolation_depth).floor() as usize;

        // Compute global peak for intensity normalization
        let samples = sound.samples();
        let mean: f64 = samples.iter().sum::<f64>() / nx as f64;
        let global_peak = samples
            .iter()
            .map(|&s| (s - mean).abs())
            .fold(0.0, f64::max);

        if global_peak == 0.0 {
            return Self::empty(xmin, xmax, dt, pitch_floor, pitch_ceiling);
        }

        // Adjust max_candidates if needed
        let max_candidates = max_candidates.max((pitch_ceiling / pitch_floor).floor() as usize);

        // Number of samples in the longest period (for local peak calculation)
        let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
        let halfnsamp_period = nsamp_period / 2 + 1;
        let halfnsamp_window = nsamp_window / 2;

        // Process each frame using FCC
        let mut frames = Vec::with_capacity(number_of_frames);
        for iframe in 0..number_of_frames {
            let time = t1 + iframe as f64 * dt;
            let frame = compute_pitch_frame_fcc(
                samples,
                dx,
                x1,
                nx,
                time,
                pitch_floor,
                pitch_ceiling,
                max_candidates,
                voicing_threshold,
                octave_cost,
                nsamp_window,
                halfnsamp_window,
                halfnsamp_period,
                maximum_lag,
                brent_ixmax,
                global_peak,
            );
            frames.push(frame);
        }

        // Run path finder (Viterbi)
        pitch_path_finder(
            &mut frames,
            silence_threshold,
            voicing_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
            pitch_ceiling,
            dt,
        );

        Self {
            frames,
            start_time: t1,
            time_step: dt,
            pitch_floor,
            pitch_ceiling,
            xmin,
            xmax,
        }
    }

    /// Get pitch value at a specific time
    pub fn get_value_at_time(
        &self,
        time: f64,
        unit: PitchUnit,
        interpolation: Interpolation,
    ) -> Option<f64> {
        if self.frames.is_empty() {
            return None;
        }

        let position = (time - self.start_time) / self.time_step;

        if position < -0.5 || position > self.frames.len() as f64 - 0.5 {
            return None;
        }

        let pitch_values: Vec<f64> = self
            .frames
            .iter()
            .map(|f| {
                if f.candidates.is_empty() {
                    f64::NAN
                } else {
                    let freq = f.candidates[0].frequency;
                    if freq > 0.0 && freq < self.pitch_ceiling {
                        freq
                    } else {
                        f64::NAN
                    }
                }
            })
            .collect();

        let hz = interpolation.interpolate_with_undefined(&pitch_values, position.max(0.0))?;
        Some(unit.from_hertz(hz))
    }

    /// Get the pitch value at a specific frame
    pub fn get_value_at_frame(&self, frame: usize) -> Option<f64> {
        self.frames.get(frame).and_then(|f| {
            if f.candidates.is_empty() {
                None
            } else {
                let freq = f.candidates[0].frequency;
                if freq > 0.0 && freq < self.pitch_ceiling {
                    Some(freq)
                } else {
                    None
                }
            }
        })
    }

    /// Get the strength (autocorrelation) at a specific frame
    pub fn get_strength_at_frame(&self, frame: usize) -> Option<f64> {
        self.frames.get(frame).and_then(|f| {
            if f.candidates.is_empty() {
                None
            } else {
                Some(f.candidates[0].strength)
            }
        })
    }

    /// Check if a frame is voiced
    pub fn is_voiced(&self, frame: usize) -> bool {
        self.get_value_at_frame(frame).is_some()
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
            .filter_map(|f| self.frame_frequency(f))
            .reduce(f64::min)
    }

    /// Get maximum pitch value (over voiced frames)
    pub fn max(&self) -> Option<f64> {
        self.frames
            .iter()
            .filter_map(|f| self.frame_frequency(f))
            .reduce(f64::max)
    }

    /// Get mean pitch value (over voiced frames)
    pub fn mean(&self) -> Option<f64> {
        let voiced: Vec<f64> = self
            .frames
            .iter()
            .filter_map(|f| self.frame_frequency(f))
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
            .filter(|f| self.frame_frequency(f).is_some())
            .count()
    }

    fn frame_frequency(&self, frame: &PitchFrame) -> Option<f64> {
        if frame.candidates.is_empty() {
            return None;
        }
        let freq = frame.candidates[0].frequency;
        if freq > 0.0 && freq < self.pitch_ceiling {
            Some(freq)
        } else {
            None
        }
    }

    /// Get the pitch floor used for analysis
    pub fn pitch_floor(&self) -> f64 {
        self.pitch_floor
    }

    /// Get the pitch ceiling used for analysis
    pub fn pitch_ceiling(&self) -> f64 {
        self.pitch_ceiling
    }

    /// Get a reference to the frames
    pub fn frames(&self) -> &[PitchFrame] {
        &self.frames
    }
}

/// Compute pitch candidates for a single frame (matches Sound_into_PitchFrame)
#[allow(clippy::too_many_arguments)]
fn compute_pitch_frame(
    samples: &[f64],
    dx: f64,
    x1: f64,
    time: f64,
    pitch_floor: f64,
    _pitch_ceiling: f64,
    max_candidates: usize,
    voicing_threshold: f64,
    octave_cost: f64,
    nsamp_window: usize,
    halfnsamp_window: usize,
    nsamp_period: usize,
    halfnsamp_period: usize,
    maximum_lag: usize,
    brent_ixmax: usize,
    global_peak: f64,
    window: &[f64],
    window_r: &[f64],
    fft: &mut Fft,
    nsamp_fft: usize,
) -> PitchFrame {
    let nx = samples.len();

    // Sample indices (matching Praat's Sampled_xToLowIndex)
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean (looking one longest period to both sides)
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize) as usize).min(nx);
    let local_mean: f64 = if end_mean > start_mean {
        samples[start_mean..end_mean].iter().sum::<f64>() / (end_mean - start_mean) as f64
    } else {
        0.0
    };

    // Copy window to frame and subtract local mean
    let start_sample = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let end_sample = ((left_sample + halfnsamp_window as isize) as usize).min(nx);

    let mut frame_data = vec![0.0; nsamp_fft];
    for (j, i) in (start_sample..end_sample).enumerate() {
        if j < nsamp_window && j < window.len() {
            frame_data[j] = (samples[i] - local_mean) * window[j];
        }
    }

    // Compute local peak (looking half a period to both sides)
    let peak_start = halfnsamp_window.saturating_sub(halfnsamp_period);
    let peak_end = (halfnsamp_window + halfnsamp_period).min(nsamp_window);
    let mut local_peak = 0.0;
    for j in peak_start..peak_end {
        let value = frame_data[j].abs();
        if value > local_peak {
            local_peak = value;
        }
    }

    let intensity = if local_peak > global_peak {
        1.0
    } else {
        local_peak / global_peak
    };

    // Initialize candidates with voiceless option
    let mut candidates = vec![PitchCandidate {
        frequency: 0.0,
        strength: 0.0,
    }];

    // If silence, return early
    if local_peak == 0.0 {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // Compute autocorrelation via FFT
    let ac = fft.autocorrelation(&frame_data);

    // Normalize autocorrelation
    let mut r = vec![0.0; 2 * nsamp_window + 1];
    let r_offset = nsamp_window; // r[r_offset + i] = r[i], r[r_offset - i] = r[-i]
    r[r_offset] = 1.0;

    if ac[0] > 0.0 {
        for i in 1..=brent_ixmax.min(ac.len() - 1).min(nsamp_window) {
            // Praat: r[i] = ac[i+1] / (ac[1] * windowR[i+1])
            // In our 0-based indexing: r[i] = ac[i] / (ac[0] * window_r[i])
            if i < window_r.len() && window_r[i].abs() > 1e-10 {
                let normalized = ac[i] / (ac[0] * window_r[i]);
                r[r_offset + i] = normalized;
                r[r_offset - i] = normalized;
            }
        }
    }

    // Find maxima in the autocorrelation
    let mut imax = vec![0usize; max_candidates + 1];
    imax[0] = 0;

    for i in 2..maximum_lag.min(brent_ixmax) {
        let ri = r[r_offset + i];
        let ri_prev = r[r_offset + i - 1];
        let ri_next = r[r_offset + i + 1];

        // Not too unvoiced? And is it a maximum?
        if ri > 0.5 * voicing_threshold && ri > ri_prev && ri >= ri_next {
            // Parabolic interpolation for first estimate
            let dr = 0.5 * (ri_next - ri_prev);
            let d2r = ri - ri_prev + ri - ri_next;

            if d2r > 0.0 {
                let frequency_of_maximum = 1.0 / dx / (i as f64 + dr / d2r);

                // Sinc interpolation for strength
                let strength_of_maximum = sinc_interpolate(&r, r_offset, 1.0 / dx / frequency_of_maximum, 30);
                let strength_of_maximum = if strength_of_maximum > 1.0 {
                    1.0 / strength_of_maximum
                } else {
                    strength_of_maximum
                };

                // Find a place for this candidate
                let mut place = 0;
                if candidates.len() < max_candidates {
                    candidates.push(PitchCandidate {
                        frequency: 0.0,
                        strength: 0.0,
                    });
                    place = candidates.len() - 1;
                } else {
                    // Find weakest candidate
                    let mut weakest = 2.0;
                    for iweak in 1..candidates.len() {
                        let local_strength = candidates[iweak].strength
                            - octave_cost * (pitch_floor / candidates[iweak].frequency).log2();
                        if local_strength < weakest {
                            weakest = local_strength;
                            place = iweak;
                        }
                    }
                    // Check if new candidate is stronger
                    if strength_of_maximum - octave_cost * (pitch_floor / frequency_of_maximum).log2()
                        <= weakest
                    {
                        place = 0;
                    }
                }

                if place > 0 {
                    candidates[place].frequency = frequency_of_maximum;
                    candidates[place].strength = strength_of_maximum;
                    imax[place] = i;
                }
            }
        }
    }

    // Second pass: refine with sinc interpolation (simplified version)
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let lag = 1.0 / dx / candidates[i].frequency;
            // Use parabolic + sinc refinement
            let (refined_lag, refined_strength) =
                improve_maximum(&r, r_offset, lag, brent_ixmax);
            if refined_lag > 0.0 {
                candidates[i].frequency = 1.0 / dx / refined_lag;
                let strength = if refined_strength > 1.0 {
                    1.0 / refined_strength
                } else {
                    refined_strength
                };
                candidates[i].strength = strength;
            }
        }
    }

    PitchFrame {
        candidates,
        intensity,
    }
}

/// Compute pitch candidates for a single frame from multiple channels (AC method)
/// Sums autocorrelation (power spectra) across channels before finding peaks
#[allow(clippy::too_many_arguments)]
fn compute_pitch_frame_multichannel(
    channel_samples: &[&[f64]],
    dx: f64,
    x1: f64,
    time: f64,
    pitch_floor: f64,
    _pitch_ceiling: f64,
    max_candidates: usize,
    voicing_threshold: f64,
    octave_cost: f64,
    nsamp_window: usize,
    halfnsamp_window: usize,
    nsamp_period: usize,
    halfnsamp_period: usize,
    maximum_lag: usize,
    brent_ixmax: usize,
    global_peak: f64,
    window: &[f64],
    window_r: &[f64],
    fft: &mut Fft,
    nsamp_fft: usize,
) -> PitchFrame {
    let nx = channel_samples[0].len();

    // Sample indices
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean for each channel
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize) as usize).min(nx);
    let local_means: Vec<f64> = channel_samples
        .iter()
        .map(|samples| {
            if end_mean > start_mean {
                samples[start_mean..end_mean].iter().sum::<f64>() / (end_mean - start_mean) as f64
            } else {
                0.0
            }
        })
        .collect();

    // Copy window to frame and subtract local mean for each channel
    let start_sample = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let end_sample = ((left_sample + halfnsamp_window as isize) as usize).min(nx);

    let mut frame_data_per_channel: Vec<Vec<f64>> = Vec::with_capacity(channel_samples.len());
    for (ch, samples) in channel_samples.iter().enumerate() {
        let mut frame_data = vec![0.0; nsamp_fft];
        for (j, i) in (start_sample..end_sample).enumerate() {
            if j < nsamp_window && j < window.len() {
                frame_data[j] = (samples[i] - local_means[ch]) * window[j];
            }
        }
        frame_data_per_channel.push(frame_data);
    }

    // Compute local peak across all channels
    let peak_start = halfnsamp_window.saturating_sub(halfnsamp_period);
    let peak_end = (halfnsamp_window + halfnsamp_period).min(nsamp_window);
    let mut local_peak = 0.0;
    for frame_data in &frame_data_per_channel {
        for j in peak_start..peak_end {
            let value = frame_data[j].abs();
            if value > local_peak {
                local_peak = value;
            }
        }
    }

    let intensity = if local_peak > global_peak {
        1.0
    } else {
        local_peak / global_peak
    };

    // Initialize candidates with voiceless option
    let mut candidates = vec![PitchCandidate {
        frequency: 0.0,
        strength: 0.0,
    }];

    if local_peak == 0.0 {
        return PitchFrame { candidates, intensity };
    }

    // Compute autocorrelation via FFT, summing power across channels
    let frame_refs: Vec<&[f64]> = frame_data_per_channel.iter().map(|v| v.as_slice()).collect();
    let ac = fft.autocorrelation_multichannel(&frame_refs);

    // Normalize autocorrelation
    let mut r = vec![0.0; 2 * nsamp_window + 1];
    let r_offset = nsamp_window;
    r[r_offset] = 1.0;

    if ac[0] > 0.0 {
        for i in 1..=brent_ixmax.min(ac.len() - 1).min(nsamp_window) {
            if i < window_r.len() && window_r[i].abs() > 1e-10 {
                let normalized = ac[i] / (ac[0] * window_r[i]);
                r[r_offset + i] = normalized;
                r[r_offset - i] = normalized;
            }
        }
    }

    // Find maxima in the autocorrelation (same as single-channel)
    let mut imax = vec![0usize; max_candidates + 1];
    imax[0] = 0;

    for i in 2..maximum_lag.min(brent_ixmax) {
        let ri = r[r_offset + i];
        let ri_prev = r[r_offset + i - 1];
        let ri_next = r[r_offset + i + 1];

        if ri > 0.5 * voicing_threshold && ri > ri_prev && ri >= ri_next {
            let dr = 0.5 * (ri_next - ri_prev);
            let d2r = ri - ri_prev + ri - ri_next;

            if d2r > 0.0 {
                let frequency_of_maximum = 1.0 / dx / (i as f64 + dr / d2r);
                let strength_of_maximum = sinc_interpolate(&r, r_offset, 1.0 / dx / frequency_of_maximum, 30);
                let strength_of_maximum = if strength_of_maximum > 1.0 {
                    1.0 / strength_of_maximum
                } else {
                    strength_of_maximum
                };

                let mut place = 0;
                if candidates.len() < max_candidates {
                    candidates.push(PitchCandidate {
                        frequency: 0.0,
                        strength: 0.0,
                    });
                    place = candidates.len() - 1;
                } else {
                    let mut weakest = 2.0;
                    for iweak in 1..candidates.len() {
                        let local_strength = candidates[iweak].strength
                            - octave_cost * (pitch_floor / candidates[iweak].frequency).log2();
                        if local_strength < weakest {
                            weakest = local_strength;
                            place = iweak;
                        }
                    }
                    if strength_of_maximum - octave_cost * (pitch_floor / frequency_of_maximum).log2()
                        <= weakest
                    {
                        place = 0;
                    }
                }

                if place > 0 {
                    candidates[place].frequency = frequency_of_maximum;
                    candidates[place].strength = strength_of_maximum;
                    imax[place] = i;
                }
            }
        }
    }

    // Second pass: refine with sinc interpolation
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let lag = 1.0 / dx / candidates[i].frequency;
            let (refined_lag, refined_strength) = improve_maximum(&r, r_offset, lag, brent_ixmax);
            if refined_lag > 0.0 {
                candidates[i].frequency = 1.0 / dx / refined_lag;
                let strength = if refined_strength > 1.0 {
                    1.0 / refined_strength
                } else {
                    refined_strength
                };
                candidates[i].strength = strength;
            }
        }
    }

    PitchFrame { candidates, intensity }
}

/// Compute pitch candidates for a single frame using FCC with multiple channels
#[allow(clippy::too_many_arguments)]
fn compute_pitch_frame_fcc_multichannel(
    channel_samples: &[&[f64]],
    dx: f64,
    x1: f64,
    nx: usize,
    time: f64,
    pitch_floor: f64,
    _pitch_ceiling: f64,
    max_candidates: usize,
    voicing_threshold: f64,
    octave_cost: f64,
    nsamp_window: usize,
    halfnsamp_window: usize,
    _halfnsamp_period: usize,
    maximum_lag: usize,
    brent_ixmax: usize,
    global_peak: f64,
) -> PitchFrame {
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean for each channel
    let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize) as usize).min(nx);
    let local_means: Vec<f64> = channel_samples
        .iter()
        .map(|samples| {
            if end_mean > start_mean {
                samples[start_mean..end_mean].iter().sum::<f64>() / (end_mean - start_mean) as f64
            } else {
                0.0
            }
        })
        .collect();

    // FCC start position
    let dt_window = nsamp_window as f64 * dx;
    let start_time = time - 0.5 * (1.0 / pitch_floor + dt_window);
    let start_sample_fcc = ((start_time - x1) / dx).floor() as isize;
    let start_sample_fcc = start_sample_fcc.max(0) as usize;

    let local_span = maximum_lag + nsamp_window;
    let local_span = local_span.min(nx.saturating_sub(start_sample_fcc));
    let local_maximum_lag = local_span.saturating_sub(nsamp_window);

    // Compute local peak across all channels
    let peak_start = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let peak_end = ((left_sample + halfnsamp_window as isize) as usize).min(nx);
    let mut local_peak = 0.0;
    for (ch, samples) in channel_samples.iter().enumerate() {
        for i in peak_start..peak_end {
            let value = (samples[i] - local_means[ch]).abs();
            if value > local_peak {
                local_peak = value;
            }
        }
    }

    let intensity = if local_peak > global_peak { 1.0 } else { local_peak / global_peak };

    let mut candidates = vec![PitchCandidate { frequency: 0.0, strength: 0.0 }];

    if local_peak == 0.0 || local_maximum_lag < 2 {
        return PitchFrame { candidates, intensity };
    }

    let offset = start_sample_fcc;
    if offset + nsamp_window > nx || offset + local_maximum_lag + nsamp_window > nx {
        return PitchFrame { candidates, intensity };
    }

    // Compute sumx2 and initial sumy2 across all channels
    let mut sumx2: f64 = 0.0;
    for (ch, samples) in channel_samples.iter().enumerate() {
        for i in 0..nsamp_window {
            let x = samples[offset + i] - local_means[ch];
            sumx2 += x * x;
        }
    }

    if sumx2 == 0.0 {
        return PitchFrame { candidates, intensity };
    }

    let mut sumy2 = sumx2;

    // Allocate correlation array
    let mut r = vec![0.0; 2 * local_maximum_lag + 1];
    let r_offset = local_maximum_lag;
    r[r_offset] = 1.0;

    // Compute cross-correlation for each lag, summing across channels
    for lag in 1..=local_maximum_lag {
        // Update sumy2 incrementally for each channel
        for (ch, samples) in channel_samples.iter().enumerate() {
            let y0 = samples[offset + lag - 1] - local_means[ch];
            let y_new_idx = offset + lag + nsamp_window - 1;
            if y_new_idx < nx {
                let y_new = samples[y_new_idx] - local_means[ch];
                sumy2 += y_new * y_new - y0 * y0;
            }
        }

        // Compute cross-correlation product across all channels
        let mut product: f64 = 0.0;
        for (ch, samples) in channel_samples.iter().enumerate() {
            for j in 0..nsamp_window {
                let x = samples[offset + j] - local_means[ch];
                let y = samples[offset + lag + j] - local_means[ch];
                product += x * y;
            }
        }

        let norm = (sumx2 * sumy2).sqrt();
        if norm > 0.0 {
            let corr = product / norm;
            r[r_offset + lag] = corr;
            r[r_offset - lag] = corr;
        }
    }

    // Find maxima (same as single-channel)
    for i in 2..local_maximum_lag.min(brent_ixmax) {
        let ri = r[r_offset + i];
        let ri_prev = r[r_offset + i - 1];
        let ri_next = if r_offset + i + 1 < r.len() { r[r_offset + i + 1] } else { 0.0 };

        if ri > 0.5 * voicing_threshold && ri > ri_prev && ri >= ri_next {
            let dr = 0.5 * (ri_next - ri_prev);
            let d2r = ri - ri_prev + ri - ri_next;

            if d2r > 0.0 {
                let frequency_of_maximum = 1.0 / dx / (i as f64 + dr / d2r);
                let strength_of_maximum = sinc_interpolate(&r, r_offset, 1.0 / dx / frequency_of_maximum, 30);
                let strength_of_maximum = if strength_of_maximum > 1.0 {
                    1.0 / strength_of_maximum
                } else {
                    strength_of_maximum
                };

                let mut place = 0;
                if candidates.len() < max_candidates {
                    candidates.push(PitchCandidate { frequency: 0.0, strength: 0.0 });
                    place = candidates.len() - 1;
                } else {
                    let mut weakest = 2.0;
                    for iweak in 1..candidates.len() {
                        let local_strength = candidates[iweak].strength
                            - octave_cost * (pitch_floor / candidates[iweak].frequency).log2();
                        if local_strength < weakest {
                            weakest = local_strength;
                            place = iweak;
                        }
                    }
                    if strength_of_maximum - octave_cost * (pitch_floor / frequency_of_maximum).log2() <= weakest {
                        place = 0;
                    }
                }

                if place > 0 {
                    candidates[place].frequency = frequency_of_maximum;
                    candidates[place].strength = strength_of_maximum;
                }
            }
        }
    }

    // Second pass: refine
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let lag = 1.0 / dx / candidates[i].frequency;
            let (refined_lag, refined_strength) = improve_maximum(&r, r_offset, lag, brent_ixmax);
            if refined_lag > 0.0 {
                candidates[i].frequency = 1.0 / dx / refined_lag;
                candidates[i].strength = if refined_strength > 1.0 {
                    1.0 / refined_strength
                } else {
                    refined_strength
                };
            }
        }
    }

    PitchFrame { candidates, intensity }
}

/// Compute pitch candidates for a single frame using FCC (Forward Cross-Correlation)
/// This matches Praat's FCC_ACCURATE method used by Harmonicity CC
#[allow(clippy::too_many_arguments)]
fn compute_pitch_frame_fcc(
    samples: &[f64],
    dx: f64,
    x1: f64,
    nx: usize,
    time: f64,
    pitch_floor: f64,
    _pitch_ceiling: f64,
    max_candidates: usize,
    voicing_threshold: f64,
    octave_cost: f64,
    nsamp_window: usize,
    halfnsamp_window: usize,
    _halfnsamp_period: usize,
    maximum_lag: usize,
    brent_ixmax: usize,
    global_peak: f64,
) -> PitchFrame {
    // Sample indices (matching Praat's Sampled_xToLowIndex)
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean (looking one longest period to both sides)
    let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize) as usize).min(nx);
    let local_mean: f64 = if end_mean > start_mean {
        samples[start_mean..end_mean].iter().sum::<f64>() / (end_mean - start_mean) as f64
    } else {
        0.0
    };

    // For FCC, we need to determine the start position for cross-correlation
    // startTime = t - 0.5 * (1.0 / pitchFloor + dt_window)
    let dt_window = nsamp_window as f64 * dx;
    let start_time = time - 0.5 * (1.0 / pitch_floor + dt_window);
    let start_sample_fcc = ((start_time - x1) / dx).floor() as isize;
    let start_sample_fcc = start_sample_fcc.max(0) as usize;

    // Local span for FCC
    let local_span = maximum_lag + nsamp_window;
    let local_span = local_span.min(nx.saturating_sub(start_sample_fcc));
    let local_maximum_lag = local_span.saturating_sub(nsamp_window);

    // Compute local peak for intensity (looking at the window around the frame center)
    let peak_start = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let peak_end = ((left_sample + halfnsamp_window as isize) as usize).min(nx);
    let mut local_peak = 0.0;
    for i in peak_start..peak_end {
        let value = (samples[i] - local_mean).abs();
        if value > local_peak {
            local_peak = value;
        }
    }

    let intensity = if local_peak > global_peak {
        1.0
    } else {
        local_peak / global_peak
    };

    // Initialize candidates with voiceless option
    let mut candidates = vec![PitchCandidate {
        frequency: 0.0,
        strength: 0.0,
    }];

    // If silence, return early
    if local_peak == 0.0 || local_maximum_lag < 2 {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // Compute FCC (Forward Cross-Correlation) in time domain
    // r[i] = sum(x[j] * y[i+j]) / sqrt(sumx² * sumy²)
    let offset = start_sample_fcc;

    // Check bounds
    if offset + nsamp_window > nx || offset + local_maximum_lag + nsamp_window > nx {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // Compute sum of squares for the first window (x)
    let mut sumx2: f64 = 0.0;
    for i in 0..nsamp_window {
        let x = samples[offset + i] - local_mean;
        sumx2 += x * x;
    }

    if sumx2 == 0.0 {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // Initialize sumy2 (at zero lag, equals sumx2)
    let mut sumy2 = sumx2;

    // Allocate correlation array (symmetric around zero)
    let mut r = vec![0.0; 2 * local_maximum_lag + 1];
    let r_offset = local_maximum_lag;
    r[r_offset] = 1.0; // r[0] = 1.0

    // Compute cross-correlation for each lag
    for lag in 1..=local_maximum_lag {
        // Update sumy2 incrementally
        // sumy2 += y[lag + nsamp_window - 1]² - y[lag - 1]²
        let y0 = samples[offset + lag - 1] - local_mean;
        let y_new_idx = offset + lag + nsamp_window - 1;
        if y_new_idx < nx {
            let y_new = samples[y_new_idx] - local_mean;
            sumy2 += y_new * y_new - y0 * y0;
        }

        // Compute cross-correlation product
        let mut product: f64 = 0.0;
        for j in 0..nsamp_window {
            let x = samples[offset + j] - local_mean;
            let y = samples[offset + lag + j] - local_mean;
            product += x * y;
        }

        // Normalize
        let norm = (sumx2 * sumy2).sqrt();
        if norm > 0.0 {
            let corr = product / norm;
            r[r_offset + lag] = corr;
            r[r_offset - lag] = corr; // Symmetric
        }
    }

    // Find maxima in the correlation (same as AC method)
    for i in 2..local_maximum_lag.min(brent_ixmax) {
        let ri = r[r_offset + i];
        let ri_prev = r[r_offset + i - 1];
        let ri_next = if r_offset + i + 1 < r.len() {
            r[r_offset + i + 1]
        } else {
            0.0
        };

        // Not too unvoiced? And is it a maximum?
        if ri > 0.5 * voicing_threshold && ri > ri_prev && ri >= ri_next {
            // Parabolic interpolation for first estimate
            let dr = 0.5 * (ri_next - ri_prev);
            let d2r = ri - ri_prev + ri - ri_next;

            if d2r > 0.0 {
                let frequency_of_maximum = 1.0 / dx / (i as f64 + dr / d2r);

                // Sinc interpolation for strength
                let strength_of_maximum =
                    sinc_interpolate(&r, r_offset, 1.0 / dx / frequency_of_maximum, 30);
                let strength_of_maximum = if strength_of_maximum > 1.0 {
                    1.0 / strength_of_maximum
                } else {
                    strength_of_maximum
                };

                // Find a place for this candidate
                let mut place = 0;
                if candidates.len() < max_candidates {
                    candidates.push(PitchCandidate {
                        frequency: 0.0,
                        strength: 0.0,
                    });
                    place = candidates.len() - 1;
                } else {
                    // Find weakest candidate
                    let mut weakest = 2.0;
                    for iweak in 1..candidates.len() {
                        let local_strength = candidates[iweak].strength
                            - octave_cost * (pitch_floor / candidates[iweak].frequency).log2();
                        if local_strength < weakest {
                            weakest = local_strength;
                            place = iweak;
                        }
                    }
                    // Check if new candidate is stronger
                    if strength_of_maximum
                        - octave_cost * (pitch_floor / frequency_of_maximum).log2()
                        <= weakest
                    {
                        place = 0;
                    }
                }

                if place > 0 {
                    candidates[place].frequency = frequency_of_maximum;
                    candidates[place].strength = strength_of_maximum;
                }
            }
        }
    }

    // Refine candidates with parabolic interpolation
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let lag = 1.0 / dx / candidates[i].frequency;
            let (refined_lag, refined_strength) =
                improve_maximum(&r, r_offset, lag, brent_ixmax);
            if refined_lag > 0.0 {
                candidates[i].frequency = 1.0 / dx / refined_lag;
                let strength = if refined_strength > 1.0 {
                    1.0 / refined_strength
                } else {
                    refined_strength
                };
                candidates[i].strength = strength;
            }
        }
    }

    PitchFrame {
        candidates,
        intensity,
    }
}

/// Sinc interpolation with raised cosine window (matches Praat's NUM_interpolate_sinc)
///
/// This interpolates between samples using a windowed sinc function, which is the
/// theoretically optimal interpolation for band-limited signals.
///
/// Arguments:
/// - `r`: the array of samples (with offset applied to indexing)
/// - `offset`: offset to add to indices when accessing `r`
/// - `x`: the (fractional) position to interpolate at
/// - `max_depth`: maximum number of samples to use on each side of x
fn sinc_interpolate(r: &[f64], offset: usize, x: f64, max_depth: i32) -> f64 {
    let midleft = x.floor() as isize;
    let midright = midleft + 1;

    // Convert to 1-based indexing like Praat (r[1..n] in Praat corresponds to r[0..n-1] here)
    // The offset is the center of the symmetric autocorrelation array
    let n = r.len();

    // Check bounds (in Praat's 1-based terms: 1 <= x <= y.size)
    // Since we have offset, actual positions in r are: offset - max_depth to offset + max_depth
    let idx_midleft = offset as isize + midleft;
    let idx_midright = offset as isize + midright;

    if idx_midleft < 0 || idx_midright >= n as isize {
        // Out of bounds - return nearest valid value
        if idx_midleft < 0 {
            return r.get(0).copied().unwrap_or(0.0);
        } else {
            return r.get(n - 1).copied().unwrap_or(0.0);
        }
    }

    // If x is exactly an integer, return that sample
    if x == midleft as f64 {
        return r[idx_midleft as usize];
    }

    // Determine actual depth based on available samples
    let mut max_depth = max_depth as isize;
    // Limit by samples available on left (midright - 1 from position 0)
    max_depth = max_depth.min(idx_midright);
    // Limit by samples available on right (n - midleft - 1)
    max_depth = max_depth.min(n as isize - 1 - idx_midleft);

    if max_depth <= 0 {
        // Nearest neighbor
        let nearest = x.round() as isize;
        let idx_nearest = (offset as isize + nearest) as usize;
        return r.get(idx_nearest).copied().unwrap_or(0.0);
    }
    if max_depth == 1 {
        // Linear interpolation
        let frac = x - midleft as f64;
        return r[idx_midleft as usize] * (1.0 - frac) + r[idx_midright as usize] * frac;
    }

    // Full sinc interpolation with raised cosine window
    let left = idx_midright - max_depth;  // leftmost sample index
    let right = idx_midleft + max_depth;  // rightmost sample index

    let mut result = 0.0;
    let depth_float = max_depth as f64 + 0.5;  // Praat uses maxDepth + 0.5

    // Left half: from midleft down to left
    {
        let window_phase_step = PI / depth_float;
        let sin_window_phase_step = window_phase_step.sin();
        let cos_window_phase_step = window_phase_step.cos();

        // Initialize sinc phase
        let left_phase_initial = PI * (x - midleft as f64);
        let mut left_phase = left_phase_initial;
        let mut half_sin_left_phase = 0.5 * left_phase_initial.sin();

        // Initialize window phase
        let window_phase_initial = left_phase_initial / depth_float;
        let mut sin_window_phase = window_phase_initial.sin();
        let mut cos_window_phase = window_phase_initial.cos();

        let mut ix = idx_midleft;
        while ix >= left {
            // Compute sinc * window contribution
            let sinc_times_window = if left_phase.abs() < 1e-10 {
                // Avoid division by zero at phase = 0
                0.5 * (1.0 + cos_window_phase)
            } else {
                half_sin_left_phase / left_phase * (1.0 + cos_window_phase)
            };
            result += r[ix as usize] * sinc_times_window;

            // Update sinc phase (adding pi flips the sign of sin)
            left_phase += PI;
            half_sin_left_phase = -half_sin_left_phase;

            // Update window phase using trig identities
            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;

            ix -= 1;
        }
    }

    // Right half: from midright up to right
    {
        let window_phase_step = PI / depth_float;
        let sin_window_phase_step = window_phase_step.sin();
        let cos_window_phase_step = window_phase_step.cos();

        // Initialize sinc phase
        let right_phase_initial = PI * (midright as f64 - x);
        let mut right_phase = right_phase_initial;
        let mut half_sin_right_phase = 0.5 * right_phase_initial.sin();

        // Initialize window phase
        let window_phase_initial = right_phase_initial / depth_float;
        let mut sin_window_phase = window_phase_initial.sin();
        let mut cos_window_phase = window_phase_initial.cos();

        let mut ix = idx_midright;
        while ix <= right {
            // Compute sinc * window contribution
            let sinc_times_window = if right_phase.abs() < 1e-10 {
                0.5 * (1.0 + cos_window_phase)
            } else {
                half_sin_right_phase / right_phase * (1.0 + cos_window_phase)
            };
            result += r[ix as usize] * sinc_times_window;

            // Update sinc phase
            right_phase += PI;
            half_sin_right_phase = -half_sin_right_phase;

            // Update window phase
            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;

            ix += 1;
        }
    }

    result
}

/// Improve maximum using parabolic interpolation
fn improve_maximum(r: &[f64], offset: usize, x: f64, _brent_ixmax: usize) -> (f64, f64) {
    let ix = x.round() as isize;
    let idx = (offset as isize + ix) as usize;

    if idx < 1 || idx >= r.len() - 1 {
        return (x, r.get(idx).copied().unwrap_or(0.0));
    }

    let y_prev = r[idx - 1];
    let y_mid = r[idx];
    let y_next = r[idx + 1];

    // Parabolic interpolation
    let dy = 0.5 * (y_next - y_prev);
    let d2y = 2.0 * y_mid - y_prev - y_next;

    if d2y > 0.0 {
        let x_refined = ix as f64 + dy / d2y;
        let y_refined = y_mid + 0.5 * dy * dy / d2y;
        (x_refined, y_refined)
    } else {
        (x, y_mid)
    }
}

/// Path finder using Viterbi algorithm (matches Pitch_pathFinder)
fn pitch_path_finder(
    frames: &mut [PitchFrame],
    silence_threshold: f64,
    voicing_threshold: f64,
    octave_cost: f64,
    octave_jump_cost: f64,
    voiced_unvoiced_cost: f64,
    ceiling: f64,
    dt: f64,
) {
    if frames.is_empty() {
        return;
    }

    let num_frames = frames.len();

    // Time step correction (Praat: 0.01 / my dx)
    let time_step_correction = 0.01 / dt;
    let octave_jump_cost = octave_jump_cost * time_step_correction;
    let voiced_unvoiced_cost = voiced_unvoiced_cost * time_step_correction;

    // Find max candidates
    let max_candidates = frames.iter().map(|f| f.candidates.len()).max().unwrap_or(1);

    // Initialize delta (local scores)
    let mut delta: Vec<Vec<f64>> = Vec::with_capacity(num_frames);
    let mut psi: Vec<Vec<usize>> = Vec::with_capacity(num_frames);

    for _frame in frames.iter() {
        delta.push(vec![f64::NEG_INFINITY; max_candidates]);
        psi.push(vec![0; max_candidates]);
    }

    // Compute initial local scores (matching Praat)
    for (iframe, frame) in frames.iter().enumerate() {
        let unvoiced_strength = if silence_threshold <= 0.0 {
            0.0
        } else {
            let intensity_factor = frame.intensity / (silence_threshold / (1.0 + voicing_threshold));
            voicing_threshold + (2.0 - intensity_factor).max(0.0)
        };

        for (icand, cand) in frame.candidates.iter().enumerate() {
            let voiceless = cand.frequency <= 0.0 || cand.frequency >= ceiling;
            delta[iframe][icand] = if voiceless {
                unvoiced_strength
            } else {
                cand.strength - octave_cost * (ceiling / cand.frequency).log2()
            };
        }
    }

    // Forward pass
    for iframe in 1..num_frames {
        let prev_frame = &frames[iframe - 1];
        let cur_frame = &frames[iframe];

        for icand2 in 0..cur_frame.candidates.len() {
            let f2 = cur_frame.candidates[icand2].frequency;
            let current_voiceless = f2 <= 0.0 || f2 >= ceiling;

            let mut maximum = f64::NEG_INFINITY;
            let mut place = 0;

            for icand1 in 0..prev_frame.candidates.len() {
                let f1 = prev_frame.candidates[icand1].frequency;
                let previous_voiceless = f1 <= 0.0 || f1 >= ceiling;

                let transition_cost = if current_voiceless {
                    if previous_voiceless {
                        0.0 // both voiceless
                    } else {
                        voiced_unvoiced_cost // voiced-to-unvoiced
                    }
                } else if previous_voiceless {
                    voiced_unvoiced_cost // unvoiced-to-voiced
                } else {
                    // both voiced: octave jump cost
                    octave_jump_cost * (f1 / f2).log2().abs()
                };

                let value = delta[iframe - 1][icand1] - transition_cost + delta[iframe][icand2];
                if value > maximum {
                    maximum = value;
                    place = icand1;
                }
            }

            delta[iframe][icand2] = maximum;
            psi[iframe][icand2] = place;
        }
    }

    // Find best final state
    let mut place = 0;
    let mut maximum = delta[num_frames - 1].get(0).copied().unwrap_or(f64::NEG_INFINITY);
    for (icand, &score) in delta[num_frames - 1].iter().enumerate() {
        if score > maximum {
            maximum = score;
            place = icand;
        }
    }

    // Backtrack and swap candidates so winner is at index 0
    for iframe in (0..num_frames).rev() {
        // Swap candidate at 'place' with candidate at 0
        if place != 0 && place < frames[iframe].candidates.len() {
            frames[iframe].candidates.swap(0, place);
        }
        place = psi[iframe][place];
    }
}

// Add to_pitch method to Sound
impl Sound {
    /// Compute pitch contour from this sound
    pub fn to_pitch(&self, time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch {
        Pitch::from_sound(self, time_step, pitch_floor, pitch_ceiling)
    }
}

/// Compute pitch from multiple channel sounds with Praat-compatible autocorrelation summing
///
/// This function matches Praat's behavior for multi-channel audio files.
/// Praat computes the autocorrelation for each channel separately, then sums them
/// (not averages) before finding pitch candidates.
///
/// For stereo: `AC(ch1) + AC(ch2)` (correct for Praat compatibility)
/// vs: `AC((ch1+ch2)/2)` (what you get with sample averaging)
///
/// # Arguments
/// * `sounds` - Slice of Sound objects (one per channel)
/// * `time_step` - Time between analysis frames (0.0 for automatic)
/// * `pitch_floor` - Minimum expected pitch (Hz)
/// * `pitch_ceiling` - Maximum expected pitch (Hz)
///
/// # Example
/// ```ignore
/// use praatfan_core::{Sound, pitch_from_channels};
///
/// // Load stereo file keeping channels separate
/// let channels = Sound::from_file_channels("stereo.wav").unwrap();
///
/// // Compute pitch with proper stereo handling
/// let pitch = pitch_from_channels(&channels, 0.0, 75.0, 600.0);
/// ```
pub fn pitch_from_channels(
    sounds: &[Sound],
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
) -> Pitch {
    Pitch::from_channels_with_method(
        sounds,
        time_step,
        pitch_floor,
        pitch_ceiling,
        MAX_CANDIDATES,
        0.03,  // silenceThreshold
        0.45,  // voicingThreshold
        0.01,  // octaveCost
        0.35,  // octaveJumpCost
        0.14,  // voicedUnvoicedCost
        3.0,   // periods_per_window
        PitchMethod::AcHanning,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_pure_tone() {
        let freq = 200.0;
        let sound = Sound::create_tone(freq, 0.5, 44100.0, 0.5, 0.0);
        let pitch = sound.to_pitch(0.0, 75.0, 600.0);

        assert!(pitch.num_frames() > 0);

        let mut voiced_count = 0;
        let mut sum = 0.0;

        for i in 0..pitch.num_frames() {
            if let Some(f0) = pitch.get_value_at_frame(i) {
                voiced_count += 1;
                sum += f0;
            }
        }

        assert!(voiced_count > pitch.num_frames() / 2);

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

        for i in 0..pitch.num_frames() {
            assert!(!pitch.is_voiced(i), "Frame {} should be unvoiced", i);
        }
    }
}
