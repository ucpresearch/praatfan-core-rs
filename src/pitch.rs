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
use num_complex::Complex;
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
        // - brent_depth: sinc interpolation depth for Brent peak refinement
        //   (matches Praat's NUM_PEAK_INTERPOLATE_* → NUM_VALUE_INTERPOLATE_*)
        let (periods_per_window, interpolation_depth, brent_depth) = match method {
            PitchMethod::AcHanning => (periods_per_window, 0.5, 70_usize),   // SINC70
            PitchMethod::AcGauss => (periods_per_window * 2.0, 0.25, 700_usize), // SINC700
            PitchMethod::FccAccurate => (periods_per_window, 1.0, 700_usize), // SINC700
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
                brent_depth,
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
        let window_autocorr = fft.autocorrelation_circular(&window_fft);
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

        // Pre-allocate workspace buffers (reused across all frames)
        let mut frame_data = vec![0.0; nsamp_fft];
        let mut fft_buffer = vec![Complex::new(0.0, 0.0); nsamp_fft];
        let mut ac_output = vec![0.0; nsamp_fft];
        let mut r_buf = vec![0.0; 2 * nsamp_window + 1];

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
                brent_depth,
                global_peak,
                &window,
                &window_r_norm,
                &mut fft,
                nsamp_fft,
                &mut frame_data,
                &mut fft_buffer,
                &mut ac_output,
                &mut r_buf,
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
                700, // SINC700 for FCC_ACCURATE
            );
        }

        // Match Praat's parameter validation (use first sound for timing)
        let sound = &sounds[0];
        let pitch_floor = pitch_floor.max(10.0);
        let pitch_ceiling = pitch_ceiling.min(0.5 / sound.dx());

        // Method-specific parameters
        let (periods_per_window, interpolation_depth, brent_depth) = match method {
            PitchMethod::AcHanning => (periods_per_window, 0.5, 70_usize),
            PitchMethod::AcGauss => (periods_per_window * 2.0, 0.25, 700_usize),
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
        let window_autocorr = fft.autocorrelation_circular(&window_fft);
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

        // Pre-allocate workspace buffers (reused across all frames)
        let nchan = sounds.len();
        let mut frame_data_pool: Vec<Vec<f64>> = (0..nchan).map(|_| vec![0.0; nsamp_fft]).collect();
        let mut fft_buffer = vec![Complex::new(0.0, 0.0); nsamp_fft];
        let mut power_buffer = vec![Complex::new(0.0, 0.0); nsamp_fft];
        let mut ac_output = vec![0.0; nsamp_fft];
        let mut r_buf = vec![0.0; 2 * nsamp_window + 1];

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
                brent_depth,
                global_peak,
                &window,
                &window_r_norm,
                &mut fft,
                nsamp_fft,
                &mut frame_data_pool,
                &mut fft_buffer,
                &mut power_buffer,
                &mut ac_output,
                &mut r_buf,
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
        brent_depth: usize,
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
        let nsamp_window_raw = (dt_window / dx).floor() as usize;
        if nsamp_window_raw < 4 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }

        // Truncate to even (matching Praat: halfnsamp_window = raw/2 - 1; nsamp_window = half*2)
        let halfnsamp_window = nsamp_window_raw / 2 - 1;
        if halfnsamp_window < 2 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }
        let nsamp_window = halfnsamp_window * 2;

        // Maximum lag (matching Praat line 335)
        let maximum_lag = ((nsamp_window as f64 / periods_per_window).floor() as usize + 2).min(nsamp_window);
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

        // Collect samples from all channels
        let channel_samples: Vec<&[f64]> = sounds.iter().map(|s| s.samples()).collect();

        // Pre-allocate workspace buffers for FCC multichannel
        let nchan = sounds.len();
        let max_span = maximum_lag + nsamp_window;
        let mut mean_sub_pool: Vec<Vec<f64>> = (0..nchan).map(|_| vec![0.0; max_span]).collect();
        let mut r_buf = vec![0.0; 2 * nsamp_window + 1];

        // FFT buffers for O(n log n) cross-correlation
        // FFT size must be >= nsamp_window + max_span to avoid circular aliasing
        let fcc_fft_size = (nsamp_window + max_span).next_power_of_two();
        let mut fft = Fft::new();
        let mut fft_buffer = vec![Complex::new(0.0, 0.0); fcc_fft_size];
        let mut fft_a_buf = vec![Complex::new(0.0, 0.0); fcc_fft_size];
        let mut power_buffer = vec![Complex::new(0.0, 0.0); fcc_fft_size];
        let mut cum_sq_buf = vec![0.0; max_span + 1];

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
                brent_depth,
                global_peak,
                dt_window,
                &mut mean_sub_pool,
                &mut r_buf,
                &mut fft,
                &mut fft_buffer,
                &mut fft_a_buf,
                &mut power_buffer,
                &mut cum_sq_buf,
                fcc_fft_size,
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
        brent_depth: usize,
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
        let nsamp_window_raw = (dt_window / dx).floor() as usize;
        if nsamp_window_raw < 4 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }

        // Truncate to even (matching Praat: halfnsamp_window = raw/2 - 1; nsamp_window = half*2)
        let halfnsamp_window = nsamp_window_raw / 2 - 1;
        if halfnsamp_window < 2 {
            return Self::empty(xmin, xmax, time_step, pitch_floor, pitch_ceiling);
        }
        let nsamp_window = halfnsamp_window * 2;

        // Maximum lag (matching Praat line 335)
        let maximum_lag = ((nsamp_window as f64 / periods_per_window).floor() as usize + 2).min(nsamp_window);

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

        // Pre-allocate workspace buffers for FCC (reused across all frames)
        let max_span = maximum_lag + nsamp_window;
        let mut mean_sub_buf = vec![0.0; max_span];
        let mut r_buf = vec![0.0; 2 * nsamp_window + 1];

        // FFT buffers for O(n log n) cross-correlation
        // FFT size must be >= nsamp_window + max_span to avoid circular aliasing
        let fcc_fft_size = (nsamp_window + max_span).next_power_of_two();
        let mut fft = Fft::new();
        let mut fft_buffer = vec![Complex::new(0.0, 0.0); fcc_fft_size];
        let mut fft_a_buf = vec![Complex::new(0.0, 0.0); fcc_fft_size];
        let mut cum_sq_buf = vec![0.0; max_span + 1];

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
                brent_depth,
                global_peak,
                dt_window,
                &mut mean_sub_buf,
                &mut r_buf,
                &mut fft,
                &mut fft_buffer,
                &mut fft_a_buf,
                &mut cum_sq_buf,
                fcc_fft_size,
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
    brent_depth: usize,
    global_peak: f64,
    window: &[f64],
    window_r: &[f64],
    fft: &mut Fft,
    _nsamp_fft: usize,
    // Pre-allocated buffers (caller must ensure correct sizes)
    frame_data: &mut [f64],
    fft_buffer: &mut [Complex<f64>],
    ac: &mut [f64],
    r: &mut [f64],
) -> PitchFrame {
    let nx = samples.len();
    let r_offset = nsamp_window;

    // Sample indices (matching Praat's Sampled_xToLowIndex)
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean (looking one longest period to both sides)
    // Praat: startSample = rightSample - nsamp_period (1-based inclusive)
    //        endSample = leftSample + nsamp_period   (1-based inclusive)
    //        divisor = 2 * nsamp_period (always, not actual count)
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize + 1) as usize).min(nx);
    let local_mean: f64 = if end_mean > start_mean {
        samples[start_mean..end_mean].iter().sum::<f64>() / (2 * nsamp_period) as f64
    } else {
        0.0
    };

    // Copy window to frame and subtract local mean
    let start_sample = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let end_sample = ((left_sample + halfnsamp_window as isize) as usize).min(nx);

    // Clear frame_data and fill with windowed samples
    for x in frame_data.iter_mut() {
        *x = 0.0;
    }
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

    // Compute autocorrelation via FFT (in-place, no allocation)
    fft.autocorrelation_circular_into(frame_data, fft_buffer, ac);

    // Normalize autocorrelation into pre-allocated r buffer
    r.fill(0.0);
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

    // Second pass: for extra precision, maximize sinc interpolation (matches Praat)
    // Use imax[i] (discrete lag from first pass) as starting point, not recomputed lag
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            // Adaptive depth: use SINC700 for high frequencies (Praat line 254)
            let depth = if candidates[i].frequency > 0.3 / dx { 700 } else { brent_depth };
            let (refined_lag, refined_strength) =
                improve_maximum(&r, r_offset, imax[i] as f64, brent_ixmax, depth);
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
    brent_depth: usize,
    global_peak: f64,
    window: &[f64],
    window_r: &[f64],
    fft: &mut Fft,
    _nsamp_fft: usize,
    // Pre-allocated buffers
    frame_data_pool: &mut [Vec<f64>],
    fft_buffer: &mut [Complex<f64>],
    power_buffer: &mut [Complex<f64>],
    ac: &mut [f64],
    r: &mut [f64],
) -> PitchFrame {
    let nx = channel_samples[0].len();
    let r_offset = nsamp_window;

    // Sample indices
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean for each channel
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize + 1) as usize).min(nx);
    let local_means: Vec<f64> = channel_samples
        .iter()
        .map(|samples| {
            if end_mean > start_mean {
                samples[start_mean..end_mean].iter().sum::<f64>() / (2 * nsamp_period) as f64
            } else {
                0.0
            }
        })
        .collect();

    // Copy window to frame and subtract local mean for each channel (into pre-allocated buffers)
    let start_sample = (right_sample - halfnsamp_window as isize).max(0) as usize;
    let end_sample = ((left_sample + halfnsamp_window as isize) as usize).min(nx);

    for (ch, samples) in channel_samples.iter().enumerate() {
        let frame_data = &mut frame_data_pool[ch];
        for x in frame_data.iter_mut() {
            *x = 0.0;
        }
        for (j, i) in (start_sample..end_sample).enumerate() {
            if j < nsamp_window && j < window.len() {
                frame_data[j] = (samples[i] - local_means[ch]) * window[j];
            }
        }
    }

    // Compute local peak across all channels
    let peak_start = halfnsamp_window.saturating_sub(halfnsamp_period);
    let peak_end = (halfnsamp_window + halfnsamp_period).min(nsamp_window);
    let mut local_peak = 0.0;
    for frame_data in frame_data_pool.iter() {
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

    // Compute autocorrelation via FFT, summing power across channels (in-place)
    let frame_refs: Vec<&[f64]> = frame_data_pool.iter().map(|v| v.as_slice()).collect();
    fft.autocorrelation_circular_multichannel_into(&frame_refs, fft_buffer, power_buffer, ac);

    // Normalize autocorrelation into pre-allocated r buffer
    r.fill(0.0);
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

    // Second pass: for extra precision, maximize sinc interpolation (matches Praat)
    // Use imax[i] (discrete lag from first pass) as starting point, not recomputed lag
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            // Adaptive depth: use SINC700 for high frequencies (Praat line 254)
            let depth = if candidates[i].frequency > 0.3 / dx { 700 } else { brent_depth };
            let (refined_lag, refined_strength) = improve_maximum(&r, r_offset, imax[i] as f64, brent_ixmax, depth);
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
///
/// Uses FFT-based autocorrelation O(n log n) with cumulative-sum normalization,
/// summing power spectra across channels before inverse FFT.
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
    brent_depth: usize,
    global_peak: f64,
    dt_window: f64,
    // Pre-allocated buffers
    mean_sub_pool: &mut [Vec<f64>],
    r: &mut [f64],
    fft: &mut Fft,
    fft_buffer: &mut [Complex<f64>],
    fft_a: &mut [Complex<f64>],
    power_buffer: &mut [Complex<f64>],
    cum_sq: &mut [f64],
    fft_size: usize,
) -> PitchFrame {
    let r_offset = nsamp_window;
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean for each channel
    let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize + 1) as usize).min(nx);
    let local_means: Vec<f64> = channel_samples
        .iter()
        .map(|samples| {
            if end_mean > start_mean {
                samples[start_mean..end_mean].iter().sum::<f64>() / (2 * nsamp_period) as f64
            } else {
                0.0
            }
        })
        .collect();

    // FCC start position (uses original dt_window, not truncated nsamp_window * dx)
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

    // Pre-subtract mean for each channel into pre-allocated buffers
    let span = local_maximum_lag + nsamp_window;
    for (ch, samples) in channel_samples.iter().enumerate() {
        let ms = &mut mean_sub_pool[ch];
        for i in 0..span {
            ms[i] = samples[offset + i] - local_means[ch];
        }
    }

    // Compute sumx2 across all channels
    let mut sumx2: f64 = 0.0;
    for ms in mean_sub_pool.iter() {
        for i in 0..nsamp_window {
            sumx2 += ms[i] * ms[i];
        }
    }

    if sumx2 == 0.0 {
        return PitchFrame { candidates, intensity };
    }

    // === FFT-based cross-correlation, summing across channels ===
    // For each channel: cross_corr = IFFT(FFT(A_ch) * conj(FFT(B_ch)))
    // where A_ch = first nsamp_window samples, B_ch = full span
    // Sum cross-spectra across channels, then single IFFT.

    // Clear power accumulator (reused for cross-spectrum sum)
    for p in power_buffer.iter_mut().take(fft_size) {
        *p = Complex::new(0.0, 0.0);
    }

    for ms in mean_sub_pool.iter() {
        // FFT(A): first nsamp_window samples
        for i in 0..fft_size {
            fft_a[i] = if i < nsamp_window {
                Complex::new(ms[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.fft_forward_inplace(fft_a, fft_size);

        // FFT(B): full span
        for i in 0..fft_size {
            fft_buffer[i] = if i < span {
                Complex::new(ms[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.fft_forward_inplace(fft_buffer, fft_size);

        // Accumulate cross-spectrum: conj(FFT(A)) * FFT(B) → cross_corr[lag] = Σ_i a[i]*b[i+lag]
        for i in 0..fft_size {
            power_buffer[i] += fft_a[i].conj() * fft_buffer[i];
        }
    }

    // Inverse FFT on accumulated cross-spectrum
    fft.ifft_inplace(power_buffer, fft_size);
    let fft_scale = 1.0 / fft_size as f64;

    // Compute cumulative sum of squares across all channels for energy normalization
    cum_sq[0] = 0.0;
    for i in 0..span {
        let mut sq_sum = 0.0;
        for ms in mean_sub_pool.iter() {
            sq_sum += ms[i] * ms[i];
        }
        cum_sq[i + 1] = cum_sq[i] + sq_sum;
    }

    // Initialize correlation array
    r.fill(0.0);
    r[r_offset] = 1.0;

    // Normalize each lag
    // numerator = Σ_ch Σ_{i=0}^{nsamp_window-1} x_ch[i]*x_ch[i+lag]
    // sumy2 = Σ_ch Σ_{i=lag}^{lag+nsamp_window-1} x_ch[i]²
    for lag in 1..=local_maximum_lag {
        let corr = power_buffer[lag].re * fft_scale;
        let e2 = cum_sq[(lag + nsamp_window).min(span)] - cum_sq[lag];
        let norm = (sumx2 * e2).sqrt();
        if norm > 0.0 {
            let normalized = corr / norm;
            r[r_offset + lag] = normalized;
            r[r_offset - lag] = normalized;
        }
    }

    // Track discrete lag positions for each candidate (imax)
    let mut imax: Vec<usize> = vec![0; max_candidates + 1];

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
                    if place < imax.len() {
                        imax[place] = i;
                    }
                }
            }
        }
    }

    // Second pass: for extra precision, maximize sinc interpolation (matches Praat)
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let ixmid = if i < imax.len() && imax[i] > 0 {
                imax[i] as f64
            } else {
                1.0 / dx / candidates[i].frequency
            };
            // Adaptive depth: use SINC700 for high frequencies (Praat line 254)
            let depth = if candidates[i].frequency > 0.3 / dx { 700 } else { brent_depth };
            let (refined_lag, refined_strength) = improve_maximum(&r, r_offset, ixmid, brent_ixmax, depth);
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
///
/// Uses FFT-based autocorrelation O(n log n) with cumulative-sum normalization
/// instead of O(n²) time-domain cross-correlation.
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
    brent_depth: usize,
    global_peak: f64,
    dt_window: f64,
    // Pre-allocated buffers
    mean_sub: &mut [f64],
    r: &mut [f64],
    fft: &mut Fft,
    fft_buffer: &mut [Complex<f64>],
    fft_a: &mut [Complex<f64>],
    cum_sq: &mut [f64],
    fft_size: usize,
) -> PitchFrame {
    let r_offset = nsamp_window;

    // Sample indices (matching Praat's Sampled_xToLowIndex)
    let left_sample = ((time - x1) / dx).floor() as isize;
    let right_sample = left_sample + 1;

    // Compute local mean (looking one longest period to both sides)
    let nsamp_period = (1.0 / dx / pitch_floor).floor() as usize;
    let start_mean = (right_sample - nsamp_period as isize).max(0) as usize;
    let end_mean = ((left_sample + nsamp_period as isize + 1) as usize).min(nx);
    let local_mean: f64 = if end_mean > start_mean {
        samples[start_mean..end_mean].iter().sum::<f64>() / (2 * nsamp_period) as f64
    } else {
        0.0
    };

    // FCC start position (uses original dt_window, not truncated nsamp_window * dx)
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

    let offset = start_sample_fcc;

    // Check bounds
    if offset + nsamp_window > nx || offset + local_maximum_lag + nsamp_window > nx {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // Pre-subtract local mean into buffer
    let span = local_maximum_lag + nsamp_window;
    for i in 0..span {
        mean_sub[i] = samples[offset + i] - local_mean;
    }

    // Compute sum of squares for the first window (x) — check for silence
    let mut sumx2: f64 = 0.0;
    for i in 0..nsamp_window {
        sumx2 += mean_sub[i] * mean_sub[i];
    }

    if sumx2 == 0.0 {
        return PitchFrame {
            candidates,
            intensity,
        };
    }

    // === FFT-based cross-correlation ===
    // FCC computes: r(lag) = Σ_{i=0}^{nsamp_window-1} x[i]*x[i+lag] / sqrt(sumx2 * sumy2)
    // Numerator via FFT: IFFT(conj(FFT(A)) * FFT(B))[lag] where
    //   A = mean_sub[0:nsamp_window] zero-padded, B = mean_sub[0:span] zero-padded
    // Denominator: sumx2 (constant) * sumy2(lag) via cumulative sum of squares

    // FFT(A): the first nsamp_window samples, zero-padded (using pre-allocated buffer)
    for i in 0..fft_size {
        fft_a[i] = if i < nsamp_window {
            Complex::new(mean_sub[i], 0.0)
        } else {
            Complex::new(0.0, 0.0)
        };
    }
    fft.fft_forward_inplace(fft_a, fft_size);

    // FFT(B): the full span, zero-padded
    for i in 0..fft_size {
        fft_buffer[i] = if i < span {
            Complex::new(mean_sub[i], 0.0)
        } else {
            Complex::new(0.0, 0.0)
        };
    }
    fft.fft_forward_inplace(fft_buffer, fft_size);

    // Cross-spectrum: conj(FFT(A)) * FFT(B) → cross_corr[lag] = Σ_i a[i]*b[i+lag]
    for i in 0..fft_size {
        fft_buffer[i] = fft_a[i].conj() * fft_buffer[i];
    }

    // Inverse FFT
    fft.ifft_inplace(fft_buffer, fft_size);
    let fft_scale = 1.0 / fft_size as f64;

    // Compute cumulative sum of squares for denominator energy normalization
    // cum_sq[k] = Σᵢ₌₀^{k-1} mean_sub[i]²
    cum_sq[0] = 0.0;
    for i in 0..span {
        cum_sq[i + 1] = cum_sq[i] + mean_sub[i] * mean_sub[i];
    }

    // Initialize correlation array
    r.fill(0.0);
    r[r_offset] = 1.0;

    // Normalize each lag
    // numerator = cross_corr[lag] = Σ_{i=0}^{nsamp_window-1} x[i]*x[i+lag]
    // sumy2 = Σ_{i=lag}^{lag+nsamp_window-1} x[i]² = cum_sq[lag+nsamp_window] - cum_sq[lag]
    for lag in 1..=local_maximum_lag {
        let corr = fft_buffer[lag].re * fft_scale;
        let e2 = cum_sq[(lag + nsamp_window).min(span)] - cum_sq[lag];
        let norm = (sumx2 * e2).sqrt();
        if norm > 0.0 {
            let normalized = corr / norm;
            r[r_offset + lag] = normalized;
            r[r_offset - lag] = normalized;
        }
    }

    // Track discrete lag positions for each candidate (imax)
    let mut imax: Vec<usize> = vec![0; max_candidates + 1];

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
                    if place < imax.len() {
                        imax[place] = i;
                    }
                }
            }
        }
    }

    // Refine candidates with sinc interpolation + Brent optimization (matches Praat)
    for i in 1..candidates.len() {
        if candidates[i].frequency > 0.0 {
            let ixmid = if i < imax.len() && imax[i] > 0 {
                imax[i] as f64
            } else {
                1.0 / dx / candidates[i].frequency
            };
            // Adaptive depth: use SINC700 for high frequencies (Praat line 254)
            let depth = if candidates[i].frequency > 0.3 / dx { 700 } else { brent_depth };
            let (refined_lag, refined_strength) =
                improve_maximum(&r, r_offset, ixmid, brent_ixmax, depth);
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

    // Hoist common sin/cos step computation (shared by both halves)
    let window_phase_step = PI / depth_float;
    let sin_window_phase_step = window_phase_step.sin();
    let cos_window_phase_step = window_phase_step.cos();

    // Left half: from midleft down to left
    // Note: left_phase can never be 0 in the loop because the early return at
    // line 2021 (`if x == midleft as f64`) handles that case. The initial phase
    // is PI * (x - midleft) where x != midleft, and subsequent iterations add PI.
    {
        let left_phase_initial = PI * (x - midleft as f64);
        let mut left_phase = left_phase_initial;
        let mut half_sin_left_phase = 0.5 * left_phase_initial.sin();

        let window_phase_initial = left_phase_initial / depth_float;
        let mut sin_window_phase = window_phase_initial.sin();
        let mut cos_window_phase = window_phase_initial.cos();

        let mut ix = idx_midleft;
        while ix >= left {
            let sinc_times_window = half_sin_left_phase / left_phase * (1.0 + cos_window_phase);
            // SAFETY: ix is bounded by left..=idx_midleft, validated at lines 2045-2046
            result += unsafe { *r.get_unchecked(ix as usize) } * sinc_times_window;

            left_phase += PI;
            half_sin_left_phase = -half_sin_left_phase;

            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;

            ix -= 1;
        }
    }

    // Right half: from midright up to right
    {
        let right_phase_initial = PI * (midright as f64 - x);
        let mut right_phase = right_phase_initial;
        let mut half_sin_right_phase = 0.5 * right_phase_initial.sin();

        let window_phase_initial = right_phase_initial / depth_float;
        let mut sin_window_phase = window_phase_initial.sin();
        let mut cos_window_phase = window_phase_initial.cos();

        let mut ix = idx_midright;
        while ix <= right {
            let sinc_times_window = half_sin_right_phase / right_phase * (1.0 + cos_window_phase);
            // SAFETY: ix is bounded by idx_midright..=right, validated at lines 2045-2046
            result += unsafe { *r.get_unchecked(ix as usize) } * sinc_times_window;

            right_phase += PI;
            half_sin_right_phase = -half_sin_right_phase;

            let next_sin = cos_window_phase * sin_window_phase_step + sin_window_phase * cos_window_phase_step;
            let next_cos = cos_window_phase * cos_window_phase_step - sin_window_phase * sin_window_phase_step;
            sin_window_phase = next_sin;
            cos_window_phase = next_cos;

            ix += 1;
        }
    }

    result
}

/// Brent's method for univariate minimization (matches Praat's NUMminimize_brent)
///
/// Finds x in [a, b] that minimizes f(x). Returns (x_min, f_min).
/// Closely modeled after the netlib code by Oleg Keselyov.
fn minimize_brent<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> (f64, f64) {
    const GOLDEN: f64 = 1.0 - 0.6180339887498948482; // 1 - golden ratio
    let sqrt_epsilon = f64::EPSILON.sqrt();
    const ITERMAX: usize = 60;

    let mut a = a;
    let mut b = b;

    // First step - golden section
    let mut v = a + GOLDEN * (b - a);
    let mut fv = f(v);
    let mut x = v;
    let mut w = v;
    let mut fx = fv;
    let mut fw = fv;

    for _ in 0..ITERMAX {
        let middle_range = (a + b) / 2.0;
        let tol_act = sqrt_epsilon * x.abs() + tol / 3.0;
        let range = b - a;
        if (x - middle_range).abs() + range / 2.0 <= 2.0 * tol_act {
            return (x, fx);
        }

        // Obtain the golden section step
        let mut new_step = GOLDEN * if x < middle_range { b - x } else { a - x };

        // Decide if the parabolic interpolation can be tried
        if (x - w).abs() >= tol_act {
            let t = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * t;
            q = 2.0 * (q - t);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            if p.abs() < (new_step * q).abs()
                && p > q * (a - x + 2.0 * tol_act)
                && p < q * (b - x - 2.0 * tol_act)
            {
                new_step = p / q;
            }
        }

        // Adjust the step to be not less than tolerance
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }

        // Obtain the next approximation to min
        let t = x + new_step;
        let ft = f(t);

        if ft <= fx {
            if t < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = t;
            fv = fw;
            fw = fx;
            fx = ft;
        } else {
            if t < x {
                a = t;
            } else {
                b = t;
            }
            if ft <= fw || w == x {
                v = w;
                w = t;
                fv = fw;
                fw = ft;
            } else if ft <= fv || v == x || v == w {
                v = t;
                fv = ft;
            }
        }
    }

    (x, fx)
}

/// Improve maximum using sinc interpolation + Brent optimization
/// (matches Praat's NUMimproveMaximum / NUMimproveExtremum)
///
/// Arguments:
/// - `r`: correlation array with offset
/// - `offset`: center position in r (r[offset + lag] = correlation at lag)
/// - `x`: fractional lag position (will be rounded to find ixmid)
/// - `brent_ixmax`: maximum lag index available in r
/// - `brent_depth`: interpolation depth (70 = SINC70, 700 = SINC700, 1 = parabolic)
fn improve_maximum(r: &[f64], offset: usize, x: f64, _brent_ixmax: usize, brent_depth: usize) -> (f64, f64) {
    let ixmid = x.round() as isize;

    // Boundary checks (matching Praat's NUMimproveExtremum)
    // In our representation, valid range is 0..r.len(), with offset as center
    if ixmid <= -(offset as isize) {
        let idx = 0;
        return (-(offset as f64), r[idx]);
    }
    if ixmid >= (r.len() - offset) as isize {
        let idx = r.len() - 1;
        return ((idx as isize - offset as isize) as f64, r[idx]);
    }

    let idx = (offset as isize + ixmid) as usize;

    // For no interpolation (depth 0) or edge cases
    if brent_depth == 0 {
        return (ixmid as f64, r[idx]);
    }

    // For parabolic interpolation (depth 1)
    if brent_depth <= 1 {
        if idx < 1 || idx >= r.len() - 1 {
            return (ixmid as f64, r[idx]);
        }
        let dy = 0.5 * (r[idx + 1] - r[idx - 1]);
        let d2y = 2.0 * r[idx] - r[idx - 1] - r[idx + 1];
        if d2y > 0.0 {
            let x_refined = ixmid as f64 + dy / d2y;
            let y_refined = r[idx] + 0.5 * dy * dy / d2y;
            return (x_refined, y_refined);
        }
        return (ixmid as f64, r[idx]);
    }

    // Sinc interpolation + Brent optimization
    // We search for the maximum over [ixmid-1, ixmid+1] by minimizing -sinc_interpolate
    let search_lo = (ixmid - 1) as f64;
    let search_hi = (ixmid + 1) as f64;

    // Clamp search range to valid data range
    let data_lo = -(offset as f64);
    let data_hi = (r.len() - 1 - offset) as f64;
    let search_lo = search_lo.max(data_lo);
    let search_hi = search_hi.min(data_hi);
    if search_lo >= search_hi {
        return (ixmid as f64, r[idx]);
    }

    let depth = brent_depth as i32;
    let (xmid, neg_ymid) = minimize_brent(
        &|pos| -sinc_interpolate(r, offset, pos, depth),
        search_lo,
        search_hi,
        1e-10,
    );

    (xmid, -neg_ymid)
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
