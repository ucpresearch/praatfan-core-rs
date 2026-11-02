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

use crate::interpolation::{sampled_window_samples, vector_extremum, vector_value_at_x, Interpolation, PeakInterpolation};
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
    /// Time domain (Praat's xmin/xmax: the analysed sound's domain)
    xmin: f64,
    xmax: f64,
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
    /// * `periods_per_window` - Number of periods per analysis window.
    ///   **Use 4.5 for the AC method**, which is Praat's own default here; its
    ///   `Sound: To Harmonicity (ac)...` command refuses anything below 3.0
    ///   ("The number of periods per window must be at least 3.0"). The check
    ///   lives in Praat's command layer, not in `Sound_to_Harmonicity_ac()`,
    ///   which is why parselmouth -- calling the C function directly --
    ///   defaults this to 1.0 and accepts it. 1.0 is the correct default for
    ///   the **cc** method, not this one. Agreement with parselmouth does not
    ///   depend on it: on `one_two_three_four_five.wav` (voiced frames) the
    ///   max abs diff is 6e-11 dB at 4.5, 6e-9 dB at 3.0 and 1e-6 dB at 1.0
    ///   (before 0.1.15: 0.014 / 5.11 / 11.05 dB). See `DIVERGENCES.md` SS4.
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
    /// **Note:** This uses the FCC_ACCURATE method (forward cross-correlation)
    /// which computes correlation in the time domain without windowing.
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz), determines max lag for correlation
    /// * `silence_threshold` - Threshold for silence detection (typically 0.1)
    /// * `periods_per_window` - Number of periods per analysis window
    ///   (1.0 -- Praat's own default for the cc method, and unlike the ac
    ///   method Praat imposes no lower bound here)
    pub fn from_sound_cc(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        // Method 3 = FCC_ACCURATE (Forward Cross-Correlation)
        Self::from_pitch_method(
            sound,
            PitchMethod::FccAccurate,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
        )
    }

    /// Like [`from_sound_ac`](Self::from_sound_ac), but the amplitude
    /// reference for the silence gate is decoupled from the whole-file peak.
    ///
    /// - `Some(x)`: use `x` as the amplitude reference (must be finite, > 0).
    /// - `None`: estimate a speech reference internally
    ///   ([`crate::estimate_speech_reference`] with default parameters),
    ///   falling back to the legacy whole-file peak on an all-zero signal.
    pub fn from_sound_ac_referenced(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
        reference_peak: Option<f64>,
    ) -> crate::Result<Self> {
        Self::from_pitch_method_referenced(
            sound,
            PitchMethod::AcGauss,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
            reference_peak,
        )
    }

    /// Like [`from_sound_cc`](Self::from_sound_cc), but with a decoupled
    /// amplitude reference. See
    /// [`from_sound_ac_referenced`](Self::from_sound_ac_referenced).
    pub fn from_sound_cc_referenced(
        sound: &Sound,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
        reference_peak: Option<f64>,
    ) -> crate::Result<Self> {
        Self::from_pitch_method_referenced(
            sound,
            PitchMethod::FccAccurate,
            time_step,
            min_pitch,
            silence_threshold,
            periods_per_window,
            reference_peak,
        )
    }

    /// Internal: referenced-variant counterpart of `from_pitch_method`.
    fn from_pitch_method_referenced(
        sound: &Sound,
        method: PitchMethod,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
        reference_peak: Option<f64>,
    ) -> crate::Result<Self> {
        let pitch_ceiling = 0.5 / sound.dx();

        let pitch = Pitch::from_sound_with_method_referenced(
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
            reference_peak,
        )?;

        let mut values = Vec::with_capacity(pitch.num_frames());
        for frame in pitch.frames() {
            let hnr = Self::strength_to_hnr(frame);
            values.push(hnr);
        }

        Ok(Self {
            values,
            start_time: pitch.start_time(),
            time_step: pitch.time_step(),
            min_pitch,
            xmin: pitch.xmin(),
            xmax: pitch.xmax(),
        })
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
            xmin: pitch.xmin(),
            xmax: pitch.xmax(),
        }
    }

    /// Compute harmonicity from multiple Sound channels
    ///
    /// This matches Praat's behavior for multi-channel audio: each channel's
    /// autocorrelation is computed separately, then summed (not averaged).
    ///
    /// # Arguments
    /// * `sounds` - Slice of Sound objects (one per channel)
    /// * `method` - AC or CC method
    /// * `time_step` - Time between analysis frames (0.0 for automatic)
    /// * `min_pitch` - Minimum expected pitch (Hz)
    /// * `silence_threshold` - Threshold for silence detection
    /// * `periods_per_window` - Number of periods per analysis window
    pub fn from_channels(
        sounds: &[Sound],
        method: PitchMethod,
        time_step: f64,
        min_pitch: f64,
        silence_threshold: f64,
        periods_per_window: f64,
    ) -> Self {
        if sounds.is_empty() {
            return Self {
                values: Vec::new(),
                start_time: 0.0,
                time_step: if time_step > 0.0 { time_step } else { 0.01 },
                min_pitch,
                xmin: 0.0,
                xmax: 0.0,
            };
        }

        // For single channel, use the standard method
        if sounds.len() == 1 {
            return Self::from_pitch_method(
                &sounds[0],
                method,
                time_step,
                min_pitch,
                silence_threshold,
                periods_per_window,
            );
        }

        // Multi-channel: use pitch from channels
        let pitch_ceiling = 0.5 / sounds[0].dx();
        let pitch = Pitch::from_channels_with_method(
            sounds,
            time_step,
            min_pitch,
            pitch_ceiling,
            15,                 // maxnCandidates
            silence_threshold,  // silenceThreshold
            0.0,                // voicingThreshold
            0.0,                // octaveCost
            0.0,                // octaveJumpCost
            0.0,                // voicedUnvoicedCost
            periods_per_window,
            method,
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
            xmin: pitch.xmin(),
            xmax: pitch.xmax(),
        }
    }

    /// Convert pitch frame strength to HNR in dB
    ///
    /// Reproduces `Sound_to_Harmonicity.cpp` exactly, including both of
    /// Praat's boundary markers:
    ///
    /// * unvoiced (first candidate at 0 Hz) -> `-200.0`
    /// * `r <= 1e-15` -> `-150.0`
    /// * `r > 1.0 - 1e-15` -> `+150.0`
    /// * otherwise `10 * log10(r / (1 - r))`
    ///
    /// Both markers are effectively unreachable on speech, because the pitch
    /// frame routine reflects over-1 correlations around 1 (`r -> 1/r`) and
    /// bounds the lag search by `brent_ixmax`. Measured: 0 frames at or above
    /// 150 dB on the standard fixture at `periods_per_window` 1.0 / 3.0 / 4.5.
    ///
    /// The sibling clean-room package (`praatfan`, MIT) reports `NaN` and
    /// `-200.0` at these two boundaries instead. That choice is deliberate
    /// there and deliberately not adopted here -- see `DIVERGENCES.md` SS1-3.
    /// Do not "fix" this to match it without reading that file first.
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
                10.0 * libm::log10(r / (1.0 - r))
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
            xmin: pitch.xmin(),
            xmax: pitch.xmax(),
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
    ///
    /// Praat's `Harmonicity: Get value at time...`: a plain
    /// `Vector_getValueAtX`, so -200 (unvoiced) frames are interpolated in
    /// like any other value (e.g. -35.6 dB a quarter of the way from a
    /// 12.4 dB frame to a -200 dB one). Use `get_value_at_frame` (None for
    /// -200) to tell unvoiced frames apart.
    pub fn get_value_at_time(&self, time: f64, interpolation: Interpolation) -> Option<f64> {
        vector_value_at_x(&self.values, self.start_time, self.time_step, time, interpolation)
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

    /// Start of the time domain (Praat's xmin)
    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    /// End of the time domain (Praat's xmax)
    pub fn xmax(&self) -> f64 {
        self.xmax
    }

    /// `Harmonicity_getSoundingValues`: the frames in `[from_time, to_time]`
    /// (equal or reversed: everything) whose value is not -200.
    fn sounding_values(&self, mut from_time: f64, mut to_time: f64) -> Vec<f64> {
        if from_time >= to_time {
            from_time = self.xmin;
            to_time = self.xmax;
        }
        let (imin, imax) =
            sampled_window_samples(self.values.len(), self.start_time, self.time_step, from_time, to_time);
        if imin > imax {
            return Vec::new();
        }
        self.values[(imin - 1) as usize..imax as usize]
            .iter()
            .copied()
            .filter(|&v| v != -200.0)
            .collect()
    }

    /// Praat's `Harmonicity: Get mean...`: mean of the sounding frames
    /// (-200 frames excluded) in `[from_time, to_time]`.
    pub fn get_mean(&self, from_time: f64, to_time: f64) -> Option<f64> {
        let v = self.sounding_values(from_time, to_time);
        if v.is_empty() {
            None
        } else {
            Some(crate::utils::praat_mean(&v))
        }
    }

    /// Praat's `Harmonicity: Get standard deviation...` (sounding frames,
    /// n - 1 denominator).
    pub fn get_standard_deviation(&self, from_time: f64, to_time: f64) -> Option<f64> {
        let v = self.sounding_values(from_time, to_time);
        // Praat's NUMstdev: sumsq / (n - 1). With no sounding frames that is
        // 0 / -1 = -0.0 (Praat reports -0); with one it is 0 / 0 (undefined).
        if v.is_empty() {
            return Some(-0.0);
        }
        if v.len() == 1 {
            return None;
        }
        let mean = crate::utils::praat_mean(&v);
        let mut sumsq = crate::utils::CompSum::default();
        for &x in &v {
            sumsq.add((x - mean) * (x - mean));
        }
        Some(libm::sqrt(sumsq.value() / (v.len() - 1) as f64))
    }

    fn extremum(&self, from_time: f64, to_time: f64, interpolation: PeakInterpolation, is_max: bool) -> (f64, f64) {
        vector_extremum(
            &self.values,
            self.start_time,
            self.time_step,
            self.xmin,
            self.xmax,
            from_time,
            to_time,
            interpolation,
            is_max,
        )
    }

    /// Praat's `Harmonicity: Get minimum...` (`Vector_getMinimum`; as in
    /// Praat, -200 frames take part).
    pub fn get_minimum(&self, from_time: f64, to_time: f64, interpolation: PeakInterpolation) -> Option<f64> {
        Some(self.extremum(from_time, to_time, interpolation, false).0).filter(|v| !v.is_nan())
    }

    /// Praat's `Harmonicity: Get maximum...` (`Vector_getMaximum`).
    pub fn get_maximum(&self, from_time: f64, to_time: f64, interpolation: PeakInterpolation) -> Option<f64> {
        Some(self.extremum(from_time, to_time, interpolation, true).0).filter(|v| !v.is_nan())
    }

    /// Praat's `Harmonicity: Get time of minimum...`.
    pub fn get_time_of_minimum(&self, from_time: f64, to_time: f64, interpolation: PeakInterpolation) -> Option<f64> {
        Some(self.extremum(from_time, to_time, interpolation, false).1).filter(|v| !v.is_nan())
    }

    /// Praat's `Harmonicity: Get time of maximum...`.
    pub fn get_time_of_maximum(&self, from_time: f64, to_time: f64, interpolation: PeakInterpolation) -> Option<f64> {
        Some(self.extremum(from_time, to_time, interpolation, true).1).filter(|v| !v.is_nan())
    }

    /// Minimum over the whole domain, parabolic (Praat's defaults).
    pub fn min(&self) -> Option<f64> {
        self.get_minimum(0.0, 0.0, PeakInterpolation::Parabolic)
    }

    /// Maximum over the whole domain, parabolic (Praat's defaults).
    pub fn max(&self) -> Option<f64> {
        self.get_maximum(0.0, 0.0, PeakInterpolation::Parabolic)
    }

    /// Mean of the sounding frames over the whole domain (Praat's `Get mean`).
    pub fn mean(&self) -> Option<f64> {
        self.get_mean(0.0, 0.0)
    }

    /// Standard deviation of the sounding frames over the whole domain.
    pub fn standard_deviation(&self) -> Option<f64> {
        self.get_standard_deviation(0.0, 0.0)
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

/// Compute harmonicity from multiple channel sounds using AC method
///
/// This function matches Praat's behavior for multi-channel audio files.
///
/// # Arguments
/// * `sounds` - Slice of Sound objects (one per channel)
/// * `time_step` - Time between frames (0.0 for automatic)
/// * `min_pitch` - Minimum expected pitch (Hz)
/// * `silence_threshold` - Threshold for silence detection (typically 0.1)
/// * `periods_per_window` - Periods per window (typically 3.0 for AC)
pub fn harmonicity_from_channels_ac(
    sounds: &[Sound],
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
) -> Harmonicity {
    Harmonicity::from_channels(
        sounds,
        PitchMethod::AcGauss,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
    )
}

/// Compute harmonicity from multiple channel sounds using CC method
///
/// This function matches Praat's behavior for multi-channel audio files.
///
/// # Arguments
/// * `sounds` - Slice of Sound objects (one per channel)
/// * `time_step` - Time between frames (0.0 for automatic)
/// * `min_pitch` - Minimum expected pitch (Hz)
/// * `silence_threshold` - Threshold for silence detection (typically 0.1)
/// * `periods_per_window` - Periods per window (typically 1.0 for CC)
pub fn harmonicity_from_channels_cc(
    sounds: &[Sound],
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,
    periods_per_window: f64,
) -> Harmonicity {
    Harmonicity::from_channels(
        sounds,
        PitchMethod::FccAccurate,
        time_step,
        min_pitch,
        silence_threshold,
        periods_per_window,
    )
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
            let tone = 0.3 * libm::sin(2.0 * PI * 200.0 * t);

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
