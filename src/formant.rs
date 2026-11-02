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

use crate::interpolation::{sampled_extremum, sampled_mean_interpolated, sampled_quantile, sampled_value_at_x, sampled_window_samples, Interpolation};
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
    /// Intensity of the frame (sum of squared samples before windowing)
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

    /// Number of real formants in the frame (Praat's `numberOfFormants`).
    /// Frames are padded with NaN entries up to the analysis maximum, so this
    /// counts the leading defined ones.
    pub fn num_defined_formants(&self) -> usize {
        self.formants.iter().take_while(|p| !p.frequency.is_nan()).count()
    }

    /// Get the intensity of this frame (sum of squared samples before windowing)
    pub fn intensity(&self) -> f64 {
        self.intensity
    }

    /// Get the formants in this frame as a slice
    pub fn formants(&self) -> &[FormantPoint] {
        &self.formants
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
    /// Time domain (Praat's xmin/xmax: the analysed sound's)
    xmin: f64,
    xmax: f64,
}

impl Formant {
    /// Compute formants from a Sound using Burg's LPC method
    ///
    /// # Arguments
    /// * `sound` - Input audio signal
    /// * `time_step` - Time between analysis frames (0.0 for automatic: 0.25 × window_length)
    /// * `max_num_formants` - Maximum number of formants to track (typically 5)
    /// * `max_formant_hz` - Maximum formant frequency (Hz). Use ~5000 for an adult male, ~5500 for an adult female voice
    /// * `window_length` - Analysis window duration in seconds (typically 0.025)
    /// * `pre_emphasis_from` - Pre-emphasis frequency (Hz), typically 50
    ///
    /// # Algorithm
    /// For each frame:
    /// 1. Apply pre-emphasis filter
    /// 2. Extract frame with Gaussian window
    /// 3. Compute LPC coefficients (order = 2 × max_num_formants)
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
        Self::from_sound_burg_impl(
            sound,
            time_step,
            2 * max_num_formants,
            max_formant_hz,
            window_length,
            pre_emphasis_from,
            None,
            false,
        )
    }

    /// Like [`Formant::from_sound_burg`], but takes Praat's real-valued
    /// "Max. number of formants" (e.g. 5.5). As in Praat, the LPC order is
    /// `round(2 * max_num_formants)` and a frame holds up to
    /// `(order + 1) / 2` formants.
    pub fn from_sound_burg_f(
        sound: &Sound,
        time_step: f64,
        max_num_formants: f64,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Self {
        let poles = libm::round(2.0 * max_num_formants).max(1.0) as usize;
        Self::from_sound_burg_impl(
            sound,
            time_step,
            poles,
            max_formant_hz,
            window_length,
            pre_emphasis_from,
            None,
            false,
        )
    }

    /// Like [`Formant::from_sound_burg`] but forces the analysis frame grid (first-frame
    /// time + number-of-frames) to the given values AND subtracts the frame
    /// mean before windowing (matches Praat's `Sound_into_LPC` code path used
    /// by `Sound_to_FormantPath_any`). Do NOT use this for standalone
    /// formant analysis — it differs from `Sound_to_Formant_any` by exactly
    /// one `Vector_subtractMean` call (Praat `Sound_and_LPC.cpp:471`).
    pub fn from_sound_burg_with_grid(
        sound: &Sound,
        time_step: f64,
        max_num_formants: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
        forced_t1: f64,
        forced_num_frames: usize,
    ) -> Self {
        Self::from_sound_burg_impl(
            sound,
            time_step,
            2 * max_num_formants,
            max_formant_hz,
            window_length,
            pre_emphasis_from,
            Some((forced_t1, forced_num_frames)),
            true,
        )
    }

    fn from_sound_burg_impl(
        sound: &Sound,
        time_step: f64,
        number_of_poles: usize,
        max_formant_hz: f64,
        window_length: f64,
        pre_emphasis_from: f64,
        forced_grid: Option<(f64, usize)>,
        subtract_mean_per_frame: bool,
    ) -> Self {
        // No parameter clamping: Praat (Sound_to_Formant.cpp) uses the values
        // as given. Up to (poles + 1) / 2 formants per frame (Formant_create).
        let number_of_poles = number_of_poles.max(1);
        let max_num_formants = (number_of_poles + 1) / 2;

        // Time step: default is 0.25 × window_length (Praat convention)
        let time_step = if time_step <= 0.0 {
            window_length / 4.0
        } else {
            time_step
        };

        // Praat resamples to 2 × max_formant before LPC analysis
        // This is critical for accurate formant estimation
        // Sound_to_Formant_any: resample to 2 x ceiling, upwards too (an 8 kHz
        // file with a 5500 Hz ceiling is analysed at 11 kHz), unless the
        // ceiling is <= 0 or already the Nyquist frequency.
        let nyquist_in = 0.5 * sound.sample_rate();
        let (resampled, max_formant_hz) =
            if max_formant_hz <= 0.0 || (max_formant_hz / nyquist_in - 1.0).abs() < 1e-12 {
                (sound.clone(), nyquist_in)
            } else {
                (sound.resample(2.0 * max_formant_hz), max_formant_hz)
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
        let x1_sound = emphasized.x1();

        // LPC order: 2 × max_formants (Praat convention for Sound -> Formant)
        // Each formant needs 2 poles (complex conjugate pair)
        let lpc_order = number_of_poles;

        // CRITICAL: window_length parameter is actually halfdt_window in Praat!
        // The actual analysis window is TWICE this value.
        // See Sound_to_Formant.cpp: dt_window = 2.0 * halfdt_window
        let halfdt_window = window_length;
        let mut dt_window = 2.0 * halfdt_window;
        let mut nsamp_window = (dt_window / dx).floor() as usize;
        // Praat computes halfnsamp_window here, *before* the one-frame fallback
        // below resets nsamp_window, and never updates it.
        let halfnsamp_window = nsamp_window / 2;

        // Frame timing - exactly as Praat does it (verified with source code)
        // physicalDuration = nx * dx
        let physical_duration = samples.len() as f64 * dx;
        let (first_frame_time, num_frames) = if let Some((t1, nf)) = forced_grid {
            (t1, nf)
        } else {
            // nFrames = 1 + floor((physicalDuration - dt_window) / dt), which
            // Praat computes in signed arithmetic; if it is < 1 Praat analyses
            // the whole sound as one frame.
            let nf = 1 + libm::floor((physical_duration - dt_window) / time_step) as i64;
            if nf < 1 {
                dt_window = physical_duration;
                nsamp_window = samples.len();
                (x1_sound + 0.5 * physical_duration, 1)
            } else {
                // t1 = x1 + 0.5 * (physicalDuration - dx - (numberOfFrames - 1) * dt)
                let nf = nf as usize;
                let t1 = x1_sound + 0.5 * (physical_duration - dx - (nf - 1) as f64 * time_step);
                (t1, nf)
            }
        };
        let _ = dt_window;

        // Generate Praat's exact Gaussian window for formant analysis.
        //
        // FormantPath candidates go through `Sound_into_LPC`
        // (LPC/Sound_and_LPC.cpp), which sizes everything from the window
        // *duration*: windowDuration = min(2 * analysisWidth, nx * dx); the
        // frame and the Gaussian (Sound_createSimple / Sound_createGaussian)
        // have round(windowDuration * fs) samples; the frame starts at
        // Sampled_xToNearestIndex(t - windowDuration / 2).
        let lpc_window_duration = (2.0 * window_length).min(dx * samples.len() as f64);
        let frame_window_len = if subtract_mean_per_frame {
            libm::round(lpc_window_duration * sample_rate) as usize
        } else {
            nsamp_window
        };
        let window = praat_formant_window(frame_window_len);

        if num_frames == 0 || samples.is_empty() {
            return Self {
                frames: Vec::new(),
                start_time: sound.start_time(),
                time_step,
                max_num_formants,
                max_formant_hz,
                xmin: sound.start_time(),
                xmax: sound.end_time(),
            };
        }

        // Every frame is independent: map over them in parallel (order kept).
        let frames = crate::par::map_init(num_frames, || (), |_, frame_idx| {
            let frame_time = first_frame_time + frame_idx as f64 * time_step;

            // Sample extraction semantics differ between Praat's two code paths:
            //
            // * `Sound_to_Formant.cpp` (standalone To Formant (burg)) uses
            //   `Sampled_xToLowIndex` = `floor((t-x1)/dx) + 1` (1-based), giving
            //   startSample = leftSample + 1 - halfnsamp_window (our default).
            //
            // * `Sound_into_LPC` (used by FormantPath) uses `Sound_into_Sound`
            //   which calls `Sampled_xToNearestIndex` = `round((t-x1)/dx) + 1`,
            //   yielding a start offset that can differ by 1 sample when the
            //   real-valued sample position has fractional part < 0.5.
            //
            // `subtract_mean_per_frame` doubles as the FormantPath flag, so we
            // switch sample-extraction convention accordingly.
            let (start_sample_raw, end_sample_raw) = if subtract_mean_per_frame {
                // Praat `Sound_into_Sound (sound, sframe, t - 0.5 * windowDuration)`:
                // index = Melder_iround ((x - x1) / dx + 1.0)  (1-based, ties up)
                let x = frame_time - 0.5 * lpc_window_duration;
                let index_1based = libm::floor((x - x1_sound) / dx + 1.0 + 0.5) as isize;
                let start0 = index_1based - 1;
                (start0, start0 + frame_window_len as isize - 1)
            } else {
                let left_sample = ((frame_time - x1_sound) / dx).floor() as isize;
                let right_sample = left_sample + 1;
                (
                    right_sample - halfnsamp_window as isize,
                    left_sample + halfnsamp_window as isize,
                )
            };

            // Raw frame extraction: for the FormantPath path
            // (`subtract_mean_per_frame = true`), Praat's `Sound_into_Sound`
            // ZERO-PADS out-of-bounds samples to produce a fixed-length
            // `nsamp_window` buffer. For standalone, we simply clamp to
            // available samples (matching `Sound_to_Formant_any`).
            let raw_frame: Vec<f64> = if subtract_mean_per_frame {
                let mut out = Vec::with_capacity(frame_window_len);
                for i in 0..frame_window_len {
                    let j = start_sample_raw + i as isize;
                    let v = if j < 0 || (j as usize) >= samples.len() {
                        0.0
                    } else {
                        samples[j as usize]
                    };
                    out.push(v);
                }
                out
            } else {
                let start_sample = start_sample_raw.max(0) as usize;
                let end_sample = (end_sample_raw as usize).min(samples.len().saturating_sub(1));
                if end_sample >= start_sample {
                    samples[start_sample..=end_sample].to_vec()
                } else {
                    Vec::new()
                }
            };
            let actual_frame_length = raw_frame.len();

            // Check maximum intensity on raw (un-windowed) pre-emphasized samples.
            // Praat skips Burg analysis if all samples are zero (Sound_to_Formant.cpp line 342).
            let mut maximum_intensity = 0.0_f64;
            for &value in raw_frame.iter() {
                let intensity = value * value;
                if intensity > maximum_intensity {
                    maximum_intensity = intensity;
                }
            }

            // Extract windowed frame - optionally subtract frame mean, then
            // apply Gaussian window. Praat's `Sound_into_LPC` (used by
            // FormantPath's candidate LPCs) subtracts the mean of the full
            // `nsamp_window`-sample frame (including any zero-padding) before
            // windowing, via `Vector_subtractMean`
            // (`LPC/Sound_and_LPC.cpp:471`). The standalone
            // `Sound_to_Formant_any` does NOT subtract the mean at all.
            let mean = if subtract_mean_per_frame && !raw_frame.is_empty() {
                raw_frame.iter().sum::<f64>() / raw_frame.len() as f64
            } else {
                0.0
            };

            let mut windowed = Vec::with_capacity(actual_frame_length);
            for i in 0..actual_frame_length {
                let w = if i < window.len() { window[i] } else { 0.0 };
                let s = (raw_frame[i] - mean) * w;
                windowed.push(s);
            }

            // Skip Burg analysis if all samples are zero (Burg cannot stand all zeroes).
            // Matches Praat's Sound_to_Formant.cpp line 342-343.
            // `energy` is filled below with the LPC residual (gain) so that it
            // matches Praat's `Formant_Frame.intensity` semantics (used by
            // FormantPath's intensity modulation).
            let mut energy = 0.0_f64;
            let formant_points = if maximum_intensity == 0.0 {
                Vec::new()
            } else if let Some(lpc_result) = lpc_burg(&windowed, lpc_order) {
                energy = lpc_result.gain;
                // Find formants from LPC polynomial roots
                let candidates = lpc_to_formants(&lpc_result.coefficients, sample_rate);

                // Praat (burg() in Sound_to_Formant.cpp and
                // Roots_into_Formant_Frame): keep roots with
                // 50 <= f <= nyquist - 50 (safety margin 50 Hz); no
                // bandwidth test.
                let nyquist = 0.5 * sample_rate;
                let mut valid_formants: Vec<FormantCandidate> = candidates
                    .into_iter()
                    .filter(|f| {
                        f.frequency >= 50.0
                            && f.frequency <= nyquist - 50.0
                            && !f.bandwidth.is_nan()
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

            FormantFrame::new(formants, energy)
        });

        Self {
            frames,
            start_time: first_frame_time,
            time_step,
            max_num_formants,
            max_formant_hz,
            xmin: sound.start_time(),
            xmax: sound.end_time(),
        }
    }

    /// Compute Burg-LPC formants for each ceiling in `max_formants_hz`.
    ///
    /// Equivalent to calling [`Formant::from_sound_burg`] once per ceiling with
    /// all other parameters fixed. Results are returned in the same order as
    /// the input slice. With the `parallel` feature, ceilings (and the frames
    /// within each) are processed concurrently; otherwise serially.
    pub fn from_sound_burg_multi(
        sound: &Sound,
        time_step: f64,
        max_num_formants: usize,
        max_formants_hz: &[f64],
        window_length: f64,
        pre_emphasis_from: f64,
    ) -> Vec<Self> {
        crate::par::map(max_formants_hz, |&hz| {
            Self::from_sound_burg(
                sound,
                time_step,
                max_num_formants,
                hz,
                window_length,
                pre_emphasis_from,
            )
        })
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

        // Praat: Formant_getValueAtTime -> Sampled_getValueAtX (always
        // linear; "nearest" here selects the nearest frame), with each frame
        // converted to `unit` before interpolating.
        sampled_value_at_x(
            self.frames.len(),
            self.start_time,
            self.time_step,
            time,
            interpolation != Interpolation::Nearest,
            |i| {
                self.frames[i]
                    .get_formant(formant_number)
                    .map(|p| unit.from_hertz(p.frequency))
            },
        )
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

        // Praat: Formant_getBandwidthAtTime -> Sampled_getValueAtX. In a
        // non-Hertz unit the bandwidth is the width of [f - b/2, f + b/2]
        // on that scale, the lower edge floored at 0 Hz
        // (structFormant::v_getValueAtSample).
        sampled_value_at_x(
            self.frames.len(),
            self.start_time,
            self.time_step,
            time,
            interpolation != Interpolation::Nearest,
            |i| {
                self.frames[i].get_formant(formant_number).map(|p| {
                    if unit == FrequencyUnit::Hertz {
                        p.bandwidth
                    } else {
                        let fleft = p.frequency - 0.5 * p.bandwidth;
                        let fleft = if fleft <= 0.0 { 0.0 } else { unit.from_hertz(fleft) };
                        unit.from_hertz(p.frequency + 0.5 * p.bandwidth) - fleft
                    }
                })
            },
        )
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

    /// Start of the time domain (Praat's xmin)
    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    /// End of the time domain (Praat's xmax)
    pub fn xmax(&self) -> f64 {
        self.xmax
    }

    /// Formant `n` of frame `i` (1-based) in `unit`; NaN if absent.
    fn unit_value(&self, n: usize, i: isize, unit: FrequencyUnit) -> f64 {
        self.frames
            .get((i - 1) as usize)
            .and_then(|f| if n <= f.num_defined_formants() { f.get_formant(n) } else { None })
            .map(|p| unit.from_hertz(p.frequency))
            .unwrap_or(f64::NAN)
    }

    /// Bandwidth `n` of frame `i` in `unit` (Praat: in Bark, the Bark width
    /// of [f - b/2, f + b/2], lower edge floored at 0 Hz).
    fn unit_bandwidth(&self, n: usize, i: isize, unit: FrequencyUnit) -> f64 {
        self.frames
            .get((i - 1) as usize)
            .and_then(|f| if n <= f.num_defined_formants() { f.get_formant(n) } else { None })
            .map(|p| {
                if unit == FrequencyUnit::Hertz {
                    p.bandwidth
                } else {
                    let fl = p.frequency - 0.5 * p.bandwidth;
                    let fl = if fl <= 0.0 { 0.0 } else { unit.from_hertz(fl) };
                    unit.from_hertz(p.frequency + 0.5 * p.bandwidth) - fl
                }
            })
            .unwrap_or(f64::NAN)
    }

    fn n_ok(&self, n: usize) -> bool {
        n >= 1 && n <= self.max_num_formants
    }

    /// Praat's `Formant: Get mean...`: mean of the linearly interpolated
    /// track of formant `n` over `[from_time, to_time]` (`from >= to`:
    /// whole domain), in `unit`.
    pub fn get_mean(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit) -> Option<f64> {
        if !self.n_ok(n) {
            return None;
        }
        sampled_mean_interpolated(self.frames.len(), self.start_time, self.time_step, self.xmin, self.xmax,
            from_time, to_time, |i| self.unit_value(n, i, unit))
    }

    fn extremum(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool, is_max: bool) -> Option<(f64, f64)> {
        if !self.n_ok(n) {
            return None;
        }
        let (v, t) = sampled_extremum(self.frames.len(), self.start_time, self.time_step, self.xmin, self.xmax,
            from_time, to_time, interpolate, is_max, |i| self.unit_value(n, i, unit))?;
        Some((v, t))
    }

    /// `Formant_getMinimumAndTime`: a value <= 0 is undefined; the time of
    /// that extremum stays defined, as in Praat.
    fn extremum_value(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool, is_max: bool) -> Option<f64> {
        let (v, _) = self.extremum(n, from_time, to_time, unit, interpolate, is_max)?;
        if v <= 0.0 { None } else { Some(v) }
    }

    /// Praat's `Formant: Get minimum...` (`interpolate`: "Parabolic" vs "None").
    pub fn get_minimum(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool) -> Option<f64> {
        self.extremum_value(n, from_time, to_time, unit, interpolate, false)
    }

    /// Praat's `Formant: Get maximum...`.
    pub fn get_maximum(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool) -> Option<f64> {
        self.extremum_value(n, from_time, to_time, unit, interpolate, true)
    }

    /// Praat's `Formant: Get time of minimum...`.
    pub fn get_time_of_minimum(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool) -> Option<f64> {
        self.extremum(n, from_time, to_time, unit, interpolate, false).map(|x| x.1)
    }

    /// Praat's `Formant: Get time of maximum...`.
    pub fn get_time_of_maximum(&self, n: usize, from_time: f64, to_time: f64, unit: FrequencyUnit, interpolate: bool) -> Option<f64> {
        self.extremum(n, from_time, to_time, unit, interpolate, true).map(|x| x.1)
    }

    /// Praat's `Formant: Get quantile...` (frames with formant `n` in the range).
    pub fn get_quantile(&self, n: usize, from_time: f64, to_time: f64, quantile: f64, unit: FrequencyUnit) -> Option<f64> {
        if !self.n_ok(n) {
            return None;
        }
        sampled_quantile(self.frames.len(), self.start_time, self.time_step, self.xmin, self.xmax,
            from_time, to_time, quantile, |i| self.unit_value(n, i, unit))
    }

    /// Praat's `Formant: Get quantile of bandwidth...`.
    pub fn get_quantile_of_bandwidth(&self, n: usize, from_time: f64, to_time: f64, quantile: f64, unit: FrequencyUnit) -> Option<f64> {
        if !self.n_ok(n) {
            return None;
        }
        sampled_quantile(self.frames.len(), self.start_time, self.time_step, self.xmin, self.xmax,
            from_time, to_time, quantile, |i| self.unit_bandwidth(n, i, unit))
    }

    /// Praat's `Formant: Get standard deviation...` (`Formant_getStandardDeviation`):
    /// spread of the frame values around the interpolated mean, n - 1
    /// denominator.
    pub fn get_standard_deviation(&self, n: usize, mut from_time: f64, mut to_time: f64, unit: FrequencyUnit) -> Option<f64> {
        if !self.n_ok(n) || from_time.is_nan() || to_time.is_nan() {
            return None;
        }
        if from_time >= to_time {
            from_time = self.xmin;
            to_time = self.xmax;
        }
        let (imin, imax) = sampled_window_samples(self.frames.len(), self.start_time, self.time_step, from_time, to_time);
        if imin > imax {
            return None;
        }
        let mean = self.get_mean(n, from_time, to_time, unit).unwrap_or(f64::NAN);
        let mut sum = crate::utils::CompSum::default();
        let mut count = 0usize;
        for i in imin..=imax {
            let f = self.unit_value(n, i, unit);
            if f.is_nan() || f == 0.0 {
                continue;
            }
            count += 1;
            sum.add((f - mean) * (f - mean));
        }
        if count > 1 { Some(libm::sqrt(sum.value() / (count - 1) as f64)) } else { None }
    }

    /// Get the maximum formant frequency setting used for analysis
    pub fn max_formant_hz(&self) -> f64 {
        self.max_formant_hz
    }

    /// Get the analysis frames as a slice
    pub fn frames(&self) -> &[FormantFrame] {
        &self.frames
    }

    /// Access a single frame by 0-based index
    pub fn frame(&self, index: usize) -> Option<&FormantFrame> {
        self.frames.get(index)
    }

    /// Build a new Formant by copying one frame per frame from a source selector.
    /// `source_frames[t]` yields the FormantFrame to use at frame `t`. The grid
    /// (start_time, time_step, max_num_formants, max_formant_hz) is taken from `template`.
    pub fn from_frame_selector<F>(template: &Formant, num_frames: usize, mut source_frames: F) -> Self
    where
        F: FnMut(usize) -> FormantFrame,
    {
        let mut frames = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            frames.push(source_frames(t));
        }
        Self {
            frames,
            start_time: template.start_time,
            time_step: template.time_step,
            max_num_formants: template.max_num_formants,
            max_formant_hz: template.max_formant_hz,
            xmin: template.xmin,
            xmax: template.xmax,
        }
    }
}

// Add to_formant_burg method to Sound
impl Sound {
    /// Compute formant tracks from this sound using Burg's LPC method
    ///
    /// # Arguments
    /// * `time_step` - Time between frames (0.0 for automatic)
    /// * `max_num_formants` - Maximum formants to find (typically 5)
    /// * `max_formant_hz` - Maximum formant frequency (~5000 Hz for an adult male, ~5500 Hz for an adult female)
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
        let _mean_f1 = formant.get_mean(1, 0.0, 0.0, FrequencyUnit::Hertz);
        let _min_f1 = formant.get_minimum(1, 0.0, 0.0, FrequencyUnit::Hertz, true);
        let _max_f1 = formant.get_maximum(1, 0.0, 0.0, FrequencyUnit::Hertz, true);

        // Invalid formant number should return None
        assert!(formant.get_mean(0, 0.0, 0.0, FrequencyUnit::Hertz).is_none());
        assert!(formant.get_mean(10, 0.0, 0.0, FrequencyUnit::Hertz).is_none());
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
            samples[i] = 0.5 * crate::math::sin(2.0 * std::f64::consts::PI * 150.0 * t)
                + 0.3 * crate::math::sin(2.0 * std::f64::consts::PI * 300.0 * t)
                + 0.2 * crate::math::sin(2.0 * std::f64::consts::PI * 450.0 * t)
                + 0.1 * crate::math::sin(2.0 * std::f64::consts::PI * 600.0 * t);
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
