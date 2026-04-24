//! Multi-ceiling Burg formant analysis with Viterbi-optimal path selection.
//!
//! Mirrors Praat's `FormantPath` (LPC/FormantPath.cpp). A FormantPath holds
//! one Burg `Formant` analysis per ceiling (`middle * exp(±i * step)` for
//! `i = 1..N`) plus an integer path indicating which candidate to use at each
//! frame. The default path selects the middle candidate everywhere; call
//! `path_finder` to compute the Viterbi-optimal path using a weighted sum of
//! stress, Q-factor, frequency-change and ceiling-change costs.
//!
//! Frame-level accuracy: reuses `Formant::from_sound_burg_multi` which matches
//! single-ceiling Burg output exactly. The path selector adds Viterbi logic
//! matching `FormantPath_pathFinder` in Praat.
//!
//! # Parity with parselmouth
//!
//! Parselmouth (0.4.x → Praat 6.1.38) does **not** expose a native
//! `FormantPath` class — users must go through `parselmouth.praat.call(...)`.
//! This module provides the same functionality as a first-class Rust/Python/
//! WASM type.
//!
//! Four structural details separate `FormantPath`'s internal formant
//! analysis from the standalone `Sound: To Formant (burg)`. All four are
//! implemented here, matching Praat's `Sound_to_FormantPath_any` +
//! `Sound_into_LPC` code paths (rather than the simpler
//! `Sound_to_Formant_any` path used by standalone):
//!
//! 1. **Shared frame grid.** Every candidate's LPC object uses the
//!    middle-ceiling-resampled sound's `(t1, numberOfFrames)`, even though
//!    each candidate has its own resampled `dx`.
//! 2. **Mean subtraction per frame.** `Sound_into_LPC` applies
//!    `Vector_subtractMean` to each frame buffer before windowing; standalone
//!    does not.
//! 3. **Nearest-index sample extraction.** `Sound_into_LPC` copies the frame
//!    starting at `round((t - 0.5*windowDuration - x1)/dx)` (via
//!    `Sampled_xToNearestIndex`), while standalone uses
//!    `floor((t - x1)/dx) + 1 - halfnsamp_window` (via `Sampled_xToLowIndex`).
//! 4. **Zero-padded out-of-bounds samples.** `Sound_into_Sound` zero-pads any
//!    samples outside `[1, my nx]`, and the zero-padded samples are included
//!    in the subsequent mean calculation.
//!
//! Bandwidth semantics also matched: Praat's C++ `norm(complex)` returns
//! `|z|²` (not `|z|`), so `bandwidth = -log(|z|²) * nyquist / π`
//! (`LPC/Roots_and_Formant.cpp:49`). FormantPath stress uses the bandwidth
//! as the per-data-point σ in its weighted Legendre fit, so a half-scale
//! bandwidth silently halves σ and breaks stress parity.
//!
//! Observed parity on Tom's Diner (129 s, 25 825 frames, default weights
//! 0.5): F1/F2/F3 median error 0.00 Hz, 90 % of frames within 1 Hz, 96 %
//! within 5 Hz. Remaining divergences are near-tie cases in Viterbi
//! back-tracking that depend on sub-Hz stress/Q numerical precision.
//!
//! # References
//!
//! - Boersma, P. & Weenink, D. (2024). *Praat: doing phonetics by computer.*
//!   <https://www.fon.hum.uva.nl/praat/>
//! - Weenink, D. (2015). Improved formant frequency measurements of short
//!   segments. *Proceedings of ICPhS 2015*, Glasgow.
//! - Escudero, P., Boersma, P., Rauber, A. S., & Bion, R. A. H. (2009).
//!   A cross-dialect acoustic description of vowels. *J. Acoust. Soc. Am.
//!   126*(3), 1379–1393.
//! - Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
//!   A Python interface to Praat. *Journal of Phonetics, 71*, 1–15.
//! - Forney, G. D. (1973). The Viterbi Algorithm. *Proceedings of the IEEE,
//!   61*(3), 268–278.

use crate::formant::{Formant, FormantFrame, FormantPoint};
use crate::formant_modeler::FormantModeler;
use crate::sound::Sound;

/// Transition cost cutoff (1 / "transition cost maximum"). Praat uses 100.0.
const TRANSITION_COST_CUTOFF: f64 = 100.0;
/// Q-factor normalization cutoff: per-track Q above this saturates at 1.
const Q_CUTOFF: f64 = 20.0;
/// Stress normalization cutoff: stress above this saturates at 1.
const STRESS_CUTOFF: f64 = 100.0;

/// A FormantPath: Burg formant candidates at a bundle of ceiling frequencies
/// plus a per-frame path pointing to the currently selected candidate.
#[derive(Debug, Clone)]
pub struct FormantPath {
    candidates: Vec<Formant>,
    ceilings: Vec<f64>,
    /// Per-frame selected candidate, 0-based indexing into `candidates`.
    path: Vec<usize>,
    start_time: f64,
    time_step: f64,
    num_frames: usize,
    max_num_formants: usize,
}

impl FormantPath {
    /// Build a FormantPath from a Sound using Praat's `To FormantPath (burg)`.
    ///
    /// `number_of_steps_up_down` controls how many ceilings are placed on each
    /// side of `middle_formant_ceiling`; total candidates = `2*N + 1`. Ceilings
    /// are `middle * exp((i - N) * ceiling_step_size)` for `i = 0..2*N`.
    ///
    /// `max_num_formants` is the integer number of formants per frame (Praat
    /// uses a floating-point "5.0" but truncates).
    pub fn from_sound_burg(
        sound: &Sound,
        time_step: f64,
        max_num_formants: usize,
        middle_formant_ceiling: f64,
        window_length: f64,
        pre_emphasis_from: f64,
        ceiling_step_size: f64,
        number_of_steps_up_down: usize,
    ) -> Self {
        assert!(time_step > 0.0, "time_step must be positive");
        assert!(
            ceiling_step_size > 0.0,
            "ceiling_step_size must be positive"
        );

        let ceilings = generate_ceilings(
            middle_formant_ceiling,
            ceiling_step_size,
            number_of_steps_up_down,
        );

        // Compute the middle candidate's frame grid once. All candidates —
        // INCLUDING the middle — must use this same grid AND go through the
        // mean-subtracting Burg path (matches Praat's `Sound_to_FormantPath_any`
        // which routes every candidate through `Sound_into_LPC`, not the
        // standalone `Sound_to_Formant_any`).
        //
        // We first compute the middle candidate with the standalone path just
        // to recover its `(t1, num_frames)`; then we recompute ALL candidates
        // (middle included) through the FormantPath-accurate `with_grid` path.
        // The throwaway standalone computation is cheap (one ceiling) and the
        // double work avoids a second grid-derivation formula here.
        let middle = number_of_steps_up_down;
        let (start_time, time_step_actual, num_frames) = {
            let tmp = Formant::from_sound_burg(
                sound,
                time_step,
                max_num_formants,
                middle_formant_ceiling,
                window_length,
                pre_emphasis_from,
            );
            (tmp.start_time(), tmp.time_step(), tmp.num_frames())
        };

        let candidates: Vec<Formant> = compute_candidates_with_shared_grid(
            sound,
            time_step,
            max_num_formants,
            &ceilings,
            window_length,
            pre_emphasis_from,
            start_time,
            num_frames,
        );

        let path = vec![middle; num_frames];

        Self {
            candidates,
            ceilings,
            path,
            start_time,
            time_step: time_step_actual,
            num_frames,
            max_num_formants,
        }
    }

    /// Number of candidate analyses (= `2 * number_of_steps_up_down + 1`).
    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Ceiling frequencies in Hz, in ascending order.
    pub fn ceilings(&self) -> &[f64] {
        &self.ceilings
    }

    /// Access one candidate Formant (0-based).
    pub fn candidate(&self, index: usize) -> &Formant {
        &self.candidates[index]
    }

    /// Current path: per-frame candidate index (0-based).
    pub fn path(&self) -> &[usize] {
        &self.path
    }

    /// Force every frame overlapping `[t_min, t_max]` to select `candidate`.
    pub fn set_path(&mut self, t_min: f64, t_max: f64, candidate: usize) {
        assert!(candidate < self.num_candidates(), "candidate out of range");
        let (imin, imax) = self.frame_range(t_min, t_max);
        for i in imin..=imax {
            if i < self.path.len() {
                self.path[i] = candidate;
            }
        }
    }

    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    pub fn max_num_formants(&self) -> usize {
        self.max_num_formants
    }

    /// Viterbi-optimal path selection mirroring `FormantPath_pathFinder`.
    ///
    /// `parameters[i]` is the Legendre polynomial order used to fit track i+1
    /// during stress computation. Pass an empty vector to disable stress-based
    /// costs (then `stress_weight` is ignored).
    pub fn path_finder(
        &mut self,
        q_weight: f64,
        frequency_change_weight: f64,
        stress_weight: f64,
        ceiling_change_weight: f64,
        intensity_modulation_step_size: f64,
        window_length: f64,
        parameters: &[i64],
        power: f64,
    ) {
        let n_cand = self.num_candidates();
        if n_cand == 0 || self.num_frames == 0 {
            return;
        }

        let middle = n_cand / 2; // same as numberOfStepsUpDown for 2N+1 candidates
        let num_tracks_fit = if parameters.is_empty() {
            0
        } else {
            parameters.len().min(self.max_num_formants)
        };

        // Intensity range across all frames in the middle candidate.
        let mut i_min = f64::INFINITY;
        let mut i_max = f64::NEG_INFINITY;
        if intensity_modulation_step_size > 0.0 {
            let mid_c = &self.candidates[middle];
            for t in 0..self.num_frames {
                if let Some(frame) = mid_c.frame(t) {
                    let intensity = frame.intensity();
                    if intensity < i_min {
                        i_min = intensity;
                    }
                    if intensity > i_max {
                        i_max = intensity;
                    }
                }
            }
        }
        let has_intensity_difference = i_max > i_min;
        let db_mid = if has_intensity_difference && i_min > 0.0 {
            0.5 * 10.0 * (i_max * i_min).log10()
        } else {
            0.0
        };

        // Pre-compute Q-sums (num_tracks_fit defaulting to max_num_formants when
        // parameters is empty — mirrors Praat's `numberOfTracks` fallback).
        let q_tracks = if num_tracks_fit > 0 {
            num_tracks_fit
        } else {
            self.max_num_formants
        };
        let qsums: Option<Vec<Vec<f64>>> = if q_weight > 0.0 {
            Some(self.compute_qsums(q_tracks))
        } else {
            None
        };

        // Pre-compute stress matrix (n_cand × num_frames) when stress is used.
        let stresses: Option<Vec<Vec<f64>>> = if stress_weight > 0.0 && num_tracks_fit > 0 {
            Some(self.compute_stresses(window_length, parameters, power, num_tracks_fit))
        } else {
            None
        };

        let ceilings_range = self.ceilings[n_cand - 1] - self.ceilings[0];

        // Viterbi: delta[i][t] = min cost to reach candidate i at frame t.
        let mut delta = vec![vec![0.0_f64; self.num_frames]; n_cand];
        let mut psi = vec![vec![0_usize; self.num_frames]; n_cand];

        // Static cost at every (iformant, itime).
        let static_cost = |this: &Self, iformant: usize, itime: usize| -> f64 {
            let frame = this.candidates[iformant].frame(itime);
            let frame = match frame {
                Some(f) => f,
                None => return 0.0,
            };
            let mut w_intensity = 1.0;
            if has_intensity_difference && intensity_modulation_step_size > 0.0 {
                let intensity = frame.intensity();
                if intensity > 0.0 {
                    let dbi = 10.0 * (intensity / 2e-5).log10();
                    w_intensity = num_sigmoid((dbi - db_mid) / intensity_modulation_step_size);
                } else {
                    w_intensity = 0.0;
                }
            }
            let mut costs = 0.0;
            if stress_weight > 0.0 {
                if let Some(ref s) = stresses {
                    let v = s[iformant][itime];
                    if v.is_finite() {
                        costs += stress_weight * (v / STRESS_CUTOFF).min(1.0);
                    }
                }
            }
            if q_weight > 0.0 {
                if let Some(ref q) = qsums {
                    costs -= q_weight * (q[iformant][itime] / Q_CUTOFF).min(1.0);
                }
            }
            w_intensity * costs
        };

        // t = 0: initialize with static cost only.
        for iformant in 0..n_cand {
            delta[iformant][0] = static_cost(self, iformant, 0);
        }

        // t = 1..num_frames-1: best predecessor + transition + static cost.
        for itime in 1..self.num_frames {
            // Access frames at itime once per candidate.
            let frames_i: Vec<Option<&FormantFrame>> = self
                .candidates
                .iter()
                .map(|c| c.frame(itime))
                .collect();
            let frames_j: Vec<Option<&FormantFrame>> = self
                .candidates
                .iter()
                .map(|c| c.frame(itime - 1))
                .collect();
            for iformant in 0..n_cand {
                let fi = frames_i[iformant];
                let num_tracks_i = match fi {
                    Some(f) => self.max_num_formants.min(f.num_formants()),
                    None => 0,
                };

                let mut deltamin = f64::INFINITY;
                let mut minpos = 0usize;

                for jformant in 0..n_cand {
                    let fj = frames_j[jformant];
                    let mut transition_costs = delta[jformant][itime - 1];
                    if frequency_change_weight > 0.0 {
                        if let (Some(ffi), Some(ffj)) = (fi, fj) {
                            let ntracks =
                                ffj.num_formants().min(num_tracks_i).min(self.max_num_formants);
                            if ntracks > 0 {
                                let mut fcost = 0.0;
                                for itrack in 1..=ntracks {
                                    let pi = ffi.get_formant(itrack).copied().unwrap_or_default();
                                    let pj = ffj.get_formant(itrack).copied().unwrap_or_default();
                                    let dif = (pi.frequency - pj.frequency).abs();
                                    let sum = pi.frequency + pj.frequency;
                                    if sum > 0.0 {
                                        let bw = (pi.bandwidth * pj.bandwidth).max(0.0).sqrt();
                                        fcost += bw * dif / sum;
                                    }
                                }
                                fcost /= ntracks as f64;
                                transition_costs += frequency_change_weight
                                    * (fcost / TRANSITION_COST_CUTOFF).min(1.0);
                            }
                        }
                    }
                    if ceiling_change_weight > 0.0 && ceilings_range > 0.0 {
                        let ceiling_change_cost =
                            (self.ceilings[iformant] - self.ceilings[jformant]).abs()
                                / ceilings_range;
                        transition_costs += ceiling_change_weight * ceiling_change_cost;
                    }
                    if transition_costs < deltamin {
                        deltamin = transition_costs;
                        minpos = jformant;
                    }
                }

                // Add static-cost components for (iformant, itime). Note Praat
                // adds these twice at t > 0 — once already embedded in static
                // cost via intensity-scaled stress/q and once unscaled below —
                // we reproduce that exactly so numbers match.
                if stress_weight > 0.0 {
                    if let Some(ref s) = stresses {
                        let v = s[iformant][itime];
                        if v.is_finite() {
                            deltamin += stress_weight * (v / STRESS_CUTOFF).min(1.0);
                        }
                    }
                }
                if q_weight > 0.0 {
                    if let Some(ref q) = qsums {
                        deltamin -= q_weight * (q[iformant][itime] / Q_CUTOFF).min(1.0);
                    }
                }

                delta[iformant][itime] = static_cost(self, iformant, itime) + deltamin;
                psi[iformant][itime] = minpos;
            }
        }

        // Backtrack from the argmax of the last column. Praat uses `NUMmaxPos`
        // here (LPC/FormantPath.cpp:260); combined with `psi` storing argmin
        // transitions, this yields the path whose endpoint has the maximum
        // accumulated delta while all predecessors are chosen by minimum cost.
        // See docs/RECIPE_formant_path.md in praatfan-core-clean for the
        // rationale.
        let last = self.num_frames - 1;
        let mut best = 0usize;
        let mut best_val = delta[0][last];
        for iformant in 1..n_cand {
            if delta[iformant][last] > best_val {
                best_val = delta[iformant][last];
                best = iformant;
            }
        }
        self.path[last] = best;
        for t in (1..self.num_frames).rev() {
            self.path[t - 1] = psi[self.path[t]][t];
        }
    }

    /// Build a new Formant by selecting, at each frame, the formants from the
    /// candidate currently indicated by `self.path`.
    pub fn extract_formant(&self) -> Formant {
        // Use candidate 0 as the template (all candidates share the same grid).
        let template = if self.candidates.is_empty() {
            // Degenerate case: return an empty Formant with the middle ceiling.
            return Formant::from_sound_burg(
                &Sound::from_samples(&[], 22050.0),
                self.time_step,
                self.max_num_formants,
                self.ceilings
                    .get(self.num_candidates() / 2)
                    .copied()
                    .unwrap_or(5500.0),
                0.025,
                50.0,
            );
        } else {
            &self.candidates[0]
        };
        let n = self.num_frames;
        let max_formants = self.max_num_formants;
        Formant::from_frame_selector(template, n, |t| {
            let idx = self.path.get(t).copied().unwrap_or(self.num_candidates() / 2);
            let source = &self.candidates[idx];
            let src_frame = source.frame(t).cloned();
            match src_frame {
                Some(f) => f,
                None => FormantFrame::new(
                    vec![
                        FormantPoint {
                            frequency: f64::NAN,
                            bandwidth: f64::NAN,
                        };
                        max_formants
                    ],
                    0.0,
                ),
            }
        })
    }

    /// Stress of one candidate over [t_min, t_max].
    ///
    /// Matches `FormantPath_getStressOfCandidate`. `from_formant`/`to_formant`
    /// are 1-based formant track ranges (0 means "auto = full range of tracks
    /// in `parameters`").
    pub fn get_stress_of_candidate(
        &self,
        t_min: f64,
        t_max: f64,
        from_formant: usize,
        to_formant: usize,
        parameters: &[i64],
        power: f64,
        candidate: usize,
    ) -> f64 {
        assert!(candidate < self.num_candidates(), "candidate out of range");
        let fm = FormantModeler::from_formant(
            &self.candidates[candidate],
            t_min,
            t_max,
            parameters,
        );
        let (from_t, to_t) = resolve_track_range(from_formant, to_formant, parameters.len());
        fm.stress(from_t, to_t, power)
    }

    /// Stress for every candidate over [t_min, t_max].
    pub fn get_stress_of_candidates(
        &self,
        t_min: f64,
        t_max: f64,
        from_formant: usize,
        to_formant: usize,
        parameters: &[i64],
        power: f64,
    ) -> Vec<f64> {
        (0..self.num_candidates())
            .map(|c| {
                self.get_stress_of_candidate(t_min, t_max, from_formant, to_formant, parameters, power, c)
            })
            .collect()
    }

    // ---- internals -------------------------------------------------------

    fn frame_range(&self, t_min: f64, t_max: f64) -> (usize, usize) {
        if self.num_frames == 0 {
            return (0, 0);
        }
        let imin = (((t_min - self.start_time) / self.time_step).ceil()).max(0.0) as isize;
        let imax = (((t_max - self.start_time) / self.time_step).floor())
            .min((self.num_frames - 1) as f64) as isize;
        (
            imin.max(0) as usize,
            (imax.max(0) as usize).min(self.num_frames.saturating_sub(1)),
        )
    }

    /// Matrix[candidate][frame] of per-frame Q sums (average of f/bw over up to
    /// `num_tracks` tracks). Mirrors `FormantPath_to_Matrix_qSums`.
    fn compute_qsums(&self, num_tracks: usize) -> Vec<Vec<f64>> {
        let n_cand = self.num_candidates();
        let n_frames = self.num_frames;
        let mut out = vec![vec![0.0_f64; n_frames]; n_cand];
        for c in 0..n_cand {
            let candidate = &self.candidates[c];
            for t in 0..n_frames {
                let frame = match candidate.frame(t) {
                    Some(f) => f,
                    None => continue,
                };
                let current_n = num_tracks.min(frame.num_formants());
                if current_n == 0 {
                    continue;
                }
                let mut qsum = 0.0;
                for itrack in 1..=current_n {
                    if let Some(p) = frame.get_formant(itrack) {
                        if p.frequency.is_finite() && p.bandwidth.is_finite() && p.bandwidth > 0.0 {
                            qsum += p.frequency / p.bandwidth;
                        }
                    }
                }
                out[c][t] = qsum / current_n as f64;
            }
        }
        out
    }

    /// Matrix[candidate][frame] of per-frame stress values. Mirrors
    /// `FormantPath_to_Matrix_stress`.
    fn compute_stresses(
        &self,
        window_length: f64,
        parameters: &[i64],
        power: f64,
        num_tracks_fit: usize,
    ) -> Vec<Vec<f64>> {
        // Resolve from/to formant indices (skip zero parameters at start/end).
        let mut from_formant = 1usize;
        while from_formant <= parameters.len() && parameters[from_formant - 1] <= 0 {
            from_formant += 1;
        }
        let mut to_formant = num_tracks_fit;
        while to_formant > 0 && parameters[to_formant - 1] <= 0 {
            to_formant -= 1;
        }

        let n_cand = self.num_candidates();
        let n_frames = self.num_frames;
        let mut out = vec![vec![f64::NAN; n_frames]; n_cand];
        if from_formant > to_formant {
            return out;
        }

        for c in 0..n_cand {
            let candidate = &self.candidates[c];
            for t in 0..n_frames {
                let time = self.start_time + t as f64 * self.time_step;
                let start_time = time - 0.5 * window_length;
                let end_time = time + 0.5 * window_length;
                let fm = FormantModeler::from_formant(
                    candidate,
                    start_time,
                    end_time,
                    parameters,
                );
                out[c][t] = fm.stress(from_formant, to_formant, power);
            }
        }
        out
    }
}

fn compute_candidates_with_shared_grid(
    sound: &Sound,
    time_step: f64,
    max_num_formants: usize,
    ceilings: &[f64],
    window_length: f64,
    pre_emphasis_from: f64,
    forced_t1: f64,
    forced_num_frames: usize,
) -> Vec<Formant> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        ceilings
            .par_iter()
            .map(|&hz| {
                Formant::from_sound_burg_with_grid(
                    sound,
                    time_step,
                    max_num_formants,
                    hz,
                    window_length,
                    pre_emphasis_from,
                    forced_t1,
                    forced_num_frames,
                )
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        ceilings
            .iter()
            .map(|&hz| {
                Formant::from_sound_burg_with_grid(
                    sound,
                    time_step,
                    max_num_formants,
                    hz,
                    window_length,
                    pre_emphasis_from,
                    forced_t1,
                    forced_num_frames,
                )
            })
            .collect()
    }
}

fn generate_ceilings(
    middle_ceiling: f64,
    step_size: f64,
    number_of_steps_up_down: usize,
) -> Vec<f64> {
    let n_total = 2 * number_of_steps_up_down + 1;
    let mut out = vec![0.0_f64; n_total];
    let mid = number_of_steps_up_down;
    out[mid] = middle_ceiling;
    for istep in 1..=number_of_steps_up_down {
        out[mid + istep] = middle_ceiling * (step_size * istep as f64).exp();
        out[mid - istep] = middle_ceiling * (-step_size * istep as f64).exp();
    }
    out
}

fn resolve_track_range(from_formant: usize, to_formant: usize, n_params: usize) -> (usize, usize) {
    // Praat's `checkTrackAutoRange`: from=0 & to=0 means full range.
    if from_formant == 0 && to_formant == 0 {
        (1, n_params.max(1))
    } else {
        (from_formant.max(1), to_formant.max(from_formant.max(1)))
    }
}

fn num_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Sound {
    /// Compute a FormantPath (multi-ceiling Burg analysis) from this sound.
    ///
    /// Parameters follow Praat's `Sound: To FormantPath (burg)`:
    /// * `time_step` — seconds between frames
    /// * `max_num_formants` — per-frame formant cap (typically 5)
    /// * `middle_formant_ceiling` — Hz (typically 5000/5500)
    /// * `window_length` — analysis window duration (typically 0.025)
    /// * `pre_emphasis_from` — pre-emphasis Hz (typically 50)
    /// * `ceiling_step_size` — log-step for ceilings (typically 0.05)
    /// * `number_of_steps_up_down` — total ceilings = `2*N + 1`
    pub fn to_formant_path_burg(
        &self,
        time_step: f64,
        max_num_formants: usize,
        middle_formant_ceiling: f64,
        window_length: f64,
        pre_emphasis_from: f64,
        ceiling_step_size: f64,
        number_of_steps_up_down: usize,
    ) -> FormantPath {
        FormantPath::from_sound_burg(
            self,
            time_step,
            max_num_formants,
            middle_formant_ceiling,
            window_length,
            pre_emphasis_from,
            ceiling_step_size,
            number_of_steps_up_down,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ceilings() {
        let c = generate_ceilings(5500.0, 0.05, 4);
        assert_eq!(c.len(), 9);
        assert!((c[4] - 5500.0).abs() < 1e-9);
        // Ascending
        for i in 1..c.len() {
            assert!(c[i] > c[i - 1]);
        }
        // Geometric
        for i in 1..=4 {
            assert!((c[4 + i] - 5500.0 * (0.05 * i as f64).exp()).abs() < 1e-9);
            assert!((c[4 - i] - 5500.0 * (-0.05 * i as f64).exp()).abs() < 1e-9);
        }
    }

    #[test]
    fn test_formant_path_basic() {
        let sound = Sound::create_tone(200.0, 0.5, 16000.0, 0.5, 0.0);
        let fp = sound.to_formant_path_burg(0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4);
        assert_eq!(fp.num_candidates(), 9);
        assert_eq!(fp.ceilings().len(), 9);
        assert!(fp.num_frames() > 0);
        // Initial path is all middle candidate.
        let middle = 4;
        assert!(fp.path().iter().all(|&c| c == middle));
    }

    #[test]
    fn test_extract_before_pathfinder_returns_middle() {
        let sound = Sound::create_tone(200.0, 0.5, 16000.0, 0.5, 0.0);
        let fp = sound.to_formant_path_burg(0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4);
        let extracted = fp.extract_formant();
        assert_eq!(extracted.num_frames(), fp.num_frames());
        // Extracted should equal the middle candidate frame-for-frame.
        let middle = fp.candidate(4);
        for t in 0..fp.num_frames() {
            let a = extracted.get_value_at_frame(1, t);
            let b = middle.get_value_at_frame(1, t);
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(x), Some(y)) = (a, b) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_path_finder_runs() {
        let sound = Sound::create_tone(200.0, 0.5, 16000.0, 0.5, 0.0);
        let mut fp = sound.to_formant_path_burg(0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4);
        fp.path_finder(1.0, 1.0, 1.0, 1.0, 5.0, 0.035, &[3, 3, 3, 3], 1.25);
        assert_eq!(fp.path().len(), fp.num_frames());
        for &c in fp.path() {
            assert!(c < fp.num_candidates());
        }
    }
}
