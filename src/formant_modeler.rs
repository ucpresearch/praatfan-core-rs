//! Per-ceiling formant-track modeling for FormantPath stress computation.
//!
//! Fits a Legendre polynomial of a caller-chosen order to each formant track
//! (F1, F2, …) over a time interval, weighted by `1/bandwidth`. Used solely to
//! compute the stress value that drives FormantPath's Viterbi optimum-path
//! selection, not as a general curve-fitter.
//!
//! Mirrors the Praat subset needed:
//! - `FormantModeler_create` / `Formant_to_FormantModeler`
//! - `series_fit` with Legendre basis on x scaled to [-1, 1]
//! - `DataModeler_getChiSquaredQ`
//! - `DataModeler_getVarianceOfParameters`
//! - `FormantModeler_getStress`
//!
//! Weighting: `ONE_OVER_SIGMA` with `sigma = bandwidth`.
//! Covariance: SVD pseudo-inverse of the weighted design matrix, squared.
//! Stress: `sqrt(pow(var/numFree, power) * (chisq_weighted / ndfTotal))`.

use nalgebra::{DMatrix, DVector};

use crate::formant::{Formant, FormantFrame};

/// Legendre polynomial value P_k(xs) for k = 0..(n-1).
/// Returns exactly `n` basis-function values into `terms`.
/// `xs` is assumed to be in [-1, 1].
fn legendre_basis(xs: f64, n: usize, terms: &mut [f64]) {
    debug_assert_eq!(terms.len(), n);
    if n == 0 {
        return;
    }
    terms[0] = 1.0;
    if n == 1 {
        return;
    }
    let two_x = 2.0 * xs;
    terms[1] = xs;
    let mut f2 = xs;
    let mut d = 1.0_f64;
    for ipar in 2..n {
        let f1 = d;
        d += 1.0;
        f2 += two_x;
        terms[ipar] = (f2 * terms[ipar - 1] - f1 * terms[ipar - 2]) / d;
    }
}

/// One fitted formant track.
struct TrackFit {
    /// Model coefficients (Legendre basis, 0..num_params-1).
    _parameters: Vec<f64>,
    /// Diagonal of the parameter covariance matrix (variance estimate per parameter).
    cov_diag: Vec<f64>,
    /// Chi-squared for this track: sum of `((y - model)/sigma)^2` over valid data points.
    chisq: f64,
    /// Degrees of freedom: `num_valid_data_points - num_free_params` (clamped at 0).
    dof: f64,
    /// Number of free parameters used in the fit.
    num_free_params: usize,
    /// True if the fit could be performed (enough valid points, invertible system).
    fitted: bool,
}

/// Formant-track modeler for a single Formant candidate over a [tmin, tmax] window.
pub struct FormantModeler {
    xmin: f64,
    xmax: f64,
    tracks: Vec<TrackFit>,
}

impl FormantModeler {
    /// Build and fit a FormantModeler from a Formant over [tmin, tmax] using
    /// `parameters[i]` as the number of Legendre coefficients for track `i`
    /// (1-based track, matching Praat). Tracks with `parameters[i] == 0` are
    /// represented but never contribute to stress (caller skips them).
    pub fn from_formant(formant: &Formant, tmin: f64, tmax: f64, parameters: &[i64]) -> Self {
        let (ifmin, ifmax) = window_samples(formant, tmin, tmax);
        let num_data_points = if ifmax >= ifmin {
            ifmax - ifmin + 1
        } else {
            0
        };

        let tracks = parameters
            .iter()
            .enumerate()
            .map(|(track_idx_0, &npar)| {
                let track_1based = track_idx_0 + 1;
                fit_track(
                    formant,
                    ifmin,
                    ifmax,
                    num_data_points,
                    track_1based,
                    npar as usize,
                    tmin,
                    tmax,
                )
            })
            .collect();

        Self {
            xmin: tmin,
            xmax: tmax,
            tracks,
        }
    }

    /// Domain used for Legendre x-scaling (for reference only).
    pub fn domain(&self) -> (f64, f64) {
        (self.xmin, self.xmax)
    }

    /// Combined variance of parameters (sum of diagonal covariances) for tracks
    /// in `from_track..=to_track` (1-based, inclusive). Returns `(variance,
    /// num_free_params)`. Variance is undefined (NaN) if any track failed to fit
    /// or has no free parameters.
    pub fn variance_of_parameters(&self, from_track: usize, to_track: usize) -> (f64, usize) {
        let mut variance = 0.0;
        let mut num_free = 0;
        let mut any_defined = false;
        for t in from_track..=to_track {
            if t == 0 || t > self.tracks.len() {
                continue;
            }
            let track = &self.tracks[t - 1];
            if !track.fitted {
                continue;
            }
            any_defined = true;
            for &v in &track.cov_diag {
                variance += v;
            }
            num_free += track.num_free_params;
        }
        if !any_defined {
            (f64::NAN, 0)
        } else {
            (variance, num_free)
        }
    }

    /// Chi-squared weighted across tracks, and total degrees of freedom.
    /// Mirrors `FormantModeler_getChiSquaredQ`: only returns a defined chisq
    /// when EVERY track in the range has `isdefined (chisqi)` (i.e. fitted).
    pub fn chi_squared(&self, from_track: usize, to_track: usize) -> (f64, f64) {
        let mut chisq_accum = 0.0;
        let mut ndf_total = 0.0;
        let mut number_of_defined = 0;
        let expected = to_track.saturating_sub(from_track) + 1;
        for t in from_track..=to_track {
            if t == 0 || t > self.tracks.len() {
                continue;
            }
            let track = &self.tracks[t - 1];
            if !track.fitted {
                continue;
            }
            let df = track.dof;
            let chisqi = track.chisq;
            chisq_accum += df * chisqi;
            ndf_total += df;
            number_of_defined += 1;
        }
        if number_of_defined == expected && ndf_total > 0.0 {
            (chisq_accum / ndf_total, ndf_total)
        } else {
            (f64::NAN, ndf_total)
        }
    }

    /// Praat `FormantModeler_getStress` over tracks [from_track, to_track]:
    ///
    ///     stress = sqrt( (var / num_free)^power * (chisq / ndf) )
    ///
    /// Returns NaN when variance, chisq, num_free, or ndf are undefined /
    /// non-positive (matches Praat's `isdefined` gate).
    pub fn stress(&self, from_track: usize, to_track: usize, power: f64) -> f64 {
        let (var, num_free) = self.variance_of_parameters(from_track, to_track);
        let (chisq, ndf) = self.chi_squared(from_track, to_track);
        if var.is_nan() || chisq.is_nan() || num_free == 0 || ndf < 0.0 {
            return f64::NAN;
        }
        let ratio = var / num_free as f64;
        let ratio_pow = ratio.powf(power);
        let term = ratio_pow * (chisq / ndf);
        if term < 0.0 {
            f64::NAN
        } else {
            term.sqrt()
        }
    }
}

/// Emulate Praat's `Sampled_getWindowSamples`: return (ifmin, ifmax) as 0-based
/// frame indices inclusive. Frames are centred at `start_time + i * time_step`;
/// we include frame `i` iff `tmin - eps <= frame_time <= tmax + eps` (Praat
/// uses half-open tolerance `0.5*dx`).
fn window_samples(formant: &Formant, tmin: f64, tmax: f64) -> (usize, usize) {
    let n = formant.num_frames();
    if n == 0 {
        return (1, 0);
    }
    let ts = formant.time_step();
    let x1 = formant.start_time();
    let ifmin = (((tmin - x1) / ts).ceil()).max(0.0) as isize;
    let ifmax = (((tmax - x1) / ts).floor()).min((n - 1) as f64) as isize;
    if ifmax < ifmin {
        return (1, 0);
    }
    (ifmin as usize, ifmax as usize)
}

fn fit_track(
    formant: &Formant,
    ifmin: usize,
    ifmax: usize,
    num_data_points: usize,
    track_1based: usize,
    num_params: usize,
    tmin: f64,
    tmax: f64,
) -> TrackFit {
    // Zero-order fit: no parameters, nothing meaningful to produce.
    if num_params == 0 || num_data_points == 0 {
        return TrackFit {
            _parameters: Vec::new(),
            cov_diag: Vec::new(),
            chisq: 0.0,
            dof: 0.0,
            num_free_params: 0,
            fitted: false,
        };
    }

    // Gather valid (x, y, sigma) triples. Invalid points: missing formant, NaN frequency.
    let ts = formant.time_step();
    let x1 = formant.start_time();
    let mut xs = Vec::with_capacity(num_data_points);
    let mut ys = Vec::with_capacity(num_data_points);
    let mut sigmas = Vec::with_capacity(num_data_points);
    for iframe in ifmin..=ifmax {
        let frame: &FormantFrame = match formant.frame(iframe) {
            Some(f) => f,
            None => continue,
        };
        if track_1based > frame.num_formants() {
            continue;
        }
        let fp = match frame.get_formant(track_1based) {
            Some(fp) => fp,
            None => continue,
        };
        if !fp.frequency.is_finite() {
            continue;
        }
        let sigma = fp.bandwidth;
        if !sigma.is_finite() || sigma <= 0.0 {
            // Praat requires defined sigma for ONE_OVER_SIGMA; skip if unusable.
            continue;
        }
        let x = x1 + iframe as f64 * ts;
        xs.push(x);
        ys.push(fp.frequency);
        sigmas.push(sigma);
    }

    let n_valid = xs.len();
    if n_valid == 0 || n_valid < num_params {
        return TrackFit {
            _parameters: vec![0.0; num_params],
            cov_diag: vec![0.0; num_params],
            chisq: 0.0,
            dof: 0.0,
            num_free_params: num_params,
            fitted: false,
        };
    }

    // Build weighted design matrix A[i][k] = L_k(xs_i) * w_i, b[i] = y_i * w_i,
    // where xs is scaled to [-1, 1] via Legendre scaling.
    let width = tmax - tmin;
    let mut design = DMatrix::<f64>::zeros(n_valid, num_params);
    let mut target = DVector::<f64>::zeros(n_valid);
    let mut basis = vec![0.0_f64; num_params];
    for i in 0..n_valid {
        let xs_scaled = if width > 0.0 {
            (2.0 * xs[i] - tmin - tmax) / width
        } else {
            0.0
        };
        legendre_basis(xs_scaled, num_params, &mut basis);
        let w = 1.0 / sigmas[i];
        for k in 0..num_params {
            design[(i, k)] = basis[k] * w;
        }
        target[i] = ys[i] * w;
    }

    // SVD-based least-squares with zeroing of small singular values. Praat's
    // `Formant_to_FormantModeler` sets `DataModeler.tolerance = 1e-5` (see
    // `LPC/FormantModeler.cpp`), which is applied as a RATIO cutoff
    // `sv[k] < sv[0] * tolerance => zero` (`dwsys/SVD.cpp:SVD_zeroSmallSingularValues`).
    let svd = design.clone().svd(true, true);
    let tolerance = 1e-5_f64;
    let u = match &svd.u {
        Some(u) => u,
        None => {
            return TrackFit {
                _parameters: vec![0.0; num_params],
                cov_diag: vec![0.0; num_params],
                chisq: 0.0,
                dof: 0.0,
                num_free_params: num_params,
                fitted: false,
            };
        }
    };
    let v_t = match &svd.v_t {
        Some(v_t) => v_t,
        None => {
            return TrackFit {
                _parameters: vec![0.0; num_params],
                cov_diag: vec![0.0; num_params],
                chisq: 0.0,
                dof: 0.0,
                num_free_params: num_params,
                fitted: false,
            };
        }
    };

    let sv = &svd.singular_values;
    let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = s_max * tolerance;
    let nrank = sv.len();

    // Solve: x = V * diag(1/sv_k) * U^T * b, with 1/sv_k = 0 for tiny sv_k.
    let utb = u.transpose() * &target;
    let mut weighted = DVector::<f64>::zeros(nrank);
    let mut inv_sv = vec![0.0_f64; nrank];
    for k in 0..nrank {
        if sv[k] > cutoff {
            inv_sv[k] = 1.0 / sv[k];
            weighted[k] = utb[k] * inv_sv[k];
        }
    }
    let v = v_t.transpose();
    let solution = &v * weighted;

    // Parameter covariance diagonal via SVD: (V diag(1/sv^2) V^T)_{kk}
    // = sum_j V[k][j]^2 * (1/sv_j)^2.
    let mut cov_diag = vec![0.0_f64; num_params];
    for k in 0..num_params {
        let mut sum = 0.0;
        for j in 0..nrank {
            if inv_sv[j] != 0.0 {
                let vij = v[(k, j)];
                sum += vij * vij * inv_sv[j] * inv_sv[j];
            }
        }
        cov_diag[k] = sum;
    }

    // Compute chi-squared: sum_i ((y_i - model(x_i)) / sigma_i)^2.
    let mut chisq = 0.0;
    let mut basis = vec![0.0_f64; num_params];
    for i in 0..n_valid {
        let xs_scaled = if width > 0.0 {
            (2.0 * xs[i] - tmin - tmax) / width
        } else {
            0.0
        };
        legendre_basis(xs_scaled, num_params, &mut basis);
        let mut model = 0.0;
        for k in 0..num_params {
            model += solution[k] * basis[k];
        }
        let z = (ys[i] - model) / sigmas[i];
        chisq += z * z;
    }

    let num_free_params = num_params;
    let dof = (n_valid as f64 - num_free_params as f64).max(0.0);

    let mut params = vec![0.0_f64; num_params];
    for k in 0..num_params {
        params[k] = solution[k];
    }

    TrackFit {
        _parameters: params,
        cov_diag,
        chisq,
        dof,
        num_free_params,
        fitted: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legendre_matches_recurrence() {
        // P_0 = 1, P_1 = x, P_2 = 0.5*(3x^2-1), P_3 = 0.5*(5x^3-3x)
        let mut terms = vec![0.0; 4];
        legendre_basis(0.3, 4, &mut terms);
        assert!((terms[0] - 1.0).abs() < 1e-12);
        assert!((terms[1] - 0.3).abs() < 1e-12);
        assert!((terms[2] - 0.5 * (3.0 * 0.09 - 1.0)).abs() < 1e-12);
        assert!((terms[3] - 0.5 * (5.0 * 0.027 - 3.0 * 0.3)).abs() < 1e-12);
    }
}
