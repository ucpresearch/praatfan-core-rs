//! Interpolation methods for querying sampled data
//!
//! This module provides interpolation functions matching Praat's conventions.
//! These are used when querying analysis results (pitch, formants, etc.) at
//! arbitrary time points.

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest neighbor (no interpolation)
    Nearest,
    /// Linear interpolation between adjacent samples
    #[default]
    Linear,
    /// Cubic interpolation using 4 neighboring samples
    Cubic,
    /// Sinc interpolation (band-limited)
    Sinc70,
    /// Higher quality sinc interpolation
    Sinc700,
}

impl Interpolation {
    /// Interpolate a value from a uniformly-sampled array
    ///
    /// # Arguments
    /// * `samples` - The array of samples
    /// * `position` - The fractional position in the array (0.0 = first sample)
    ///
    /// # Returns
    /// The interpolated value, or None if the position is outside the valid range
    pub fn interpolate(self, samples: &[f64], position: f64) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let max_pos = (n - 1) as f64;

        // Clamp position to valid range
        if position < 0.0 || position > max_pos {
            return None;
        }

        match self {
            Interpolation::Nearest => {
                let idx = position.round() as usize;
                let idx = idx.min(n - 1);
                Some(samples[idx])
            }

            Interpolation::Linear => Some(linear_interpolate(samples, position)),

            Interpolation::Cubic => Some(cubic_interpolate(samples, position)),

            Interpolation::Sinc70 => Some(sinc_interpolate(samples, position, 70)),

            Interpolation::Sinc700 => Some(sinc_interpolate(samples, position, 700)),
        }
    }

    /// Interpolate a value, skipping undefined (NaN) values
    ///
    /// This is important for analysis results where some frames may be undefined
    /// (e.g., unvoiced frames in pitch analysis).
    pub fn interpolate_with_undefined(self, samples: &[f64], position: f64) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let max_pos = (n - 1) as f64;

        if position < 0.0 || position > max_pos {
            return None;
        }

        match self {
            Interpolation::Nearest => {
                let idx = position.round() as usize;
                let idx = idx.min(n - 1);
                let val = samples[idx];
                if val.is_nan() {
                    None
                } else {
                    Some(val)
                }
            }

            Interpolation::Linear => {
                linear_interpolate_with_undefined(samples, position)
            }

            Interpolation::Cubic => {
                // For cubic, fall back to linear if we can't get all 4 defined points
                cubic_interpolate_with_undefined(samples, position)
            }

            // Sinc interpolation with undefined values is complex; fall back to linear
            Interpolation::Sinc70 | Interpolation::Sinc700 => {
                linear_interpolate_with_undefined(samples, position)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Praat-exact query semantics
// ---------------------------------------------------------------------------

impl Interpolation {
    /// Praat's `kVector_valueInterpolation_to_interpolationDepth`.
    pub(crate) fn praat_depth(self) -> isize {
        match self {
            Interpolation::Nearest => 0,
            Interpolation::Linear => 1,
            Interpolation::Cubic => 2,
            Interpolation::Sinc70 => 70,
            Interpolation::Sinc700 => 700,
        }
    }
}

/// Port of Praat's `NUM_interpolate_sinc` (melder/NUMinterpol.cpp, 6.1.38),
/// including its depth-0 (nearest) and depth-2 (cubic) special cases.
/// `x` is a 1-based real index into `y`.
pub(crate) fn num_interpolate_sinc(y: &[f64], x: f64, max_depth: isize) -> f64 {
    let n = y.len() as isize;
    if n < 1 {
        return f64::NAN;
    }
    if x < 1.0 {
        return y[0];
    }
    if x > n as f64 {
        return y[(n - 1) as usize];
    }
    let midleft = x.floor() as isize;
    let midright = midleft + 1;
    if x == midleft as f64 {
        return y[(midleft - 1) as usize];
    }
    let depth = max_depth.min(midright - 1).min(n - midleft);
    let at = |i: isize| y[(i - 1) as usize];
    if depth <= 0 {
        return at((x + 0.5).floor() as isize);
    }
    if depth == 1 {
        return at(midleft) + (x - midleft as f64) * (at(midright) - at(midleft));
    }
    if depth == 2 {
        let yl = at(midleft);
        let yr = at(midright);
        let dyl = 0.5 * (yr - at(midleft - 1));
        let dyr = 0.5 * (at(midright + 1) - yl);
        let fil = x - midleft as f64;
        let fir = midright as f64 - x;
        return yl * fir + yr * fil
            - fil * fir * (0.5 * (dyr - dyl) + (fil - 0.5) * (dyl + dyr - 2.0 * (yr - yl)));
    }
    crate::sound::sinc_interpolate_1based(y, x, depth)
}

/// Praat's `Vector_getValueAtX` for a single-channel vector of frames
/// (Intensity, Harmonicity): no undefined handling, defined on
/// `[x1 - dx/2, x1 + (n - 1/2) dx]`.
pub(crate) fn vector_value_at_x(
    values: &[f64],
    x1: f64,
    dx: f64,
    x: f64,
    interpolation: Interpolation,
) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let left_edge = x1 - 0.5 * dx;
    let right_edge = left_edge + values.len() as f64 * dx;
    if x < left_edge || x > right_edge {
        return None;
    }
    let index_real = (x - x1) / dx + 1.0;
    let v = num_interpolate_sinc(values, index_real, interpolation.praat_depth());
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}

/// Praat's `Sampled_getValueAtX` (fon/Sampled.cpp): linear interpolation
/// between the nearer and the farther frame. Undefined if the nearer frame is
/// undefined; the nearer value if the farther frame is undefined or off the
/// edge. `interpolate = false` gives the nearest frame. `get(i)` (0-based)
/// returns the frame value already converted to the requested unit.
pub(crate) fn sampled_value_at_x(
    n: usize,
    x1: f64,
    dx: f64,
    x: f64,
    interpolate: bool,
    get: impl Fn(usize) -> Option<f64>,
) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let index_real = (x - x1) / dx + 1.0;
    let fetch = |i: isize| -> Option<f64> {
        if i < 1 || i > n as isize {
            None
        } else {
            get((i - 1) as usize).filter(|v| !v.is_nan())
        }
    };
    if !interpolate {
        return fetch((index_real + 0.5).floor() as isize);
    }
    let left = index_real.floor() as isize;
    let mut phase = index_real - left as f64;
    let (near, far) = if phase < 0.5 {
        (left, left + 1)
    } else {
        phase = 1.0 - phase;
        (left + 1, left)
    };
    let near_value = fetch(near)?;
    match fetch(far) {
        None => Some(near_value),
        Some(far_value) => Some(near_value + phase * (far_value - near_value)),
    }
}

/// Peak interpolation for extremum queries (Praat's `kVector_peakInterpolation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeakInterpolation {
    /// The extreme frame value itself
    None,
    /// Parabolic refinement through three frames (Praat's default)
    #[default]
    Parabolic,
    /// Cubic interpolation refined by Brent's method
    Cubic,
    /// Sinc70 interpolation refined by Brent's method
    Sinc70,
    /// Sinc700 interpolation refined by Brent's method
    Sinc700,
}

impl PeakInterpolation {
    fn depth(self) -> i32 {
        match self {
            PeakInterpolation::None => 0,
            PeakInterpolation::Parabolic => 1,
            PeakInterpolation::Cubic => 2,
            PeakInterpolation::Sinc70 => 3,
            PeakInterpolation::Sinc700 => 4,
        }
    }
}

/// Port of Praat's `NUMimproveExtremum` (melder/NUMinterpol.cpp). `ixmid` is
/// 1-based. Returns `(value, real_index_1based)`.
pub(crate) fn num_improve_extremum(y: &[f64], ixmid: isize, peak: PeakInterpolation, is_max: bool) -> (f64, f64) {
    let n = y.len() as isize;
    if ixmid <= 1 {
        return (y[0], 1.0);
    }
    if ixmid >= n {
        return (y[(n - 1) as usize], n as f64);
    }
    let at = |i: isize| y[(i - 1) as usize];
    match peak.depth() {
        0 => (at(ixmid), ixmid as f64),
        1 => {
            let dy = 0.5 * (at(ixmid + 1) - at(ixmid - 1));
            let d2y = 2.0 * at(ixmid) - at(ixmid - 1) - at(ixmid + 1);
            (at(ixmid) + 0.5 * dy * dy / d2y, ixmid as f64 + dy / d2y)
        }
        d => {
            let vdepth = match d {
                2 => 2,
                3 => 70,
                _ => 700,
            };
            let f = |x: f64| {
                let v = num_interpolate_sinc(y, x, vdepth);
                if is_max { -v } else { v }
            };
            let (xr, res) = crate::pitch::minimize_brent(&f, (ixmid - 1) as f64, (ixmid + 1) as f64, 1e-10);
            (if is_max { -res } else { res }, xr)
        }
    }
}

/// `Sampled_getWindowSamples`: 1-based `(imin, imax)` of the frames whose
/// centres lie in `[tmin, tmax]`, clipped to `1..=n` (may be empty: imin > imax).
pub(crate) fn sampled_window_samples(n: usize, x1: f64, dx: f64, tmin: f64, tmax: f64) -> (isize, isize) {
    let ixmin_real = 1.0 + libm::ceil((tmin - x1) / dx);
    let ixmax_real = 1.0 + libm::floor((tmax - x1) / dx);
    let imin = if ixmin_real < 1.0 { 1 } else { ixmin_real as isize };
    let imax = if ixmax_real > n as f64 { n as isize } else { ixmax_real as isize };
    (imin, imax)
}

/// Port of Praat's `Vector_getMinimumAndX` / `Vector_getMaximumAndX` for one
/// channel. `[dom_xmin, dom_xmax]` is the object's time domain; `tmin >= tmax`
/// means the whole domain. Returns `(value, time)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vector_extremum(
    y: &[f64],
    x1: f64,
    dx: f64,
    dom_xmin: f64,
    dom_xmax: f64,
    mut tmin: f64,
    mut tmax: f64,
    peak: PeakInterpolation,
    is_max: bool,
) -> (f64, f64) {
    if y.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    if tmin >= tmax {
        tmin = dom_xmin;
        tmax = dom_xmax;
    }
    let n = y.len();
    let (mut imin, mut imax) = sampled_window_samples(n, x1, dx, tmin, tmax);
    if imin > imax {
        let vi = if peak != PeakInterpolation::None { Interpolation::Linear } else { Interpolation::Nearest };
        let yl = vector_value_at_x(y, x1, dx, tmin, vi).unwrap_or(f64::NAN);
        let yr = vector_value_at_x(y, x1, dx, tmax, vi).unwrap_or(f64::NAN);
        // std::min (a, b) = b < a ? b : a;  std::max (a, b) = a < b ? b : a
        let value = if is_max { if yl < yr { yr } else { yl } } else if yr < yl { yr } else { yl };
        let x = if yl == yr {
            (tmin + tmax) / 2.0
        } else if (is_max && yl > yr) || (!is_max && yl < yr) {
            tmin
        } else {
            tmax
        };
        return (value, x);
    }
    let at = |i: isize| y[(i - 1) as usize];
    let better = |a: f64, b: f64| if is_max { a > b } else { a < b };
    let mut extremum = at(imin);
    let mut x = imin as f64;
    if better(at(imax), extremum) {
        extremum = at(imax);
        x = imax as f64;
    }
    if imin == 1 {
        imin += 1;
    }
    if imax == n as isize {
        imax -= 1;
    }
    for i in imin..=imax {
        let is_peak = if is_max {
            at(i) > at(i - 1) && at(i) >= at(i + 1)
        } else {
            at(i) < at(i - 1) && at(i) <= at(i + 1)
        };
        if is_peak {
            let (local, i_real) = num_improve_extremum(y, i, peak, is_max);
            if better(local, extremum) {
                extremum = local;
                x = i_real;
            }
        }
    }
    let x = (x1 + (x - 1.0) * dx).clamp(tmin, tmax);
    (extremum, x)
}

/// Port of Praat's `Sampled_getMean` with `interpolate = true`
/// (`Sampled_getSumAndDefinitionRange`): the mean of the linearly
/// interpolated contour over `[tmin, tmax]` (`tmin >= tmax`: the whole
/// domain). `get(i)` (1-based) returns the frame value in the averaging unit,
/// NaN if undefined. None if nothing is defined in the range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_mean_interpolated(
    n: usize,
    x1: f64,
    dx: f64,
    dom_xmin: f64,
    dom_xmax: f64,
    xmin: f64,
    xmax: f64,
    get: impl Fn(isize) -> f64,
) -> Option<f64> {
    let (s, r) = sampled_sum_and_range(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax, get)?;
    if r <= 0.0 { None } else { Some(s / r) }
}

/// `Sampled_getSumAndDefinitionRange` (interpolate = true): area under the
/// linearly interpolated curve and the length over which it is defined, in
/// frame units. None if the range does not intersect the domain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_sum_and_range(
    n: usize,
    x1: f64,
    dx: f64,
    dom_xmin: f64,
    dom_xmax: f64,
    mut xmin: f64,
    mut xmax: f64,
    get: impl Fn(isize) -> f64,
) -> Option<(f64, f64)> {
    use crate::utils::CompSum;
    let value_at = |i: isize| if i < 1 || i > n as isize { f64::NAN } else { get(i) };
    if xmin >= xmax {
        xmin = dom_xmin;
        xmax = dom_xmax;
    }
    // Function_intersectRangeWithDomain
    if xmin < dom_xmin {
        xmin = dom_xmin;
    }
    if xmax > dom_xmax {
        xmax = dom_xmax;
    }
    if xmax <= xmin {
        return None;
    }
    let (mut sum, mut range) = (CompSum::default(), CompSum::default());
    let (imin, imax) = sampled_window_samples(n, x1, dx, xmin, xmax);
    if imin <= imax {
        let left_edge = x1 - 0.5 * dx;
        let right_edge = left_edge + n as f64 * dx;
        for isamp in imin..=imax {
            let v = value_at(isamp);
            if !v.is_nan() {
                range.add(1.0);
                sum.add(v);
            }
        }
        if xmin > left_edge {
            let mut phase = (x1 + (imin - 1) as f64 * dx - xmin) / dx;
            let right_value = value_at(imin);
            let left_value = value_at(imin - 1);
            if !right_value.is_nan() {
                range.add(-0.5);
                sum.add(-0.5 * right_value);
                if !left_value.is_nan() {
                    range.add(phase);
                    sum.add(phase * (right_value + 0.5 * phase * (left_value - right_value)));
                } else {
                    if phase > 0.5 {
                        phase = 0.5;
                    }
                    range.add(phase);
                    sum.add(phase * right_value);
                }
            } else if !left_value.is_nan() && phase > 0.5 {
                range.add(phase - 0.5);
                sum.add((phase - 0.5) * left_value);
            }
        }
        if xmax < right_edge {
            let mut phase = (xmax - (x1 + (imax - 1) as f64 * dx)) / dx;
            let left_value = value_at(imax);
            let right_value = value_at(imax + 1);
            if !left_value.is_nan() {
                range.add(-0.5);
                sum.add(-0.5 * left_value);
                if !right_value.is_nan() {
                    range.add(phase);
                    sum.add(phase * (left_value + 0.5 * phase * (right_value - left_value)));
                } else {
                    if phase > 0.5 {
                        phase = 0.5;
                    }
                    range.add(phase);
                    sum.add(phase * left_value);
                }
            } else if !right_value.is_nan() && phase > 0.5 {
                range.add(phase - 0.5);
                sum.add((phase - 0.5) * right_value);
            }
        }
    } else {
        // No frame centre in [xmin, xmax]: mean of the interpolated values.
        let left_value = value_at(imax);
        let right_value = value_at(imin);
        let mut phase1 = (xmin - (x1 + (imax - 1) as f64 * dx)) / dx;
        let mut phase2 = (xmax - (x1 + (imax - 1) as f64 * dx)) / dx;
        if imin == imax + 1 {
            if !left_value.is_nan() {
                if !right_value.is_nan() {
                    range.add(phase2 - phase1);
                    sum.add((phase2 - phase1) * (left_value + 0.5 * (phase1 + phase2) * (right_value - left_value)));
                } else if phase1 < 0.5 {
                    if phase2 > 0.5 {
                        phase2 = 0.5;
                    }
                    range.add(phase2 - phase1);
                    sum.add((phase2 - phase1) * left_value);
                }
            } else if !right_value.is_nan() && phase2 > 0.5 {
                if phase1 < 0.5 {
                    phase1 = 0.5;
                }
                range.add(phase2 - phase1);
                sum.add((phase2 - phase1) * right_value);
            }
        }
    }
    Some((sum.value(), range.value()))
}

/// Praat's `Function_unidirectionalAutowindow` + `Function_intersectRangeWithDomain`
/// + `Sampled_getWindowSamples` (`autoWindowDomainSamples`). Returns the 1-based
/// frame range and the clipped `[xmin, xmax]`, or None if nothing intersects.
pub(crate) fn sampled_window_in_domain(
    n: usize,
    x1: f64,
    dx: f64,
    dom_xmin: f64,
    dom_xmax: f64,
    mut xmin: f64,
    mut xmax: f64,
) -> Option<(isize, isize, f64, f64)> {
    if xmin >= xmax {
        xmin = dom_xmin;
        xmax = dom_xmax;
    }
    if xmin < dom_xmin {
        xmin = dom_xmin;
    }
    if xmax > dom_xmax {
        xmax = dom_xmax;
    }
    if xmax <= xmin {
        return None;
    }
    let (imin, imax) = sampled_window_samples(n, x1, dx, xmin, xmax);
    Some((imin, imax, xmin, xmax))
}

/// `Sampled_countDefinedSamples`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_count_defined(
    n: usize, x1: f64, dx: f64, dom_xmin: f64, dom_xmax: f64, xmin: f64, xmax: f64,
    get: impl Fn(isize) -> f64,
) -> usize {
    match sampled_window_in_domain(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax) {
        Some((imin, imax, _, _)) if imin <= imax => (imin..=imax).filter(|&i| !get(i).is_nan()).count(),
        _ => 0,
    }
}

/// Praat's `NUMquantile` on sorted values.
pub(crate) fn num_quantile(a: &[f64], factor: f64) -> f64 {
    let n = a.len();
    if n < 1 {
        return 0.0;
    }
    if n == 1 {
        return a[0];
    }
    let place = factor * n as f64 + 0.5;
    let mut left = libm::floor(place) as isize;
    if left < 1 {
        left = 1;
    }
    if left >= n as isize {
        left = n as isize - 1;
    }
    let (l, r) = (a[(left - 1) as usize], a[left as usize]);
    if r == l {
        return l;
    }
    l + (place - left as f64) * (r - l)
}

/// `Sampled_getQuantile`: quantile of the defined frame values in the range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_quantile(
    n: usize, x1: f64, dx: f64, dom_xmin: f64, dom_xmax: f64, xmin: f64, xmax: f64, quantile: f64,
    get: impl Fn(isize) -> f64,
) -> Option<f64> {
    let (imin, imax, _, _) = sampled_window_in_domain(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax)?;
    let mut v: Vec<f64> = (imin..=imax).map(&get).filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(num_quantile(&v, quantile))
}

/// `Sampled_getStandardDeviation` (interpolate = true).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_stdev_interpolated(
    n: usize, x1: f64, dx: f64, dom_xmin: f64, dom_xmax: f64, xmin: f64, xmax: f64,
    get: impl Fn(isize) -> f64,
) -> Option<f64> {
    let (sum, range) = sampled_sum_and_range(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax, &get)?;
    if range < 2.0 {
        return None;
    }
    let mean = sum / range;
    // Sampled_getSum2AndDefinitionRange: the same integral over (v - mean)^2
    let (sum2, range2) = sampled_sum_and_range(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax, |i| {
        let v = get(i);
        let d = v - mean;
        d * d
    })?;
    Some(libm::sqrt(sum2 / (range2 - 1.0)))
}

/// `Sampled_getMinimumAndX` / `Sampled_getMaximumAndX`: extremum of the
/// defined frames in the range, optionally refined parabolically (also using a
/// frame just outside the range), compared with the interpolated values at
/// the range ends. Returns `(value, time)`; None if nothing is defined.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_extremum(
    n: usize, x1: f64, dx: f64, dom_xmin: f64, dom_xmax: f64, xmin: f64, xmax: f64,
    interpolate: bool, is_max: bool,
    get: impl Fn(isize) -> f64,
) -> Option<(f64, f64)> {
    if xmin.is_nan() || xmax.is_nan() {
        return None;
    }
    let (imin, imax, xmin, xmax) = sampled_window_in_domain(n, x1, dx, dom_xmin, dom_xmax, xmin, xmax)?;
    let sentinel = if is_max { -1e301 } else { 1e301 };
    let better = |a: f64, b: f64| if is_max { a > b } else { a < b };
    let mut ext = sentinel;
    let mut xext = 0.0;
    let at = |i: isize| if i < 1 || i > n as isize { f64::NAN } else { get(i) };
    let value_at_x = |x: f64, interp: bool| -> f64 {
        if x < dom_xmin || x > dom_xmax {
            return f64::NAN;
        }
        sampled_value_at_x(n, x1, dx, x, interp, |i| Some(get(i as isize + 1))).unwrap_or(f64::NAN)
    };
    if imin > imax {
        let fl = value_at_x(xmin, interpolate);
        let fr = value_at_x(xmax, interpolate);
        if !fl.is_nan() && better(fl, ext) {
            ext = fl;
            xext = xmin;
        }
        if !fr.is_nan() && better(fr, ext) {
            ext = fr;
            xext = xmax;
        }
    } else {
        for i in imin..=imax {
            let fmid = at(i);
            if fmid.is_nan() {
                continue;
            }
            if !interpolate {
                if better(fmid, ext) {
                    ext = fmid;
                    xext = i as f64;
                }
            } else {
                let fl = if i <= 1 { f64::NAN } else { at(i - 1) };
                let fr = if i >= n as isize { f64::NAN } else { at(i + 1) };
                if fl.is_nan() || fr.is_nan() {
                    if better(fmid, ext) {
                        ext = fmid;
                        xext = i as f64;
                    }
                } else if (is_max && fmid > fl && fmid >= fr) || (!is_max && fmid < fl && fmid <= fr) {
                    let (local, i_real) = num_improve_extremum(&[fl, fmid, fr], 2, PeakInterpolation::Parabolic, is_max);
                    if better(local, ext) {
                        ext = local;
                        xext = i_real + i as f64 - 2.0;
                    }
                }
            }
        }
        xext = x1 + (xext - 1.0) * dx;
        if interpolate {
            let fl = value_at_x(xmin, true);
            let fr = value_at_x(xmax, true);
            if !fl.is_nan() && better(fl, ext) {
                ext = fl;
                xext = xmin;
            }
            if !fr.is_nan() && better(fr, ext) {
                ext = fr;
                xext = xmax;
            }
        }
        xext = xext.clamp(xmin, xmax);
    }
    if ext == sentinel {
        None
    } else {
        Some((ext, xext))
    }
}

/// Linear interpolation between two values
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Linear interpolation in an array
fn linear_interpolate(samples: &[f64], position: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return samples[0];
    }

    let idx = position.floor() as usize;
    if idx >= n - 1 {
        return samples[n - 1];
    }

    let frac = position - idx as f64;
    lerp(samples[idx], samples[idx + 1], frac)
}

/// Linear interpolation, skipping NaN values
fn linear_interpolate_with_undefined(samples: &[f64], position: f64) -> Option<f64> {
    let n = samples.len();
    if n == 0 {
        return None;
    }

    let idx = position.floor() as usize;
    let idx = idx.min(n - 1);
    let frac = position - idx as f64;

    // Get the two neighboring values
    let v0 = samples[idx];
    let v1 = if idx + 1 < n { samples[idx + 1] } else { v0 };

    // Handle undefined values
    match (v0.is_nan(), v1.is_nan()) {
        (true, true) => None,
        (true, false) => Some(v1),
        (false, true) => Some(v0),
        (false, false) => Some(lerp(v0, v1, frac)),
    }
}

/// Cubic interpolation using Catmull-Rom spline
///
/// This provides C1 continuity (smooth first derivative).
fn cubic_interpolate(samples: &[f64], position: f64) -> f64 {
    let n = samples.len();
    if n < 2 {
        return if n == 1 { samples[0] } else { 0.0 };
    }
    if n < 4 {
        return linear_interpolate(samples, position);
    }

    let idx = position.floor() as isize;
    let frac = position - idx as f64;

    // Get 4 surrounding points, clamping at boundaries
    let get_sample = |i: isize| -> f64 {
        let clamped = i.max(0).min(n as isize - 1) as usize;
        samples[clamped]
    };

    let y0 = get_sample(idx - 1);
    let y1 = get_sample(idx);
    let y2 = get_sample(idx + 1);
    let y3 = get_sample(idx + 2);

    // Catmull-Rom spline coefficients
    let t = frac;
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom formula
    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;

    a0 * t3 + a1 * t2 + a2 * t + a3
}

/// Cubic interpolation with undefined value handling
fn cubic_interpolate_with_undefined(samples: &[f64], position: f64) -> Option<f64> {
    let n = samples.len();
    if n < 2 {
        return if n == 1 && !samples[0].is_nan() {
            Some(samples[0])
        } else {
            None
        };
    }

    let idx = position.floor() as isize;

    // Check if we have 4 defined points for cubic interpolation
    let get_sample = |i: isize| -> Option<f64> {
        if i < 0 || i >= n as isize {
            None
        } else {
            let val = samples[i as usize];
            if val.is_nan() { None } else { Some(val) }
        }
    };

    // Try to get all 4 points
    if let (Some(y0), Some(y1), Some(y2), Some(y3)) = (
        get_sample(idx - 1),
        get_sample(idx),
        get_sample(idx + 1),
        get_sample(idx + 2),
    ) {
        let frac = position - idx as f64;
        let t = frac;
        let t2 = t * t;
        let t3 = t2 * t;

        let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
        let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let a2 = -0.5 * y0 + 0.5 * y2;
        let a3 = y1;

        return Some(a0 * t3 + a1 * t2 + a2 * t + a3);
    }

    // Fall back to linear interpolation
    linear_interpolate_with_undefined(samples, position)
}

/// Sinc (band-limited) interpolation
///
/// This provides the theoretically optimal interpolation for band-limited signals.
fn sinc_interpolate(samples: &[f64], position: f64, num_terms: usize) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return samples[0];
    }

    let idx = position.floor() as isize;
    let frac = position - idx as f64;

    // If we're exactly on a sample, return it
    if frac.abs() < 1e-10 {
        let clamped_idx = idx.max(0).min(n as isize - 1) as usize;
        return samples[clamped_idx];
    }

    let mut sum = 0.0;
    let mut weight_sum = 0.0;

    let half_terms = num_terms as isize / 2;
    let start = (idx - half_terms).max(0);
    let end = (idx + half_terms + 1).min(n as isize);

    for i in start..end {
        let x = position - i as f64;
        let sinc = if x.abs() < 1e-10 {
            1.0
        } else {
            let pi_x = std::f64::consts::PI * x;
            libm::sin(pi_x) / pi_x
        };

        // Apply a windowing function (Hanning) to reduce ringing
        let window_pos = (i - start) as f64 / (end - start - 1).max(1) as f64;
        let window = 0.5 * (1.0 - libm::cos(2.0 * std::f64::consts::PI * window_pos));

        let weight = sinc * window;
        sum += samples[i as usize] * weight;
        weight_sum += weight;
    }

    if weight_sum.abs() > 1e-10 {
        sum / weight_sum
    } else {
        samples[idx.max(0).min(n as isize - 1) as usize]
    }
}

/// Parabolic interpolation to find the peak of a discrete signal
///
/// Given three consecutive samples where the middle one is the maximum,
/// returns the fractional offset from the center sample and the interpolated
/// peak value.
///
/// # Returns
/// (offset, peak_value) where offset is in the range [-0.5, 0.5]
pub fn parabolic_peak(y0: f64, y1: f64, y2: f64) -> (f64, f64) {
    // For a parabola passing through (-1, y0), (0, y1), (1, y2):
    // The peak is at offset = (y0 - y2) / (2 * (y0 + y2 - 2*y1))
    let denominator = 2.0 * (y0 + y2 - 2.0 * y1);

    if denominator.abs() < 1e-10 {
        // Flat or nearly flat - return center
        return (0.0, y1);
    }

    let offset = (y0 - y2) / denominator;
    // Peak value: y1 + (y0 - y2)^2 / (8 * (y0 + y2 - 2*y1))
    let peak = y1 - (y0 - y2) * (y0 - y2) / (4.0 * denominator);

    (offset.clamp(-0.5, 0.5), peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_interpolation() {
        let samples = vec![0.0, 1.0, 2.0, 3.0];

        // At sample points
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 0.0).unwrap(),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 1.0).unwrap(),
            1.0,
            epsilon = 1e-10
        );

        // Between samples
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 0.5).unwrap(),
            0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            Interpolation::Linear.interpolate(&samples, 1.5).unwrap(),
            1.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_cubic_interpolation() {
        // For a parabola, cubic interpolation should be exact
        let samples: Vec<f64> = (0..10).map(|i| (i as f64).powi(2)).collect();

        for i in 1..8 {
            let pos = i as f64 + 0.5;
            let interpolated = Interpolation::Cubic.interpolate(&samples, pos).unwrap();
            let expected = pos.powi(2);
            assert_relative_eq!(interpolated, expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_interpolation_with_nan() {
        let samples = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

        // Should return None when both neighbors are NaN
        // In this case, position 2.5 has NaN on left
        let result = Interpolation::Linear.interpolate_with_undefined(&samples, 2.5);
        assert!(result.is_some()); // Should fall back to the defined value

        // Position at a NaN value itself with nearest
        let result = Interpolation::Nearest.interpolate_with_undefined(&samples, 2.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_parabolic_peak() {
        // Perfect parabola centered at x=0 with peak at 10
        // y = 10 - x^2
        let y0 = 10.0 - 1.0; // x = -1
        let y1 = 10.0; // x = 0
        let y2 = 10.0 - 1.0; // x = 1

        let (offset, peak) = parabolic_peak(y0, y1, y2);
        assert_relative_eq!(offset, 0.0, epsilon = 1e-10);
        assert_relative_eq!(peak, 10.0, epsilon = 1e-10);

        // Asymmetric case
        // y = 10 - (x - 0.3)^2 at x = -1, 0, 1
        let true_peak_x: f64 = 0.3;
        let y0 = 10.0 - (-1.0_f64 - true_peak_x).powi(2);
        let y1 = 10.0 - (0.0_f64 - true_peak_x).powi(2);
        let y2 = 10.0 - (1.0_f64 - true_peak_x).powi(2);

        let (offset, _peak) = parabolic_peak(y0, y1, y2);
        assert_relative_eq!(offset, true_peak_x, epsilon = 1e-10);
    }
}
