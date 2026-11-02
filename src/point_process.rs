//! Glottal pulses (PointProcess), jitter, shimmer and Praat's voice report.
//!
//! Ports of Praat 6.1.38: `Sound_Pitch_to_PointProcess_cc` and
//! `Sound_to_PointProcess_periodic_cc` (fon/Pitch_to_PointProcess.cpp,
//! fon/Sound_to_PointProcess.cpp), the PointProcess period statistics
//! (fon/PointProcess.cpp), jitter and shimmer (fon/VoiceAnalysis.cpp,
//! fon/AmplitudeTier.cpp) and `Sound_Pitch_PointProcess_voiceReport`.
//!
//! Indices in comments are Praat's (1-based); the code keeps them 1-based
//! where that makes the correspondence obvious.

use crate::pitch::{Pitch, StrengthUnit};
use crate::utils::CompSum;
use crate::{Interpolation, PitchUnit, Sound};

/// A sorted sequence of time points (glottal pulses), as Praat's PointProcess.
#[derive(Debug, Clone)]
pub struct PointProcess {
    xmin: f64,
    xmax: f64,
    t: Vec<f64>,
}

/// Shimmer measures (Praat's `PointProcess_Sound_getShimmer_multi`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Shimmer {
    pub local: Option<f64>,
    pub local_db: Option<f64>,
    pub apq3: Option<f64>,
    pub apq5: Option<f64>,
    pub apq11: Option<f64>,
    pub dda: Option<f64>,
}

/// The numbers in Praat's "Voice report".
#[derive(Debug, Clone, Default)]
pub struct VoiceReport {
    pub from_time: f64,
    pub to_time: f64,
    pub median_pitch: Option<f64>,
    pub mean_pitch: Option<f64>,
    pub stdev_pitch: Option<f64>,
    pub minimum_pitch: Option<f64>,
    pub maximum_pitch: Option<f64>,
    pub number_of_pulses: usize,
    pub number_of_periods: usize,
    pub mean_period: Option<f64>,
    pub stdev_period: Option<f64>,
    pub unvoiced_frames: usize,
    pub total_frames: usize,
    pub fraction_of_locally_unvoiced_frames: Option<f64>,
    pub number_of_voice_breaks: usize,
    pub voice_breaks_duration: f64,
    pub degree_of_voice_breaks: Option<f64>,
    pub jitter_local: Option<f64>,
    pub jitter_local_absolute: Option<f64>,
    pub jitter_rap: Option<f64>,
    pub jitter_ppq5: Option<f64>,
    pub jitter_ddp: Option<f64>,
    pub shimmer: Shimmer,
    pub mean_autocorrelation: Option<f64>,
    pub mean_noise_to_harmonics_ratio: Option<f64>,
    pub mean_harmonics_to_noise_ratio: Option<f64>,
}

fn def(v: f64) -> Option<f64> {
    if v.is_finite() { Some(v) } else { None }
}

impl PointProcess {
    /// An empty PointProcess on the domain `[xmin, xmax]`.
    pub fn new(xmin: f64, xmax: f64) -> Self {
        Self { xmin, xmax, t: Vec::new() }
    }

    /// From explicit times (sorted, duplicates dropped, as `addPoint` does).
    pub fn from_times(xmin: f64, xmax: f64, times: &[f64]) -> Self {
        let mut p = Self::new(xmin, xmax);
        for &t in times {
            p.add_point(t);
        }
        p
    }

    pub fn times(&self) -> &[f64] {
        &self.t
    }
    pub fn num_points(&self) -> usize {
        self.t.len()
    }
    pub fn xmin(&self) -> f64 {
        self.xmin
    }
    pub fn xmax(&self) -> f64 {
        self.xmax
    }

    fn at(&self, i: isize) -> f64 {
        self.t[(i - 1) as usize]
    }

    /// `PointProcess_getLowIndex` (1-based; 0 if before the first point).
    pub fn low_index(&self, t: f64) -> isize {
        let n = self.t.len() as isize;
        if n == 0 || t < self.at(1) {
            return 0;
        }
        if t >= self.at(n) {
            return n;
        }
        let (mut left, mut right) = (1isize, n);
        while left < right - 1 {
            let mid = (left + right) / 2;
            if t >= self.at(mid) { left = mid } else { right = mid }
        }
        left
    }

    /// `PointProcess_getHighIndex` (1-based; n + 1 if after the last point).
    pub fn high_index(&self, t: f64) -> isize {
        let n = self.t.len() as isize;
        if n == 0 {
            return 0;
        }
        if t <= self.at(1) {
            return 1;
        }
        if t > self.at(n) {
            return n + 1;
        }
        let (mut left, mut right) = (1isize, n);
        while left < right - 1 {
            let mid = (left + right) / 2;
            if t > self.at(mid) { left = mid } else { right = mid }
        }
        right
    }

    /// `PointProcess_addPoint`: insert keeping order; an exact duplicate is not added.
    pub fn add_point(&mut self, t: f64) {
        if t.is_nan() {
            return;
        }
        if self.t.is_empty() || t >= *self.t.last().unwrap() {
            self.t.push(t);
            return;
        }
        let left = self.low_index(t);
        if left == 0 || self.at(left) != t {
            self.t.insert(left as usize, t);
        }
    }

    /// Praat's unidirectional autowindow: an empty or reversed range means
    /// the whole domain.
    fn window(&self, tmin: f64, tmax: f64) -> (f64, f64) {
        if tmin >= tmax { (self.xmin, self.xmax) } else { (tmin, tmax) }
    }

    /// `PointProcess_getWindowPoints`: (first, last), 1-based.
    fn window_points(&self, tmin: f64, tmax: f64) -> (isize, isize) {
        (self.high_index(tmin), self.low_index(tmax))
    }

    /// `PointProcess_isPeriod` for the interval from point `ileft` to `ileft + 1`.
    fn is_period(&self, ileft: isize, pmin: f64, pmax: f64, max_period_factor: f64) -> bool {
        let n = self.t.len() as isize;
        let iright = ileft + 1;
        if ileft < 1 || iright > n {
            return false;
        }
        if pmin == pmax {
            return true;
        }
        let interval = self.at(iright) - self.at(ileft);
        if interval <= 0.0 || interval < pmin || interval > pmax {
            return false;
        }
        if max_period_factor.is_nan() || max_period_factor < 1.0 {
            return true;
        }
        let prev = if ileft <= 1 { f64::NAN } else { self.at(ileft) - self.at(ileft - 1) };
        let next = if iright >= n { f64::NAN } else { self.at(iright + 1) - self.at(iright) };
        let mut pf = if prev.is_finite() && prev > 0.0 { interval / prev } else { f64::NAN };
        let mut nf = if next.is_finite() && next > 0.0 { interval / next } else { f64::NAN };
        if pf.is_nan() && nf.is_nan() {
            return true;
        }
        if pf.is_finite() && pf > 0.0 && pf < 1.0 {
            pf = 1.0 / pf;
        }
        if nf.is_finite() && nf > 0.0 && nf < 1.0 {
            nf = 1.0 / nf;
        }
        !(pf.is_finite() && pf > max_period_factor && nf.is_finite() && nf > max_period_factor)
    }

    /// Praat's `Get number of periods...`.
    pub fn get_number_of_periods(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, max_period_factor: f64) -> usize {
        let (tmin, tmax) = self.window(tmin, tmax);
        let (first, last) = self.window_points(tmin, tmax);
        (first..last).filter(|&i| self.is_period(i, pmin, pmax, max_period_factor)).count()
    }

    /// Praat's `Get mean period...`.
    pub fn get_mean_period(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, max_period_factor: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        let (first, last) = self.window_points(tmin, tmax);
        let mut sum = CompSum::default();
        let mut n = 0usize;
        for i in first..last {
            if self.is_period(i, pmin, pmax, max_period_factor) {
                n += 1;
                sum.add(self.at(i + 1) - self.at(i));
            }
        }
        if n > 0 { Some(sum.value() / n as f64) } else { None }
    }

    /// Praat's `Get stdev period...`.
    pub fn get_stdev_period(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, max_period_factor: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        let (first, last) = self.window_points(tmin, tmax);
        let mut sum = CompSum::default();
        let mut n = 0usize;
        for i in first..last {
            if self.is_period(i, pmin, pmax, max_period_factor) {
                n += 1;
                sum.add(self.at(i + 1) - self.at(i));
            }
        }
        if n < 2 {
            return None;
        }
        let mean = sum.value() / n as f64;
        let mut sum2 = CompSum::default();
        for i in first..last {
            if self.is_period(i, pmin, pmax, max_period_factor) {
                let d = self.at(i + 1) - self.at(i) - mean;
                sum2.add(d * d);
            }
        }
        Some(libm::sqrt(sum2.value() / (n - 1) as f64))
    }

    fn ok_factor(p1: f64, p2: f64, pmin: f64, pmax: f64, mpf: f64) -> bool {
        let f = if p1 > p2 { p1 / p2 } else { p2 / p1 };
        pmin == pmax || (p1 >= pmin && p1 <= pmax && p2 >= pmin && p2 <= pmax && f <= mpf)
    }

    /// Mean absolute difference of consecutive periods (seconds), and the
    /// number of periods used; shared by jitter (local) and (local, absolute).
    fn jitter_sum_local(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        let (first, last) = self.window_points(tmin, tmax);
        let mut n_periods = last - first; // pointNumbers.size() - 1
        if n_periods < 2 {
            return None;
        }
        let mut sum = CompSum::default();
        for i in first + 1..last {
            let p1 = self.at(i) - self.at(i - 1);
            let p2 = self.at(i + 1) - self.at(i);
            if Self::ok_factor(p1, p2, pmin, pmax, mpf) {
                sum.add((p1 - p2).abs());
            } else {
                n_periods -= 1;
            }
        }
        if n_periods < 2 {
            return None;
        }
        Some(sum.value() / (n_periods - 1) as f64)
    }

    /// Praat's `Get jitter (local)...` (a fraction; Praat prints it as %).
    pub fn get_jitter_local(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        let num = self.jitter_sum_local(tmin, tmax, pmin, pmax, mpf)?;
        def(num / self.get_mean_period(tmin, tmax, pmin, pmax, mpf).unwrap_or(f64::NAN))
    }

    /// Praat's `Get jitter (local, absolute)...` (seconds).
    pub fn get_jitter_local_absolute(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        self.jitter_sum_local(tmin, tmax, pmin, pmax, mpf)
    }

    /// Praat's `Get jitter (rap)...`.
    pub fn get_jitter_rap(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        let (first, last) = self.window_points(tmin, tmax);
        let mut n_periods = last - first;
        if n_periods < 3 {
            return None;
        }
        let mut sum = CompSum::default();
        for i in first + 2..last {
            let p1 = self.at(i - 1) - self.at(i - 2);
            let p2 = self.at(i) - self.at(i - 1);
            let p3 = self.at(i + 1) - self.at(i);
            let f1 = if p1 > p2 { p1 / p2 } else { p2 / p1 };
            let f2 = if p2 > p3 { p2 / p3 } else { p3 / p2 };
            if pmin == pmax
                || (p1 >= pmin && p1 <= pmax && p2 >= pmin && p2 <= pmax && p3 >= pmin && p3 <= pmax && f1 <= mpf && f2 <= mpf)
            {
                sum.add((p2 - (p1 + p2 + p3) / 3.0).abs());
            } else {
                n_periods -= 1;
            }
        }
        if n_periods < 3 {
            return None;
        }
        def(sum.value() / (n_periods - 2) as f64 / self.get_mean_period(tmin, tmax, pmin, pmax, mpf).unwrap_or(f64::NAN))
    }

    /// Praat's `Get jitter (ppq5)...`.
    pub fn get_jitter_ppq5(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        let (tmin, tmax) = self.window(tmin, tmax);
        let (first, last) = self.window_points(tmin, tmax);
        let mut n_periods = last - first;
        if n_periods < 5 {
            return None;
        }
        let mut sum = CompSum::default();
        for i in first + 5..=last {
            let p: [f64; 5] = [
                self.at(i - 4) - self.at(i - 5),
                self.at(i - 3) - self.at(i - 4),
                self.at(i - 2) - self.at(i - 3),
                self.at(i - 1) - self.at(i - 2),
                self.at(i) - self.at(i - 1),
            ];
            let fac = |a: f64, b: f64| if a > b { a / b } else { b / a };
            let ok = pmin == pmax
                || (p.iter().all(|&x| x >= pmin && x <= pmax)
                    && fac(p[0], p[1]) <= mpf
                    && fac(p[1], p[2]) <= mpf
                    && fac(p[2], p[3]) <= mpf
                    && fac(p[3], p[4]) <= mpf);
            if ok {
                sum.add((p[2] - (p[0] + p[1] + p[2] + p[3] + p[4]) / 5.0).abs());
            } else {
                n_periods -= 1;
            }
        }
        if n_periods < 5 {
            return None;
        }
        def(sum.value() / (n_periods - 4) as f64 / self.get_mean_period(tmin, tmax, pmin, pmax, mpf).unwrap_or(f64::NAN))
    }

    /// Praat's `Get jitter (ddp)...` = 3 x rap.
    pub fn get_jitter_ddp(&self, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<f64> {
        self.get_jitter_rap(tmin, tmax, pmin, pmax, mpf).map(|r| 3.0 * r)
    }

    /// `PointProcess_Sound_to_AmplitudeTier_period`: Hann-windowed RMS at each
    /// pulse whose two neighbouring periods qualify. None if fewer than 3
    /// pulses are in the range (Praat: "Too few pulses").
    fn amplitude_tier(&self, sound: &Sound, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64) -> Option<Vec<(f64, f64)>> {
        let (first, last) = self.window_points(tmin, tmax);
        if last - first + 1 < 3 {
            return None;
        }
        let mut tier: Vec<(f64, f64)> = Vec::new();
        for i in first + 1..last {
            let p1 = self.at(i) - self.at(i - 1);
            let p2 = self.at(i + 1) - self.at(i);
            if Self::ok_factor(p1, p2, pmin, pmax, mpf) {
                let peak = hann_windowed_rms(sound, self.at(i), 0.2 * p1, 0.2 * p2);
                if peak.is_finite() && peak > 0.0 {
                    // RealTier_addPoint keeps times sorted; these already are.
                    tier.push((self.at(i), peak));
                }
            }
        }
        Some(tier)
    }

    /// Praat's six shimmer measures (`PointProcess_Sound_getShimmer_multi`).
    pub fn get_shimmer(&self, sound: &Sound, tmin: f64, tmax: f64, pmin: f64, pmax: f64, mpf: f64, maf: f64) -> Shimmer {
        let (tmin, tmax) = self.window(tmin, tmax);
        let Some(pts) = self.amplitude_tier(sound, tmin, tmax, pmin, pmax, mpf) else {
            return Shimmer::default();
        };
        let apq3 = shimmer_apq(&pts, pmin, pmax, maf, 1);
        Shimmer {
            local: shimmer_local(&pts, pmin, pmax, maf),
            local_db: shimmer_local_db(&pts, pmin, pmax, maf),
            apq3,
            apq5: shimmer_apq(&pts, pmin, pmax, maf, 2),
            apq11: shimmer_apq(&pts, pmin, pmax, maf, 5),
            dda: apq3.map(|v| 3.0 * v),
        }
    }

    /// `PointProcess_getCountAndFractionOfVoiceBreaks`: (count, total break
    /// duration, range duration).
    fn voice_breaks(&self, tmin: f64, tmax: f64, max_period: f64) -> (usize, f64, f64) {
        let (first, last) = self.window_points(tmin, tmax);
        let (mut count, mut num, mut den) = (0usize, 0.0, 0.0);
        if last - first + 1 > 1 {
            den = tmax - tmin;
            let mut previous_voiced = true;
            for i in first + 1..last {
                let period = self.at(i) - self.at(i - 1);
                if period > max_period {
                    num += period;
                    if previous_voiced {
                        count += 1;
                        previous_voiced = false;
                    }
                } else {
                    previous_voiced = true;
                }
            }
        }
        (count, num, den)
    }
}

// ---------------------------------------------------------------------------
// Shimmer helpers (AmplitudeTier_getShimmer_*)
// ---------------------------------------------------------------------------

fn amp_factor(a: f64, b: f64) -> f64 {
    if a > b { a / b } else { b / a }
}

/// Mean amplitude over points 1 .. n-1 (Praat excludes the last point).
fn shimmer_denominator(pts: &[(f64, f64)]) -> f64 {
    let mut den = CompSum::default();
    let mut n = 0usize;
    for &(_, a) in pts.iter().take(pts.len().saturating_sub(1)) {
        den.add(a);
        n += 1;
    }
    den.value() / n as f64
}

fn shimmer_local(pts: &[(f64, f64)], pmin: f64, pmax: f64, maf: f64) -> Option<f64> {
    let mut num = CompSum::default();
    let mut n = 0usize;
    for i in 1..pts.len() {
        let p = pts[i].0 - pts[i - 1].0;
        if pmin == pmax || (p >= pmin && p <= pmax) {
            let (a1, a2) = (pts[i - 1].1, pts[i].1);
            if amp_factor(a1, a2) <= maf {
                num.add((a1 - a2).abs());
                n += 1;
            }
        }
    }
    if n < 1 {
        return None;
    }
    let numerator = num.value() / n as f64;
    let den = shimmer_denominator(pts);
    if den == 0.0 { None } else { def(numerator / den) }
}

fn shimmer_local_db(pts: &[(f64, f64)], pmin: f64, pmax: f64, maf: f64) -> Option<f64> {
    let mut sum = CompSum::default();
    let mut n = 0usize;
    for i in 1..pts.len() {
        let p = pts[i].0 - pts[i - 1].0;
        if pmin == pmax || (p >= pmin && p <= pmax) {
            let (a1, a2) = (pts[i - 1].1, pts[i].1);
            if amp_factor(a1, a2) <= maf {
                sum.add(crate::math::log10(a1 / a2).abs());
                n += 1;
            }
        }
    }
    if n < 1 { None } else { def(20.0 * (sum.value() / n as f64)) }
}

/// apq3 (`half` = 1), apq5 (2) and apq11 (5): deviation of each amplitude
/// from the average of the 2*half+1 surrounding ones.
fn shimmer_apq(pts: &[(f64, f64)], pmin: f64, pmax: f64, maf: f64, half: usize) -> Option<f64> {
    let n_pts = pts.len();
    let mut num = CompSum::default();
    let mut n = 0usize;
    if n_pts >= 2 * half + 1 {
        for c in half..n_pts - half {
            let lo = c - half;
            let hi = c + half;
            let periods_ok = pmin == pmax
                || (lo..hi).all(|k| {
                    let p = pts[k + 1].0 - pts[k].0;
                    p >= pmin && p <= pmax
                });
            if !periods_ok {
                continue;
            }
            if !(lo..hi).all(|k| amp_factor(pts[k].1, pts[k + 1].1) <= maf) {
                continue;
            }
            // Praat's grouping of the sums (AmplitudeTier.cpp)
            let a: Vec<f64> = (lo..=hi).map(|k| pts[k].1).collect();
            let avg = match half {
                1 => (a[0] + a[1] + a[2]) / 3.0,
                2 => ((a[0] + a[1] + a[2]) + (a[3] + a[4])) / 5.0,
                _ => (((a[0] + a[1] + a[2]) + (a[3] + a[4] + a[5])) + ((a[6] + a[7] + a[8]) + (a[9] + a[10]))) / 11.0,
            };
            num.add((pts[c].1 - avg).abs());
            n += 1;
        }
    }
    if n < 1 {
        return None;
    }
    let numerator = num.value() / n as f64;
    let den = shimmer_denominator(pts);
    if den == 0.0 { None } else { def(numerator / den) }
}

// ---------------------------------------------------------------------------
// Sound helpers
// ---------------------------------------------------------------------------

/// `Sampled_xToLowIndex` / `Sampled_xToHighIndex` / `Sampled_xToNearestIndex` (1-based).
fn low_index(s: &Sound, x: f64) -> isize {
    libm::floor((x - s.x1()) / s.dx() + 1.0) as isize
}
fn high_index(s: &Sound, x: f64) -> isize {
    libm::ceil((x - s.x1()) / s.dx() + 1.0) as isize
}
fn nearest_index(s: &Sound, x: f64) -> isize {
    libm::floor((x - s.x1()) / s.dx() + 1.0 + 0.5) as isize
}

/// `Sound_getHannWindowedRms` (fon/AmplitudeTier.cpp).
fn hann_windowed_rms(s: &Sound, tmid: f64, width_left: f64, width_right: f64) -> f64 {
    let n = s.num_samples();
    let (imin, imax) = crate::interpolation::sampled_window_samples(n, s.x1(), s.dx(), tmid - width_left, tmid + width_right);
    if imax - imin + 1 < 3 {
        return f64::NAN;
    }
    let (x1, dx, y) = (s.x1(), s.dx(), s.samples());
    let (mut ss, mut ws) = (CompSum::default(), CompSum::default());
    for i in imin..=imax {
        let t = x1 + (i - 1) as f64 * dx;
        let width = if t < tmid { width_left } else { width_right };
        let phase = (t - tmid) / width;
        let w = 0.5 + 0.5 * crate::math::cos(std::f64::consts::PI * phase);
        let v = y[(i - 1) as usize] * w;
        ss.add(v * v);
        ws.add(w * w);
    }
    libm::sqrt(ss.value() / ws.value())
}

/// `findExtremum_3` for one channel; returns a 1-based real index into the
/// `n` samples starting after offset `d`, or 0 if n <= 0.
fn find_extremum_3(y: &[f64], d: isize, n: isize, include_max: bool, include_min: bool) -> f64 {
    let at = |i: isize| y[(d + i - 1) as usize];
    let include_all = include_max == include_min;
    if n < 3 {
        if n <= 0 {
            return 0.0;
        } else if n == 1 {
            return 1.0;
        }
        let (x1, x2) = (at(1), at(2));
        let xl = if include_all { x1.abs() } else if include_max { x1 } else { -x1 };
        let xr = if include_all { x2.abs() } else if include_max { x2 } else { -x2 };
        return if xl > xr { 1.0 } else if xl < xr { 2.0 } else { 1.5 };
    }
    let (mut minimum, mut maximum) = (at(1), at(1));
    let (mut imin, mut imax) = (1isize, 1isize);
    for i in 2..=n {
        let v = at(i);
        if v < minimum {
            minimum = v;
            imin = i;
        }
        if v > maximum {
            maximum = v;
            imax = i;
        }
    }
    if minimum == maximum {
        return 0.5 * (n as f64 + 1.0);
    }
    let iextr = if include_all {
        if minimum.abs() > maximum.abs() { imin } else { imax }
    } else if include_max {
        imax
    } else {
        imin
    };
    if iextr == 1 {
        return 1.0;
    }
    if iextr == n {
        return n as f64;
    }
    let (vm, vl, vr) = (at(iextr), at(iextr - 1), at(iextr + 1));
    iextr as f64 + 0.5 * (vr - vl) / (2.0 * vm - vl - vr)
}

/// `Sound_findExtremum` (fon/Pitch_to_PointProcess.cpp), mono.
fn sound_find_extremum(s: &Sound, tmin: f64, tmax: f64, include_max: bool, include_min: bool) -> f64 {
    let nx = s.num_samples() as isize;
    let imin = low_index(s, tmin).max(1);
    let imax = high_index(s, tmax).min(nx);
    let iext = find_extremum_3(s.samples(), imin - 1, imax - imin + 1, include_max, include_min);
    if iext != 0.0 {
        s.x1() + ((imin - 1) as f64 + iext - 1.0) * s.dx()
    } else {
        (tmin + tmax) / 2.0
    }
}

/// `Sound_findMaximumCorrelation`, mono. Returns (correlation, tout, peak).
fn sound_find_maximum_correlation(s: &Sound, t1: f64, window_length: f64, tmin2: f64, tmax2: f64) -> (f64, Option<f64>, f64) {
    let y = s.samples();
    let nx = y.len() as isize;
    let half = 0.5 * window_length;
    let ileft1 = nearest_index(s, t1 - half);
    let iright1 = nearest_index(s, t1 + half);
    let ileft2min = low_index(s, tmin2 - half);
    let ileft2max = high_index(s, tmax2 - half);
    let mut max_corr = -1.0;
    let (mut r1_best, mut r3_best, mut ir) = (f64::NAN, f64::NAN, f64::NAN);
    let (mut r2, mut r3) = (0.0f64, 0.0f64);
    let mut peak = 0.0;
    for ileft2 in ileft2min..=ileft2max {
        let (mut norm1, mut norm2, mut product, mut local_peak) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut i2 = ileft2;
        for i1 in ileft1..=iright1 {
            if !(i1 < 1 || i1 > nx || i2 < 1 || i2 > nx) {
                let a1 = y[(i1 - 1) as usize];
                let a2 = y[(i2 - 1) as usize];
                norm1 += a1 * a1;
                norm2 += a2 * a2;
                product += a1 * a2;
                if a2.abs() > local_peak {
                    local_peak = a2.abs();
                }
            }
            i2 += 1;
        }
        let r1 = r2;
        r2 = r3;
        r3 = if product != 0.0 { product / libm::sqrt(norm1 * norm2) } else { 0.0 };
        if r2 > max_corr && r2 >= r1 && r2 >= r3 {
            r1_best = r1;
            max_corr = r2;
            r3_best = r3;
            ir = (ileft2 - 1) as f64;
            peak = local_peak;
        }
    }
    let mut tout = None;
    if max_corr > -1.0 {
        let d2r = 2.0 * max_corr - r1_best - r3_best;
        if d2r != 0.0 {
            let dr = 0.5 * (r3_best - r1_best);
            max_corr += 0.5 * dr * dr / d2r;
            ir += dr / d2r;
        }
        tout = Some(t1 + (ir - ileft1 as f64) * s.dx());
    }
    (max_corr, tout, peak)
}

/// `Pitch_getVoicedIntervalAfter`.
fn voiced_interval_after(p: &Pitch, after: f64) -> Option<(f64, f64)> {
    let nx = p.num_frames() as isize;
    let (x1, dx) = (p.start_time(), p.time_step());
    let mut ileft = libm::ceil((after - x1) / dx + 1.0) as isize; // Sampled_xToHighIndex
    if ileft > nx {
        return None;
    }
    if ileft < 1 {
        ileft = 1;
    }
    let voiced = |i: isize| p.get_value_at_frame((i - 1) as usize).is_some();
    while ileft <= nx && !voiced(ileft) {
        ileft += 1;
    }
    if ileft > nx {
        return None;
    }
    let mut iright = ileft;
    while iright <= nx && voiced(iright) {
        iright += 1;
    }
    iright -= 1;
    let mut tleft = x1 + (ileft - 1) as f64 * dx - 0.5 * dx;
    let mut tright = x1 + (iright - 1) as f64 * dx + 0.5 * dx;
    if tleft >= p.xmax() - 0.5 * dx {
        return None;
    }
    if tleft < p.xmin() {
        tleft = p.xmin();
    }
    if tright > p.xmax() {
        tright = p.xmax();
    }
    if tright <= after {
        return None;
    }
    Some((tleft, tright))
}

impl PointProcess {
    /// Praat's `Sound & Pitch: To PointProcess (cc)` (mono sound).
    pub fn from_sound_pitch_cc(sound: &Sound, pitch: &Pitch) -> Self {
        let mut point = Self::new(sound.start_time(), sound.end_time());
        let f0_at = |t: f64| pitch.get_value_at_time(t, PitchUnit::Hertz, Interpolation::Linear);
        // Vector_getAbsoluteExtremum (sound, xmin, xmax, NONE)
        let global_peak = sound.samples().iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        let mut t = pitch.xmin();
        let mut added_right = -1e308;
        while let Some((tleft, tright)) = voiced_interval_after(pitch, t) {
            let tmiddle = (tleft + tright) / 2.0;
            let Some(f0middle) = f0_at(tmiddle) else { break };
            let mut tmax = sound_find_extremum(sound, tmiddle - 0.5 / f0middle, tmiddle + 0.5 / f0middle, true, true);
            point.add_point(tmax);
            let tsave = tmax;
            // leftwards
            loop {
                let Some(f0) = f0_at(tmax) else { break };
                let (corr, tout, peak) =
                    sound_find_maximum_correlation(sound, tmax, 1.0 / f0, tmax - 1.25 / f0, tmax - 0.8 / f0);
                if let Some(to) = tout {
                    tmax = to;
                }
                if corr == -1.0 {
                    tmax -= 1.0 / f0;
                }
                if tmax < tleft {
                    if corr > 0.7 && peak > 0.023333 * global_peak && tmax - added_right > 0.8 / f0 {
                        point.add_point(tmax);
                    }
                    break;
                }
                if corr > 0.3 && (peak == 0.0 || peak > 0.01 * global_peak) && tmax - added_right > 0.8 / f0 {
                    point.add_point(tmax);
                }
            }
            tmax = tsave;
            // rightwards
            loop {
                let Some(f0) = f0_at(tmax) else { break };
                let (corr, tout, peak) =
                    sound_find_maximum_correlation(sound, tmax, 1.0 / f0, tmax + 0.8 / f0, tmax + 1.25 / f0);
                if let Some(to) = tout {
                    tmax = to;
                }
                if corr == -1.0 {
                    tmax += 1.0 / f0;
                }
                if tmax > tright {
                    if corr > 0.7 && peak > 0.023333 * global_peak {
                        point.add_point(tmax);
                        added_right = tmax;
                    }
                    break;
                }
                if corr > 0.3 && (peak == 0.0 || peak > 0.01 * global_peak) {
                    point.add_point(tmax);
                    added_right = tmax;
                }
            }
            t = tright;
        }
        point
    }
}

impl Sound {
    /// Praat's `Sound: To PointProcess (periodic, cc)`: a default pitch
    /// analysis (`To Pitch`, time step 0) followed by
    /// [`PointProcess::from_sound_pitch_cc`].
    pub fn to_point_process_periodic_cc(&self, pitch_floor: f64, pitch_ceiling: f64) -> PointProcess {
        let pitch = self.to_pitch(0.0, pitch_floor, pitch_ceiling);
        PointProcess::from_sound_pitch_cc(self, &pitch)
    }
}

/// Praat's `Voice report` (Sound + Pitch + PointProcess). `from_time ==
/// to_time` means the whole sound. Defaults in Praat's dialog: floor 75,
/// ceiling 600, maximum period factor 1.3, maximum amplitude factor 1.6,
/// silence threshold 0.03, voicing threshold 0.45.
#[allow(clippy::too_many_arguments)]
pub fn voice_report(
    sound: &Sound,
    pitch: &Pitch,
    pulses: &PointProcess,
    mut from_time: f64,
    mut to_time: f64,
    floor: f64,
    ceiling: f64,
    max_period_factor: f64,
    max_amplitude_factor: f64,
    silence_threshold: f64,
    voicing_threshold: f64,
) -> VoiceReport {
    if from_time >= to_time {
        from_time = sound.start_time();
        to_time = sound.end_time();
    }
    let (tmin, tmax) = (from_time, to_time);
    let (pmin, pmax) = (0.8 / ceiling, 1.25 / floor);
    let (first, last) = pulses.window_points(tmin, tmax);
    let (unv, tot) = pitch.fraction_of_locally_unvoiced_frames(tmin, tmax, ceiling, silence_threshold, voicing_threshold);
    let (breaks, break_dur, break_den) = pulses.voice_breaks(tmin, tmax, pmax);
    VoiceReport {
        from_time: tmin,
        to_time: tmax,
        median_pitch: pitch.get_quantile(tmin, tmax, 0.5, PitchUnit::Hertz),
        mean_pitch: pitch.get_mean(tmin, tmax, PitchUnit::Hertz),
        stdev_pitch: pitch.get_standard_deviation(tmin, tmax, PitchUnit::Hertz),
        minimum_pitch: pitch.get_minimum(tmin, tmax, PitchUnit::Hertz, true),
        maximum_pitch: pitch.get_maximum(tmin, tmax, PitchUnit::Hertz, true),
        number_of_pulses: (last - first + 1).max(0) as usize,
        number_of_periods: pulses.get_number_of_periods(tmin, tmax, pmin, pmax, max_period_factor),
        mean_period: pulses.get_mean_period(tmin, tmax, pmin, pmax, max_period_factor),
        stdev_period: pulses.get_stdev_period(tmin, tmax, pmin, pmax, max_period_factor),
        unvoiced_frames: unv,
        total_frames: tot,
        fraction_of_locally_unvoiced_frames: if tot > 0 { Some(unv as f64 / tot as f64) } else { None },
        number_of_voice_breaks: breaks,
        voice_breaks_duration: break_dur,
        degree_of_voice_breaks: if break_den > 0.0 { Some(break_dur / break_den) } else { None },
        jitter_local: pulses.get_jitter_local(tmin, tmax, pmin, pmax, max_period_factor),
        jitter_local_absolute: pulses.get_jitter_local_absolute(tmin, tmax, pmin, pmax, max_period_factor),
        jitter_rap: pulses.get_jitter_rap(tmin, tmax, pmin, pmax, max_period_factor),
        jitter_ppq5: pulses.get_jitter_ppq5(tmin, tmax, pmin, pmax, max_period_factor),
        jitter_ddp: pulses.get_jitter_ddp(tmin, tmax, pmin, pmax, max_period_factor),
        shimmer: pulses.get_shimmer(sound, tmin, tmax, pmin, pmax, max_period_factor, max_amplitude_factor),
        mean_autocorrelation: pitch.get_mean_strength(tmin, tmax, StrengthUnit::Autocorrelation),
        mean_noise_to_harmonics_ratio: pitch.get_mean_strength(tmin, tmax, StrengthUnit::NoiseToHarmonicsRatio),
        mean_harmonics_to_noise_ratio: pitch.get_mean_strength(tmin, tmax, StrengthUnit::HarmonicsToNoiseDb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_point_sorts_and_drops_duplicates() {
        let p = PointProcess::from_times(0.0, 1.0, &[0.3, 0.1, 0.2, 0.2, 0.05]);
        assert_eq!(p.times(), &[0.05, 0.1, 0.2, 0.3]);
        assert_eq!(p.low_index(0.2), 3);
        assert_eq!(p.high_index(0.2), 3);
        assert_eq!(p.high_index(0.21), 4);
        assert_eq!(p.low_index(0.01), 0);
    }

    #[test]
    fn jitter_of_alternating_periods() {
        // periods 0.010, 0.011, 0.010, ... (20 periods)
        let mut t = vec![0.1];
        for k in 0..20 {
            let last = *t.last().unwrap();
            t.push(last + if k % 2 == 0 { 0.010 } else { 0.011 });
        }
        let p = PointProcess::from_times(0.0, 1.0, &t);
        let abs = p.get_jitter_local_absolute(0.0, 0.0, 0.0001, 0.02, 1.3).unwrap();
        assert!((abs - 0.001).abs() < 1e-12, "{abs}");
        let mean = p.get_mean_period(0.0, 0.0, 0.0001, 0.02, 1.3).unwrap();
        assert!((mean - 0.0105).abs() < 1e-12, "{mean}");
        let local = p.get_jitter_local(0.0, 0.0, 0.0001, 0.02, 1.3).unwrap();
        assert!((local - 0.001 / 0.0105).abs() < 1e-12);
        // ddp is 3 x rap by definition
        let rap = p.get_jitter_rap(0.0, 0.0, 0.0001, 0.02, 1.3).unwrap();
        let ddp = p.get_jitter_ddp(0.0, 0.0, 0.0001, 0.02, 1.3).unwrap();
        assert!((ddp - 3.0 * rap).abs() < 1e-15);
        // too few pulses: undefined
        let short = PointProcess::from_times(0.0, 1.0, &[0.1, 0.11]);
        assert!(short.get_jitter_local(0.0, 0.0, 0.0001, 0.02, 1.3).is_none());
    }
}
