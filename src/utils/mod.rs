//! Utility modules for signal processing
//!
//! This module contains low-level signal processing utilities used by
//! the analysis algorithms.

pub mod fft;
pub mod fft_fftpack;
pub mod lpc;

pub use fft::{Fft, FftDirection};
pub use lpc::{lpc_burg, lpc_to_formants};

/// Praat's `NUMfpp -> eps`: LAPACK `dlamch ("Epsilon")`, the relative
/// machine precision 2^-53 (half of `f64::EPSILON`) on IEEE doubles with
/// rounding. Used by Brent's method and root polishing.
pub(crate) const PRAAT_EPS: f64 = f64::EPSILON / 2.0;

/// Neumaier-compensated accumulator: stands in for Praat's `longdouble`
/// (80-bit on x86-64) running sums. Terms are f64 (Praat rounds each product
/// to double before adding it to the long-double sum); the accumulation error
/// is far below the 64-bit mantissa of x87 long double.
#[derive(Clone, Copy, Default)]
pub(crate) struct CompSum {
    s: f64,
    c: f64,
}

impl CompSum {
    #[inline(always)]
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.s + x;
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }
    #[inline(always)]
    pub(crate) fn value(self) -> f64 {
        self.s + self.c
    }
}


/// Mean as Praat's `NUMmean` computes it: a long-double sum divided by `n`,
/// rounded to double. Emulated with a compensated (two-sum) accumulation and
/// an FMA-corrected division, so the result is the correctly rounded mean
/// in all but pathological cases -- in particular, a constant run returns
/// exactly that constant (a naive f64 mean of 0.2 repeated does not, which
/// leaves ~1e-17 residue after centring where Praat has exact zeros).
pub(crate) fn praat_mean(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &v in x {
        let t = s + v;
        let z = t - s;
        c += (s - (t - z)) + (v - z);
        s = t;
    }
    let nf = n as f64;
    let q = s / nf;
    let rem = libm::fma(-q, nf, s); // exact: s - q * n
    q + (rem + c) / nf
}

#[cfg(test)]
mod mean_tests {
    use super::praat_mean;

    #[test]
    fn constant_run_is_exact() {
        for &c in &[0.2, 0.1, -0.7, 1.0 / 3.0, 1e-5] {
            for n in [1usize, 3, 7, 100, 1001, 12345] {
                let v = vec![c; n];
                assert_eq!(praat_mean(&v), c, "c={c} n={n}");
            }
        }
    }
}

/// Praat's `Sampled_xToLowIndex` as a 0-based index:
/// `Melder_ifloor ((x - x1) / dx + 1.0) - 1`.
///
/// The `+ 1.0` comes before the floor on purpose. When `(x - x1) / dx`
/// falls a few ulps below an integer (a frame centre on a sample tie),
/// adding 1.0 rounds it up, and Praat takes the next sample.
/// `floor((x - x1) / dx)` would take the one before.
#[inline]
pub fn praat_low_index0(x: f64, x1: f64, dx: f64) -> isize {
    libm::floor((x - x1) / dx + 1.0) as isize - 1
}
