//! FFTPACK-derived real-FFT kernels, ported from speexdsp-rs (MIT) and
//! promoted from f32 to f64 to match Praat's `NUMfft_core.h`.
//!
//! Praat's `dwsys/NUMfft_core.h` is a manual port of the original Ogg/Vorbis
//! `vorbis_smallft.cpp` (which is itself a C port of FFTPACK's `drftf1` /
//! `drftb1`, originally in Fortran by Paul Swarztrauber at NCAR). Praat's
//! edit (djmw, 2004-05-11) was to change all local `float` variables to
//! `double` to raise numerical precision.
//!
//! speexdsp-rs's `fft/src/{smallft,dradb,dradf}.rs` is a c2rust translation
//! of the SAME upstream C (via speexdsp, which shares the vorbis smallft).
//! By vendoring those three files and mechanically promoting `f32 → f64` we
//! obtain an algorithm that matches Praat's edit bit-for-bit.
//!
//! Why this matters: rustfft (Cooley-Tukey split-radix) produces the same
//! DFT as FFTPACK's mixed-radix routine up to rounding, but individual bin
//! values differ numerically. For `Sound::fft_lowpass_filter` that zeroes
//! high-frequency bins and IFFTs back, these rounding differences propagate
//! into ~1e-5 per-sample deviations — directly causing the residual ~0.3 Hz
//! formant gap vs Praat's FormantPath output.
//!
//! Packed output layout (drftf1 forward FFT):
//!
//! ```text
//! data[0] = DC
//! data[1] = Re(bin 1)      data[2] = Im(bin 1)
//! data[3] = Re(bin 2)      data[4] = Im(bin 2)
//! ...
//! data[n-2] = Re(bin n/2-1)  data[n-1] = Im(bin n/2-1)    (or Nyquist for even n)
//! ```
//!
//! Praat's `NUMforwardRealFastFourierTransform` rotates this so that
//! `data[1]` (0-based) becomes Nyquist — our wrapper does the same rotation
//! so that callers see Praat's 1-based packed layout:
//!
//! ```text
//! data[1] = DC (was data[0])
//! data[2] = Nyquist (was data[n-1] for even n)
//! data[3] = Re(bin 1), data[4] = Im(bin 1), ...
//! ```

pub mod dradb;
pub mod dradf;
pub mod smallft;

/// Reusable trigonometric cache for real-FFT calls on a fixed length `n`.
/// Computing the cache is ~O(n); amortize across multiple FFTs of same size.
pub struct RealFftPlan {
    n: usize,
    /// Full 3n buffer: [0..n] is `ch` scratch, [n..3n] is the `wa` twiddle factors
    /// populated by `fdrffti`. Matches speexdsp-rs's `DrftLookup::trigcache` layout.
    trigcache: Vec<f64>,
    /// Copy of `trigcache[n..]` taken once, so the kernels can read twiddles
    /// while `trigcache[..n]` serves as their `ch` scratch.
    wa: Vec<f64>,
    splitcache: Vec<i32>,
}

impl RealFftPlan {
    pub fn new(n: usize) -> Self {
        let mut plan = Self {
            n,
            trigcache: vec![0.0_f64; 3 * n.max(1)],
            wa: Vec::new(),
            splitcache: vec![0_i32; 32],
        };
        smallft::fdrffti(n, &mut plan.trigcache, &mut plan.splitcache);
        plan.wa = plan.trigcache[n.min(plan.trigcache.len())..].to_vec();
        plan
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Forward real FFT, in place, on `data` of length `n`. Output follows
    /// `drftf1`'s packed layout: `[DC, Re1, Im1, Re2, Im2, ..., Nyquist]`
    /// (length `n`). NOT Praat's rotated layout — use [`realft_forward_praat`]
    /// for that.
    pub fn forward(&mut self, data: &mut [f64]) {
        debug_assert_eq!(data.len(), self.n);
        if self.n <= 1 {
            return;
        }
        // `ch` must be a separate scratch of length ≥ n. drftf1 passes data
        // around via both c and ch, so the twiddles live in their own copy
        // (`wa`) to avoid aliasing the scratch region.
        smallft::drftf1(
            self.n as i32,
            data,
            &mut self.trigcache,
            &mut self.wa,
            &mut self.splitcache,
        );
    }

    /// Backward (inverse) real FFT, in place. Output NOT normalized — caller
    /// must divide by `n` to recover the original signal.
    pub fn backward(&mut self, data: &mut [f64]) {
        debug_assert_eq!(data.len(), self.n);
        if self.n <= 1 {
            return;
        }
        smallft::drftb1(
            self.n as i32,
            data,
            &mut self.trigcache,
            &mut self.wa,
            &mut self.splitcache,
        );
    }
}

/// Praat `NUMforwardRealFastFourierTransform` — forward FFT followed by the
/// Nyquist-to-index-1 rotation that gives the 1-based packed layout Praat's
/// downstream code expects.
///
/// In-place on `data` (length n, 0-based buffer of n f64). After return,
/// `data` contains `[DC, Nyquist, Re1, Im1, Re2, Im2, ...]` — 0-based.
/// (Praat's code indexes this 1-based as `data[1]=DC, data[2]=Nyquist, ...`.)
pub fn realft_forward_praat(plan: &mut RealFftPlan, data: &mut [f64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    plan.forward(data);
    // Praat: tmp = data[size]; for i = size downto 3 (1-based): data[i]=data[i-1]; data[2]=tmp.
    // In 0-based: tmp = data[n-1]; for i = n-1 downto 2: data[i]=data[i-1]; data[1]=tmp.
    let tmp = data[n - 1];
    for i in (2..n).rev() {
        data[i] = data[i - 1];
    }
    data[1] = tmp;
}

/// Praat `NUMreverseRealFastFourierTransform` — inverse rotation + backward
/// FFT. Output NOT normalized: caller must divide by `n`.
pub fn realft_backward_praat(plan: &mut RealFftPlan, data: &mut [f64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    // Un-rotate: tmp = data[1] (0-based); for i = 1 to n-2: data[i]=data[i+1]; data[n-1]=tmp.
    let tmp = data[1];
    for i in 1..n - 1 {
        data[i] = data[i + 1];
    }
    data[n - 1] = tmp;
    plan.backward(data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let n = 32;
        let original: Vec<f64> = (0..n).map(|i| libm::sin(i as f64 * 0.1)).collect();
        let mut data = original.clone();
        let mut plan = RealFftPlan::new(n);
        plan.forward(&mut data);
        plan.backward(&mut data);
        for (i, (&o, &r)) in original.iter().zip(data.iter()).enumerate() {
            let scaled = r / n as f64;
            assert!(
                (o - scaled).abs() < 1e-13,
                "mismatch at {}: orig={} after-roundtrip/n={}",
                i,
                o,
                scaled
            );
        }
    }

    /// Every length 1..=700, not just powers of two: odd and prime factors
    /// take the general-radix paths (dradfg/dradbg), which had two c2rust
    /// translation bugs (forward panic at n = 9, 18, 27, ...; inverse wrong
    /// for any odd factor).
    #[test]
    fn test_all_lengths_against_direct_dft() {
        for n in 1..=700usize {
            let x: Vec<f64> = (0..n)
                .map(|i| libm::sin(0.37 * i as f64) + 0.5 * libm::cos(1.3 * i as f64 + 0.2))
                .collect();
            let mut data = x.clone();
            let mut plan = RealFftPlan::new(n);
            plan.forward(&mut data);

            // Direct DFT in drftf1's packed layout [DC, Re1, Im1, ..., (Nyquist)].
            let mut expected = vec![0.0; n];
            for k in 0..=n / 2 {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, &v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
                    re += v * libm::cos(a);
                    im += v * libm::sin(a);
                }
                if k == 0 {
                    expected[0] = re;
                } else if 2 * k == n {
                    expected[n - 1] = re;
                } else {
                    expected[2 * k - 1] = re;
                    expected[2 * k] = im;
                }
            }
            let scale = expected.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for i in 0..n {
                assert!(
                    (data[i] - expected[i]).abs() <= 1e-10 * scale,
                    "forward n={} bin {}: got {}, expected {}",
                    n, i, data[i], expected[i]
                );
            }

            plan.backward(&mut data);
            for i in 0..n {
                assert!(
                    (data[i] / n as f64 - x[i]).abs() <= 1e-10,
                    "roundtrip n={} sample {}: got {}, expected {}",
                    n, i, data[i] / n as f64, x[i]
                );
            }
        }
    }

    #[test]
    fn test_dc_component() {
        let n = 16;
        let mut data = vec![1.0_f64; n];
        let mut plan = RealFftPlan::new(n);
        plan.forward(&mut data);
        assert!((data[0] - n as f64).abs() < 1e-10, "DC bin = {}", data[0]);
        for i in 1..n {
            assert!(data[i].abs() < 1e-10, "bin {} should be 0, got {}", i, data[i]);
        }
    }

    #[test]
    fn test_praat_rotation_roundtrip() {
        let n = 32;
        let original: Vec<f64> = (0..n).map(|i| libm::cos(i as f64 * 0.2)).collect();
        let mut data = original.clone();
        let mut plan = RealFftPlan::new(n);
        realft_forward_praat(&mut plan, &mut data);
        realft_backward_praat(&mut plan, &mut data);
        for (i, (&o, &r)) in original.iter().zip(data.iter()).enumerate() {
            let scaled = r / n as f64;
            assert!(
                (o - scaled).abs() < 1e-12,
                "mismatch at {}: orig={} after-roundtrip/n={}",
                i, o, scaled
            );
        }
    }
}
