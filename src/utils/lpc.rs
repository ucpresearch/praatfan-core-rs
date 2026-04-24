//! Linear Predictive Coding (LPC) utilities
//!
//! This module provides LPC analysis using Burg's method, which is used
//! for formant estimation. The implementation matches Praat's VECburg
//! and polynomial root-finding exactly.

use num_complex::Complex;
use std::f64::consts::PI;

use faer::Mat as FaerMat;

/// Kahan-Neumaier compensated addition.
///
/// Updates `sum` and the compensation term `c` to accumulate `value` with
/// error smaller than a single f64 ulp, emulating what Praat gets "for free"
/// from its `longdouble` accumulators in VECburg / its sum-of-squares loops.
#[inline(always)]
fn kahan_add(sum: &mut f64, c: &mut f64, value: f64) {
    let t = *sum + value;
    if sum.abs() >= value.abs() {
        *c += (*sum - t) + value;
    } else {
        *c += (value - t) + *sum;
    }
    *sum = t;
}

/// Result of LPC analysis
#[derive(Debug, Clone)]
pub struct LpcResult {
    /// LPC coefficients (matching Praat's convention: a[1..m], no leading 1.0)
    pub coefficients: Vec<f64>,
    /// Prediction error (xms)
    pub gain: f64,
}

/// Compute LPC coefficients using Burg's method
///
/// This implementation matches Praat's VECburg exactly.
/// Returns coefficients a[1..m] where the LPC polynomial is:
/// A(z) = 1 + a[1]*z^-1 + a[2]*z^-2 + ... + a[m]*z^-m
///
/// # Arguments
/// * `samples` - Input signal samples (1-indexed internally to match Praat)
/// * `order` - LPC order (number of poles)
///
/// # Returns
/// LPC result containing coefficients and prediction error
pub fn lpc_burg(samples: &[f64], order: usize) -> Option<LpcResult> {
    let n = samples.len();
    let m = order;

    // Initialize coefficients to 0
    let mut a = vec![0.0; m];

    if n <= 2 {
        if !a.is_empty() {
            a[0] = -1.0;
        }
        let xms = if n == 2 {
            0.5 * (samples[0] * samples[0] + samples[1] * samples[1])
        } else if n == 1 {
            samples[0] * samples[0]
        } else {
            0.0
        };
        return Some(LpcResult {
            coefficients: a,
            gain: xms,
        });
    }

    // Initialize forward (b1) and backward (b2) prediction errors
    // Praat uses 1-based indexing, we'll use 0-based but follow same logic
    let mut b1 = vec![0.0; n];
    let mut b2 = vec![0.0; n];

    // Compute initial power using Kahan-Neumaier compensated summation.
    // This matches Praat's `longdouble p` accumulator precision on
    // platforms with 80-bit long double (IA-32/x86-64 Linux), which was
    // the source of ~1-Hz-scale formant divergences in long frames.
    let mut p_sum = 0.0_f64;
    let mut p_c = 0.0_f64;
    for j in 0..n {
        let term = samples[j] * samples[j];
        kahan_add(&mut p_sum, &mut p_c, term);
    }
    let p = p_sum + p_c;

    let mut xms = p / n as f64;
    if xms <= 0.0 {
        return Some(LpcResult {
            coefficients: a,
            gain: xms,
        });
    }

    // Initialize b1 and b2 (Praat's initialization pattern)
    // b1[0] = x[0]
    // b2[n-2] = x[n-1]
    // for j = 1 to n-2: b1[j] = b2[j-1] = x[j]
    b1[0] = samples[0];
    b2[n - 2] = samples[n - 1];
    for j in 1..n - 1 {
        b1[j] = samples[j];
        b2[j - 1] = samples[j];
    }

    // Storage for previous coefficients
    let mut aa = vec![0.0; m];

    for i in 0..m {
        // Compute reflection coefficient (Praat uses positive sign: 2*num/den)
        // Use Kahan-Neumaier summation for `num` and `den` accumulators —
        // matches Praat's `longdouble num, denum`.
        let mut num = 0.0_f64;
        let mut num_c = 0.0_f64;
        let mut den = 0.0_f64;
        let mut den_c = 0.0_f64;
        for j in 0..n - i - 1 {
            kahan_add(&mut num, &mut num_c, b1[j] * b2[j]);
            kahan_add(&mut den, &mut den_c, b1[j] * b1[j] + b2[j] * b2[j]);
        }
        let num = num + num_c;
        let den = den + den_c;

        if den <= 0.0 {
            return Some(LpcResult {
                coefficients: a,
                gain: 0.0,
            });
        }

        // Praat: a[i] = 2.0 * num / denum (positive!)
        a[i] = 2.0 * num / den;
        xms *= 1.0 - a[i] * a[i];

        // Update previous coefficients: a[j] = aa[j] - a[i] * aa[i-j-1]
        // (Praat's indexing: for j = 1 to i-1: a[j] = aa[j] - a[i] * aa[i-j])
        for j in 0..i {
            a[j] = aa[j] - a[i] * aa[i - j - 1];
        }

        if i < m - 1 {
            // Save coefficients for next iteration
            for j in 0..=i {
                aa[j] = a[j];
            }
            // Update prediction errors
            // Praat: for j = 1 to n-i-2: b1[j] -= aa[i]*b2[j]; b2[j] = b2[j+1] - aa[i]*b1[j+1]
            for j in 0..n - i - 2 {
                b1[j] -= aa[i] * b2[j];
                b2[j] = b2[j + 1] - aa[i] * b1[j + 1];
            }
        }
    }

    // Note: Praat's Sound_and_LPC.cpp negates coefficients at the end,
    // but VECburg (used by Sound_to_Formant.cpp) may not.
    // The lpc_to_formants function will handle the polynomial sign convention.
    // For now, return coefficients as-is (positive convention from Burg algorithm).

    Some(LpcResult {
        coefficients: a,
        gain: xms,
    })
}

/// A formant (resonance) with frequency and bandwidth
#[derive(Debug, Clone, Copy)]
pub struct FormantCandidate {
    /// Frequency in Hz
    pub frequency: f64,
    /// Bandwidth in Hz
    pub bandwidth: f64,
}

/// Extract formant frequencies from LPC coefficients
///
/// This matches Praat's burg() function in Sound_to_Formant.cpp:
/// 1. Convert LPC coefficients to polynomial (negate and reverse)
/// 2. Find polynomial roots using eigenvalue method
/// 3. Fix roots into unit circle
/// 4. Extract formants from roots with positive imaginary part
///
/// # Arguments
/// * `coefficients` - LPC coefficients from lpc_burg (a[0..m-1])
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Vector of formant candidates sorted by frequency
pub fn lpc_to_formants(coefficients: &[f64], sample_rate: f64) -> Vec<FormantCandidate> {
    if coefficients.is_empty() {
        return Vec::new();
    }

    let m = coefficients.len();
    let nyquist = sample_rate / 2.0;

    // Construct polynomial coefficients (Praat's convention from Sound_to_Formant.cpp):
    // polynomial[i] = -coefficients[size - i + 1] for i = 1..size (1-indexed)
    // polynomial[size + 1] = 1.0
    // Converting to 0-indexed:
    // poly_coeffs[i] = -coefficients[m - 1 - i] for i = 0..m-1
    // poly_coeffs[m] = 1.0
    let mut poly_coeffs = vec![0.0; m + 1];
    for i in 0..m {
        poly_coeffs[i] = -coefficients[m - 1 - i];
    }
    poly_coeffs[m] = 1.0;

    // Find roots using eigenvalue method (companion matrix)
    let mut roots = find_polynomial_roots_eigen(&poly_coeffs);

    // Polish roots using Newton-Raphson (matches Praat's Roots_Polynomial_polish)
    polish_roots(&poly_coeffs, &mut roots);

    // Fix roots into unit circle (Praat's Roots_fixIntoUnitCircle)
    // Praat: roots[i] = 1.0 / conj(roots[i]) = roots[i] / |roots[i]|^2
    // This keeps the sign of the imaginary part (important for formant extraction)
    for root in &mut roots {
        let mag = root.norm();
        if mag > 1.0 {
            // Map to 1/conj(z) = z / |z|^2 (NOT conj(z) / |z|^2)
            *root = *root / (mag * mag);
        }
    }

    // Convert roots to formants (Praat's method)
    let safety_margin = 50.0; // Praat uses 50 Hz safety margin
    let mut formants = Vec::new();

    for root in &roots {
        // Only consider roots with non-negative imaginary part (upper half-plane)
        if root.im < 0.0 {
            continue;
        }

        // Frequency from angle: f = |atan2(im, re)| * nyquist / π
        let frequency = root.im.atan2(root.re).abs() * nyquist / PI;

        // Apply safety margin (Praat: f >= 50 && f <= nyquist - 50)
        if frequency < safety_margin || frequency > nyquist - safety_margin {
            continue;
        }

        // Bandwidth from squared magnitude: bw = -log(|z|²) * nyquist / π.
        // Praat uses C++ `norm(complex)` which returns |z|² (NOT |z|), so the
        // Rust `root.norm()` (magnitude) gives HALF of Praat's bandwidth. See
        // LPC/Roots_and_Formant.cpp:49 and test in formant_modeler — bandwidth
        // is the fit weight for FormantPath stress and must match Praat.
        let norm_sq = root.re * root.re + root.im * root.im;
        let bandwidth = if norm_sq > 0.0 {
            -(norm_sq.ln()) * nyquist / PI
        } else {
            nyquist // Maximum bandwidth
        };

        formants.push(FormantCandidate {
            frequency,
            bandwidth,
        });
    }

    // Sort by frequency
    formants.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());

    formants
}

/// Evaluate polynomial and its derivative at a complex point using Horner's method
///
/// This matches Praat's Polynomial_evaluateWithDerivative_z from Roots.cpp
fn evaluate_polynomial_with_derivative(
    coefficients: &[f64],
    z: Complex<f64>,
) -> (Complex<f64>, Complex<f64>) {
    let n = coefficients.len();
    if n == 0 {
        return (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
    }

    // coefficients[0] is constant term, coefficients[n-1] is leading coefficient
    // Horner's method: p(z) = c[n-1]*z^(n-1) + ... + c[1]*z + c[0]
    //                       = ((...((c[n-1])*z + c[n-2])*z + ...)*z + c[1])*z + c[0]
    let x = z.re;
    let y = z.im;

    // Start with leading coefficient
    let mut pr = coefficients[n - 1];
    let mut pi = 0.0;
    let mut dpr = 0.0;
    let mut dpi = 0.0;

    for i in (0..n - 1).rev() {
        // Update derivative: dp = dp * z + p
        let tr = dpr;
        dpr = dpr * x - dpi * y + pr;
        dpi = tr * y + dpi * x + pi;

        // Update value: p = p * z + c[i]
        let tr = pr;
        pr = pr * x - pi * y + coefficients[i];
        pi = tr * y + pi * x;
    }

    (Complex::new(pr, pi), Complex::new(dpr, dpi))
}

/// Polish a complex root using Newton-Raphson iteration
///
/// This matches Praat's Polynomial_polish_complexroot_nr from Roots.cpp
fn polish_complex_root(coefficients: &[f64], root: &mut Complex<f64>, max_iter: usize) {
    let eps = 1e-15; // Machine epsilon approximation
    let mut best = *root;
    let mut min_residual = f64::MAX;

    for _ in 0..max_iter {
        let (p, dp) = evaluate_polynomial_with_derivative(coefficients, *root);
        let residual = p.norm();

        // Stop if residual is increasing or not improving
        if residual >= min_residual || (min_residual - residual).abs() < eps {
            *root = best;
            return;
        }

        min_residual = residual;
        best = *root;

        // Stop if derivative is zero (can't continue)
        if dp.norm() == 0.0 {
            return;
        }

        // Newton-Raphson step: root = root - p(root) / p'(root)
        *root = *root - p / dp;
    }
}

/// Polish a real root using Newton-Raphson iteration
///
/// This matches Praat's Polynomial_polish_realroot from Roots.cpp
fn polish_real_root(coefficients: &[f64], root: &mut f64, max_iter: usize) {
    let eps = 1e-15;
    let mut best = *root;
    let mut min_residual = f64::MAX;

    for _ in 0..max_iter {
        // Evaluate polynomial and derivative at real root
        let n = coefficients.len();
        if n == 0 {
            return;
        }

        let mut p = coefficients[n - 1];
        let mut dp = 0.0;

        for i in (0..n - 1).rev() {
            dp = dp * (*root) + p;
            p = p * (*root) + coefficients[i];
        }

        let residual = p.abs();

        if residual >= min_residual || (min_residual - residual).abs() < eps {
            *root = best;
            return;
        }

        min_residual = residual;
        best = *root;

        if dp.abs() == 0.0 {
            return;
        }

        *root = *root - p / dp;
    }
}

/// Polish all roots using Newton-Raphson iteration
///
/// This matches Praat's Roots_Polynomial_polish from Roots.cpp
/// Complex roots come in conjugate pairs, so we only need to polish
/// one of each pair and mirror the result.
fn polish_roots(coefficients: &[f64], roots: &mut [Complex<f64>]) {
    const MAX_ITER: usize = 80;

    let mut i = 0;
    while i < roots.len() {
        let im = roots[i].im;
        let re = roots[i].re;

        if im.abs() > 1e-15 {
            // Complex root - polish it
            polish_complex_root(coefficients, &mut roots[i], MAX_ITER);

            // Check if next root is conjugate pair
            if i + 1 < roots.len() {
                let next_im = roots[i + 1].im;
                let next_re = roots[i + 1].re;
                if (next_im + im).abs() < 1e-10 && (next_re - re).abs() < 1e-10 {
                    // Mirror the polished result to conjugate
                    roots[i + 1] = roots[i].conj();
                    i += 1;
                }
            }
        } else {
            // Real root
            let mut real_root = roots[i].re;
            polish_real_root(coefficients, &mut real_root, MAX_ITER);
            roots[i] = Complex::new(real_root, 0.0);
        }

        i += 1;
    }
}

/// Find polynomial roots using the companion matrix eigenvalue method
///
/// This matches Praat's Polynomial_to_Roots which uses LAPACK's dhseqr
/// (QR algorithm on Hessenberg matrix). We use our own QR implementation
/// with bounded iteration count, which is WASM-compatible and avoids
/// potential hangs on pathological matrices.
fn find_polynomial_roots_eigen(coefficients: &[f64]) -> Vec<Complex<f64>> {
    let n = coefficients.len() - 1; // Degree of polynomial

    if n == 0 {
        return Vec::new();
    }

    let leading = coefficients[n];
    if leading.abs() < 1e-15 {
        // Reduce degree
        return find_polynomial_roots_eigen(&coefficients[..n]);
    }

    // Normalize coefficients
    let normalized: Vec<f64> = coefficients.iter().map(|&c| c / leading).collect();

    if n == 1 {
        // Linear: c0 + c1*z = 0 => z = -c0/c1 = -c0 (since c1 = 1)
        return vec![Complex::new(-normalized[0], 0.0)];
    }

    if n == 2 {
        // Quadratic
        let a = 1.0;
        let b = normalized[1];
        let c = normalized[0];
        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let sqrt_d = discriminant.sqrt();
            return vec![
                Complex::new((-b + sqrt_d) / 2.0, 0.0),
                Complex::new((-b - sqrt_d) / 2.0, 0.0),
            ];
        } else {
            let sqrt_d = (-discriminant).sqrt();
            return vec![
                Complex::new(-b / 2.0, sqrt_d / 2.0),
                Complex::new(-b / 2.0, -sqrt_d / 2.0),
            ];
        }
    }

    // Build companion matrix for polynomial c0 + c1*z + ... + c(n-1)*z^(n-1) + z^n
    // Praat uses the form from Polynomial_to_Roots in Roots.cpp:
    // [  0   0  ...  0  -c0   ]
    // [  1   0  ...  0  -c1   ]
    // [  0   1  ...  0  -c2   ]
    // ...
    // [  0   0  ...  1  -c(n-1)]
    //
    // Eigenvalues via faer's Hessenberg + Francis QR, which is the same
    // algorithm family LAPACK's dhseqr uses and gives precision ~= machine
    // epsilon on the eigenvalues. faer is pure Rust and WASM-compatible.
    let comp = FaerMat::<f64>::from_fn(n, n, |i, j| {
        if j == n - 1 {
            -normalized[i]
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });
    match comp.eigenvalues() {
        Ok(ev) => ev,
        Err(_) => Vec::new(), // degenerate matrix — caller gets no formants for this frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lpc_burg_basic() {
        // Simple test signal
        let samples: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 8000.0).sin())
            .collect();

        let result = lpc_burg(&samples, 10).unwrap();
        assert_eq!(result.coefficients.len(), 10);
        assert!(result.gain > 0.0);
    }

    #[test]
    fn test_polynomial_roots_quadratic() {
        // x^2 - 5x + 6 = 0 has roots 2 and 3
        let coeffs = vec![6.0, -5.0, 1.0];
        let roots = find_polynomial_roots_eigen(&coeffs);

        assert_eq!(roots.len(), 2);

        let mut real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_relative_eq!(real_roots[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(real_roots[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_roots_complex() {
        // x^2 + 1 = 0 has roots i and -i
        let coeffs = vec![1.0, 0.0, 1.0];
        let roots = find_polynomial_roots_eigen(&coeffs);

        assert_eq!(roots.len(), 2);

        for root in &roots {
            assert_relative_eq!(root.re, 0.0, epsilon = 1e-10);
            assert_relative_eq!(root.im.abs(), 1.0, epsilon = 1e-10);
        }
    }
}
