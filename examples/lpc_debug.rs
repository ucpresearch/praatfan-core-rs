use praat_core::Sound;
use std::f64::consts::PI;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    println!("Sound info:");
    println!("  Duration: {:.3}s", sound.duration());
    println!("  Sample rate: {} Hz", sound.sample_rate());
    println!();

    // Match the actual formant analysis: resample then pre-emphasize
    let target_sr = 2.0 * 5500.0; // 11000 Hz
    let resampled = sound.resample(target_sr);
    let resampled = resampled.pre_emphasis(50.0);

    println!("After resampling:");
    println!("  Sample rate: {} Hz", resampled.sample_rate());
    println!("  Duration: {:.3}s", resampled.duration());

    // Check raw resampled samples
    let center_idx = (0.316 * resampled.sample_rate()) as usize;
    println!("  Samples around t=0.316:");
    for i in (center_idx - 5)..=(center_idx + 5) {
        println!("    [{}] = {:.6}", i, resampled.samples()[i]);
    }
    println!();

    // Extract a single frame at t=0.316 (Praat's frame 30) - should have speech
    let frame_time = 0.316;
    let window_length = 0.025;
    let window_samples = (window_length * resampled.sample_rate()).round() as usize;
    let half_window = window_samples / 2;

    let center_sample = ((frame_time - resampled.start_time()) * resampled.sample_rate()).round() as usize;
    let start_sample = center_sample.saturating_sub(half_window);
    let end_sample = (center_sample + half_window).min(resampled.samples().len());

    println!("Frame at t={:.4}:", frame_time);
    println!("  Window samples: {}", window_samples);
    println!("  Center sample: {}", center_sample);
    println!("  Sample range: {}..{}", start_sample, end_sample);
    println!();

    // Create window and extract samples
    let window = praat_formant_window(window_samples);
    let mut windowed = Vec::with_capacity(window_samples);
    for (i, sample_idx) in (start_sample..end_sample).enumerate() {
        let w = if i < window.len() { window[i] } else { 0.0 };
        windowed.push(resampled.samples()[sample_idx] * w);
    }
    while windowed.len() < window_samples {
        windowed.push(0.0);
    }

    println!("Windowed signal:");
    println!("  Len: {}", windowed.len());
    println!("  First 5: {:?}", &windowed[..5.min(windowed.len())]);
    println!("  Energy: {:.6}", windowed.iter().map(|x| x*x).sum::<f64>());
    println!();

    // Compute LPC
    let lpc_order = 2 * 5 + 2; // 12
    println!("LPC order: {}", lpc_order);

    if let Some(lpc) = lpc_burg(&windowed, lpc_order) {
        println!("LPC coefficients:");
        for (i, c) in lpc.coefficients.iter().enumerate() {
            println!("  a[{}] = {:.6}", i, c);
        }
        println!("  Gain: {:.6}", lpc.gain);
        println!();

        // Find roots
        let roots = find_lpc_roots(&lpc.coefficients);
        println!("Polynomial roots ({} total):", roots.len());
        for (i, root) in roots.iter().enumerate() {
            let mag = root.norm();
            let angle = root.im.atan2(root.re);
            let freq = angle * resampled.sample_rate() / (2.0 * PI);
            let bw = if mag > 0.0 {
                -(mag.ln()) * resampled.sample_rate() / PI
            } else {
                f64::MAX
            };
            println!("  Root {}: {:.4} + {:.4}i  mag={:.4}  freq={:.1} Hz  bw={:.1} Hz",
                     i, root.re, root.im, mag, freq, bw);
        }

        // Filter to formants
        println!();
        println!("Filtered formants:");
        let formants = extract_formants(&roots, resampled.sample_rate(), 5500.0);
        for (i, f) in formants.iter().enumerate() {
            println!("  F{}: {:.1} Hz, B{}: {:.1} Hz", i+1, f.0, i+1, f.1);
        }
    }
}

fn praat_formant_window(size: usize) -> Vec<f64> {
    let edge = (-12.0_f64).exp();
    let imid = (size as f64 - 1.0) / 2.0;
    let denom = (size + 1) as f64;
    (0..size)
        .map(|i| {
            let diff = i as f64 - imid;
            let gaussian = (-48.0 * diff * diff / (denom * denom)).exp();
            (gaussian - edge) / (1.0 - edge)
        })
        .collect()
}

struct LpcResult {
    coefficients: Vec<f64>,
    gain: f64,
}

fn lpc_burg(samples: &[f64], order: usize) -> Option<LpcResult> {
    let n = samples.len();
    if n == 0 || order == 0 || order >= n {
        return None;
    }

    let mut ef: Vec<f64> = samples.to_vec();
    let mut eb: Vec<f64> = samples.to_vec();
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut error: f64 = samples.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    if error == 0.0 {
        return Some(LpcResult { coefficients: a, gain: 0.0 });
    }

    let mut a_new = vec![0.0; order + 1];

    for m in 1..=order {
        let mut num = 0.0;
        let mut den = 0.0;

        for i in m..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }

        if den == 0.0 {
            break;
        }

        let k = -2.0 * num / den;

        if k.abs() >= 1.0 {
            let k = k.signum() * 0.9999;
            error *= 1.0 - k * k;
        } else {
            error *= 1.0 - k * k;
        }

        a_new[0] = 1.0;
        for i in 1..m {
            a_new[i] = a[i] + k * a[m - i];
        }
        a_new[m] = k;

        for i in 0..=m {
            a[i] = a_new[i];
        }

        for i in (m..n).rev() {
            let ef_new = ef[i] + k * eb[i - 1];
            let eb_new = eb[i - 1] + k * ef[i];
            ef[i] = ef_new;
            eb[i] = eb_new;
        }
    }

    Some(LpcResult { coefficients: a, gain: error.sqrt() })
}

use num_complex::Complex;

fn find_lpc_roots(coefficients: &[f64]) -> Vec<Complex<f64>> {
    // Reverse coefficients for polynomial: z^n + a1*z^(n-1) + ... + an
    let reversed: Vec<f64> = coefficients.iter().rev().copied().collect();
    find_polynomial_roots(&reversed)
}

fn find_polynomial_roots(coefficients: &[f64]) -> Vec<Complex<f64>> {
    let n = coefficients.len() - 1;
    if n == 0 {
        return Vec::new();
    }

    let leading = coefficients[n];
    if leading.abs() < 1e-10 {
        return find_polynomial_roots(&coefficients[..n]);
    }

    // Build companion matrix
    let mut companion = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for i in 1..n {
        companion[i][i - 1] = Complex::new(1.0, 0.0);
    }
    for i in 0..n {
        companion[i][n - 1] = Complex::new(-coefficients[i] / leading, 0.0);
    }

    // Eigenvalues via QR iteration
    eigenvalues_qr(&companion)
}

fn eigenvalues_qr(matrix: &[Vec<Complex<f64>>]) -> Vec<Complex<f64>> {
    let n = matrix.len();
    if n == 0 {
        return Vec::new();
    }

    let mut a: Vec<Vec<Complex<f64>>> = matrix.to_vec();
    let max_iterations = 100 * n;
    let tolerance = 1e-10;

    for _ in 0..max_iterations {
        let mut converged = true;
        for i in 1..n {
            if a[i][i - 1].norm() > tolerance {
                converged = false;
                break;
            }
        }
        if converged {
            break;
        }

        let shift = if n >= 2 {
            wilkinson_shift(a[n - 2][n - 2], a[n - 2][n - 1], a[n - 1][n - 2], a[n - 1][n - 1])
        } else {
            a[n - 1][n - 1]
        };

        for i in 0..n {
            a[i][i] -= shift;
        }

        let (q, r) = qr_decomposition(&a);
        a = matrix_multiply(&r, &q);

        for i in 0..n {
            a[i][i] += shift;
        }
    }

    (0..n).map(|i| a[i][i]).collect()
}

fn wilkinson_shift(a: Complex<f64>, b: Complex<f64>, c: Complex<f64>, d: Complex<f64>) -> Complex<f64> {
    let trace = a + d;
    let det = a * d - b * c;
    let discriminant = trace * trace - Complex::new(4.0, 0.0) * det;
    let sqrt_disc = complex_sqrt(discriminant);
    let e1 = (trace + sqrt_disc) / Complex::new(2.0, 0.0);
    let e2 = (trace - sqrt_disc) / Complex::new(2.0, 0.0);
    if (e1 - d).norm() < (e2 - d).norm() { e1 } else { e2 }
}

fn complex_sqrt(z: Complex<f64>) -> Complex<f64> {
    Complex::from_polar(z.norm().sqrt(), z.arg() / 2.0)
}

fn qr_decomposition(a: &[Vec<Complex<f64>>]) -> (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f64>>>) {
    let n = a.len();
    let mut q: Vec<Vec<Complex<f64>>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) }).collect())
        .collect();
    let mut r: Vec<Vec<Complex<f64>>> = a.to_vec();

    for k in 0..n.saturating_sub(1) {
        let mut x: Vec<Complex<f64>> = (k..n).map(|i| r[i][k]).collect();
        let norm_x = x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm_x < 1e-15 { continue; }
        let alpha = if x[0].re >= 0.0 { -norm_x } else { norm_x };
        x[0] -= Complex::new(alpha, 0.0);
        let norm_v = x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm_v < 1e-15 { continue; }
        for v in &mut x { *v /= norm_v; }

        for j in k..n {
            let mut dot = Complex::new(0.0, 0.0);
            for i in 0..x.len() { dot += x[i].conj() * r[k + i][j]; }
            dot *= Complex::new(2.0, 0.0);
            for i in 0..x.len() { r[k + i][j] -= dot * x[i]; }
        }
        for j in 0..n {
            let mut dot = Complex::new(0.0, 0.0);
            for i in 0..x.len() { dot += x[i].conj() * q[k + i][j]; }
            dot *= Complex::new(2.0, 0.0);
            for i in 0..x.len() { q[k + i][j] -= dot * x[i]; }
        }
    }

    let mut q_result = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            q_result[i][j] = q[j][i].conj();
        }
    }
    (q_result, r)
}

fn matrix_multiply(a: &[Vec<Complex<f64>>], b: &[Vec<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
    let n = a.len();
    let mut result = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn extract_formants(roots: &[Complex<f64>], sample_rate: f64, max_formant: f64) -> Vec<(f64, f64)> {
    let mut formants = Vec::new();

    for root in roots {
        let mag = root.norm();
        // Only consider roots inside or on unit circle
        if mag > 1.01 { continue; }
        // Only consider upper half-plane (positive imaginary)
        if root.im <= 0.0 { continue; }

        let angle = root.im.atan2(root.re);
        let freq = angle * sample_rate / (2.0 * PI);

        // Skip very low or very high frequencies
        if freq < 50.0 || freq >= max_formant { continue; }

        let bw = if mag > 0.0 {
            -(mag.ln()) * sample_rate / PI
        } else {
            sample_rate
        };

        // Skip if bandwidth is too large
        if bw >= sample_rate / 2.0 { continue; }

        formants.push((freq, bw));
    }

    formants.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    formants
}
