use praat_core::Sound;
use std::f64::consts::PI;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    println!("Sound info:");
    println!("  Duration: {:.3}s", sound.duration());
    println!("  Sample rate: {} Hz", sound.sample_rate());
    println!();

    // Match formant analysis: resample then pre-emphasize
    let target_sr = 2.0 * 5500.0; // 11000 Hz
    let resampled = sound.resample(target_sr);
    let resampled = resampled.pre_emphasis(50.0);

    println!("After processing:");
    println!("  Sample rate: {} Hz", resampled.sample_rate());
    println!();

    // Extract frame 20 (Praat's time 0.216)
    // Praat's x1 = 0.026, dx = 0.01
    // Frame 20 time = 0.026 + 19 * 0.01 = 0.216 (Praat is 1-indexed)
    let frame_time = 0.216;
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
        println!();

        // Find roots using our Aberth-Ehrlich implementation
        let roots = find_lpc_roots(&lpc.coefficients);
        println!("All polynomial roots ({} total):", roots.len());

        // Sort by frequency for easier reading
        let mut root_info: Vec<_> = roots.iter().map(|root| {
            let mag = root.norm();
            let angle = root.im.atan2(root.re);
            let freq = angle * resampled.sample_rate() / (2.0 * PI);
            let bw = if mag > 0.0 && mag < 1.0 {
                -(mag.ln()) * resampled.sample_rate() / PI
            } else {
                f64::MAX
            };
            (root, mag, freq, bw)
        }).collect();

        // Show all roots
        for (root, mag, freq, bw) in &root_info {
            let inside = if *mag <= 1.01 { "inside" } else { "outside" };
            let upper = if root.im > 0.0 { "upper" } else { "lower" };
            println!("  {:.4} + {:.4}i  mag={:.4}  freq={:.1} Hz  bw={:.1} Hz  [{}, {}]",
                     root.re, root.im, mag, freq, if *bw < 10000.0 { *bw } else { f64::INFINITY }, inside, upper);
        }
        println!();

        // Filter to formants (same logic as in lpc_to_formants)
        println!("Filtered formants (inside unit circle, upper half, 50 < f < 5500):");
        let max_formant = 5500.0;
        let max_bandwidth = resampled.sample_rate() / 2.0;

        let mut formants: Vec<(f64, f64)> = root_info
            .iter()
            .filter(|(root, mag, freq, bw)| {
                *mag <= 1.01
                    && root.im > 0.0
                    && *freq > 50.0
                    && *freq < max_formant
                    && *bw > 0.0
                    && *bw < max_bandwidth
            })
            .map(|(_, _, freq, bw)| (*freq, *bw))
            .collect();

        formants.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (i, (freq, bw)) in formants.iter().enumerate() {
            println!("  F{}: {:.1} Hz, B{}: {:.1} Hz", i + 1, freq, i + 1, bw);
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
        return Some(LpcResult { coefficients: a });
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

        let mut k = -2.0 * num / den;

        if k.abs() >= 1.0 {
            k = k.signum() * 0.9999;
        }

        error *= 1.0 - k * k;

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

    Some(LpcResult { coefficients: a })
}

use num_complex::Complex;

fn find_lpc_roots(coefficients: &[f64]) -> Vec<Complex<f64>> {
    let reversed: Vec<f64> = coefficients.iter().rev().copied().collect();
    aberth_ehrlich(&reversed)
}

fn aberth_ehrlich(coefficients: &[f64]) -> Vec<Complex<f64>> {
    let n = coefficients.len() - 1;
    if n == 0 {
        return Vec::new();
    }

    let leading = coefficients[n];
    if leading.abs() < 1e-10 {
        return aberth_ehrlich(&coefficients[..n]);
    }

    // Estimate root radius
    let mut max_ratio = 0.0_f64;
    for i in 0..n {
        let ratio = (coefficients[i] / leading).abs();
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    let radius = (1.0 + max_ratio).max(1.0);

    // Initialize roots on a circle
    let mut roots: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = 2.0 * PI * (k as f64 + 0.5) / n as f64;
            let perturb = 0.1 * (k as f64 * 0.7).sin();
            let r = radius * (0.9 + 0.1 * perturb);
            Complex::new(r * angle.cos(), r * angle.sin())
        })
        .collect();

    let max_iterations = 50;
    let tolerance = 1e-14;

    for _ in 0..max_iterations {
        let mut max_delta = 0.0_f64;

        for k in 0..n {
            let z_k = roots[k];
            let (p, dp) = evaluate_polynomial_and_derivative(coefficients, z_k);

            if p.norm() < tolerance * leading.abs() {
                continue;
            }

            let newton_step = if dp.norm() > 1e-30 { p / dp } else { p };

            let mut aberth_sum = Complex::new(0.0, 0.0);
            for j in 0..n {
                if j != k {
                    let diff = z_k - roots[j];
                    if diff.norm() > 1e-30 {
                        aberth_sum += Complex::new(1.0, 0.0) / diff;
                    }
                }
            }

            let denominator = Complex::new(1.0, 0.0) - newton_step * aberth_sum;
            let delta = if denominator.norm() > 1e-30 {
                newton_step / denominator
            } else {
                newton_step * Complex::new(0.5, 0.0)
            };

            roots[k] -= delta;
            let delta_mag = delta.norm();
            if delta_mag > max_delta {
                max_delta = delta_mag;
            }
        }

        if max_delta < tolerance {
            break;
        }
    }

    // Polish roots
    for root in &mut roots {
        *root = polish_root(coefficients, *root);
    }

    roots
}

fn polish_root(coefficients: &[f64], mut root: Complex<f64>) -> Complex<f64> {
    let max_iterations = 10;
    let tolerance = 1e-12;

    for _ in 0..max_iterations {
        let (p, dp) = evaluate_polynomial_and_derivative(coefficients, root);

        if dp.norm() < tolerance {
            break;
        }

        let delta = p / dp;
        root -= delta;

        if delta.norm() < tolerance * root.norm().max(1.0) {
            break;
        }
    }

    root
}

fn evaluate_polynomial_and_derivative(
    coefficients: &[f64],
    z: Complex<f64>,
) -> (Complex<f64>, Complex<f64>) {
    let n = coefficients.len();
    if n == 0 {
        return (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
    }

    let mut p = Complex::new(coefficients[n - 1], 0.0);
    let mut dp = Complex::new(0.0, 0.0);

    for i in (0..n - 1).rev() {
        dp = dp * z + p;
        p = p * z + Complex::new(coefficients[i], 0.0);
    }

    (p, dp)
}
