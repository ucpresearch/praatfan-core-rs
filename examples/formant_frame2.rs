use praatfan_core::Sound;
use std::f64::consts::PI;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    // Process exactly as formant analysis does
    let target_sr = 2.0 * 5500.0;
    let resampled = sound.resample(target_sr);
    let emphasized = resampled.pre_emphasis(50.0);

    let sample_rate = emphasized.sample_rate();
    let samples = emphasized.samples();

    let window_length = 0.025;
    let window_samples = (window_length * sample_rate).round() as usize;
    let half_window = window_samples / 2;

    // Frame 2 at t=0.036
    let frame_time = 0.036;
    let center_sample = ((frame_time - emphasized.start_time()) * sample_rate).round() as usize;

    println!("Frame 2 (t=0.036) extraction:");
    println!("  Sample rate: {}", sample_rate);
    println!("  Window samples: {}", window_samples);
    println!("  Center sample: {}", center_sample);
    println!("  Half window: {}", half_window);

    let start_sample = center_sample.saturating_sub(half_window);
    let end_sample = (center_sample + half_window).min(samples.len());

    println!("  Range: {}..{}", start_sample, end_sample);

    // Create window
    let edge = (-12.0_f64).exp();
    let imid = (window_samples as f64 - 1.0) / 2.0;
    let denom = (window_samples + 1) as f64;
    let window: Vec<f64> = (0..window_samples)
        .map(|i| {
            let diff = i as f64 - imid;
            let gaussian = (-48.0 * diff * diff / (denom * denom)).exp();
            (gaussian - edge) / (1.0 - edge)
        })
        .collect();

    // Extract and window
    let mut windowed = Vec::with_capacity(window_samples);
    for (i, sample_idx) in (start_sample..end_sample).enumerate() {
        let w = if i < window.len() { window[i] } else { 0.0 };
        windowed.push(samples[sample_idx] * w);
    }
    while windowed.len() < window_samples {
        windowed.push(0.0);
    }

    let energy: f64 = windowed.iter().map(|x| x * x).sum();
    println!("  Energy: {:.6}", energy);
    println!();

    // Compute LPC using Burg
    let lpc_order = 12;
    if let Some(lpc) = lpc_burg(&windowed, lpc_order) {
        println!("LPC coefficients:");
        for (i, c) in lpc.coefficients.iter().enumerate() {
            println!("  a[{}] = {:.6}", i, c);
        }

        // Find roots
        let roots = find_roots(&lpc.coefficients);
        println!();
        println!("Formant candidates (inside unit circle, upper half, freq > 50):");
        let mut formants: Vec<(f64, f64)> = Vec::new();
        for root in &roots {
            let mag = root.norm();
            if mag > 1.01 || root.im <= 0.0 {
                continue;
            }
            let angle = root.im.atan2(root.re);
            let freq = angle * sample_rate / (2.0 * PI);
            if freq < 50.0 || freq >= 5500.0 {
                continue;
            }
            let bw = if mag > 0.0 { -(mag.ln()) * sample_rate / PI } else { sample_rate };
            if bw < 0.0 || bw >= 5500.0 || bw >= freq * 2.0 {
                continue;
            }
            formants.push((freq, bw));
        }
        formants.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (i, (freq, bw)) in formants.iter().take(5).enumerate() {
            println!("  F{}: {:.1} Hz, B{}: {:.1} Hz", i + 1, freq, i + 1, bw);
        }
    }
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

fn find_roots(coefficients: &[f64]) -> Vec<Complex<f64>> {
    let n = coefficients.len() - 1;
    if n == 0 {
        return Vec::new();
    }
    let leading = coefficients[0]; // LPC has a[0]=1
    let mut roots: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = 2.0 * PI * (k as f64 + 0.5) / n as f64;
            Complex::new(0.9 * angle.cos(), 0.9 * angle.sin())
        })
        .collect();

    // Aberth-Ehrlich
    for _ in 0..50 {
        let mut max_delta = 0.0;
        for k in 0..n {
            let z = roots[k];
            let (p, dp) = eval_poly(coefficients, z);
            if p.norm() < 1e-14 {
                continue;
            }
            let newton = if dp.norm() > 1e-30 { p / dp } else { p };
            let mut aberth_sum = Complex::new(0.0, 0.0);
            for j in 0..n {
                if j != k {
                    let diff = z - roots[j];
                    if diff.norm() > 1e-30 {
                        aberth_sum += Complex::new(1.0, 0.0) / diff;
                    }
                }
            }
            let denom = Complex::new(1.0, 0.0) - newton * aberth_sum;
            let delta = if denom.norm() > 1e-30 { newton / denom } else { newton * 0.5 };
            roots[k] -= delta;
            if delta.norm() > max_delta {
                max_delta = delta.norm();
            }
        }
        if max_delta < 1e-14 {
            break;
        }
    }
    roots
}

fn eval_poly(coeffs: &[f64], z: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let n = coeffs.len();
    let mut p = Complex::new(coeffs[n - 1], 0.0);
    let mut dp = Complex::new(0.0, 0.0);
    for i in (0..n - 1).rev() {
        dp = dp * z + p;
        p = p * z + Complex::new(coeffs[i], 0.0);
    }
    (p, dp)
}
