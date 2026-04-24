//! Dump our FFTPACK output on a controlled input so Praat can compute the
//! same transform for bit-parity comparison. Run as:
//!
//!     cargo run --release --example fft_dump -- <n>
//!
//! Writes CSV `idx,value` to stdout: first 8192 rows are `drftf1_forward`
//! output (Praat-rotated layout), next 8192 rows are the inverse of that.

use praatfan_core::utils::fft_fftpack::{realft_backward_praat, realft_forward_praat, RealFftPlan};
use std::env;

fn main() {
    let n: usize = env::args()
        .nth(1)
        .as_deref()
        .unwrap_or("8192")
        .parse()
        .expect("need n");

    // Deterministic synthetic signal: sum of two sinusoids + chirp-like modulation
    let mut data: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 3.1 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 17.5 * t).cos()
                + 0.1 * ((i as f64 * 0.01).sin())
        })
        .collect();

    let mut plan = RealFftPlan::new(n);
    let before = data.clone();
    realft_forward_praat(&mut plan, &mut data);
    println!("# forward FFT output (Praat layout) — n={}", n);
    for (i, v) in data.iter().enumerate() {
        println!("f,{},{:.17e}", i, v);
    }

    // Round-trip
    realft_backward_praat(&mut plan, &mut data);
    let scale = 1.0 / n as f64;
    println!("# round-trip (after / n)");
    for (i, (v, o)) in data.iter().zip(before.iter()).enumerate() {
        let recovered = v * scale;
        println!("rt,{},{:.17e},{:.17e},{:.17e}", i, recovered, o, recovered - o);
    }
}
