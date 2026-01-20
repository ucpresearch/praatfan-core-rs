use praatfan_core::Sound;
use praatfan_core::window::praat_formant_window;

// Re-export LPC functions for debugging
mod lpc_debug {
    pub use praatfan_core::utils::lpc::{lpc_burg, lpc_to_formants};
}

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    println!("Original sample rate: {}", sound.sample_rate());

    // Resample to 11000 Hz like Praat does for 5500 Hz max formant
    let resampled = sound.resample(11000.0);
    println!("Resampled to: {} Hz", resampled.sample_rate());

    // Apply pre-emphasis
    let emphasized = resampled.pre_emphasis(50.0);

    // Get a frame around t=0.20 (voiced speech region)
    let target_time = 0.20;
    let window_length = 0.025;
    let window_samples = (window_length * 11000.0_f64).round() as usize;  // 275
    let half_window = window_samples / 2;

    let window = praat_formant_window(window_samples);

    // Get center sample
    let center_sample = ((target_time - emphasized.start_time()) * 11000.0).round() as usize;
    let samples = emphasized.samples();

    println!("\nFrame at t={} (center_sample={})", target_time, center_sample);
    println!("Window samples: {}", window_samples);

    // Extract windowed frame
    let start_sample = center_sample.saturating_sub(half_window);
    let end_sample = (center_sample + half_window).min(samples.len());

    let mut windowed: Vec<f64> = Vec::with_capacity(window_samples);
    for (i, sample_idx) in (start_sample..end_sample).enumerate() {
        let w = if i < window.len() { window[i] } else { 0.0 };
        windowed.push(samples[sample_idx] * w);
    }
    while windowed.len() < window_samples {
        windowed.push(0.0);
    }

    // Add dither like formant.rs does
    let dither_amplitude = 1e-10;
    for (i, s) in windowed.iter_mut().enumerate() {
        let dither = dither_amplitude * ((i as f64 * 0.7).sin() + (i as f64 * 1.3).cos());
        *s += dither;
    }

    println!("\nWindowed samples (first 10):");
    for i in 0..10 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    println!("\nCenter windowed samples:");
    for i in 135..=140 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    // Compute LPC
    let lpc_order = 12;  // 2 * 5 + 2
    match lpc_debug::lpc_burg(&windowed, lpc_order) {
        Some(result) => {
            println!("\nLPC coefficients (order={}):", lpc_order);
            for (i, c) in result.coefficients.iter().enumerate() {
                println!("  a[{}] = {:.10e}", i, c);
            }
            println!("\nGain: {:.10e}", result.gain);

            // Get formants
            let formants = lpc_debug::lpc_to_formants(&result.coefficients, 11000.0);
            println!("\nFormants found: {}", formants.len());
            for (i, f) in formants.iter().take(5).enumerate() {
                println!("  F{}: {:.1} Hz (bandwidth: {:.1} Hz)", i + 1, f.frequency, f.bandwidth);
            }
        }
        None => {
            println!("LPC failed!");
        }
    }
}
