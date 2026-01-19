use praat_core::Sound;
use praat_core::window::praat_formant_window;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    println!("Original: {} samples, {} Hz", sound.samples().len(), sound.sample_rate());

    let resampled = sound.resample(11000.0);
    println!("Resampled: {} samples, {} Hz", resampled.samples().len(), resampled.sample_rate());

    let emphasized = resampled.pre_emphasis(50.0);
    let samples = emphasized.samples();

    let window_samples = 275;
    let half_window = window_samples / 2;
    let window = praat_formant_window(window_samples);

    println!("\nWindow samples: {}, half_window: {}", window_samples, half_window);

    // Frame at t=0.45 (good region)
    let t = 0.45;
    let center_sample = (t * 11000.0_f64).round() as usize;
    let start = center_sample.saturating_sub(half_window);

    println!("\nFrame at t={}: center={}, start={}", t, center_sample, start);

    let mut windowed = Vec::with_capacity(window_samples);
    let mut energy = 0.0;
    for i in 0..window_samples {
        let idx = start + i;
        let s = if idx < samples.len() {
            samples[idx] * window[i]
        } else {
            0.0
        };
        windowed.push(s);
        energy += s * s;
    }

    println!("First 5 windowed samples (Rust):");
    for i in 0..5 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    println!("\nCenter windowed samples (Rust):");
    for i in 137..142 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    println!("\nEnergy: {:.10e}", energy);

    // Frame at t=0.836 (problem region)
    let t = 0.836;
    let center_sample = (t * 11000.0_f64).round() as usize;
    let start = center_sample.saturating_sub(half_window);

    println!("\n\nFrame at t={} (problem region): center={}, start={}", t, center_sample, start);

    let mut windowed = Vec::with_capacity(window_samples);
    let mut energy = 0.0;
    for i in 0..window_samples {
        let idx = start + i;
        let s = if idx < samples.len() {
            samples[idx] * window[i]
        } else {
            0.0
        };
        windowed.push(s);
        energy += s * s;
    }

    println!("First 5 windowed samples (Rust):");
    for i in 0..5 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    println!("\nCenter windowed samples (Rust):");
    for i in 137..142 {
        println!("  windowed[{}] = {:.10e}", i, windowed[i]);
    }

    println!("\nEnergy: {:.10e}", energy);
}
