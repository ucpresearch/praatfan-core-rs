use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let resampled = sound.resample(11000.0);
    let samples = resampled.samples();

    // Check around center sample for t=0.45 (sample 4950)
    println!("Samples around center=4950 (t=0.45):");
    println!("\nBEFORE pre-emphasis (Rust resampled):");
    for i in 4945..4955 {
        println!("  sample[{}] = {:.10e}", i, samples[i]);
    }

    // Also apply pre-emphasis and check
    let emphasized = resampled.pre_emphasis(50.0);
    let emph_samples = emphasized.samples();

    println!("\nAFTER pre-emphasis (Rust):");
    for i in 4945..4955 {
        println!("  sample[{}] = {:.10e}", i, emph_samples[i]);
    }

    let alpha = (-2.0 * std::f64::consts::PI * 50.0 / 11000.0).exp();
    println!("\nPre-emphasis alpha = {:.10}", alpha);
    println!("Manual check: sample_after[4950] = sample_before[4950] - alpha * sample_before[4949]");
    println!("  Expected: {:.10e}", samples[4950] - alpha * samples[4949]);
    println!("  Actual:   {:.10e}", emph_samples[4950]);
}
