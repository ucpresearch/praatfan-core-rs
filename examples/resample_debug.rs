use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    println!("Original sound:");
    println!("  Sample rate: {} Hz", sound.sample_rate());
    println!("  Samples: {}", sound.samples().len());
    println!("  Duration: {:.4}s", sound.duration());

    // Show original samples around t=0.316
    let frame_time = 0.316;
    let center_idx = (frame_time * sound.sample_rate()) as usize;
    println!("\n  Samples around t=0.316 (original):");
    for i in center_idx.saturating_sub(3)..=(center_idx + 3).min(sound.samples().len() - 1) {
        println!("    [{}] = {:.6}", i, sound.samples()[i]);
    }

    // Pre-emphasize
    let pre_emph = sound.pre_emphasis(50.0);
    println!("\nAfter pre-emphasis:");
    println!("  Sample rate: {} Hz", pre_emph.sample_rate());
    println!("  Samples: {}", pre_emph.samples().len());

    let center_idx = (frame_time * pre_emph.sample_rate()) as usize;
    println!("\n  Samples around t=0.316 (pre-emph):");
    for i in center_idx.saturating_sub(3)..=(center_idx + 3).min(pre_emph.samples().len() - 1) {
        println!("    [{}] = {:.6}", i, pre_emph.samples()[i]);
    }

    // Resample to 11000 Hz
    let resampled = pre_emph.resample(11000.0);
    println!("\nAfter resampling to 11000 Hz:");
    println!("  Sample rate: {} Hz", resampled.sample_rate());
    println!("  Samples: {}", resampled.samples().len());

    let center_idx = (frame_time * resampled.sample_rate()) as usize;
    println!("\n  Samples around t=0.316 (resampled):");
    for i in center_idx.saturating_sub(5)..=(center_idx + 5).min(resampled.samples().len() - 1) {
        println!("    [{}] = {:.6}", i, resampled.samples()[i]);
    }

    // Calculate energy in window
    let window_samples = (0.025 * resampled.sample_rate()) as usize;
    let half_window = window_samples / 2;
    let start = center_idx.saturating_sub(half_window);
    let end = (center_idx + half_window).min(resampled.samples().len());
    let energy: f64 = resampled.samples()[start..end].iter().map(|x| x * x).sum();
    println!("\n  Energy (no window) from {} to {}: {:.6}", start, end, energy);

    // Compare with Praat values:
    // Praat samples around 3476: -0.013 to -0.065 range
    // Praat energy: 0.43
}
