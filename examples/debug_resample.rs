use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    println!("Original: {} samples at {} Hz", sound.samples().len(), sound.sample_rate());

    // Get original sample around equivalent time position
    let orig_idx = ((0.45 * 24000.0) as usize).saturating_sub(5);
    println!("\nOriginal samples around t=0.45:");
    for i in orig_idx..(orig_idx + 10) {
        if i < sound.samples().len() {
            println!("  orig[{}] (t={:.4}) = {:.10e}", i, i as f64 / 24000.0, sound.samples()[i]);
        }
    }

    let resampled = sound.resample(11000.0);
    println!("\nResampled: {} samples at {} Hz", resampled.samples().len(), resampled.sample_rate());

    // Resampled samples
    println!("\nResampled samples around t=0.45 (sample ~4950):");
    for i in 4945..4955 {
        if i < resampled.samples().len() {
            println!("  resampled[{}] = {:.10e}", i, resampled.samples()[i]);
        }
    }
}
