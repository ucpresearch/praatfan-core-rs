use praatfan_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    // Check original samples
    println!("Original samples around 10800:");
    for i in 10795..10805 {
        println!("  orig[{}] = {:.10e}", i, sound.samples()[i]);
    }

    // Resample
    let resampled = sound.resample(11000.0);

    println!("\nResampled samples around 4950:");
    for i in 4945..4955 {
        println!("  resampled[{}] = {:.10e}", i, resampled.samples()[i]);
    }

    // The ratio calculation
    let ratio = 11000.0 / 24000.0;
    println!("\nRatio: {}", ratio);
    println!("For output index 4950, input index = {}", 4950.0 / ratio);
    println!("For output index 4949, input index = {}", 4949.0 / ratio);

    // Check if the FFT filter is actually doing something
    // by looking at samples that should be filtered
    println!("\nExpected: Praat's value at sample 4950 is -3.49e-2");
    println!("Our value at sample 4950: {:.10e}", resampled.samples()[4950]);
}
