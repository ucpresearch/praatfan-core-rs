use praatfan_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    println!("Rust loading:");
    println!("  Sample rate: {} Hz", sound.sample_rate());
    println!("  Num samples: {}", sound.samples().len());

    // Find min and max
    let min = sound.samples().iter().cloned().fold(f64::INFINITY, f64::min);
    let max = sound.samples().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Values range: [{:.4}, {:.4}]", min, max);

    println!("  First 5 samples: {:?}", &sound.samples()[..5]);
    println!("  Samples at t=0.316: {:?}", &sound.samples()[7581..7586]);

    // Calculate pre-emphasis manually
    let alpha = (-2.0 * std::f64::consts::PI * 50.0 / sound.sample_rate()).exp();
    println!("\nalpha = {:.6}", alpha);
    println!("Manual pre-emph at 7582: {} - {} * {} = {:.6}",
             sound.samples()[7582],
             alpha,
             sound.samples()[7581],
             sound.samples()[7582] - alpha * sound.samples()[7581]);

    // Compare with pre_emphasis function
    let pre_emp = sound.pre_emphasis(50.0);
    println!("pre_emphasis() at 7582: {:.6}", pre_emp.samples()[7582]);
}
