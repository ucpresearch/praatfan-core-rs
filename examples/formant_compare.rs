use praatfan_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Our structure:");
    println!("  x1: {:.6}", formant.start_time());
    println!("  nx: {}", formant.num_frames());
    println!();

    // Get raw frame values (no interpolation)
    println!("Raw frame F1 values (first 5 frames):");
    for i in 0..5 {
        let f1 = formant.get_value_at_frame(1, i);
        let t = formant.get_time_from_frame(i);
        match f1 {
            Some(f) => println!("  Frame {} (t={:.4}): F1={:.1}", i, t, f),
            None => println!("  Frame {} (t={:.4}): F1=NaN", i, t),
        }
    }
    println!();

    // Query at specific times with interpolation
    println!("Interpolated F1 at query times:");
    let times = [0.0, 0.01, 0.02, 0.026, 0.03, 0.036, 0.04, 0.05];
    for &t in &times {
        let f1 = formant.get_value_at_time(
            1, t,
            praatfan_core::FrequencyUnit::Hertz,
            praatfan_core::Interpolation::Linear
        );
        match f1 {
            Some(f) if f.is_finite() => println!("  t={:.3}: F1={:.1}", t, f),
            _ => println!("  t={:.3}: F1=NaN", t),
        }
    }
}
