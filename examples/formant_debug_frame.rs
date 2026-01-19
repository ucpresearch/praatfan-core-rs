use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Our formant analysis:");
    println!("  x1: {:.4}", formant.start_time());
    println!("  dx: {:.4}", formant.time_step());
    println!("  n_frames: {}", formant.num_frames());
    println!();

    // Check specific times
    for t in [0.026, 0.030, 0.036, 0.040] {
        let f1 = formant.get_value_at_time(
            1, t,
            praat_core::FrequencyUnit::Hertz,
            praat_core::Interpolation::Linear
        );
        let f2 = formant.get_value_at_time(
            2, t,
            praat_core::FrequencyUnit::Hertz,
            praat_core::Interpolation::Linear
        );
        match (f1, f2) {
            (Some(f1), Some(f2)) if f1.is_finite() && f2.is_finite() =>
                println!("t={:.3}: F1={:.1}, F2={:.1}", t, f1, f2),
            _ => println!("t={:.3}: F1/F2 undefined", t),
        }
    }

    println!();
    println!("Frame times and F1 values:");
    for i in 0..5 {
        let frame_time = formant.start_time() + i as f64 * formant.time_step();
        let f1 = formant.get_value_at_time(
            1, frame_time,
            praat_core::FrequencyUnit::Hertz,
            praat_core::Interpolation::Linear
        );
        match f1 {
            Some(f) if f.is_finite() => println!("  Frame {} (t={:.4}): F1={:.1}", i+1, frame_time, f),
            _ => println!("  Frame {} (t={:.4}): F1=NaN", i+1, frame_time),
        }
    }
}
