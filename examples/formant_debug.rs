use praat_core::{Sound, Interpolation};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    println!("Sound info:");
    println!("  Duration: {:.3}s", sound.duration());
    println!("  Sample rate: {} Hz", sound.sample_rate());
    println!();

    // Use same params as ground truth
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Formant analysis:");
    println!("  Num frames: {}", formant.num_frames());
    println!("  First frame time: {:.4}", formant.start_time());
    println!("  Time step: {:.4}", formant.time_step());
    println!();

    // Compare at Praat's exact frame times (x1=0.026, dx=0.01)
    // Praat Frame 1 (t=0.0260): F1=1054, F2=2157, F3=3304
    // Praat Frame 5 (t=0.0660): F1=1547, F2=2305, F3=3402
    // Praat Frame 10 (t=0.1160): F1=1159, F2=1928, F3=3220
    // Praat Frame 20 (t=0.2160): F1=403, F2=935, F3=3148
    // Praat Frame 30 (t=0.3160): F1=264, F2=984, F3=2402
    // Praat Frame 50 (t=0.5160): F1=281, F2=2417, F3=2812

    let praat_frame_times = [
        (1, 0.026, 1054.0, 2157.0, 3304.0),
        (5, 0.066, 1547.0, 2305.0, 3402.0),
        (10, 0.116, 1159.0, 1928.0, 3220.0),
        (20, 0.216, 403.0, 935.0, 3148.0),
        (30, 0.316, 264.0, 984.0, 2402.0),
        (50, 0.516, 281.0, 2417.0, 2812.0),
    ];

    println!("Comparing at Praat frame times:");
    println!("{:>6} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "Frame", "Time", "Praat_F1", "Ours_F1", "Praat_F2", "Ours_F2", "Praat_F3", "Ours_F3");
    println!("{}", "-".repeat(90));

    for (frame, t, praat_f1, praat_f2, praat_f3) in praat_frame_times {
        let f1 = formant.get_value_at_time(1, t, praat_core::FrequencyUnit::Hertz, Interpolation::Linear);
        let f2 = formant.get_value_at_time(2, t, praat_core::FrequencyUnit::Hertz, Interpolation::Linear);
        let f3 = formant.get_value_at_time(3, t, praat_core::FrequencyUnit::Hertz, Interpolation::Linear);

        let f1_str = f1.map_or("undef".to_string(), |v| format!("{:.0}", v));
        let f2_str = f2.map_or("undef".to_string(), |v| format!("{:.0}", v));
        let f3_str = f3.map_or("undef".to_string(), |v| format!("{:.0}", v));

        println!("{:>6} {:>8.3} {:>12.0} {:>12} {:>12.0} {:>12} {:>12.0} {:>12}",
                 frame, t, praat_f1, f1_str, praat_f2, f2_str, praat_f3, f3_str);
    }
}
