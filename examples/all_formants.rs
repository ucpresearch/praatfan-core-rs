use praatfan_core::Sound;
use praatfan_core::{FrequencyUnit, Interpolation};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    let test_times = [0.40, 0.45, 0.84, 0.85, 0.86, 1.05];

    for t in test_times {
        println!("\nt={:.2}:", t);
        for i in 1..=5 {
            let f = formant.get_value_at_time(i, t, FrequencyUnit::Hertz, Interpolation::Linear);
            print!("  F{}: {:>7} ", i, f.map_or("NaN".to_string(), |v| format!("{:.0}", v)));
        }
        println!();
    }

    // Also show frame-level data
    println!("\n\nFrame-level values around t=0.84-0.86:");
    for frame in 80..90 {
        if frame >= formant.num_frames() { break; }
        let t = formant.get_time_from_frame(frame);
        if t < 0.82 || t > 0.88 { continue; }

        print!("Frame {} (t={:.3}): ", frame, t);
        for i in 1..=5 {
            let f = formant.get_value_at_frame(i, frame);
            print!("F{}: {:>5} ", i, f.map_or("NaN".to_string(), |v| format!("{:.0}", v)));
        }
        println!();
    }
}
