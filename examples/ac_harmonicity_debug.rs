//! Debug AC harmonicity: dump all voiced frame strengths.
use praatfan_core::pitch::{Pitch, PitchMethod};
use praatfan_core::Sound;

fn main() {
    let wav = "tests/fixtures/one_two_three_four_five.wav";
    let sound = Sound::from_file(wav).expect("Failed to load audio");

    let pitch_ceiling = 0.5 * sound.sample_rate();

    let pitch = Pitch::from_sound_with_method(
        &sound, 0.01, 75.0, pitch_ceiling, 15,
        0.1, 0.0, 0.0, 0.0, 0.0, 3.0,
        PitchMethod::AcGauss,
    );

    let t1 = pitch.start_time();
    let dt = pitch.time_step();

    // Output all voiced frames in JSON for easy comparison
    println!("[");
    let mut first = true;
    for (i, frame) in pitch.frames().iter().enumerate() {
        if !frame.candidates.is_empty() && frame.candidates[0].frequency > 0.0 {
            let c = &frame.candidates[0];
            let t = t1 + i as f64 * dt;
            if !first { println!(","); }
            first = false;
            print!("  {{\"frame\": {}, \"t\": {:.6}, \"freq\": {:.18e}, \"r\": {:.18e}}}",
                i, t, c.frequency, c.strength);
        }
    }
    println!("\n]");
}
