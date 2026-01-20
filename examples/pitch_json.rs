//! Output pitch analysis as JSON for comparison with Praat.
//!
//! Usage: pitch_json <audio_file> <time_step> <pitch_floor> <pitch_ceiling>

use praat_core::{Sound, pitch_from_channels};
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct PitchOutput {
    n_frames: usize,
    start_time: f64,
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
    frames: Vec<PitchFrame>,
}

#[derive(Serialize)]
struct PitchFrame {
    time: f64,
    frequency: f64,
    strength: f64,
    voiced: bool,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <audio_file> <time_step> <pitch_floor> <pitch_ceiling>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let time_step: f64 = args[2].parse().expect("Invalid time_step");
    let pitch_floor: f64 = args[3].parse().expect("Invalid pitch_floor");
    let pitch_ceiling: f64 = args[4].parse().expect("Invalid pitch_ceiling");

    // Load channels separately to support stereo (Praat sums autocorrelations)
    let channels = Sound::from_file_channels(audio_path).expect("Failed to load audio file");
    let pitch = pitch_from_channels(&channels, time_step, pitch_floor, pitch_ceiling);

    let mut frames = Vec::with_capacity(pitch.num_frames());
    for i in 0..pitch.num_frames() {
        let time = pitch.get_time_from_frame(i);
        let frequency = pitch.get_value_at_frame(i).unwrap_or(0.0);
        let strength = pitch.get_strength_at_frame(i).unwrap_or(0.0);
        let voiced = frequency > 0.0;

        frames.push(PitchFrame {
            time,
            frequency,
            strength,
            voiced,
        });
    }

    let output = PitchOutput {
        n_frames: pitch.num_frames(),
        start_time: pitch.start_time(),
        time_step: pitch.time_step(),
        pitch_floor: pitch.pitch_floor(),
        pitch_ceiling: pitch.pitch_ceiling(),
        frames,
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
