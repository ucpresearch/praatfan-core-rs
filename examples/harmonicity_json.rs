//! Output harmonicity analysis as JSON for comparison with Praat.
//!
//! Usage: harmonicity_json <audio_file> <time_step> <min_pitch> <silence_threshold> <periods_per_window>

use praat_core::Sound;
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct HarmonicityOutput {
    n_frames: usize,
    start_time: f64,
    time_step: f64,
    min_pitch: f64,
    values: Vec<HarmonicityFrame>,
}

#[derive(Serialize)]
struct HarmonicityFrame {
    time: f64,
    hnr: f64,
    voiced: bool,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: {} <audio_file> <time_step> <min_pitch> <silence_threshold> <periods_per_window>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let time_step: f64 = args[2].parse().expect("Invalid time_step");
    let min_pitch: f64 = args[3].parse().expect("Invalid min_pitch");
    let silence_threshold: f64 = args[4].parse().expect("Invalid silence_threshold");
    let periods_per_window: f64 = args[5].parse().expect("Invalid periods_per_window");

    let sound = Sound::from_file(audio_path).expect("Failed to load audio file");
    let hnr = sound.to_harmonicity_cc(time_step, min_pitch, silence_threshold, periods_per_window);

    let mut values = Vec::with_capacity(hnr.num_frames());
    for i in 0..hnr.num_frames() {
        let time = hnr.get_time_from_frame(i);
        let hnr_val = hnr.values()[i];
        let voiced = hnr_val > -199.0;

        values.push(HarmonicityFrame {
            time,
            hnr: hnr_val,
            voiced,
        });
    }

    let output = HarmonicityOutput {
        n_frames: hnr.num_frames(),
        start_time: hnr.start_time(),
        time_step: hnr.time_step(),
        min_pitch: hnr.min_pitch(),
        values,
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
