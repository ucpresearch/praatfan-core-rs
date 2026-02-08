//! Output intensity analysis as JSON for comparison with Praat.
//!
//! Usage: intensity_json <audio_file> <min_pitch> <time_step>

use praatfan_core::{Sound, intensity_from_channels};
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct IntensityOutput {
    sample_rate: f64,
    duration: f64,
    n_samples: usize,
    intensity: IntensityData,
}

#[derive(Serialize)]
struct IntensityData {
    times: Vec<f64>,
    values: Vec<Option<f64>>,
    time_step: f64,
    min_pitch: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <audio_file> <min_pitch> <time_step>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let min_pitch: f64 = args[2].parse().expect("Invalid min_pitch");
    let time_step: f64 = args[3].parse().expect("Invalid time_step");

    // Load channels separately to match Praat's multi-channel intensity handling
    let channels = Sound::from_file_channels(audio_path).expect("Failed to load audio file");

    let intensity = intensity_from_channels(&channels, min_pitch, time_step);

    let n_frames = intensity.num_frames();
    let mut times = Vec::with_capacity(n_frames);
    let mut values = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        times.push(intensity.get_time_from_frame(i));
        let v = intensity.values()[i];
        values.push(if v > -300.0 { Some(v) } else { None });
    }

    let output = IntensityOutput {
        sample_rate: channels[0].sample_rate(),
        duration: channels[0].duration(),
        n_samples: channels[0].num_samples(),
        intensity: IntensityData {
            times,
            values,
            time_step: intensity.time_step(),
            min_pitch,
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
