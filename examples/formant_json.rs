//! Output formant analysis as JSON for comparison with Praat.
//!
//! Usage: formant_json <audio_file> <time_step> <max_formants> <max_formant_hz> <window_length> <pre_emphasis>

use praatfan_core::Sound;
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct FormantOutput {
    sample_rate: f64,
    duration: f64,
    n_samples: usize,
    formant: FormantData,
}

#[derive(Serialize)]
struct FormantData {
    times: Vec<f64>,
    time_step: f64,
    max_num_formants: usize,
    max_formant_hz: f64,
    window_length: f64,
    f1: Vec<Option<f64>>,
    f2: Vec<Option<f64>>,
    f3: Vec<Option<f64>>,
    f4: Vec<Option<f64>>,
    f5: Vec<Option<f64>>,
    b1: Vec<Option<f64>>,
    b2: Vec<Option<f64>>,
    b3: Vec<Option<f64>>,
    b4: Vec<Option<f64>>,
    b5: Vec<Option<f64>>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "Usage: {} <audio_file> <time_step> <max_formants> <max_formant_hz> <window_length> <pre_emphasis>",
            args[0]
        );
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let time_step: f64 = args[2].parse().expect("Invalid time_step");
    let max_formants: usize = args[3].parse().expect("Invalid max_formants");
    let max_formant_hz: f64 = args[4].parse().expect("Invalid max_formant_hz");
    let window_length: f64 = args[5].parse().expect("Invalid window_length");
    let pre_emphasis: f64 = args[6].parse().expect("Invalid pre_emphasis");

    let sound = Sound::from_file(audio_path).expect("Failed to load audio file");

    let formant = sound.to_formant_burg(
        time_step,
        max_formants,
        max_formant_hz,
        window_length,
        pre_emphasis,
    );

    // Get frame times
    let n_frames = formant.num_frames();
    let mut times = Vec::with_capacity(n_frames);
    let mut f1 = Vec::with_capacity(n_frames);
    let mut f2 = Vec::with_capacity(n_frames);
    let mut f3 = Vec::with_capacity(n_frames);
    let mut f4 = Vec::with_capacity(n_frames);
    let mut f5 = Vec::with_capacity(n_frames);
    let mut b1 = Vec::with_capacity(n_frames);
    let mut b2 = Vec::with_capacity(n_frames);
    let mut b3 = Vec::with_capacity(n_frames);
    let mut b4 = Vec::with_capacity(n_frames);
    let mut b5 = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        times.push(formant.get_time_from_frame(i));

        f1.push(formant.get_value_at_frame(1, i));
        f2.push(formant.get_value_at_frame(2, i));
        f3.push(formant.get_value_at_frame(3, i));
        f4.push(formant.get_value_at_frame(4, i));
        f5.push(formant.get_value_at_frame(5, i));

        b1.push(formant.get_bandwidth_at_frame(1, i));
        b2.push(formant.get_bandwidth_at_frame(2, i));
        b3.push(formant.get_bandwidth_at_frame(3, i));
        b4.push(formant.get_bandwidth_at_frame(4, i));
        b5.push(formant.get_bandwidth_at_frame(5, i));
    }

    let output = FormantOutput {
        sample_rate: sound.sample_rate(),
        duration: sound.duration(),
        n_samples: sound.num_samples(),
        formant: FormantData {
            times,
            time_step,
            max_num_formants: max_formants,
            max_formant_hz,
            window_length,
            f1,
            f2,
            f3,
            f4,
            f5,
            b1,
            b2,
            b3,
            b4,
            b5,
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
