//! Output spectrogram analysis as JSON for comparison with Praat.
//!
//! Usage: spectrogram_json <audio_file> <window_length> <max_freq> <time_step> <freq_step>
//!
//! Handles multi-channel audio with Praat-compatible power averaging.

use praatfan_core::{Sound, WindowShape, spectrogram_from_channels};
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct SpectrogramOutput {
    sample_rate: f64,
    duration: f64,
    n_samples: usize,
    n_channels: usize,
    spectrogram: SpectrogramData,
}

#[derive(Serialize)]
struct SpectrogramData {
    n_times: usize,
    n_freqs: usize,
    time_step: f64,
    freq_step: f64,
    first_time: f64,
    first_freq: f64,
    test_points: Vec<TestPoint>,
}

#[derive(Serialize)]
struct TestPoint {
    time: f64,
    freq: f64,
    power: Option<f64>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: {} <audio_file> <window_length> <max_freq> <time_step> <freq_step>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let window_length: f64 = args[2].parse().expect("Invalid window_length");
    let max_freq: f64 = args[3].parse().expect("Invalid max_freq");
    let time_step: f64 = args[4].parse().expect("Invalid time_step");
    let freq_step: f64 = args[5].parse().expect("Invalid freq_step");

    // Load channels separately for Praat-compatible multi-channel handling
    let channels = Sound::from_file_channels(audio_path).expect("Failed to load audio file");
    let n_channels = channels.len();

    // Use the first channel for metadata (all channels should have same properties)
    let sound = &channels[0];

    // Compute spectrogram with Praat-compatible power averaging for multi-channel
    let spectrogram = spectrogram_from_channels(&channels, time_step, max_freq, window_length, freq_step, WindowShape::Gaussian);

    // Get timing info
    let n_times = spectrogram.num_frames();
    let n_freqs = spectrogram.num_freq_bins();

    // Sample test points at same locations as Python script
    // Use actual grid values (snapped to nearest frame/bin) like Praat does
    let mut test_points = Vec::new();
    for t_frac in [0.25, 0.5, 0.75] {
        let t = sound.duration() * t_frac;
        let frame = spectrogram.get_frame_from_time(t);
        if frame < n_times {
            let actual_t = spectrogram.get_time_from_frame(frame);
            // Sample at various frequency bins across the spectrum
            for bin_idx in [5, 10, 20, 40, 64, 100, 150, 200] {
                if bin_idx < n_freqs {
                    let actual_f = spectrogram.get_frequency_from_bin(bin_idx);
                    let power = spectrogram.get_power_at(actual_t, actual_f);
                    test_points.push(TestPoint {
                        time: actual_t,
                        freq: actual_f,
                        power,
                    });
                }
            }
        }
    }

    let output = SpectrogramOutput {
        sample_rate: sound.sample_rate(),
        duration: sound.duration(),
        n_samples: sound.num_samples(),
        n_channels,
        spectrogram: SpectrogramData {
            n_times,
            n_freqs,
            time_step: spectrogram.time_step(),
            freq_step: spectrogram.freq_step(),
            first_time: spectrogram.start_time(),
            first_freq: spectrogram.freq_min(),
            test_points,
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
