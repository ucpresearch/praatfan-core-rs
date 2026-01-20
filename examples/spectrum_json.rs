//! Output spectrum analysis as JSON for comparison with Praat.
//!
//! Usage: spectrum_json <audio_file>

use praat_core::Sound;
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct SpectrumOutput {
    sample_rate: f64,
    duration: f64,
    n_samples: usize,
    spectrum: SpectrumData,
}

#[derive(Serialize)]
struct SpectrumData {
    center_of_gravity: Option<f64>,
    standard_deviation: Option<f64>,
    skewness: Option<f64>,
    kurtosis: Option<f64>,
    total_energy: Option<f64>,
    low_energy: Option<f64>,
    mid_energy: Option<f64>,
    high_energy: Option<f64>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <audio_file>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let sound = Sound::from_file(audio_path).expect("Failed to load audio file");
    let spectrum = sound.to_spectrum(true); // fast = power of 2

    let nyquist = sound.sample_rate() / 2.0;

    let output = SpectrumOutput {
        sample_rate: sound.sample_rate(),
        duration: sound.duration(),
        n_samples: sound.num_samples(),
        spectrum: SpectrumData {
            center_of_gravity: Some(spectrum.get_center_of_gravity(2.0)),
            standard_deviation: Some(spectrum.get_standard_deviation(2.0)),
            skewness: Some(spectrum.get_skewness(2.0)),
            kurtosis: Some(spectrum.get_kurtosis(2.0)),
            total_energy: Some(spectrum.get_total_energy()),
            low_energy: Some(spectrum.get_band_energy(0.0, 1000.0)),
            mid_energy: Some(spectrum.get_band_energy(1000.0, 4000.0)),
            high_energy: Some(spectrum.get_band_energy(4000.0, nyquist)),
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
