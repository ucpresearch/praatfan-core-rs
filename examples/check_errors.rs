use praatfan_core::{Sound, FrequencyUnit, Interpolation};
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct GroundTruth { formant: FormantInfo }

#[derive(Deserialize)]
struct FormantInfo {
    times: Vec<f64>,
    f1: Vec<Option<f64>>,
    f2: Vec<Option<f64>>,
    f3: Vec<Option<f64>>,
    time_step: f64,
    max_num_formants: usize,
    max_formant_hz: f64,
    window_length: f64,
}

struct FormantStats {
    name: String,
    total: usize,
    within_1hz: usize,
    within_5hz: usize,
    max_error: f64,
}

fn check_formant(praat_values: &[Option<f64>], times: &[f64],
                 formant: &praatfan_core::Formant, formant_num: usize, verbose: bool) -> Option<FormantStats> {
    let mut errors: Vec<(f64, f64, f64, f64)> = Vec::new(); // (error, time, expected, actual)
    for (i, &t) in times.iter().enumerate() {
        if let (Some(e), Some(a)) = (praat_values[i],
            formant.get_value_at_time(formant_num, t, FrequencyUnit::Hertz, Interpolation::Linear)) {
            errors.push(((e - a).abs(), t, e, a));
        }
    }
    if errors.is_empty() {
        return None;
    }
    errors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if verbose && errors.last().unwrap().0 > 1.0 {
        println!("    Worst errors for F{}:", formant_num);
        for (err, t, expected, actual) in errors.iter().rev().take(5) {
            println!("      t={:.3}s: expected={:.1}, actual={:.1}, error={:.1} Hz",
                     t, expected, actual, err);
        }
    }

    Some(FormantStats {
        name: format!("F{}", formant_num),
        total: errors.len(),
        within_1hz: errors.iter().filter(|e| e.0 <= 1.0).count(),
        within_5hz: errors.iter().filter(|e| e.0 <= 5.0).count(),
        max_error: errors.last().unwrap().0,
    })
}

fn test_file(audio_path: &str, ground_truth_path: &str, verbose: bool) {
    let file = match File::open(ground_truth_path) {
        Ok(f) => f,
        Err(e) => {
            println!("  Skipping: {}", e);
            return;
        }
    };
    let gt: GroundTruth = serde_json::from_reader(BufReader::new(file)).unwrap();

    let sound = match Sound::from_file(audio_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  Skipping: {}", e);
            return;
        }
    };

    if verbose {
        println!("  Loaded: {} Hz, {} samples, {:.6}s", sound.sample_rate(), sound.num_samples(), sound.duration());
    }

    let formant = sound.to_formant_burg(gt.formant.time_step, gt.formant.max_num_formants,
        gt.formant.max_formant_hz, gt.formant.window_length, 50.0);

    for fnum in 1..=3 {
        let values = match fnum {
            1 => &gt.formant.f1,
            2 => &gt.formant.f2,
            3 => &gt.formant.f3,
            _ => unreachable!(),
        };
        if let Some(stats) = check_formant(values, &gt.formant.times, &formant, fnum, verbose) {
            let pct_1hz = 100.0 * stats.within_1hz as f64 / stats.total as f64;
            let pct_5hz = 100.0 * stats.within_5hz as f64 / stats.total as f64;
            println!("  {}: {:3}/{} within 1 Hz ({:5.1}%), {:3}/{} within 5 Hz ({:5.1}%), max: {:.1} Hz",
                     stats.name, stats.within_1hz, stats.total, pct_1hz,
                     stats.within_5hz, stats.total, pct_5hz, stats.max_error);
        }
    }
}

fn main() {
    let test_cases = [
        ("one_two_three_four_five", "tests/fixtures/one_two_three_four_five.wav", false),
        ("one_two_three_four_five_16k", "tests/fixtures/one_two_three_four_five_16k.wav", false),
        ("one_two_three_four_five_44k", "tests/fixtures/one_two_three_four_five_44k.wav", false),
        ("one_two_three_four_five_48k", "tests/fixtures/one_two_three_four_five_48k.wav", false),
        ("one_two_three_four_five_8bit", "tests/fixtures/one_two_three_four_five_8bit.wav", false),
        ("one_two_three_four_five_24bit", "tests/fixtures/one_two_three_four_five_24bit.wav", false),
        ("one_two_three_four_five_32float", "tests/fixtures/one_two_three_four_five_32float.wav", false),
        ("one_two_three_four_five_stereo", "tests/fixtures/one_two_three_four_five_stereo.wav", true),
    ];

    println!("=== WAV files ===");
    for (name, audio_path, verbose) in test_cases {
        let gt_path = format!("tests/ground_truth/{}.json", name);
        println!("\n{}:", name);
        test_file(audio_path, &gt_path, verbose);
    }

    // Test non-WAV formats (use same ground truth as WAV since they're the same audio)
    println!("\n\n=== Non-WAV formats ===");
    test_non_wav_format("FLAC", "tests/fixtures/one_two_three_four_five.flac");
    test_non_wav_format("MP3", "tests/fixtures/one_two_three_four_five.mp3");
    test_non_wav_format("OGG", "tests/fixtures/one_two_three_four_five.ogg");
}

fn test_non_wav_format(format_name: &str, audio_path: &str) {
    println!("\n{}:", format_name);
    let sound = match Sound::from_file(audio_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  Error loading: {}", e);
            return;
        }
    };
    println!("  Loaded: {} Hz, {} samples, {:.3}s",
             sound.sample_rate(), sound.num_samples(), sound.duration());

    // Perform formant analysis
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    // Get a few formant values to verify it works
    let test_times = [0.5, 0.8, 1.0];
    for t in test_times {
        if let Some(f1) = formant.get_value_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear) {
            if let Some(f2) = formant.get_value_at_time(2, t, FrequencyUnit::Hertz, Interpolation::Linear) {
                println!("  t={:.1}s: F1={:.0} Hz, F2={:.0} Hz", t, f1, f2);
            }
        }
    }
}
