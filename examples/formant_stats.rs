use praatfan_core::Sound;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct GroundTruth {
    formant: FormantInfo,
}

#[derive(Deserialize)]
struct FormantInfo {
    times: Vec<f64>,
    f1: Vec<Option<f64>>,
}

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    // Load ground truth
    let file = File::open("tests/ground_truth/one_two_three_four_five.json").unwrap();
    let reader = BufReader::new(file);
    let gt: GroundTruth = serde_json::from_reader(reader).unwrap();

    // Compute our formants
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Formant comparison statistics:");
    println!("  Ground truth frames: {}", gt.formant.times.len());
    println!("  Our frames: {}", formant.num_frames());
    println!("  Our x1: {:.4}", formant.start_time());
    println!();

    let mut diffs: Vec<f64> = Vec::new();
    let mut abs_diffs: Vec<f64> = Vec::new();
    let mut matches = 0;
    let mut mismatches = 0;
    let tolerance = 100.0;

    for (i, &time) in gt.formant.times.iter().enumerate() {
        let expected = gt.formant.f1[i];
        let actual = formant.get_value_at_time(1, time, praatfan_core::FrequencyUnit::Hertz, praatfan_core::Interpolation::Linear);

        match (expected, actual) {
            (Some(e), Some(a)) if e > 0.0 && a.is_finite() => {
                let diff = a - e;
                diffs.push(diff);
                abs_diffs.push(diff.abs());

                if diff.abs() < tolerance {
                    matches += 1;
                } else {
                    mismatches += 1;
                    if mismatches <= 10 {
                        println!("  Mismatch at t={:.3}: expected={:.1}, got={:.1}, diff={:+.1}",
                                 time, e, a, diff);
                    }
                }
            }
            (Some(e), None) if e > 0.0 => {
                mismatches += 1;
                if mismatches <= 10 {
                    println!("  Mismatch at t={:.3}: expected={:.1}, got=None", time, e);
                }
            }
            (None, Some(a)) if a.is_finite() => {
                mismatches += 1;
            }
            _ => {
                // Both undefined - OK
                matches += 1;
            }
        }
    }

    if mismatches > 10 {
        println!("  ... and {} more mismatches", mismatches - 10);
    }

    println!();
    println!("Results:");
    println!("  Matches (within {}Hz): {}", tolerance, matches);
    println!("  Mismatches: {}", mismatches);
    println!("  Match rate: {:.1}%", 100.0 * matches as f64 / (matches + mismatches) as f64);
    println!();

    if !diffs.is_empty() {
        let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let mean_abs_diff: f64 = abs_diffs.iter().sum::<f64>() / abs_diffs.len() as f64;

        let mut sorted_abs = abs_diffs.clone();
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_abs = sorted_abs[sorted_abs.len() / 2];
        let p90_abs = sorted_abs[(sorted_abs.len() as f64 * 0.9) as usize];
        let max_abs = sorted_abs.last().unwrap();

        println!("Difference statistics (Hz):");
        println!("  Mean difference (bias): {:+.1}", mean_diff);
        println!("  Mean absolute difference: {:.1}", mean_abs_diff);
        println!("  Median absolute difference: {:.1}", median_abs);
        println!("  90th percentile: {:.1}", p90_abs);
        println!("  Max absolute difference: {:.1}", max_abs);
        println!();

        // Show distribution
        let under_25 = abs_diffs.iter().filter(|&&d| d < 25.0).count();
        let under_50 = abs_diffs.iter().filter(|&&d| d < 50.0).count();
        let under_100 = abs_diffs.iter().filter(|&&d| d < 100.0).count();
        let under_200 = abs_diffs.iter().filter(|&&d| d < 200.0).count();

        println!("Distribution:");
        println!("  <25 Hz:  {} ({:.1}%)", under_25, 100.0 * under_25 as f64 / abs_diffs.len() as f64);
        println!("  <50 Hz:  {} ({:.1}%)", under_50, 100.0 * under_50 as f64 / abs_diffs.len() as f64);
        println!("  <100 Hz: {} ({:.1}%)", under_100, 100.0 * under_100 as f64 / abs_diffs.len() as f64);
        println!("  <200 Hz: {} ({:.1}%)", under_200, 100.0 * under_200 as f64 / abs_diffs.len() as f64);
    }
}
