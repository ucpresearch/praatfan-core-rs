use praat_core::Sound;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct GroundTruth {
    pitch: PitchInfo,
    formant: FormantInfo,
}

#[derive(Deserialize)]
struct PitchInfo {
    times: Vec<f64>,
    values: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct FormantInfo {
    times: Vec<f64>,
    f2: Vec<Option<f64>>,
}

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let file = File::open("tests/ground_truth/one_two_three_four_five.json").unwrap();
    let reader = BufReader::new(file);
    let gt: GroundTruth = serde_json::from_reader(reader).unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    // Build a set of voiced times (where pitch is detected)
    let mut voiced_times: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for (i, &time) in gt.pitch.times.iter().enumerate() {
        if let Some(pitch) = gt.pitch.values.get(i).and_then(|v| *v) {
            if pitch > 0.0 && pitch.is_finite() {
                voiced_times.insert((time * 1000.0).round() as i64);
            }
        }
    }

    println!("Total pitch frames: {}", gt.pitch.times.len());
    println!("Voiced frames (F0 detected): {}", voiced_times.len());

    // Compare F2 values only at voiced times
    let mut abs_diffs: Vec<f64> = Vec::new();
    let mut matches = 0;
    let mut total = 0;

    for (i, &time) in gt.formant.times.iter().enumerate() {
        let time_key = (time * 1000.0).round() as i64;

        if !voiced_times.contains(&time_key) {
            continue;
        }

        let expected = gt.formant.f2.get(i).and_then(|v| *v);
        let actual = formant.get_value_at_time(2, time, praat_core::FrequencyUnit::Hertz, praat_core::Interpolation::Linear);

        match (expected, actual) {
            (Some(e), Some(a)) if e > 0.0 && a.is_finite() => {
                let diff = (a - e).abs();
                abs_diffs.push(diff);
                total += 1;
                if diff < 100.0 {
                    matches += 1;
                }
            }
            _ => {}
        }
    }

    if !abs_diffs.is_empty() {
        let mean = abs_diffs.iter().sum::<f64>() / abs_diffs.len() as f64;
        let mut sorted = abs_diffs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let rate = 100.0 * matches as f64 / total as f64;
        let max_diff = sorted.last().unwrap();
        let min_diff = sorted.first().unwrap();

        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];
        let p90 = sorted[9 * sorted.len() / 10];

        println!("\n=== F2 differences in VOICED regions only ===");
        println!("Frames compared: {}", total);
        println!("Matches (<100Hz): {}/{} ({:.1}%)", matches, total, rate);
        println!("\nError statistics:");
        println!("  Mean:   {:.1} Hz", mean);
        println!("  Median: {:.1} Hz", median);
        println!("  Min:    {:.1} Hz", min_diff);
        println!("  Max:    {:.1} Hz", max_diff);
        println!("  25th percentile: {:.1} Hz", p25);
        println!("  75th percentile: {:.1} Hz", p75);
        println!("  90th percentile: {:.1} Hz", p90);
    }
}
