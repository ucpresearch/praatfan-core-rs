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
    let file = File::open("tests/ground_truth/one_two_three_four_five.json").unwrap();
    let reader = BufReader::new(file);
    let gt: GroundTruth = serde_json::from_reader(reader).unwrap();
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    // Split into regions based on time
    let regions = [
        ("All frames", 0.0, 2.0),
        ("t >= 0.14 (voiced)", 0.14, 2.0),
        ("t >= 0.20 (mid speech)", 0.20, 2.0),
        ("0.14 <= t <= 0.40", 0.14, 0.40),
        ("0.40 <= t <= 0.80", 0.40, 0.80),
        ("0.80 <= t <= 1.20", 0.80, 1.20),
    ];

    for (name, t_min, t_max) in &regions {
        let mut abs_diffs: Vec<f64> = Vec::new();
        let mut matches = 0;
        let mut total = 0;

        for (i, &time) in gt.formant.times.iter().enumerate() {
            if time < *t_min || time > *t_max {
                continue;
            }
            
            let expected = gt.formant.f1[i];
            let actual = formant.get_value_at_time(1, time, praatfan_core::FrequencyUnit::Hertz, praatfan_core::Interpolation::Linear);

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
            
            println!("{}: {}/{} ({:.1}%), mean={:.1}, median={:.1}", 
                     name, matches, total, rate, mean, median);
        }
    }
}
