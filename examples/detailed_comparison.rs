use praat_core::Sound;
use praat_core::{FrequencyUnit, Interpolation};
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
    f2: Vec<Option<f64>>,
}

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let file = File::open("tests/ground_truth/one_two_three_four_five.json").unwrap();
    let reader = BufReader::new(file);
    let gt: GroundTruth = serde_json::from_reader(reader).unwrap();

    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    // Analyze specific regions
    let regions = [
        ("0.40-0.50 (good region)", 0.40, 0.50),
        ("0.80-0.90 (bad region)", 0.80, 0.90),
        ("1.00-1.10 (also bad?)", 1.00, 1.10),
        ("0.14-0.24 (early voiced)", 0.14, 0.24),
    ];

    for (name, t_min, t_max) in &regions {
        println!("\n=== {} ===", name);
        println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}", "Time", "Our F1", "Praat F1", "Diff", "Our F2", "Praat F2");

        for (i, &time) in gt.formant.times.iter().enumerate() {
            if time < *t_min || time >= *t_max {
                continue;
            }

            let our_f1 = formant.get_value_at_time(1, time, FrequencyUnit::Hertz, Interpolation::Linear);
            let our_f2 = formant.get_value_at_time(2, time, FrequencyUnit::Hertz, Interpolation::Linear);
            let praat_f1 = gt.formant.f1[i];
            let praat_f2 = gt.formant.f2.get(i).and_then(|x| *x);

            let diff = match (our_f1, praat_f1) {
                (Some(o), Some(p)) => format!("{:+.0}", o - p),
                _ => "N/A".to_string(),
            };

            println!("{:>6.2} {:>10} {:>10} {:>10} {:>10} {:>10}",
                     time,
                     our_f1.map_or("NaN".to_string(), |v| format!("{:.0}", v)),
                     praat_f1.map_or("NaN".to_string(), |v| format!("{:.0}", v)),
                     diff,
                     our_f2.map_or("NaN".to_string(), |v| format!("{:.0}", v)),
                     praat_f2.map_or("NaN".to_string(), |v| format!("{:.0}", v)));
        }
    }
}
