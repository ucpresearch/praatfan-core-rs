use praatfan_core::Sound;
use praatfan_core::{FrequencyUnit, Interpolation};
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

    println!("Our formant: {} frames, x1={:.4}, dx=0.01", formant.num_frames(), formant.get_time_from_frame(0));
    println!("Praat times: {} values, starting at t={:.4}", gt.formant.times.len(), gt.formant.times[0]);

    // Compare at OUR frame times (not interpolated)
    println!("\nOur frame values vs Praat interpolated to our times:");
    println!("{:>5} {:>8} {:>10} {:>10} {:>10}", "Frame", "Time", "Our F1", "Praat F1", "Diff");

    for frame in 0..25.min(formant.num_frames()) {
        let t = formant.get_time_from_frame(frame);
        let our_f1 = formant.get_value_at_frame(1, frame);
        // Get Praat's value at this same time (using interpolation in their data)
        let praat_f1 = get_praat_f1_at_time(&gt.formant, t);

        let diff = match (our_f1, praat_f1) {
            (Some(o), Some(p)) => format!("{:+.1}", o - p),
            _ => "N/A".to_string(),
        };

        println!("{:>5} {:>8.4} {:>10} {:>10} {:>10}",
                 frame,
                 t,
                 our_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)),
                 praat_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)),
                 diff);
    }

    // Now compare at PRAAT's frame times
    println!("\n\nPraat frame values vs Our interpolated to Praat times:");
    println!("{:>5} {:>8} {:>10} {:>10} {:>10}", "Frame", "Time", "Praat F1", "Our F1", "Diff");

    for i in 0..25.min(gt.formant.times.len()) {
        let t = gt.formant.times[i];
        let praat_f1 = gt.formant.f1[i];
        let our_f1 = formant.get_value_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear);

        let diff = match (praat_f1, our_f1) {
            (Some(p), Some(o)) => format!("{:+.1}", o - p),
            _ => "N/A".to_string(),
        };

        println!("{:>5} {:>8.4} {:>10} {:>10} {:>10}",
                 i,
                 t,
                 praat_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)),
                 our_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)),
                 diff);
    }
}

fn get_praat_f1_at_time(formant: &FormantInfo, time: f64) -> Option<f64> {
    // Linear interpolation in Praat's data
    if formant.times.is_empty() {
        return None;
    }

    // Find surrounding frames
    let mut lower_idx = 0;
    for (i, &t) in formant.times.iter().enumerate() {
        if t <= time {
            lower_idx = i;
        } else {
            break;
        }
    }

    if lower_idx >= formant.times.len() - 1 {
        return formant.f1[lower_idx];
    }

    let t0 = formant.times[lower_idx];
    let t1 = formant.times[lower_idx + 1];
    let v0 = formant.f1[lower_idx]?;
    let v1 = formant.f1[lower_idx + 1]?;

    let frac = (time - t0) / (t1 - t0);
    Some(v0 + frac * (v1 - v0))
}
