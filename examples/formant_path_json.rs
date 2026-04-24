//! Output FormantPath analysis as JSON for comparison with Praat.
//!
//! Runs `Sound::to_formant_path_burg` plus `path_finder` and `extract_formant`,
//! then emits per-frame candidate indices, ceiling frequencies, per-candidate
//! stresses (over the full clip), and F1/F2/F3 from the extracted (path-
//! selected) Formant.
//!
//! References:
//! - Boersma, P. & Weenink, D. (2024). *Praat: doing phonetics by computer.*
//! - Weenink, D. *FormantPath* (Praat: LPC/FormantPath.cpp, GPL-3).
//! - Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth.
//!   *Journal of Phonetics, 71*, 1–15.
//!
//! Usage: formant_path_json <audio> <time_step> <max_formants> <middle_ceiling>
//!        <window_length> <pre_emphasis> <ceiling_step_size> <steps_up_down>
//!        <q_w> <freq_change_w> <stress_w> <ceiling_change_w>
//!        <intensity_mod_step> <path_window_length> <parameters_csv> <power>

use praatfan_core::Sound;
use serde::Serialize;
use std::env;

#[derive(Serialize)]
struct Output {
    sample_rate: f64,
    duration: f64,
    num_candidates: usize,
    ceilings: Vec<f64>,
    times: Vec<f64>,
    path: Vec<usize>,               // per-frame candidate index (0-based in Rust, 1-based for Praat comparison)
    path_1based: Vec<usize>,
    candidate_stresses: Vec<f64>,   // one per candidate, computed over [start, end]
    extracted: Extracted,
}

#[derive(Serialize)]
struct Extracted {
    times: Vec<f64>,
    f1: Vec<Option<f64>>,
    f2: Vec<Option<f64>>,
    f3: Vec<Option<f64>>,
    b1: Vec<Option<f64>>,
    b2: Vec<Option<f64>>,
    b3: Vec<Option<f64>>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 17 {
        eprintln!(
            "Usage: {} <audio> <time_step> <max_formants> <middle_ceiling> \
             <window_length> <pre_emphasis> <ceiling_step_size> <steps_up_down> \
             <q_w> <freq_change_w> <stress_w> <ceiling_change_w> \
             <intensity_mod_step> <path_window_length> <parameters_csv> <power>",
            args[0]
        );
        std::process::exit(1);
    }
    let audio_path = &args[1];
    let time_step: f64 = args[2].parse().expect("time_step");
    let max_formants: usize = args[3].parse().expect("max_formants");
    let middle_ceiling: f64 = args[4].parse().expect("middle_ceiling");
    let window_length: f64 = args[5].parse().expect("window_length");
    let pre_emphasis: f64 = args[6].parse().expect("pre_emphasis");
    let ceiling_step_size: f64 = args[7].parse().expect("ceiling_step_size");
    let steps_up_down: usize = args[8].parse().expect("steps_up_down");
    let q_w: f64 = args[9].parse().expect("q_w");
    let freq_change_w: f64 = args[10].parse().expect("freq_change_w");
    let stress_w: f64 = args[11].parse().expect("stress_w");
    let ceiling_change_w: f64 = args[12].parse().expect("ceiling_change_w");
    let intensity_mod_step: f64 = args[13].parse().expect("intensity_mod_step");
    let path_window_length: f64 = args[14].parse().expect("path_window_length");
    let parameters: Vec<i64> = args[15]
        .split(',')
        .map(|s| s.trim().parse().expect("parameter"))
        .collect();
    let power: f64 = args[16].parse().expect("power");

    let sound = Sound::from_file(audio_path).expect("load audio");

    let mut fp = sound.to_formant_path_burg(
        time_step,
        max_formants,
        middle_ceiling,
        window_length,
        pre_emphasis,
        ceiling_step_size,
        steps_up_down,
    );

    fp.path_finder(
        q_w,
        freq_change_w,
        stress_w,
        ceiling_change_w,
        intensity_mod_step,
        path_window_length,
        &parameters,
        power,
    );

    let n = fp.num_frames();
    let ts = fp.time_step();
    let t0 = fp.start_time();
    let times: Vec<f64> = (0..n).map(|i| t0 + i as f64 * ts).collect();

    let path_0: Vec<usize> = fp.path().to_vec();
    let path_1: Vec<usize> = path_0.iter().map(|&c| c + 1).collect();

    // Candidate stresses over the full frame range (matches
    // `call(fp, "Get stress of candidate", tmin, tmax, 0, 0, "…", power)`).
    let t_min = sound.start_time();
    let t_max = sound.start_time() + sound.duration();
    let candidate_stresses: Vec<f64> =
        fp.get_stress_of_candidates(t_min, t_max, 0, 0, &parameters, power);

    let extracted = fp.extract_formant();
    let ex_times: Vec<f64> = (0..extracted.num_frames())
        .map(|i| extracted.get_time_from_frame(i))
        .collect();
    let mut f1 = Vec::with_capacity(extracted.num_frames());
    let mut f2 = Vec::with_capacity(extracted.num_frames());
    let mut f3 = Vec::with_capacity(extracted.num_frames());
    let mut b1 = Vec::with_capacity(extracted.num_frames());
    let mut b2 = Vec::with_capacity(extracted.num_frames());
    let mut b3 = Vec::with_capacity(extracted.num_frames());
    for i in 0..extracted.num_frames() {
        f1.push(extracted.get_value_at_frame(1, i));
        f2.push(extracted.get_value_at_frame(2, i));
        f3.push(extracted.get_value_at_frame(3, i));
        b1.push(extracted.get_bandwidth_at_frame(1, i));
        b2.push(extracted.get_bandwidth_at_frame(2, i));
        b3.push(extracted.get_bandwidth_at_frame(3, i));
    }

    let output = Output {
        sample_rate: sound.sample_rate(),
        duration: sound.duration(),
        num_candidates: fp.num_candidates(),
        ceilings: fp.ceilings().to_vec(),
        times,
        path: path_0,
        path_1based: path_1,
        candidate_stresses,
        extracted: Extracted {
            times: ex_times,
            f1,
            f2,
            f3,
            b1,
            b2,
            b3,
        },
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
