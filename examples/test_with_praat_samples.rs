use praat_core::utils::lpc::{lpc_burg, lpc_to_formants};
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct PraatFrameData {
    time: f64,
    windowed_samples: Vec<f64>,
    praat_f1: f64,
    praat_f2: f64,
}

fn main() {
    // Load Praat's exact windowed samples
    let file = File::open("/tmp/praat_frame_data.json").expect("Run the Python script first");
    let reader = BufReader::new(file);
    let data: PraatFrameData = serde_json::from_reader(reader).unwrap();

    println!("Testing with Praat's exact windowed samples at t={}", data.time);
    println!("Praat's results: F1={:.1} Hz, F2={:.1} Hz", data.praat_f1, data.praat_f2);

    println!("\nWindowed samples (first 10):");
    for i in 0..10 {
        println!("  [{}] = {:.15e}", i, data.windowed_samples[i]);
    }

    println!("\nWindowed samples (center 5):");
    for i in 135..140 {
        println!("  [{}] = {:.15e}", i, data.windowed_samples[i]);
    }

    // Run OUR LPC on Praat's exact windowed samples
    let lpc_order = 12; // 2 * 5 + 2
    let sample_rate = 11000.0;

    if let Some(lpc_result) = lpc_burg(&data.windowed_samples, lpc_order) {
        println!("\nOur LPC coefficients:");
        for (i, c) in lpc_result.coefficients.iter().enumerate() {
            println!("  a[{}] = {:.15e}", i, c);
        }
        println!("  gain = {:.15e}", lpc_result.gain);

        let formants = lpc_to_formants(&lpc_result.coefficients, sample_rate);
        println!("\nOur formants from Praat's samples:");
        for (i, f) in formants.iter().take(5).enumerate() {
            println!("  F{} = {:.1} Hz (bw={:.1} Hz)", i + 1, f.frequency, f.bandwidth);
        }

        // Compare
        if !formants.is_empty() {
            let our_f1 = formants[0].frequency;
            let our_f2 = if formants.len() > 1 { formants[1].frequency } else { 0.0 };
            println!("\nComparison:");
            println!("  F1: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                     data.praat_f1, our_f1, our_f1 - data.praat_f1);
            println!("  F2: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                     data.praat_f2, our_f2, our_f2 - data.praat_f2);
        }
    } else {
        println!("LPC failed!");
    }
}
