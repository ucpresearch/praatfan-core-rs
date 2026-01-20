use praatfan_core::utils::lpc::{lpc_burg, lpc_to_formants};
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

#[derive(Deserialize)]
struct PraatFrameData {
    time: f64,
    sample_rate: f64,
    lpc_order: usize,
    windowed_samples: Vec<f64>,
    praat_formants: PraatFormants,
    praat_bandwidths: PraatBandwidths,
}

#[derive(Deserialize)]
struct PraatFormants {
    f1: f64,
    f2: f64,
    f3: f64,
    f4: f64,
    f5: Option<f64>,
}

#[derive(Deserialize)]
struct PraatBandwidths {
    b1: f64,
    b2: f64,
}

fn main() {
    // Load Praat's exact windowed samples
    let file = File::open("/tmp/praat_exact_frame.json").expect("Run the Python script first");
    let reader = BufReader::new(file);
    let data: PraatFrameData = serde_json::from_reader(reader).unwrap();

    println!("=== Comparing with Praat's exact frame at t={} ===", data.time);
    println!("\nPraat's formants:");
    println!("  F1 = {:.1} Hz (B1 = {:.1} Hz)", data.praat_formants.f1, data.praat_bandwidths.b1);
    println!("  F2 = {:.1} Hz (B2 = {:.1} Hz)", data.praat_formants.f2, data.praat_bandwidths.b2);
    println!("  F3 = {:.1} Hz", data.praat_formants.f3);
    println!("  F4 = {:.1} Hz", data.praat_formants.f4);

    println!("\nInput samples: {} at {} Hz", data.windowed_samples.len(), data.sample_rate);
    println!("LPC order: {}", data.lpc_order);

    // Run OUR LPC on Praat's exact windowed samples
    if let Some(lpc_result) = lpc_burg(&data.windowed_samples, data.lpc_order) {
        println!("\n=== Our LPC Coefficients ===");
        for (i, c) in lpc_result.coefficients.iter().enumerate() {
            println!("  a[{}] = {:+.15e}", i, c);
        }
        println!("  gain = {:.15e}", lpc_result.gain);

        // Extract formants
        let formants = lpc_to_formants(&lpc_result.coefficients, data.sample_rate);
        println!("\n=== Our Formants ===");
        for (i, f) in formants.iter().enumerate() {
            println!("  F{} = {:.1} Hz (B{} = {:.1} Hz)", i + 1, f.frequency, i + 1, f.bandwidth);
        }

        // Compare
        if !formants.is_empty() {
            println!("\n=== Comparison ===");
            let our_f1 = formants[0].frequency;
            let our_f2 = if formants.len() > 1 { formants[1].frequency } else { 0.0 };
            let our_f3 = if formants.len() > 2 { formants[2].frequency } else { 0.0 };

            println!("  F1: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                     data.praat_formants.f1, our_f1, our_f1 - data.praat_formants.f1);
            println!("  F2: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                     data.praat_formants.f2, our_f2, our_f2 - data.praat_formants.f2);
            println!("  F3: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                     data.praat_formants.f3, our_f3, our_f3 - data.praat_formants.f3);

            // Also compare bandwidths
            if formants.len() >= 2 {
                println!("\n  B1: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                         data.praat_bandwidths.b1, formants[0].bandwidth,
                         formants[0].bandwidth - data.praat_bandwidths.b1);
                println!("  B2: Praat={:.1}, Ours={:.1}, Diff={:+.1} Hz",
                         data.praat_bandwidths.b2, formants[1].bandwidth,
                         formants[1].bandwidth - data.praat_bandwidths.b2);
            }
        }
    } else {
        println!("LPC failed!");
    }
}
