use praat_core::praat_formant_window;

fn main() {
    let size = 275;
    let window = praat_formant_window(size);

    println!("Window size: {}", window.len());
    println!("Window sum: {:.6}", window.iter().sum::<f64>());
    println!("Window sum of squares: {:.6}", window.iter().map(|x| x*x).sum::<f64>());

    println!("\nFirst 5 values:");
    for i in 0..5 {
        println!("  window[{}] = {:.6}", i, window[i]);
    }

    println!("\nMiddle values:");
    let mid = size / 2;
    for i in (mid - 2)..=(mid + 2) {
        println!("  window[{}] = {:.6}", i, window[i]);
    }

    println!("\nLast 5 values:");
    for i in (size - 5)..size {
        println!("  window[{}] = {:.6}", i, window[i]);
    }

    // Compare with expected values from Praat formula
    println!("\n--- Manual calculation ---");
    let edge = (-12.0_f64).exp();
    let imid = (size as f64 - 1.0) / 2.0;  // Our formula
    let praat_imid = 0.5 * (size as f64 + 1.0);  // Praat's formula
    println!("Our imid: {}, Praat imid: {}", imid, praat_imid);

    // Calculate for i=0 (our) = i=1 (Praat)
    let i_our = 0;
    let i_praat = 1;
    let diff_our = i_our as f64 - imid;
    let diff_praat = i_praat as f64 - praat_imid;
    let denom = (size + 1) as f64;
    let gauss_our = (-48.0 * diff_our * diff_our / (denom * denom)).exp();
    let gauss_praat = (-48.0 * diff_praat * diff_praat / (denom * denom)).exp();
    let w_our = (gauss_our - edge) / (1.0 - edge);
    let w_praat = (gauss_praat - edge) / (1.0 - edge);
    println!("At first sample: diff_our={}, diff_praat={}", diff_our, diff_praat);
    println!("  gauss_our={:.10}, gauss_praat={:.10}", gauss_our, gauss_praat);
    println!("  w_our={:.10}, w_praat={:.10}", w_our, w_praat);

    // Check middle
    let mid_i = 137;
    let diff = mid_i as f64 - imid;
    let gauss = (-48.0 * diff * diff / (denom * denom)).exp();
    let w = (gauss - edge) / (1.0 - edge);
    println!("\nAt middle (i=137): diff={}, gauss={:.6}, w={:.6}", diff, gauss, w);
    println!("Actual window[137] = {:.6}", window[137]);
}
