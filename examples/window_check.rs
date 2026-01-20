use praatfan_core::praat_formant_window;

fn main() {
    let window = praat_formant_window(275);
    
    println!("Window size: {}", window.len());
    println!("\nWindow values at select indices:");
    for &i in &[0, 50, 100, 137, 138, 175, 200, 274] {
        println!("  i={:3}: window={:.6}", i, window[i]);
    }
}
