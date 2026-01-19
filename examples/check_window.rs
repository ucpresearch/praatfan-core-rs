use praat_core::window::praat_formant_window;

fn main() {
    let window = praat_formant_window(275);
    
    println!("Our Gaussian window (first 10 and center values):");
    for i in 0..10 {
        println!("w[{}] = {:.15e}", i, window[i]);
    }
    
    println!("\nCenter values:");
    for i in 135..=140 {
        println!("w[{}] = {:.15e}", i, window[i]);
    }
}
