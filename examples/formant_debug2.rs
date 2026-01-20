// Detailed formant debug - shows LPC candidates
use praatfan_core::{Sound, FrequencyUnit, Interpolation};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    
    // Use the same parameters as Praat
    let formant = sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0);
    
    println!("Formant analysis:");
    println!("  Num frames: {}", formant.num_frames());
    println!("  Start time: {}", formant.start_time());
    
    // Get frame values at specific frame
    let frame_idx = 1; // Second frame, around t=0.03
    let frame_time = formant.start_time() + frame_idx as f64 * formant.time_step();
    
    println!("\nFrame {} (t={:.4}s):", frame_idx, frame_time);
    for i in 1..=5 {
        let f = formant.get_value_at_frame(i, frame_idx);
        let b = formant.get_bandwidth_at_frame(i, frame_idx);
        
        let f_str = f.map_or("None".to_string(), |v| format!("{:.1}", v));
        let b_str = b.map_or("None".to_string(), |v| format!("{:.1}", v));
        
        println!("  F{}: {} Hz (B: {} Hz)", i, f_str, b_str);
    }
    
    // Also check interpolated values at exactly t=0.03
    println!("\nInterpolated at t=0.03:");
    for i in 1..=5 {
        let f = formant.get_value_at_time(i, 0.03, FrequencyUnit::Hertz, Interpolation::Linear);
        let f_str = f.map_or("None".to_string(), |v| format!("{:.1}", v));
        println!("  F{}: {} Hz", i, f_str);
    }
    
    // Compare with Praat expected values
    println!("\nPraat expected at t=0.03:");
    println!("  F1: 874.7 Hz");
    println!("  F2: 2051.8 Hz");
    println!("  F3: 3227.1 Hz");
}
