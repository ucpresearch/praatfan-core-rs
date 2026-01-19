use praat_core::{Sound, Interpolation};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    
    // Use same params as ground truth
    let hnr = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0);
    
    println!("HNR analysis:");
    println!("  Num frames: {}", hnr.num_frames());
    println!("  Start time: {}", hnr.start_time());
    println!();
    
    println!("First 10 values:");
    for i in 0..10 {
        let t = i as f64 * 0.01;
        let v = hnr.get_value_at_time(t, Interpolation::Cubic);
        let v_str = v.map_or("None".to_string(), |x| format!("{:.1}", x));
        println!("  t={:.2}: HNR={}", t, v_str);
    }
    
    // Also check some later times
    println!("\nLater values:");
    for t in [0.3, 0.5, 0.8, 1.0] {
        let v = hnr.get_value_at_time(t, Interpolation::Cubic);
        let v_str = v.map_or("None".to_string(), |x| format!("{:.1}", x));
        println!("  t={:.2}: HNR={}", t, v_str);
    }
}
