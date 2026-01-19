use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    
    println!("Original signal around t=0.026:");
    let samples = sound.samples();
    let sr = sound.sample_rate();
    for i in 0..10 {
        let t = 0.02 + i as f64 * 0.001;
        let idx = ((t - sound.start_time()) * sr).round() as usize;
        let v = if idx < samples.len() { samples[idx] } else { 0.0 };
        println!("  t={:.4}: {:.10e}", t, v);
    }
    
    // Resample
    let resampled = sound.resample(11000.0);
    let samples2 = resampled.samples();
    let sr2 = resampled.sample_rate();
    
    println!("\nResampled signal around t=0.026:");
    for i in 0..10 {
        let t = 0.02 + i as f64 * 0.001;
        let idx = ((t - resampled.start_time()) * sr2).round() as usize;
        let v = if idx < samples2.len() { samples2[idx] } else { 0.0 };
        println!("  t={:.4}: {:.10e}", t, v);
    }
    
    // Pre-emphasize
    let emphasized = resampled.pre_emphasis(50.0);
    let samples3 = emphasized.samples();
    
    println!("\nPre-emphasized signal around t=0.026:");
    for i in 0..10 {
        let t = 0.02 + i as f64 * 0.001;
        let idx = ((t - emphasized.start_time()) * sr2).round() as usize;
        let v = if idx < samples3.len() { samples3[idx] } else { 0.0 };
        println!("  t={:.4}: {:.10e}", t, v);
    }
}
