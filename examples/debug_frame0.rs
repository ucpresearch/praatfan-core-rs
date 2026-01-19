use praat_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    
    // Replicate the formant processing steps
    let target_sr = 2.0 * 5500.0;
    let resampled = sound.resample(target_sr);
    let emphasized = resampled.pre_emphasis(50.0);
    
    let sample_rate = emphasized.sample_rate();
    let samples = emphasized.samples();
    
    println!("After preprocessing:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Num samples: {}", samples.len());
    println!("  Duration: {:.4}s", samples.len() as f64 / sample_rate);
    println!();
    
    // Frame 0 at t=0.026
    let window_length = 0.025;
    let window_samples = (window_length * sample_rate).round() as usize;
    let half_window = window_samples / 2;
    
    let frame_time = 0.026;
    let center_sample = ((frame_time - emphasized.start_time()) * sample_rate).round() as usize;
    
    println!("Frame 0 (t=0.026) extraction:");
    println!("  Window samples: {}", window_samples);
    println!("  Half window: {}", half_window);
    println!("  Center sample: {}", center_sample);
    
    let start_sample = center_sample.saturating_sub(half_window);
    let end_sample = (center_sample + half_window).min(samples.len());
    
    println!("  Sample range: {}..{}", start_sample, end_sample);
    
    // Check signal energy in window
    let mut energy = 0.0;
    let mut max_abs = 0.0f64;
    for i in start_sample..end_sample {
        let s = samples[i];
        energy += s * s;
        max_abs = max_abs.max(s.abs());
    }
    println!("  Energy: {:.6}", energy);
    println!("  Max abs sample: {:.6}", max_abs);
    println!();
    
    // Also check original signal at same region
    let orig_samples = sound.samples();
    let orig_sr = sound.sample_rate();
    let orig_center = ((frame_time - sound.start_time()) * orig_sr).round() as usize;
    let orig_half = ((window_length * orig_sr) / 2.0).round() as usize;
    
    println!("Original signal at frame 0:");
    let mut orig_energy = 0.0;
    for i in orig_center.saturating_sub(orig_half)..orig_center.saturating_add(orig_half).min(orig_samples.len()) {
        orig_energy += orig_samples[i] * orig_samples[i];
    }
    println!("  Energy: {:.6}", orig_energy);
}
