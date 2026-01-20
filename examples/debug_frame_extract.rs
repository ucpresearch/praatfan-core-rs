use praatfan_core::Sound;

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    println!("Original: {} samples, {} Hz, duration={:.4}s, start_time={:.6}",
             sound.samples().len(), sound.sample_rate(), sound.duration(), sound.start_time());

    let resampled = sound.resample(11000.0);
    println!("Resampled: {} samples, {} Hz, duration={:.4}s, start_time={:.6}",
             resampled.samples().len(), resampled.sample_rate(), resampled.duration(), resampled.start_time());

    let emphasized = resampled.pre_emphasis(50.0);
    println!("Emphasized: {} samples, {} Hz, duration={:.4}s, start_time={:.6}",
             emphasized.samples().len(), emphasized.sample_rate(), emphasized.duration(), emphasized.start_time());

    let sample_rate = emphasized.sample_rate();
    let window_length = 0.025;
    let time_step = 0.01;
    let window_samples = (window_length * sample_rate).round() as usize;
    let half_window = window_samples / 2;

    println!("\nWindow: {} samples, half_window={}", window_samples, half_window);

    // Calculate frame timing using Praat's formula
    let my_duration = emphasized.duration();
    let my_dx = 1.0 / sample_rate;
    let my_x1 = emphasized.start_time() + 0.5 * my_dx;
    let num_frames = ((my_duration - window_length) / time_step).floor() as usize + 1;
    let our_mid_time = my_x1 - 0.5 * my_dx + 0.5 * my_duration;
    let thy_duration = num_frames as f64 * time_step;
    let first_frame_time = our_mid_time - 0.5 * thy_duration + 0.5 * time_step;

    println!("\nPraat formula:");
    println!("  my_duration = {:.6}", my_duration);
    println!("  my_dx = {:.9}", my_dx);
    println!("  my_x1 = {:.9}", my_x1);
    println!("  num_frames = {}", num_frames);
    println!("  our_mid_time = {:.6}", our_mid_time);
    println!("  thy_duration = {:.6}", thy_duration);
    println!("  first_frame_time = {:.6}", first_frame_time);

    println!("\nFirst 5 frames:");
    for i in 0..5 {
        let frame_time = first_frame_time + i as f64 * time_step;
        let center_sample_float = (frame_time - emphasized.start_time()) * sample_rate;
        let center_sample = center_sample_float.round() as usize;
        let center_sample_clamped = center_sample.min(emphasized.samples().len().saturating_sub(1));
        let start_sample = center_sample_clamped.saturating_sub(half_window);
        let end_sample = (center_sample_clamped + half_window).min(emphasized.samples().len());

        println!("  Frame {}: t={:.4}, center_sample_raw={:.1}, center_sample={}, start={}, end={}",
                 i, frame_time, center_sample_float, center_sample_clamped, start_sample, end_sample);
    }
}
