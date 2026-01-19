use praat_core::Sound;
use praat_core::window::praat_formant_window;
use praat_core::utils::lpc::{lpc_burg, lpc_to_formants};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let resampled = sound.resample(11000.0);
    let emphasized = resampled.pre_emphasis(50.0);
    let samples = emphasized.samples();
    let sample_rate = emphasized.sample_rate();

    let window_samples = 275;
    let half_window = window_samples / 2;
    let window = praat_formant_window(window_samples);
    let lpc_order = 12;

    // Test frames 0 and 1 (which showed identical values)
    let test_centers: [usize; 2] = [176, 286];

    for &center_sample in &test_centers {
        let start_sample = center_sample.saturating_sub(half_window);
        let end_sample = (center_sample + half_window).min(samples.len());

        println!("Center sample: {} (start={}, end={})", center_sample, start_sample, end_sample);

        // Extract windowed frame
        let mut windowed: Vec<f64> = Vec::with_capacity(window_samples);
        let mut energy = 0.0;
        for (i, sample_idx) in (start_sample..end_sample).enumerate() {
            let w = if i < window.len() { window[i] } else { 0.0 };
            let s = samples[sample_idx] * w;
            windowed.push(s);
            energy += s * s;
        }
        while windowed.len() < window_samples {
            windowed.push(0.0);
        }

        // Add dither
        let dither_amplitude = 1e-10;
        for (i, s) in windowed.iter_mut().enumerate() {
            let dither = dither_amplitude * ((i as f64 * 0.7).sin() + (i as f64 * 1.3).cos());
            *s += dither;
        }

        println!("  Energy: {:.6e}", energy);
        println!("  First 5 windowed: {:?}", &windowed[0..5].iter().map(|x| format!("{:.6e}", x)).collect::<Vec<_>>());
        println!("  Center windowed: {:?}", &windowed[137..142].iter().map(|x| format!("{:.6e}", x)).collect::<Vec<_>>());

        if let Some(lpc) = lpc_burg(&windowed, lpc_order) {
            let formants = lpc_to_formants(&lpc.coefficients, sample_rate);
            println!("  LPC gain: {:.6e}", lpc.gain);
            println!("  Formants: {:?}", formants.iter().take(5).map(|f| format!("{:.1}", f.frequency)).collect::<Vec<_>>());
        } else {
            println!("  LPC FAILED");
        }
        println!();
    }
}
