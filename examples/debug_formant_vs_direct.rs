use praatfan_core::Sound;
use praatfan_core::window::praat_formant_window;
use praatfan_core::utils::lpc::{lpc_burg, lpc_to_formants};
use praatfan_core::{FrequencyUnit, Interpolation};

fn main() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();

    // Create formant object using the full pipeline
    let formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);

    println!("Formant object: {} frames, first frame at t={:.4}, dt={:.4}",
             formant.num_frames(),
             formant.get_time_from_frame(0),
             0.01);

    // Compare at several voiced time points
    let test_times = vec![0.15, 0.16, 0.17, 0.20, 0.25, 0.30, 0.40, 0.50];

    println!("\nComparison at test times (F1):");
    println!("{:6}  {:>12}  {:>12}", "Time", "Pipeline F1", "Direct LPC F1");

    // For direct LPC comparison, we need to do what the formant pipeline does
    let resampled = sound.resample(11000.0);
    let emphasized = resampled.pre_emphasis(50.0);
    let samples = emphasized.samples();
    let window_samples = 275;
    let half_window = window_samples / 2;
    let window = praat_formant_window(window_samples);
    let lpc_order = 12;

    for t in &test_times {
        let pipeline_f1 = formant.get_value_at_time(1, *t, FrequencyUnit::Hertz, Interpolation::Linear);

        // Direct LPC at this time
        let center_sample = ((t - emphasized.start_time()) * 11000.0).round() as usize;
        let start_sample = center_sample.saturating_sub(half_window);
        let end_sample = (center_sample + half_window).min(samples.len());

        let mut windowed: Vec<f64> = Vec::with_capacity(window_samples);
        for (i, sample_idx) in (start_sample..end_sample).enumerate() {
            let w = if i < window.len() { window[i] } else { 0.0 };
            windowed.push(samples[sample_idx] * w);
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

        let direct_f1 = if let Some(result) = lpc_burg(&windowed, lpc_order) {
            let formants = lpc_to_formants(&result.coefficients, 11000.0);
            formants.first().map(|f| f.frequency)
        } else {
            None
        };

        println!("{:6.2}  {:>12}  {:>12}",
                 t,
                 pipeline_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)),
                 direct_f1.map_or("NaN".to_string(), |v| format!("{:.1}", v)));
    }
}
