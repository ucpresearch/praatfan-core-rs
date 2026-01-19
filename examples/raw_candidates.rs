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
    let max_formant_hz = 5500.0;

    // Look at t=0.84 (problem region)
    let test_times = [0.45, 0.84];

    for t in test_times {
        println!("\n=== t={:.2} ===", t);

        let center_sample = ((t - emphasized.start_time()) * sample_rate).round() as usize;
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

        // Add dither (using fixed frame_idx=0 for simplicity)
        let dither_amplitude = 1e-10;
        for (i, s) in windowed.iter_mut().enumerate() {
            let dither = dither_amplitude * ((i as f64 * 0.7).sin() + (i as f64 * 1.3).cos());
            *s += dither;
        }

        if let Some(lpc) = lpc_burg(&windowed, lpc_order) {
            let candidates = lpc_to_formants(&lpc.coefficients, sample_rate);

            println!("Raw candidates from lpc_to_formants:");
            for (i, c) in candidates.iter().enumerate() {
                let filter_status = if c.frequency > 50.0
                    && c.frequency < max_formant_hz
                    && c.bandwidth > 0.0
                    && c.bandwidth < max_formant_hz {
                        if c.bandwidth < c.frequency * 2.0 {
                            "PASS"
                        } else {
                            "FAIL(bw<2f)"
                        }
                    } else {
                        "FAIL(basic)"
                    };

                let praat_filter = if c.frequency > 50.0
                    && c.frequency < max_formant_hz - 50.0  // safety margin
                    && c.bandwidth > 0.0
                    && c.bandwidth < 1.3 * max_formant_hz {
                        "PRAAT_PASS"
                    } else {
                        "PRAAT_FAIL"
                    };

                println!("  Candidate {}: f={:7.1} Hz, bw={:7.1} Hz, {} {}",
                         i, c.frequency, c.bandwidth, filter_status, praat_filter);
            }
        }
    }
}
