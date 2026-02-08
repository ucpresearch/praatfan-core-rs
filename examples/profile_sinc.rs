/// Profile sinc_interpolate and Brent+sinc — count actual iterations on real data.
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args.get(1).map(|s| s.as_str()).unwrap_or("tests/fixtures/one_two_three_four_five.wav");

    let sound = praatfan_core::Sound::from_file(std::path::Path::new(audio_path)).unwrap();

    let pitch_floor = 75.0;
    let pitch_ceiling_nyquist = 0.5 * sound.sample_rate();

    println!("Audio: {} ({:.1}s, {:.0} Hz)", audio_path, sound.duration(), sound.sample_rate());
    println!();

    // Reset counters
    praatfan_core::BRENT_TOTAL_ITERS.store(0, Ordering::Relaxed);
    praatfan_core::BRENT_TOTAL_CALLS.store(0, Ordering::Relaxed);

    // Run with Nyquist ceiling (same as harmonicity CC)
    let t_start = Instant::now();
    let pitch_nyq = praatfan_core::Pitch::from_sound_with_method(
        &sound, 0.01, pitch_floor, pitch_ceiling_nyquist,
        15, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, praatfan_core::PitchMethod::FccAccurate,
    );
    let t_nyq = t_start.elapsed();

    let total_iters = praatfan_core::BRENT_TOTAL_ITERS.load(Ordering::Relaxed);
    let total_calls = praatfan_core::BRENT_TOTAL_CALLS.load(Ordering::Relaxed);

    let frames = pitch_nyq.frames();
    let total_cands: usize = frames.iter().map(|f| f.candidates.len().saturating_sub(1)).sum();

    println!("FCC_ACCURATE ceiling=Nyquist:");
    println!("  Time: {:.3}s", t_nyq.as_secs_f64());
    println!("  Total candidates: {}", total_cands);
    println!("  Brent calls: {}", total_calls);
    println!("  Brent total iterations: {}", total_iters);
    if total_calls > 0 {
        println!("  Avg iterations per call: {:.1}", total_iters as f64 / total_calls as f64);
        println!("  Time per Brent call: {:.1} us", t_nyq.as_micros() as f64 / total_calls as f64);
    }
    println!();

    // Reset and run with ceiling=600
    praatfan_core::BRENT_TOTAL_ITERS.store(0, Ordering::Relaxed);
    praatfan_core::BRENT_TOTAL_CALLS.store(0, Ordering::Relaxed);

    let t_start = Instant::now();
    let pitch_600 = praatfan_core::Pitch::from_sound_with_method(
        &sound, 0.01, pitch_floor, 600.0,
        15, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, praatfan_core::PitchMethod::FccAccurate,
    );
    let t_600 = t_start.elapsed();

    let total_iters_600 = praatfan_core::BRENT_TOTAL_ITERS.load(Ordering::Relaxed);
    let total_calls_600 = praatfan_core::BRENT_TOTAL_CALLS.load(Ordering::Relaxed);

    let frames_600 = pitch_600.frames();
    let total_cands_600: usize = frames_600.iter().map(|f| f.candidates.len().saturating_sub(1)).sum();

    println!("FCC_ACCURATE ceiling=600:");
    println!("  Time: {:.3}s", t_600.as_secs_f64());
    println!("  Total candidates: {}", total_cands_600);
    println!("  Brent calls: {}", total_calls_600);
    println!("  Brent total iterations: {}", total_iters_600);
    if total_calls_600 > 0 {
        println!("  Avg iterations per call: {:.1}", total_iters_600 as f64 / total_calls_600 as f64);
        println!("  Time per Brent call: {:.1} us", t_600.as_micros() as f64 / total_calls_600 as f64);
    }
}
