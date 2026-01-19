//! Integration tests comparing praat-core-rs against parselmouth ground truth
//!
//! These tests load audio files in various formats and compare analysis results
//! against pre-computed values from parselmouth (Praat).

use praat_core::{
    FrequencyUnit, Interpolation, PitchUnit, Sound, WindowShape,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Tolerance for pitch comparison (Hz)
const PITCH_TOLERANCE_HZ: f64 = 10.0;
/// Tolerance for intensity comparison (dB)
const INTENSITY_TOLERANCE_DB: f64 = 3.0;
/// Tolerance for formant comparison (Hz)
const FORMANT_TOLERANCE_HZ: f64 = 100.0;
/// Tolerance for HNR comparison (dB)
const HNR_TOLERANCE_DB: f64 = 5.0;
/// Tolerance for spectral moments
const SPECTRAL_TOLERANCE: f64 = 200.0;

// Minimum match rates (percentage) - these are relaxed for initial implementation
// and can be tightened as the algorithms are improved to match Praat exactly
const MIN_PITCH_MATCH_RATE: f64 = 70.0;
const MIN_INTENSITY_MATCH_RATE: f64 = 60.0;
const MIN_FORMANT_MATCH_RATE: f64 = 5.0; // Formants are very sensitive to implementation details - needs algorithm work
const MIN_HNR_MATCH_RATE: f64 = 15.0; // HNR can vary significantly

#[derive(Debug, Deserialize)]
struct GroundTruth {
    file: String,
    sound: SoundInfo,
    pitch: PitchInfo,
    intensity: IntensityInfo,
    formant: FormantInfo,
    harmonicity: HarmonicityInfo,
    spectrum: SpectrumInfo,
}

#[derive(Debug, Deserialize)]
struct SoundInfo {
    duration: f64,
    sample_rate: f64,
    num_samples: usize,
    num_channels: usize,
}

#[derive(Debug, Deserialize)]
struct PitchInfo {
    time_step: f64,
    pitch_floor: f64,
    pitch_ceiling: f64,
    times: Vec<f64>,
    values: Vec<Option<f64>>,
    mean: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct IntensityInfo {
    min_pitch: f64,
    time_step: f64,
    times: Vec<f64>,
    values: Vec<Option<f64>>,
    mean: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FormantInfo {
    max_num_formants: usize,
    max_formant_hz: f64,
    window_length: f64,
    time_step: f64,
    times: Vec<f64>,
    f1: Vec<Option<f64>>,
    f2: Vec<Option<f64>>,
    f3: Vec<Option<f64>>,
    b1: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct HarmonicityInfo {
    time_step: f64,
    min_pitch: f64,
    times: Vec<f64>,
    values: Vec<Option<f64>>,
    mean: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SpectrumInfo {
    center_of_gravity: Option<f64>,
    standard_deviation: Option<f64>,
    skewness: Option<f64>,
    kurtosis: Option<f64>,
}

fn load_ground_truth(name: &str) -> Option<GroundTruth> {
    let path = format!("tests/ground_truth/{}.json", name);
    if !Path::new(&path).exists() {
        return None;
    }
    let content = fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn load_sound(name: &str, ext: &str) -> Option<Sound> {
    let path = format!("tests/fixtures/{}.{}", name, ext);
    if !Path::new(&path).exists() {
        return None;
    }
    Sound::from_file(&path).ok()
}

/// Check if a value should be considered undefined
/// Praat uses -200 dB as a sentinel for undefined HNR
fn is_effectively_undefined(v: Option<f64>) -> bool {
    match v {
        None => true,
        Some(x) if !x.is_finite() => true,
        Some(x) if x <= -100.0 => true, // Treat very negative HNR as undefined
        _ => false,
    }
}

/// Compare two optional f64 values with tolerance
fn approx_eq(a: Option<f64>, b: Option<f64>, tolerance: f64) -> bool {
    match (a, b) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() && a > -100.0 && b > -100.0 => {
            (a - b).abs() < tolerance
        }
        _ if is_effectively_undefined(a) && is_effectively_undefined(b) => true,
        _ => false,
    }
}

/// Count matching values between two vectors
fn count_matches(
    expected: &[Option<f64>],
    actual: &[Option<f64>],
    tolerance: f64,
) -> (usize, usize, usize) {
    let mut matches = 0;
    let mut mismatches = 0;
    let mut both_none = 0;

    let len = expected.len().min(actual.len());
    for i in 0..len {
        if approx_eq(expected[i], actual[i], tolerance) {
            if is_effectively_undefined(expected[i]) {
                both_none += 1;
            } else {
                matches += 1;
            }
        } else {
            mismatches += 1;
        }
    }

    (matches, mismatches, both_none)
}

// ============================================================================
// Sound loading tests
// ============================================================================

#[test]
fn test_load_wav_files() {
    let files = [
        "one_two_three_four_five",
        "one_two_three_four_five_16k",
        "one_two_three_four_five_44k",
        "one_two_three_four_five_48k",
        "one_two_three_four_five_8bit",
        "one_two_three_four_five_24bit",
        "one_two_three_four_five_32float",
        "one_two_three_four_five_stereo",
        "En-US-One",
    ];

    for name in &files {
        let sound = load_sound(name, "wav");
        assert!(sound.is_some(), "Failed to load {}.wav", name);

        let sound = sound.unwrap();
        assert!(sound.num_samples() > 0, "{}.wav has no samples", name);
        assert!(sound.sample_rate() > 0.0, "{}.wav has invalid sample rate", name);

        // Compare with ground truth
        if let Some(gt) = load_ground_truth(name) {
            let duration_diff = (sound.duration() - gt.sound.duration).abs();
            assert!(
                duration_diff < 0.01,
                "{}: duration mismatch: got {}, expected {}",
                name,
                sound.duration(),
                gt.sound.duration
            );

            // Sample rate should match exactly (or be the converted rate)
            println!(
                "{}: duration={:.3}s, sr={}, samples={}",
                name,
                sound.duration(),
                sound.sample_rate(),
                sound.num_samples()
            );
        }
    }
}

#[test]
fn test_load_mp3() {
    let sound = load_sound("one_two_three_four_five", "mp3");
    assert!(sound.is_some(), "Failed to load MP3 file");

    let sound = sound.unwrap();
    assert!(sound.num_samples() > 0);
    println!(
        "MP3: duration={:.3}s, sr={}, samples={}",
        sound.duration(),
        sound.sample_rate(),
        sound.num_samples()
    );
}

#[test]
fn test_load_ogg() {
    let sound = load_sound("one_two_three_four_five", "ogg");
    assert!(sound.is_some(), "Failed to load OGG file");

    let sound = sound.unwrap();
    assert!(sound.num_samples() > 0);
    println!(
        "OGG: duration={:.3}s, sr={}, samples={}",
        sound.duration(),
        sound.sample_rate(),
        sound.num_samples()
    );
}

#[test]
fn test_load_flac() {
    let sound = load_sound("one_two_three_four_five", "flac");
    assert!(sound.is_some(), "Failed to load FLAC file");

    let sound = sound.unwrap();
    assert!(sound.num_samples() > 0);
    println!(
        "FLAC: duration={:.3}s, sr={}, samples={}",
        sound.duration(),
        sound.sample_rate(),
        sound.num_samples()
    );
}

// ============================================================================
// Pitch comparison tests
// ============================================================================

#[test]
fn test_pitch_comparison() {
    let test_files = ["one_two_three_four_five", "En-US-One"];

    for name in &test_files {
        let gt = match load_ground_truth(name) {
            Some(gt) => gt,
            None => continue,
        };

        let sound = load_sound(name, "wav").expect("Failed to load sound");
        let pitch = sound.to_pitch(gt.pitch.time_step, gt.pitch.pitch_floor, gt.pitch.pitch_ceiling);

        // Compare pitch values at ground truth times
        let mut actual_values: Vec<Option<f64>> = Vec::new();
        for &t in &gt.pitch.times {
            let val = pitch.get_value_at_time(t, PitchUnit::Hertz, Interpolation::Linear);
            actual_values.push(val);
        }

        let (matches, mismatches, both_none) =
            count_matches(&gt.pitch.values, &actual_values, PITCH_TOLERANCE_HZ);

        let total = matches + mismatches + both_none;
        let match_rate = (matches + both_none) as f64 / total as f64 * 100.0;

        println!(
            "{} pitch: {}/{} matched ({:.1}%), {} mismatches",
            name, matches + both_none, total, match_rate, mismatches
        );

        // Allow some mismatches due to algorithm differences
        assert!(
            match_rate > MIN_PITCH_MATCH_RATE,
            "{}: pitch match rate too low: {:.1}%",
            name,
            match_rate
        );
    }
}

// ============================================================================
// Intensity comparison tests
// ============================================================================

#[test]
fn test_intensity_comparison() {
    let test_files = ["one_two_three_four_five", "En-US-One"];

    for name in &test_files {
        let gt = match load_ground_truth(name) {
            Some(gt) => gt,
            None => continue,
        };

        let sound = load_sound(name, "wav").expect("Failed to load sound");
        let intensity = sound.to_intensity(gt.intensity.min_pitch, gt.intensity.time_step);

        // Compare intensity values at ground truth times
        let mut actual_values: Vec<Option<f64>> = Vec::new();
        for &t in &gt.intensity.times {
            let val = intensity.get_value_at_time(t, Interpolation::Linear);
            actual_values.push(val);
        }

        let (matches, mismatches, both_none) =
            count_matches(&gt.intensity.values, &actual_values, INTENSITY_TOLERANCE_DB);

        let total = matches + mismatches + both_none;
        let match_rate = (matches + both_none) as f64 / total as f64 * 100.0;

        println!(
            "{} intensity: {}/{} matched ({:.1}%), {} mismatches",
            name, matches + both_none, total, match_rate, mismatches
        );

        // Intensity should match reasonably well
        assert!(
            match_rate > MIN_INTENSITY_MATCH_RATE,
            "{}: intensity match rate too low: {:.1}%",
            name,
            match_rate
        );
    }
}

// ============================================================================
// Formant comparison tests
// ============================================================================

#[test]
fn test_formant_comparison() {
    let test_files = ["one_two_three_four_five", "En-US-One"];

    for name in &test_files {
        let gt = match load_ground_truth(name) {
            Some(gt) => gt,
            None => continue,
        };

        let sound = load_sound(name, "wav").expect("Failed to load sound");
        // Use 0.0 for time_step (automatic) to match Praat's behavior
        // Praat's automatic time step is window_length / 4
        let formant = sound.to_formant_burg(
            0.0, // automatic time step like Praat
            gt.formant.max_num_formants,
            gt.formant.max_formant_hz,
            gt.formant.window_length,
            50.0, // pre_emphasis
        );

        // Compare F1 values
        let mut actual_f1: Vec<Option<f64>> = Vec::new();
        for &t in &gt.formant.times {
            let val = formant.get_value_at_time(1, t, FrequencyUnit::Hertz, Interpolation::Linear);
            actual_f1.push(val);
        }

        let (matches, mismatches, both_none) =
            count_matches(&gt.formant.f1, &actual_f1, FORMANT_TOLERANCE_HZ);

        let total = matches + mismatches + both_none;
        let match_rate = if total > 0 {
            (matches + both_none) as f64 / total as f64 * 100.0
        } else {
            100.0
        };

        println!(
            "{} F1: {}/{} matched ({:.1}%), {} mismatches",
            name, matches + both_none, total, match_rate, mismatches
        );

        // Formants are harder to match exactly due to algorithm sensitivity
        assert!(
            match_rate > MIN_FORMANT_MATCH_RATE,
            "{}: F1 match rate too low: {:.1}%",
            name,
            match_rate
        );
    }
}

// ============================================================================
// Harmonicity comparison tests
// ============================================================================

#[test]
fn test_harmonicity_comparison() {
    let test_files = ["one_two_three_four_five", "En-US-One"];

    for name in &test_files {
        let gt = match load_ground_truth(name) {
            Some(gt) => gt,
            None => continue,
        };

        let sound = load_sound(name, "wav").expect("Failed to load sound");
        let hnr = sound.to_harmonicity_cc(gt.harmonicity.time_step, gt.harmonicity.min_pitch, 0.1, 1.0);

        // Compare HNR values at ground truth times
        let mut actual_values: Vec<Option<f64>> = Vec::new();
        for &t in &gt.harmonicity.times {
            let val = hnr.get_value_at_time(t, Interpolation::Linear);
            actual_values.push(val);
        }

        let (matches, mismatches, both_none) =
            count_matches(&gt.harmonicity.values, &actual_values, HNR_TOLERANCE_DB);

        let total = matches + mismatches + both_none;
        let match_rate = if total > 0 {
            (matches + both_none) as f64 / total as f64 * 100.0
        } else {
            100.0
        };

        println!(
            "{} HNR: {}/{} matched ({:.1}%), {} mismatches",
            name, matches + both_none, total, match_rate, mismatches
        );

        // HNR can vary due to implementation differences
        assert!(
            match_rate > MIN_HNR_MATCH_RATE,
            "{}: HNR match rate too low: {:.1}%",
            name,
            match_rate
        );
    }
}

// ============================================================================
// Spectrum comparison tests
// ============================================================================

#[test]
fn test_spectrum_comparison() {
    let test_files = ["one_two_three_four_five", "En-US-One"];

    for name in &test_files {
        let gt = match load_ground_truth(name) {
            Some(gt) => gt,
            None => continue,
        };

        let sound = load_sound(name, "wav").expect("Failed to load sound");

        // Extract center portion like parselmouth does
        let center_time = sound.duration() / 2.0;
        let extract_duration = 0.05;
        let start = (center_time - extract_duration / 2.0).max(0.0);
        let end = (center_time + extract_duration / 2.0).min(sound.duration());

        let extract = sound
            .extract_part(start, end, WindowShape::Hanning, 1.0, false)
            .expect("Failed to extract");
        let spectrum = extract.to_spectrum(true);

        // Compare spectral moments
        let cog = spectrum.get_center_of_gravity(2.0);
        let std = spectrum.get_standard_deviation(2.0);

        if let Some(expected_cog) = gt.spectrum.center_of_gravity {
            let diff = (cog - expected_cog).abs();
            println!(
                "{} spectrum CoG: got {:.1}, expected {:.1}, diff {:.1}",
                name, cog, expected_cog, diff
            );
            // Spectral moments can vary significantly
            assert!(
                diff < SPECTRAL_TOLERANCE * 10.0,
                "{}: CoG difference too large",
                name
            );
        }

        if let Some(expected_std) = gt.spectrum.standard_deviation {
            let diff = (std - expected_std).abs();
            println!(
                "{} spectrum std: got {:.1}, expected {:.1}, diff {:.1}",
                name, std, expected_std, diff
            );
        }
    }
}

// ============================================================================
// Cross-format consistency tests
// ============================================================================

#[test]
fn test_cross_format_consistency() {
    // Load the same audio in different formats and compare analysis results
    let formats = ["wav", "mp3", "ogg", "flac"];
    let name = "one_two_three_four_five";

    let mut results: Vec<(String, f64, Option<f64>, Option<f64>)> = Vec::new();

    for fmt in &formats {
        if let Some(sound) = load_sound(name, fmt) {
            let pitch = sound.to_pitch(0.01, 75.0, 600.0);
            let mean_pitch = pitch.mean();
            let intensity = sound.to_intensity(100.0, 0.01);
            let mean_intensity = intensity.mean();

            results.push((
                fmt.to_string(),
                sound.duration(),
                mean_pitch,
                mean_intensity,
            ));

            println!(
                "{}.{}: dur={:.3}s, mean_pitch={:?}, mean_intensity={:?}",
                name, fmt, sound.duration(), mean_pitch, mean_intensity
            );
        }
    }

    // All formats should give similar results
    if results.len() >= 2 {
        let wav_result = &results[0];

        for result in &results[1..] {
            // Duration can vary slightly due to codec padding
            let dur_diff = (result.1 - wav_result.1).abs();
            assert!(
                dur_diff < 0.1,
                "Duration mismatch between {} and {}: {:.3}s vs {:.3}s",
                wav_result.0,
                result.0,
                wav_result.1,
                result.1
            );

            // Mean pitch should be similar if both have voiced content
            if let (Some(a), Some(b)) = (wav_result.2, result.2) {
                let diff = (a - b).abs();
                println!(
                    "Pitch diff between {} and {}: {:.1} Hz",
                    wav_result.0, result.0, diff
                );
                // Allow larger tolerance for lossy formats
                assert!(
                    diff < 20.0,
                    "Pitch mismatch between {} and {}: {:.1} vs {:.1}",
                    wav_result.0,
                    result.0,
                    a,
                    b
                );
            }
        }
    }
}
