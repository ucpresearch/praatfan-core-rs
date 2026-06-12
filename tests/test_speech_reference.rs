//! Shared synthetic regression test for speech-referenced normalization.
//!
//! This is the fixture from DECISIONS-speech-reference-normalization.md §5,
//! implemented identically in every praatfan-family repo: 10 minutes of
//! quiet "voiced" signal at 16 kHz with one loud 0.5 s burst at t = 300 s.
//!
//! Run with optimizations (`[profile.test]` sets opt-level 2; `cargo test
//! --release` also works) — seven pitch analyses over a 10-minute signal.

use praatfan_core::{
    estimate_speech_reference_default, Harmonicity, Pitch, PitchMethod, Sound,
};

const SR: f64 = 16000.0;
const N: usize = 16000 * 600;
const BURST_START: usize = 16000 * 300;
const BURST_LEN: usize = 16000 / 2;
/// The burst occupies [300.0, 300.5]; "near the burst" = within 2 s of it.
const NEAR_LO: f64 = 300.0 - 2.0;
const NEAR_HI: f64 = 300.5 + 2.0;

fn speech_samples() -> Vec<f64> {
    (0..N)
        .map(|i| 0.02 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / SR).sin())
        .collect()
}

fn burst_samples() -> Vec<f64> {
    let mut s = speech_samples();
    for x in &mut s[BURST_START..BURST_START + BURST_LEN] {
        *x = 0.9;
    }
    s
}

fn legacy_pitch(samples: Vec<f64>) -> Pitch {
    let sound = Sound::from_samples_owned(samples, SR);
    Pitch::from_sound(&sound, 0.01, 75.0, 600.0)
}

fn referenced_pitch(samples: Vec<f64>, reference_peak: Option<f64>) -> Pitch {
    let sound = Sound::from_samples_owned(samples, SR);
    Pitch::from_sound_with_method_referenced(
        &sound,
        0.01,
        75.0,
        600.0,
        15,
        0.03, // silenceThreshold
        0.45, // voicingThreshold
        0.01, // octaveCost
        0.35, // octaveJumpCost
        0.14, // voicedUnvoicedCost
        3.0,  // periodsPerWindow
        PitchMethod::AcHanning,
        reference_peak,
    )
    .expect("valid reference")
}

/// Times of frames whose voicing decision differs between two analyses.
fn voicing_diff_times(a: &Pitch, b: &Pitch) -> Vec<f64> {
    assert_eq!(a.num_frames(), b.num_frames());
    (0..a.num_frames())
        .filter(|&i| a.is_voiced(i) != b.is_voiced(i))
        .map(|i| a.get_time_from_frame(i))
        .collect()
}

#[test]
fn shared_synthetic_regression() {
    let speech = speech_samples();
    let burst = burst_samples();

    // Assertion 1 (the bug, demonstrated): with the legacy whole-file peak,
    // the burst changes voicing decisions far away from it.
    let legacy_speech = legacy_pitch(speech.clone());
    let legacy_burst = legacy_pitch(burst.clone());
    let legacy_diffs = voicing_diff_times(&legacy_speech, &legacy_burst);
    let far_diffs = legacy_diffs
        .iter()
        .filter(|&&t| t < 300.25 - 10.0 || t > 300.25 + 10.0)
        .count();
    assert!(
        far_diffs > 0,
        "legacy analysis should change voicing >10 s from the burst (got {} far diffs)",
        far_diffs
    );

    // Assertion 5: the estimator's reference is immune to the burst
    // (its ~8k samples are <0.1% of speech samples).
    let ref_speech = estimate_speech_reference_default(&speech, SR);
    let ref_burst = estimate_speech_reference_default(&burst, SR);
    assert!(ref_speech.reference_peak > 0.0);
    let rel = (ref_burst.reference_peak - ref_speech.reference_peak).abs()
        / ref_speech.reference_peak;
    assert!(rel < 0.10, "estimator moved by {:.1}% under the burst", rel * 100.0);

    // Assertion 2: with the default (internally estimated) reference,
    // voicing differences are confined to the burst's neighbourhood.
    let new_speech = referenced_pitch(speech.clone(), None);
    let new_burst = referenced_pitch(burst.clone(), None);
    let new_diffs = voicing_diff_times(&new_speech, &new_burst);
    let stray: Vec<f64> = new_diffs
        .iter()
        .copied()
        .filter(|&t| t < NEAR_LO || t > NEAR_HI)
        .collect();
    assert!(
        stray.is_empty(),
        "referenced analysis changed voicing outside the burst neighbourhood at {:?}",
        stray
    );

    // Assertion 3: the same explicit reference on both signals gives
    // identical output outside the burst neighbourhood.
    let r = ref_speech.reference_peak;
    let exp_speech = referenced_pitch(speech.clone(), Some(r));
    let exp_burst = referenced_pitch(burst, Some(r));
    assert_eq!(exp_speech.num_frames(), exp_burst.num_frames());
    for i in 0..exp_speech.num_frames() {
        let t = exp_speech.get_time_from_frame(i);
        if t < NEAR_LO || t > NEAR_HI {
            assert_eq!(
                exp_speech.get_value_at_frame(i).map(f64::to_bits),
                exp_burst.get_value_at_frame(i).map(f64::to_bits),
                "explicit-reference output differs at t={t}"
            );
        }
    }

    // Assertion 4: power-of-two scale invariance is bit-identical.
    let scaled: Vec<f64> = speech.iter().map(|&x| 8.0 * x).collect();
    let exp_scaled = referenced_pitch(scaled, Some(8.0 * r));
    assert_eq!(exp_speech.num_frames(), exp_scaled.num_frames());
    for i in 0..exp_speech.num_frames() {
        assert_eq!(
            exp_speech.get_value_at_frame(i).map(f64::to_bits),
            exp_scaled.get_value_at_frame(i).map(f64::to_bits),
            "frequency not scale-invariant at frame {i}"
        );
        assert_eq!(
            exp_speech.get_strength_at_frame(i).map(f64::to_bits),
            exp_scaled.get_strength_at_frame(i).map(f64::to_bits),
            "strength not scale-invariant at frame {i}"
        );
    }
}

#[test]
fn invalid_reference_peak_is_rejected() {
    let sound = Sound::from_samples_owned(speech_samples()[..16000].to_vec(), SR);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            Pitch::from_sound_with_method_referenced(
                &sound, 0.01, 75.0, 600.0, 15, 0.03, 0.45, 0.01, 0.35, 0.14, 3.0,
                PitchMethod::AcHanning,
                Some(bad),
            )
            .is_err(),
            "reference_peak={bad} should be rejected"
        );
        assert!(Harmonicity::from_sound_ac_referenced(&sound, 0.01, 75.0, 0.1, 1.0, Some(bad))
            .is_err());
        assert!(Harmonicity::from_sound_cc_referenced(&sound, 0.01, 75.0, 0.1, 1.0, Some(bad))
            .is_err());
    }
}

#[test]
fn harmonicity_referenced_smoke() {
    // 2 s of clean 120 Hz tone: HNR should be high and equal under the
    // legacy call and the referenced call with the legacy peak handed in
    // explicitly (the signal has no mean offset, so global_peak == max|x|).
    let samples: Vec<f64> = (0..32000)
        .map(|i| 0.1 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / SR).sin())
        .collect();
    let global_peak = samples.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let sound = Sound::from_samples_owned(samples, SR);

    let legacy = Harmonicity::from_sound_cc(&sound, 0.01, 75.0, 0.1, 1.0);
    let referenced =
        Harmonicity::from_sound_cc_referenced(&sound, 0.01, 75.0, 0.1, 1.0, Some(global_peak))
            .unwrap();
    assert_eq!(legacy.num_frames(), referenced.num_frames());
    for i in 0..legacy.num_frames() {
        assert_eq!(
            legacy.get_value_at_frame(i).map(f64::to_bits),
            referenced.get_value_at_frame(i).map(f64::to_bits),
            "explicit legacy peak should reproduce the legacy result at frame {i}"
        );
    }

    // And the default (estimated) reference also yields voiced, high-HNR
    // frames. AC needs >= 3 periods per window (Praat's default is 4.5).
    let default_ref =
        Harmonicity::from_sound_ac_referenced(&sound, 0.01, 75.0, 0.1, 4.5, None).unwrap();
    assert!(default_ref.num_frames() > 0);
    assert!(default_ref.mean().unwrap() > 20.0, "clean tone should have high HNR");
}
