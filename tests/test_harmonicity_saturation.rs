//! Guard: AC harmonicity must never pin on a constant near-100 dB.
//!
//! Context.  The sibling clean-room package (`praatfan` / `praatfan_rust`,
//! praatfan-core-clean) clamped the correlation to `[1e-10, 1 - 1e-10]` before
//! applying `10*log10(r/(1-r))`, so every correlation at or above `1 - 1e-10`
//! came back as *exactly* 99.99999964022885 dB.  On Buckeye conversational
//! speech read at `periods_per_window = 1.0` that pinned 62% of voiced frames
//! at that one number.
//!
//! `praatfan_gpl` was **not** affected: it already reflects an over-1
//! correlation around 1 in the pitch frame routine (matching Praat's
//! "high values due to short windows are to be reflected around 1") and
//! reports Praat's own ±150 dB boundary markers rather than a clamp.
//! Measured against parselmouth / Praat 6.1.38 on a 10 s Buckeye excerpt
//! (`s0101a.wav`, `time_step = 0.01`, `pitch_floor = 75`,
//! `silence_threshold = 0.1`, `periods_per_window = 4.5`): 399 of 989 frames
//! bit-exact, r = 0.993 over the rest, and no repeated saturation constant.
//!
//! This file pins that property so a future change to the pitch strength path
//! cannot reintroduce the clamp behaviour here.

use praatfan_core::{Harmonicity, Sound};

const SR: f64 = 16000.0;

/// The value the sibling package's old clamp produced for every correlation at
/// or above `1 - 1e-10`.  Nothing here may ever emit it repeatedly.
const CLAMP_ARTEFACT_DB: f64 = 99.999_999_640_228_85;

/// Praat's boundary markers for `Sound_to_Harmonicity_ac`.
const HNR_BOUNDARY_DB: f64 = 150.0;
const HNR_UNVOICED_DB: f64 = -200.0;

/// xorshift64* — deterministic, no dev-dependency on `rand`.
fn pseudo_noise(n: usize, amplitude: f64) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64
                / (1u64 << 53) as f64;
            (u * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn tone(n: usize, f0: f64, amplitude: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amplitude * (2.0 * std::f64::consts::PI * f0 * i as f64 / SR).sin())
        .collect()
}

/// The largest number of frames that share any one value above 50 dB.
///
/// A clamp shows up as a big cluster on a single number; a real analysis
/// spreads its high values out.
fn largest_high_value_cluster(values: &[f64]) -> (usize, f64) {
    let mut high: Vec<f64> = values.iter().copied().filter(|v| *v > 50.0).collect();
    high.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut best_n, mut best_v) = (0usize, f64::NAN);
    let mut i = 0;
    while i < high.len() {
        let mut j = i;
        while j < high.len() && high[j] == high[i] {
            j += 1;
        }
        if j - i > best_n {
            best_n = j - i;
            best_v = high[i];
        }
        i = j;
    }
    (best_n, best_v)
}

#[test]
fn white_noise_reports_no_harmonic_structure() {
    let sound = Sound::from_samples(&pseudo_noise(16000, 0.05), SR);
    for ppw in [1.0_f64, 4.5] {
        let h = Harmonicity::from_sound_ac(&sound, 0.01, 75.0, 0.1, ppw);
        let values = h.values();
        assert!(
            !values
                .iter()
                .any(|v| (*v - CLAMP_ARTEFACT_DB).abs() < 1e-9),
            "ppw={ppw}: emitted the sibling package's clamp artefact"
        );
        let voiced: Vec<f64> = values
            .iter()
            .copied()
            .filter(|v| *v > HNR_UNVOICED_DB + 1.0)
            .collect();
        let max = voiced.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            voiced.is_empty() || max < 40.0,
            "ppw={ppw}: noise reported a harmonic HNR of {max} dB"
        );
    }
}

#[test]
fn pure_tone_reads_high_but_is_not_clamped() {
    // A pure tone is the case that legitimately drives r towards 1.  It must
    // read as a large finite HNR, and must not be folded into the -200 dB
    // "unvoiced" marker or pinned at the +150 dB boundary.
    let sound = Sound::from_samples(&tone(16000, 200.0, 0.2), SR);
    for ppw in [1.0_f64, 4.5] {
        let h = Harmonicity::from_sound_ac(&sound, 0.01, 75.0, 0.1, ppw);
        let values = h.values();
        let voiced: Vec<f64> = values
            .iter()
            .copied()
            .filter(|v| *v > HNR_UNVOICED_DB + 1.0)
            .collect();
        assert!(!voiced.is_empty(), "ppw={ppw}: tone came out unvoiced");
        for v in &voiced {
            assert!(v.is_finite(), "ppw={ppw}: non-finite HNR");
            assert!(
                *v < HNR_BOUNDARY_DB,
                "ppw={ppw}: pinned at the +150 dB boundary marker"
            );
            assert!(
                (*v - CLAMP_ARTEFACT_DB).abs() > 1e-9,
                "ppw={ppw}: emitted the sibling package's clamp artefact"
            );
        }
    }
}

#[test]
fn mixed_signal_does_not_cluster_on_one_high_value() {
    // Speech-like: a decaying-amplitude tone train with additive noise, so
    // successive frames have genuinely different periodicity.  A clamp would
    // collapse the high tail onto one number.
    let n = 16000 * 4;
    let mut samples = tone(n, 130.0, 0.15);
    let noise = pseudo_noise(n, 0.02);
    for (i, s) in samples.iter_mut().enumerate() {
        let envelope = 0.3 + 0.7 * ((i as f64 / SR) * 3.0).sin().abs();
        *s = *s * envelope + noise[i];
    }
    let sound = Sound::from_samples(&samples, SR);
    for ppw in [1.0_f64, 4.5] {
        let h = Harmonicity::from_sound_ac(&sound, 0.01, 75.0, 0.1, ppw);
        let (n_cluster, value) = largest_high_value_cluster(h.values());
        let n_high = h.values().iter().filter(|v| **v > 50.0).count();
        assert!(
            n_high == 0 || n_cluster * 2 <= n_high,
            "ppw={ppw}: {n_cluster} of {n_high} high frames share the single \
             value {value} dB — that is a clamp, not an analysis"
        );
    }
}
