//! Regression test for the NIST SPHERE fallback in `Sound::from_file`.
//!
//! `symphonia`/`hound` cannot decode NIST SPHERE (and Praat itself errors on
//! shorten-compressed PCM). Commit 70aeedd wired in the MIT-licensed `desphere`
//! transcoder as an error-path fallback. This test guards that wiring end to
//! end: a real *embedded-shorten* SPHERE file must route through the fallback
//! and decode to the correct samples.
//!
//! Fixture: `tests/fixtures/tone_mono_shorten.sph` — a self-owned, license-clean
//! 400-sample mono 16 kHz 200 Hz sine (amplitude 0.5), synthesized from the
//! formula below, written as a PCM SPHERE, then shorten-compressed with the
//! canonical `shorten` encoder. Its decode was verified byte-for-byte against
//! both the `sph2pipe` oracle and the `desphere` library at generation time.

#![cfg(feature = "file-io")]

use praatfan_core::Sound;

const SR: f64 = 16000.0;
const N: usize = 400;
const FREQ: f64 = 200.0;
const AMP: f64 = 0.5;
const FIXTURE: &str = "tests/fixtures/tone_mono_shorten.sph";

/// Reconstruct the exact int16 PCM the fixture was generated from, then
/// normalize the way the hound reader does (divide by 1 << 15). One LSB of
/// tolerance absorbs any half-to-even vs half-away rounding difference between
/// the Python generator and this recomputation; it stays far below the error a
/// byte-order, scaling, or channel bug in the fallback would produce.
fn expected_normalized() -> Vec<f64> {
    (0..N)
        .map(|i| {
            let s = (AMP * 32767.0 * (2.0 * std::f64::consts::PI * FREQ * i as f64 / SR).sin())
                .round() as i32;
            s as f64 / 32768.0
        })
        .collect()
}

#[test]
fn sphere_shorten_decodes_via_desphere_fallback() {
    let sound = Sound::from_file(FIXTURE)
        .expect("embedded-shorten SPHERE should decode via the desphere fallback");

    assert_eq!(sound.sample_rate(), SR, "sample rate");
    assert_eq!(sound.samples().len(), N, "sample count");
    assert!(
        (sound.duration() - N as f64 / SR).abs() < 1e-12,
        "duration"
    );

    let expected = expected_normalized();
    let tol = 1.0 / 32768.0 + 1e-12; // one LSB
    for (i, (&got, &want)) in sound.samples().iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "sample {i}: got {got}, want {want} (diff {})",
            (got - want).abs()
        );
    }

    // Spot-check the signal shape so a silently-zeroed or constant decode fails
    // even if it happened to land within tolerance everywhere.
    assert_eq!(sound.samples()[0], 0.0, "sin(0) sample");
    let peak = sound.samples().iter().cloned().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!((peak - AMP).abs() < 2.0 / 32768.0, "peak amplitude ~0.5, got {peak}");
}

#[test]
fn sphere_shorten_decodes_per_channel() {
    let channels = Sound::from_file_channels(FIXTURE)
        .expect("embedded-shorten SPHERE should decode per-channel via desphere");
    assert_eq!(channels.len(), 1, "mono fixture yields one channel");
    assert_eq!(channels[0].samples().len(), N, "channel sample count");
    assert_eq!(channels[0].sample_rate(), SR, "channel sample rate");
}
