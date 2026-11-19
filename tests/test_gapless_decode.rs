//! Gapless decoding: Ogg Vorbis always lines up with the source WAV; MP3 does
//! with `DecodeOptions::exact_mp3_timing`, and otherwise keeps the legacy
//! (delayed) timing for Praat compatibility.
//!
//! `one_two_three_four_five_lame.mp3` is an ffmpeg/LAME encode (128 kb/s) of
//! `one_two_three_four_five.wav`. Without gapless decoding the MP3 comes out
//! 1105 samples late with 1152 extra samples, and the Ogg Vorbis fixture
//! 128 samples too long.
#![cfg(feature = "file-io")]

use praatfan_core::{DecodeOptions, Sound};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

fn corr_at(a: &[f64], b: &[f64], lag: isize) -> f64 {
    // normalized correlation of a[i] with b[i + lag]
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (i, &x) in a.iter().enumerate() {
        let j = i as isize + lag;
        if j < 0 || j as usize >= b.len() {
            continue;
        }
        let y = b[j as usize];
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

fn check_aligned(name: &str, opts: &DecodeOptions) {
    let wav = Sound::from_file(format!("{FIX}one_two_three_four_five.wav")).unwrap();
    let enc = Sound::from_file_with(format!("{FIX}{name}"), opts).unwrap();
    assert_eq!(enc.sample_rate(), wav.sample_rate(), "{name}");
    assert_eq!(enc.samples().len(), wav.samples().len(), "{name}: length");
    let c0 = corr_at(wav.samples(), enc.samples(), 0);
    assert!(c0 > 0.99, "{name}: correlation at lag 0 = {c0}");
    for lag in [-2isize, -1, 1, 2] {
        let c = corr_at(wav.samples(), enc.samples(), lag);
        assert!(c < c0, "{name}: lag {lag} correlates better ({c} >= {c0})");
    }
    let chans = Sound::from_file_channels_with(format!("{FIX}{name}"), opts).unwrap();
    assert_eq!(chans[0].samples(), enc.samples(), "{name}: channel loader");
    let bytes = std::fs::read(format!("{FIX}{name}")).unwrap();
    let ext = name.rsplit('.').next().unwrap();
    let mem = Sound::from_bytes_channels_with(&bytes, Some(ext), opts).unwrap();
    assert_eq!(mem[0].samples(), enc.samples(), "{name}: in-memory loader");
}

#[test]
fn mp3_exact_timing_is_gapless() {
    check_aligned(
        "one_two_three_four_five_lame.mp3",
        &DecodeOptions::default().exact_mp3_timing(true),
    );
}

#[test]
fn mp3_default_keeps_legacy_timing() {
    let name = format!("{FIX}one_two_three_four_five_lame.mp3");
    let wav = Sound::from_file(format!("{FIX}one_two_three_four_five.wav")).unwrap();
    let mp3 = Sound::from_file(&name).unwrap();
    assert_eq!(mp3.samples().len(), wav.samples().len() + 1152);
    let c = corr_at(wav.samples(), mp3.samples(), 1105);
    assert!(c > 0.99, "legacy MP3 should be 1105 samples late (corr {c})");
    let bytes = std::fs::read(&name).unwrap();
    let mem = Sound::from_bytes_channels(&bytes, Some("mp3")).unwrap();
    assert_eq!(mem[0].samples(), mp3.samples());
}

#[test]
fn ogg_vorbis_is_gapless_by_default() {
    check_aligned("one_two_three_four_five.ogg", &DecodeOptions::default());
    check_aligned(
        "one_two_three_four_five.ogg",
        &DecodeOptions::default().exact_mp3_timing(true),
    );
}
