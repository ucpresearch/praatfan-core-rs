//! Parallel frame analysis must be bit-identical regardless of thread count.
//!
//! Runs every parallelised analysis inside rayon pools of 1, 2, 3 and 8
//! threads (the crate uses the caller's pool when it is already on one) and
//! on the crate's own pool, and asserts the f64 bit patterns match exactly.
//! Equality with the serial (non-`parallel`) build is checked separately with
//! `examples/bitdump.rs`.

#![cfg(feature = "parallel")]

use praatfan_core::*;

fn pitch_bits(p: &Pitch, out: &mut Vec<u64>) {
    for f in p.frames() {
        out.push(f.intensity.to_bits());
        out.push(f.raw_ac_peak.to_bits());
        for c in &f.candidates {
            out.push(c.frequency.to_bits());
            out.push(c.strength.to_bits());
        }
    }
    for i in 0..p.num_frames() {
        out.push(p.get_value_at_frame(i).unwrap_or(f64::NAN).to_bits());
    }
}

fn formant_bits(f: &Formant, out: &mut Vec<u64>) {
    for fr in f.frames() {
        out.push(fr.intensity().to_bits());
        for p in fr.formants() {
            out.push(p.frequency.to_bits());
            out.push(p.bandwidth.to_bits());
        }
    }
}

/// One bit vector per analysis, labelled for readable failures.
fn run_all(sound: &Sound) -> Vec<(&'static str, Vec<u64>)> {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    let mut out = Vec::new();

    out.push(("resample", bits(sound.resample(11000.0).samples())));

    let mut v = Vec::new();
    pitch_bits(&sound.to_pitch(0.0, 75.0, 600.0), &mut v);
    out.push(("pitch_ac", v));
    let mut v = Vec::new();
    pitch_bits(
        &Pitch::from_sound_with_method(
            sound, 0.0, 75.0, 600.0, 15, 0.03, 0.45, 0.01, 0.35, 0.14, 1.0,
            PitchMethod::FccAccurate,
        ),
        &mut v,
    );
    out.push(("pitch_cc", v));

    out.push(("hnr_ac", bits(Harmonicity::from_sound_ac(sound, 0.01, 75.0, 0.1, 4.5).values())));
    out.push(("hnr_cc", bits(Harmonicity::from_sound_cc(sound, 0.01, 75.0, 0.1, 1.0).values())));
    out.push(("intensity", bits(sound.to_intensity(100.0, 0.0).values())));

    let mut v = Vec::new();
    for f in Formant::from_sound_burg_multi(sound, 0.01, 5, &[4500.0, 5500.0], 0.025, 50.0) {
        formant_bits(&f, &mut v);
    }
    out.push(("formant_multi", v));

    let fp = FormantPath::from_sound_burg(sound, 0.005, 5, 5500.0, 0.025, 50.0, 0.05, 2);
    let mut v = Vec::new();
    for c in 0..fp.num_candidates() {
        formant_bits(fp.candidate(c), &mut v);
    }
    v.extend(fp.path().iter().map(|&x| x as u64));
    out.push(("formant_path", v));

    let s = sound.to_spectrogram(0.005, 5000.0, 0.005, 20.0, WindowShape::Gaussian);
    out.push(("spectrogram", s.values().iter().flatten().map(|x| x.to_bits()).collect()));

    out
}

#[test]
fn analyses_bit_identical_across_thread_counts() {
    let sound = Sound::from_file("tests/fixtures/one_two_three_four_five.wav").unwrap();
    let reference = run_all(&sound);

    for threads in [1, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let got = pool.install(|| run_all(&sound));
        for ((name, want), (_, have)) in reference.iter().zip(&got) {
            assert_eq!(want.len(), have.len(), "{name}: length differs at {threads} threads");
            if let Some(i) = want.iter().zip(have).position(|(a, b)| a != b) {
                panic!("{name}: value {i} differs at {threads} threads");
            }
        }
    }
}
