//! Dump the exact f64 bit patterns of every analysis for one audio file.
//!
//! Used to verify that parallelisation leaves results bit-identical: build
//! once serial and once with `--features parallel`, run both (optionally at
//! several `RAYON_NUM_THREADS`), and diff the outputs.
//!
//! Usage: cargo run --release --example bitdump [--features parallel] -- <audio> [out.txt]
//!
//! Prints one `name hash count elapsed_ms` line per analysis; if `out.txt` is
//! given, also writes every value's bits there (one analysis per line).

use praatfan_core::*;
use std::fmt::Write as _;
use std::time::Instant;

struct Dump {
    full: String,
}

impl Dump {
    fn emit(&mut self, name: &str, values: &[f64], ms: f64) {
        // FNV-1a over the raw bit patterns.
        let mut h: u64 = 0xcbf29ce484222325;
        for v in values {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        println!("{:<22} {:016x} {:>9} {:>9.1}", name, h, values.len(), ms);
        let _ = write!(self.full, "{}", name);
        for v in values {
            let _ = write!(self.full, " {:016x}", v.to_bits());
        }
        self.full.push('\n');
    }
}

fn pitch_bits(p: &Pitch) -> Vec<f64> {
    let mut out = Vec::new();
    for f in p.frames() {
        out.push(f.intensity);
        out.push(f.raw_ac_peak);
        for c in &f.candidates {
            out.push(c.frequency);
            out.push(c.strength);
        }
    }
    for i in 0..p.num_frames() {
        out.push(p.get_value_at_frame(i).unwrap_or(f64::NAN));
    }
    out
}

fn formant_bits(f: &Formant) -> Vec<f64> {
    let mut out = Vec::new();
    for fr in f.frames() {
        out.push(fr.intensity());
        out.push(fr.num_formants() as f64);
        for p in fr.formants() {
            out.push(p.frequency);
            out.push(p.bandwidth);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let mut d = Dump { full: String::new() };
    let sound = Sound::from_file(path).expect("load");
    let channels = Sound::from_file_channels(path).expect("load channels");

    macro_rules! timed {
        ($e:expr) => {{
            let t = Instant::now();
            let r = $e;
            (r, t.elapsed().as_secs_f64() * 1000.0)
        }};
    }

    let (r, ms) = timed!(sound.resample(11000.0));
    d.emit("resample_11k", r.samples(), ms);

    let (p, ms) = timed!(sound.to_pitch(0.0, 75.0, 600.0));
    d.emit("pitch_ac", &pitch_bits(&p), ms);
    let (p, ms) = timed!(Pitch::from_sound_with_method(
        &sound, 0.0, 75.0, 600.0, 15, 0.03, 0.45, 0.01, 0.35, 0.14, 1.0,
        PitchMethod::FccAccurate
    ));
    d.emit("pitch_cc", &pitch_bits(&p), ms);
    let (p, ms) = timed!(Pitch::from_sound_with_method_referenced(
        &sound, 0.0, 75.0, 600.0, 15, 0.03, 0.45, 0.01, 0.35, 0.14, 3.0,
        PitchMethod::AcHanning, None
    )
    .unwrap());
    d.emit("pitch_ac_ref", &pitch_bits(&p), ms);

    let (h, ms) = timed!(Harmonicity::from_sound_ac(&sound, 0.01, 75.0, 0.1, 4.5));
    d.emit("hnr_ac", h.values(), ms);
    let (h, ms) = timed!(Harmonicity::from_sound_cc(&sound, 0.01, 75.0, 0.1, 1.0));
    d.emit("hnr_cc", h.values(), ms);

    let (i, ms) = timed!(sound.to_intensity(100.0, 0.0));
    d.emit("intensity", i.values(), ms);

    let (f, ms) = timed!(sound.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0));
    d.emit("formant_5500", &formant_bits(&f), ms);
    let (fs, ms) = timed!(Formant::from_sound_burg_multi(
        &sound, 0.01, 5, &[4500.0, 5000.0, 5500.0], 0.025, 50.0
    ));
    let all: Vec<f64> = fs.iter().flat_map(formant_bits).collect();
    d.emit("formant_multi", &all, ms);
    let (fp, ms) = timed!(FormantPath::from_sound_burg(
        &sound, 0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4
    ));
    let mut all: Vec<f64> = (0..fp.num_candidates())
        .flat_map(|c| formant_bits(fp.candidate(c)))
        .collect();
    all.extend(fp.path().iter().map(|&x| x as f64));
    d.emit("formant_path", &all, ms);

    let (s, ms) = timed!(sound.to_spectrogram(0.005, 5000.0, 0.005, 20.0, WindowShape::Gaussian));
    let all: Vec<f64> = s.values().iter().flatten().copied().collect();
    d.emit("spectrogram", &all, ms);

    if channels.len() > 1 {
        let (p, ms) = timed!(pitch_from_channels(&channels, 0.0, 75.0, 600.0));
        d.emit("mc_pitch_ac", &pitch_bits(&p), ms);
        let (h, ms) = timed!(harmonicity_from_channels_cc(&channels, 0.01, 75.0, 0.1, 1.0));
        d.emit("mc_hnr_cc", h.values(), ms);
        let (h, ms) = timed!(harmonicity_from_channels_ac(&channels, 0.01, 75.0, 0.1, 4.5));
        d.emit("mc_hnr_ac", h.values(), ms);
        let (i, ms) = timed!(intensity_from_channels(&channels, 100.0, 0.0));
        d.emit("mc_intensity", i.values(), ms);
        let (s, ms) = timed!(spectrogram_from_channels(
            &channels, 0.005, 5000.0, 0.005, 20.0, WindowShape::Gaussian
        ));
        let all: Vec<f64> = s.values().iter().flatten().copied().collect();
        d.emit("mc_spectrogram", &all, ms);
    }

    if let Some(out) = args.get(2) {
        std::fs::write(out, &d.full).expect("write dump");
    }
}
