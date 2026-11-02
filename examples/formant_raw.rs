//! Diagnostics: formants of raw f64le samples, one line per frame (F1 F2 F3, NaN if absent).
//! Usage: formant_raw in.f64 sr ceiling time_step window
use praatfan_core::Sound;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let x: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let s = Sound::from_samples(&x, a[2].parse().unwrap());
    let f = s.to_formant_burg(a[4].parse().unwrap(), 5, a[3].parse().unwrap(), a[5].parse().unwrap(), 50.0);
    for i in 0..f.num_frames() {
        let v: Vec<String> = (1..=3).map(|k| format!("{:e}", f.get_value_at_frame(k, i).unwrap_or(f64::NAN))).collect();
        println!("{}", v.join(" "));
    }
}
