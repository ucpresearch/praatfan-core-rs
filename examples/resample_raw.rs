//! Diagnostics: resample raw f64le samples. Usage: resample_raw in.f64 sr out_sr out.f64
use praatfan_core::Sound;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let x: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let s = Sound::from_samples(&x, a[2].parse().unwrap());
    let r = s.resample(a[3].parse().unwrap());
    let out: Vec<u8> = r.samples().iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[4], out).unwrap();
}
