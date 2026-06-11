use std::time::Instant;
use praatfan_core::Sound;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("path");
    let secs: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.0);

    let sound = Sound::from_file(path).expect("load");
    let dur = sound.duration();
    eprintln!("Loaded: {:.1}s, {} samples at {}Hz", dur, sound.samples().len(), sound.sample_rate());

    let chunk = if secs > 0.0 && secs < dur {
        sound.extract_part(0.0, secs, praatfan_core::WindowShape::Rectangular, 1.0, false).expect("extract")
    } else {
        sound.clone()
    };
    eprintln!("Chunk: {:.1}s, {} samples", chunk.duration(), chunk.samples().len());

    let t0 = Instant::now();
    let fmt = chunk.to_formant_burg(0.005, 5, 5500.0, 0.025, 50.0);
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("{:.1}s -> {} frames in {:.3}s ({:.1} frames/s)",
              chunk.duration(), fmt.num_frames(), elapsed, fmt.num_frames() as f64 / elapsed);
}
