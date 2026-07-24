//! praatfan-gpl-pipe: reads one JSON request from stdin, writes one JSON response to stdout.

use std::io::Read;

const HELP: &str = r#"praatfan-gpl-pipe — JSON stdin → JSON stdout acoustic analysis

USAGE:
    praatfan-gpl-pipe             read one JSON request from stdin, write one JSON response to stdout
    praatfan-gpl-pipe --help      show this message
    praatfan-gpl-pipe --version   print version and exit

REQUEST:
    { "wav_path": "audio.wav",          // any format the library reads (WAV/FLAC/MP3/OGG/SPHERE...)
      "channel": 0,                     // optional, 0-based; required for multi-channel files
      "analyses": [ { "type": "...", ...params }, ... ] }

    All analyses run against the one file; results come back in request order.
    Omitted parameters take Praat's command defaults. Unknown types or
    parameter keys are errors.

ANALYSES (defaults shown):
    pitch_ac / pitch_cc
        time_step 0 (auto), pitch_floor 75, pitch_ceiling 600,
        voicing_threshold 0.45, silence_threshold 0.03, octave_cost 0.01,
        octave_jump_cost 0.35, voiced_unvoiced_cost 0.14
        → times[], frequencies[] (0 = unvoiced), strengths[]

    formant_burg
        time_step 0 (auto), max_num_formants 5, max_formant_hz 5500,
        window_length 0.025, pre_emphasis_from 50
        → times[], n_formants[], F1..Fk[], B1..Bk[] (null = not found)

    intensity
        min_pitch 100, time_step 0 (auto)
        → times[], values[] (dB)

    harmonicity_ac / harmonicity_cc
        time_step 0.01, min_pitch 75, silence_threshold 0.1,
        periods_per_window 4.5 (ac) / 1.0 (cc)
        → times[], values[] (dB)

    spectral_moments
        times[] (required), window_length 0.025, power 2
        → times[], center_of_gravity[], standard_deviation[], skewness[], kurtosis[]

    band_energy
        times[] (required), window_length 0.025, f_min 0, f_max 0 (0 = Nyquist)
        → times[], values[] (Pa²·s, as Praat's "Get band energy")

RESPONSE:
    { "ok": true, "version": "...", "duration": ..., "sample_rate": ..., "results": [...] }
    { "ok": false, "version": "...", "error": "..." } on any failure.
    Non-finite values (absent formants, undefined moments) are JSON null.

EXAMPLES:
    echo '{"wav_path":"audio.wav","analyses":[{"type":"pitch_ac"}]}' | praatfan-gpl-pipe

    echo '{"wav_path":"audio.wav","analyses":[
      {"type":"formant_burg","max_formant_hz":5000.0},
      {"type":"spectral_moments","times":[0.10,0.15,0.20]}]}' | praatfan-gpl-pipe
"#;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    for arg in &args {
        match arg.as_str() {
            "--help" | "-h" | "help" => {
                print!("{}", HELP);
                return;
            }
            "--version" | "-V" => {
                println!("praatfan-gpl-pipe {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            other => {
                eprintln!("praatfan-gpl-pipe: unknown argument: {}", other);
                eprintln!("Try 'praatfan-gpl-pipe --help'.");
                std::process::exit(2);
            }
        }
    }

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap_or_else(|e| {
        eprintln!("failed to read stdin: {}", e);
        std::process::exit(1);
    });

    println!("{}", praatfan_core::pipe::handle_request(&input));
}
