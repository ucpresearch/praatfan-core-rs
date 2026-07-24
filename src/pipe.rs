//! Pipe (stdin JSON → stdout JSON) interface for praatfan-core.
//!
//! One request runs any number of analyses against a single audio file, so
//! callers (R via processx/jsonlite, shell, etc.) pay the file load and
//! resampling cost once per file rather than once per measure.
//!
//! Request shape:
//!
//! ```json
//! { "wav_path": "audio.wav",
//!   "channel": 0,
//!   "analyses": [
//!     { "type": "pitch_ac", "pitch_floor": 75.0, "pitch_ceiling": 600.0 },
//!     { "type": "formant_burg", "max_formant_hz": 5500.0 },
//!     { "type": "spectral_moments", "times": [0.1, 0.2, 0.3] }
//!   ] }
//! ```
//!
//! Response: `{"ok": true, "version": ..., "duration": ..., "sample_rate": ...,
//! "results": [...]}` with one result object per analysis, in request order.
//! Each result echoes its `type` and carries parallel arrays (`times` plus
//! per-analysis value arrays). Non-finite values (absent formants, undefined
//! moments) serialize as JSON `null`, which jsonlite reads back as `NA`.
//!
//! Unknown analysis types and unknown parameter keys are hard errors — a
//! typo'd knob must not silently run with defaults.
//!
//! The wire protocol matches `praatfan-open-pipe` (praatfan-core-clean); the
//! one semantic divergence is `pitch_cc`, which here runs the FCC_ACCURATE
//! cross-correlation method (this crate's only CC pitch method).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::pitch::{Pitch, PitchMethod};
use crate::sound::Sound;
use crate::spectrum::Spectrum;
use crate::window::WindowShape;

/// Praat's default maximum number of pitch candidates per frame.
const MAX_CANDIDATES: usize = 15;

// ── Request ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipeRequest {
    pub wav_path: String,
    /// Channel to read from a multi-channel file (0-based). Omit for mono.
    #[serde(default)]
    pub channel: Option<usize>,
    pub analyses: Vec<serde_json::Value>,
}

/// Parameters for `pitch_ac` / `pitch_cc`. Defaults are Praat's command
/// defaults plus the Boersma (1993) tuning values.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PitchParams {
    /// 0 = auto (0.75/floor for AC, 0.25/floor for CC).
    #[serde(default)]
    time_step: f64,
    #[serde(default = "d_75")]
    pitch_floor: f64,
    #[serde(default = "d_600")]
    pitch_ceiling: f64,
    #[serde(default = "d_0_45")]
    voicing_threshold: f64,
    #[serde(default = "d_0_03")]
    silence_threshold: f64,
    #[serde(default = "d_0_01")]
    octave_cost: f64,
    #[serde(default = "d_0_35")]
    octave_jump_cost: f64,
    #[serde(default = "d_0_14")]
    voiced_unvoiced_cost: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FormantParams {
    /// 0 = auto (25% of window length).
    #[serde(default)]
    time_step: f64,
    #[serde(default = "d_5usize")]
    max_num_formants: usize,
    #[serde(default = "d_5500")]
    max_formant_hz: f64,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default = "d_50")]
    pre_emphasis_from: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntensityParams {
    #[serde(default = "d_100")]
    min_pitch: f64,
    /// 0 = auto (0.8/min_pitch).
    #[serde(default)]
    time_step: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HarmonicityParams {
    #[serde(default = "d_0_01_step")]
    time_step: f64,
    #[serde(default = "d_75")]
    min_pitch: f64,
    #[serde(default = "d_0_1")]
    silence_threshold: f64,
    /// Praat default: 4.5 for the AC method, 1.0 for CC. Resolved per
    /// method when omitted.
    #[serde(default)]
    periods_per_window: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectralMomentsParams {
    times: Vec<f64>,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default = "d_2")]
    power: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BandEnergyParams {
    times: Vec<f64>,
    #[serde(default = "d_0_025")]
    window_length: f64,
    #[serde(default)]
    f_min: f64,
    /// 0 = up to Nyquist.
    #[serde(default)]
    f_max: f64,
}

fn d_75() -> f64 { 75.0 }
fn d_100() -> f64 { 100.0 }
fn d_600() -> f64 { 600.0 }
fn d_5500() -> f64 { 5500.0 }
fn d_50() -> f64 { 50.0 }
fn d_0_45() -> f64 { 0.45 }
fn d_0_03() -> f64 { 0.03 }
fn d_0_01() -> f64 { 0.01 }
fn d_0_35() -> f64 { 0.35 }
fn d_0_14() -> f64 { 0.14 }
fn d_0_025() -> f64 { 0.025 }
fn d_0_01_step() -> f64 { 0.01 }
fn d_0_1() -> f64 { 0.1 }
fn d_2() -> f64 { 2.0 }
fn d_5usize() -> usize { 5 }

// ── Response ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct PipeResponse {
    pub ok: bool,
    pub version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
struct PitchResult {
    r#type: &'static str,
    times: Vec<f64>,
    /// 0.0 = unvoiced frame.
    frequencies: Vec<f64>,
    strengths: Vec<f64>,
}

#[derive(Serialize)]
struct FormantResult {
    r#type: &'static str,
    times: Vec<f64>,
    n_formants: Vec<usize>,
    /// F1..Fk frequency tracks and B1..Bk bandwidth tracks; `null` where
    /// the formant was not found in a frame.
    #[serde(flatten)]
    tracks: BTreeMap<String, Vec<f64>>,
}

#[derive(Serialize)]
struct ContourResult {
    r#type: &'static str,
    times: Vec<f64>,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct SpectralMomentsResult {
    r#type: &'static str,
    times: Vec<f64>,
    center_of_gravity: Vec<f64>,
    standard_deviation: Vec<f64>,
    skewness: Vec<f64>,
    kurtosis: Vec<f64>,
}

// ── Handler ──────────────────────────────────────────────────────────

pub fn handle_request(input: &str) -> String {
    let outcome = std::panic::catch_unwind(|| handle_inner(input));
    let result = match outcome {
        Ok(r) => r,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("internal error: {}", msg))
        }
    };
    match result {
        Ok(json) => json,
        Err(e) => {
            let resp = PipeResponse {
                ok: false,
                version: env!("CARGO_PKG_VERSION"),
                error: Some(e),
                duration: None,
                sample_rate: None,
                results: None,
            };
            serde_json::to_string(&resp)
                .unwrap_or_else(|_| r#"{"ok":false,"error":"serialization failed"}"#.to_string())
        }
    }
}

fn handle_inner(input: &str) -> Result<String, String> {
    let req: PipeRequest =
        serde_json::from_str(input).map_err(|e| format!("invalid JSON request: {}", e))?;

    // Mirror praatfan-open-pipe: an omitted `channel` on a multi-channel
    // file is an error (not a silent mixdown), so callers can swap the two
    // pipe binaries without a behavior change.
    let channels = Sound::from_file_channels(&req.wav_path)
        .map_err(|e| format!("failed to load {}: {}", req.wav_path, e))?;
    let n = channels.len();
    let sound = match req.channel {
        Some(ch) => channels.into_iter().nth(ch).ok_or_else(|| {
            format!(
                "channel {} out of range: {} has {} channel(s)",
                ch, req.wav_path, n
            )
        })?,
        None => {
            if n != 1 {
                return Err(format!(
                    "{} has {} channels; specify 'channel' (0-based) to select one",
                    req.wav_path, n
                ));
            }
            channels.into_iter().next().unwrap()
        }
    };

    let mut results = Vec::with_capacity(req.analyses.len());
    for (i, spec) in req.analyses.iter().enumerate() {
        results.push(run_analysis(&sound, spec, i)?);
    }

    let resp = PipeResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        error: None,
        duration: Some(sound.duration()),
        sample_rate: Some(sound.sample_rate()),
        results: Some(results),
    };
    serde_json::to_string(&resp).map_err(|e| format!("serialization failed: {}", e))
}

fn run_analysis(
    sound: &Sound,
    spec: &serde_json::Value,
    index: usize,
) -> Result<serde_json::Value, String> {
    let obj = spec
        .as_object()
        .ok_or_else(|| format!("analyses[{}] must be an object", index))?;
    let atype = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("analyses[{}] must have a string 'type' field", index))?
        .to_string();

    let mut params = obj.clone();
    params.remove("type");
    let params = serde_json::Value::Object(params);
    let ctx = |e: String| format!("analyses[{}] ({}): {}", index, atype, e);

    let result = match atype.as_str() {
        "pitch_ac" => pitch_analysis(sound, params, PitchMethod::AcHanning, "pitch_ac"),
        "pitch_cc" => pitch_analysis(sound, params, PitchMethod::FccAccurate, "pitch_cc"),
        "formant_burg" => formant_analysis(sound, params),
        "intensity" => intensity_analysis(sound, params),
        "harmonicity_ac" => harmonicity_analysis(sound, params, true),
        "harmonicity_cc" => harmonicity_analysis(sound, params, false),
        "spectral_moments" => spectral_moments_analysis(sound, params),
        "band_energy" => band_energy_analysis(sound, params),
        other => Err(format!(
            "unknown type {:?} (expected pitch_ac, pitch_cc, formant_burg, intensity, \
             harmonicity_ac, harmonicity_cc, spectral_moments, band_energy)",
            other
        )),
    };
    result.map_err(ctx)
}

fn parse<T: for<'de> Deserialize<'de>>(params: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(params).map_err(|e| e.to_string())
}

fn to_value<T: Serialize>(result: &T) -> Result<serde_json::Value, String> {
    serde_json::to_value(result).map_err(|e| format!("serialization failed: {}", e))
}

fn pitch_analysis(
    sound: &Sound,
    params: serde_json::Value,
    method: PitchMethod,
    label: &'static str,
) -> Result<serde_json::Value, String> {
    let p: PitchParams = parse(params)?;
    if !(p.pitch_floor > 0.0 && p.pitch_ceiling > p.pitch_floor) {
        return Err(format!(
            "requires 0 < pitch_floor < pitch_ceiling, got floor={} ceiling={}",
            p.pitch_floor, p.pitch_ceiling
        ));
    }
    let periods_per_window = match method {
        PitchMethod::AcHanning => 3.0,
        _ => 2.0,
    };
    let pitch = Pitch::from_sound_with_method(
        sound,
        p.time_step,
        p.pitch_floor,
        p.pitch_ceiling,
        MAX_CANDIDATES,
        p.silence_threshold,
        p.voicing_threshold,
        p.octave_cost,
        p.octave_jump_cost,
        p.voiced_unvoiced_cost,
        periods_per_window,
        method,
    );
    let n = pitch.num_frames();
    let mut times = Vec::with_capacity(n);
    let mut frequencies = Vec::with_capacity(n);
    let mut strengths = Vec::with_capacity(n);
    for i in 0..n {
        times.push(pitch.get_time_from_frame(i));
        frequencies.push(pitch.get_value_at_frame(i).unwrap_or(0.0));
        strengths.push(pitch.get_strength_at_frame(i).unwrap_or(0.0));
    }
    to_value(&PitchResult {
        r#type: label,
        times,
        frequencies,
        strengths,
    })
}

fn formant_analysis(sound: &Sound, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let p: FormantParams = parse(params)?;
    if p.max_num_formants == 0 || p.max_formant_hz <= 0.0 || p.window_length <= 0.0 {
        return Err(format!(
            "requires max_num_formants >= 1, max_formant_hz > 0, window_length > 0, \
             got {} / {} / {}",
            p.max_num_formants, p.max_formant_hz, p.window_length
        ));
    }
    let formant = sound.to_formant_burg(
        p.time_step,
        p.max_num_formants,
        p.max_formant_hz,
        p.window_length,
        p.pre_emphasis_from,
    );
    let n = formant.num_frames();
    let times: Vec<f64> = (0..n).map(|i| formant.get_time_from_frame(i)).collect();
    let n_formants: Vec<usize> = formant.frames().iter().map(|f| f.num_formants()).collect();
    let mut tracks = BTreeMap::new();
    for fnum in 1..=p.max_num_formants {
        let freqs: Vec<f64> = (0..n)
            .map(|i| formant.get_value_at_frame(fnum, i).unwrap_or(f64::NAN))
            .collect();
        let bws: Vec<f64> = (0..n)
            .map(|i| formant.get_bandwidth_at_frame(fnum, i).unwrap_or(f64::NAN))
            .collect();
        tracks.insert(format!("F{}", fnum), freqs);
        tracks.insert(format!("B{}", fnum), bws);
    }
    to_value(&FormantResult {
        r#type: "formant_burg",
        times,
        n_formants,
        tracks,
    })
}

fn intensity_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: IntensityParams = parse(params)?;
    if p.min_pitch <= 0.0 {
        return Err(format!("requires min_pitch > 0, got {}", p.min_pitch));
    }
    let intensity = sound.to_intensity(p.min_pitch, p.time_step);
    let times: Vec<f64> = (0..intensity.num_frames())
        .map(|i| intensity.get_time_from_frame(i))
        .collect();
    to_value(&ContourResult {
        r#type: "intensity",
        times,
        values: intensity.values().to_vec(),
    })
}

fn harmonicity_analysis(
    sound: &Sound,
    params: serde_json::Value,
    ac: bool,
) -> Result<serde_json::Value, String> {
    let p: HarmonicityParams = parse(params)?;
    if p.min_pitch <= 0.0 || p.time_step <= 0.0 {
        return Err(format!(
            "requires min_pitch > 0 and time_step > 0, got {} / {}",
            p.min_pitch, p.time_step
        ));
    }
    let periods = p.periods_per_window.unwrap_or(if ac { 4.5 } else { 1.0 });
    let harmonicity = if ac {
        sound.to_harmonicity_ac(p.time_step, p.min_pitch, p.silence_threshold, periods)
    } else {
        sound.to_harmonicity_cc(p.time_step, p.min_pitch, p.silence_threshold, periods)
    };
    let times: Vec<f64> = (0..harmonicity.num_frames())
        .map(|i| harmonicity.get_time_from_frame(i))
        .collect();
    to_value(&ContourResult {
        r#type: if ac { "harmonicity_ac" } else { "harmonicity_cc" },
        times,
        values: harmonicity.values().to_vec(),
    })
}

/// Window a slice of the sound centered at `t` and transform it — clamped to
/// the signal edges, mirroring the clean crate's `get_spectrum_at_time`.
fn spectrum_at_time(sound: &Sound, t: f64, window_length: f64) -> Result<Spectrum, String> {
    let half = window_length / 2.0;
    let start = (t - half).max(0.0);
    let end = (t + half).min(sound.duration());
    let part = sound
        .extract_part(start, end, WindowShape::Rectangular, 1.0, false)
        .map_err(|e| format!("cannot extract window at t={}: {}", t, e))?;
    Ok(part.to_spectrum(true))
}

fn check_times(times: &[f64], window_length: f64) -> Result<(), String> {
    if window_length <= 0.0 {
        return Err(format!("requires window_length > 0, got {}", window_length));
    }
    match times.iter().find(|t| !t.is_finite()) {
        Some(t) => Err(format!("times must be finite, got {}", t)),
        None => Ok(()),
    }
}

fn spectral_moments_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: SpectralMomentsParams = parse(params)?;
    check_times(&p.times, p.window_length)?;
    let n = p.times.len();
    let mut result = SpectralMomentsResult {
        r#type: "spectral_moments",
        times: p.times.clone(),
        center_of_gravity: Vec::with_capacity(n),
        standard_deviation: Vec::with_capacity(n),
        skewness: Vec::with_capacity(n),
        kurtosis: Vec::with_capacity(n),
    };
    for &t in &p.times {
        let spectrum = spectrum_at_time(sound, t, p.window_length)?;
        result.center_of_gravity.push(spectrum.get_center_of_gravity(p.power));
        result.standard_deviation.push(spectrum.get_standard_deviation(p.power));
        result.skewness.push(spectrum.get_skewness(p.power));
        result.kurtosis.push(spectrum.get_kurtosis(p.power));
    }
    to_value(&result)
}

fn band_energy_analysis(
    sound: &Sound,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let p: BandEnergyParams = parse(params)?;
    check_times(&p.times, p.window_length)?;
    let mut values = Vec::with_capacity(p.times.len());
    for &t in &p.times {
        values.push(spectrum_at_time(sound, t, p.window_length)?.get_band_energy(p.f_min, p.f_max));
    }
    to_value(&ContourResult {
        r#type: "band_energy",
        times: p.times,
        values,
    })
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/one_two_three_four_five.wav"
    );

    fn request(analyses: &str) -> serde_json::Value {
        let input = format!(r#"{{"wav_path": {:?}, "analyses": {}}}"#, FIXTURE, analyses);
        serde_json::from_str(&handle_request(&input)).expect("response must be valid JSON")
    }

    fn arr_len(result: &serde_json::Value, key: &str) -> usize {
        result[key].as_array().unwrap().len()
    }

    #[test]
    fn batch_runs_all_analysis_types() {
        let resp = request(
            r#"[
                {"type": "pitch_ac"},
                {"type": "pitch_cc", "pitch_floor": 100.0, "pitch_ceiling": 400.0},
                {"type": "formant_burg", "max_num_formants": 4},
                {"type": "intensity"},
                {"type": "harmonicity_ac"},
                {"type": "harmonicity_cc"},
                {"type": "spectral_moments", "times": [0.5, 1.0]},
                {"type": "band_energy", "times": [0.5, 1.0], "f_min": 0.0, "f_max": 1000.0}
            ]"#,
        );
        assert_eq!(resp["ok"], true, "{}", resp);
        assert!(resp["duration"].as_f64().unwrap() > 0.0);
        let results = resp["results"].as_array().unwrap();
        assert_eq!(results.len(), 8);

        // Parallel arrays stay parallel, per analysis.
        let pitch = &results[0];
        assert_eq!(pitch["type"], "pitch_ac");
        let n = arr_len(pitch, "times");
        assert!(n > 0);
        assert_eq!(arr_len(pitch, "frequencies"), n);
        assert_eq!(arr_len(pitch, "strengths"), n);

        let formant = &results[2];
        let nf = arr_len(formant, "times");
        for key in ["n_formants", "F1", "F4", "B1", "B4"] {
            assert_eq!(arr_len(formant, key), nf, "length mismatch for {}", key);
        }
        assert!(formant.get("F5").is_none(), "only max_num_formants tracks");

        let moments = &results[6];
        assert_eq!(arr_len(moments, "center_of_gravity"), 2);
        assert!(moments["center_of_gravity"][0].as_f64().unwrap() > 0.0);

        let band = &results[7];
        assert!(band["values"][0].as_f64().unwrap() >= 0.0);
    }

    /// Absent formants are NaN internally; on the wire they must be `null`
    /// (jsonlite → NA), and the response must never contain a bare NaN token.
    #[test]
    fn non_finite_values_serialize_as_null() {
        let input = format!(
            r#"{{"wav_path": {:?}, "analyses": [{{"type": "formant_burg"}}]}}"#,
            FIXTURE
        );
        let raw = handle_request(&input);
        assert!(!raw.contains("NaN"), "raw JSON must not contain NaN");
        let resp: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(resp["ok"], true);
    }

    #[test]
    fn unknown_type_and_typoed_param_are_errors() {
        let resp = request(r#"[{"type": "spectrum"}]"#);
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("unknown type"));

        let resp = request(r#"[{"type": "pitch_ac", "pitch_flor": 75.0}]"#);
        assert_eq!(resp["ok"], false);
        let err = resp["error"].as_str().unwrap();
        assert!(err.contains("pitch_flor"), "error should name the bad key: {}", err);
        assert!(err.contains("analyses[0]"), "error should locate the analysis: {}", err);
    }

    #[test]
    fn missing_file_reports_error_not_panic() {
        let resp: serde_json::Value = serde_json::from_str(&handle_request(
            r#"{"wav_path": "/nonexistent.wav", "analyses": [{"type": "intensity"}]}"#,
        ))
        .unwrap();
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("/nonexistent.wav"));
    }

    /// Same contract as praatfan-open-pipe: multi-channel files require an
    /// explicit `channel`; with one, the selected channel analyzes fine.
    #[test]
    fn stereo_requires_explicit_channel() {
        let stereo = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/one_two_three_four_five-stereo.flac"
        );
        let input = format!(
            r#"{{"wav_path": {:?}, "analyses": [{{"type": "intensity"}}]}}"#,
            stereo
        );
        let resp: serde_json::Value = serde_json::from_str(&handle_request(&input)).unwrap();
        assert_eq!(resp["ok"], false, "{}", resp);
        assert!(resp["error"].as_str().unwrap().contains("channel"));

        let input = format!(
            r#"{{"wav_path": {:?}, "channel": 1, "analyses": [{{"type": "intensity"}}]}}"#,
            stereo
        );
        let resp: serde_json::Value = serde_json::from_str(&handle_request(&input)).unwrap();
        assert_eq!(resp["ok"], true, "{}", resp);
    }

    #[test]
    fn out_of_range_channel_is_an_error() {
        let input = format!(
            r#"{{"wav_path": {:?}, "channel": 7, "analyses": [{{"type": "intensity"}}]}}"#,
            FIXTURE
        );
        let resp: serde_json::Value = serde_json::from_str(&handle_request(&input)).unwrap();
        assert_eq!(resp["ok"], false);
        assert!(resp["error"].as_str().unwrap().contains("channel 7"));
    }

    /// Omitted pitch parameters must resolve to the same values as the
    /// library's own AC entry point (`Sound::to_pitch`).
    #[test]
    fn pitch_defaults_match_library_entry_point() {
        let resp = request(r#"[{"type": "pitch_ac"}]"#);
        let sound = Sound::from_file(FIXTURE).unwrap();
        let expected = sound.to_pitch(0.0, 75.0, 600.0);
        let freqs = resp["results"][0]["frequencies"].as_array().unwrap();
        assert_eq!(freqs.len(), expected.num_frames());
        for (i, got) in freqs.iter().enumerate() {
            let want = expected.get_value_at_frame(i).unwrap_or(0.0);
            assert_eq!(got.as_f64().unwrap(), want);
        }
    }

    /// Windows clamped at the signal edges must still produce moments
    /// (mirrors the clean crate's get_spectrum_at_time clamping).
    #[test]
    fn edge_windows_are_clamped_not_fatal() {
        let resp = request(r#"[{"type": "spectral_moments", "times": [0.0, 0.001]}]"#);
        assert_eq!(resp["ok"], true, "{}", resp);
        assert_eq!(arr_len(&resp["results"][0], "center_of_gravity"), 2);
    }
}
