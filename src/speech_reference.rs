//! Speech-referenced amplitude normalization
//!
//! Praat's pitch/HNR analysis references per-frame amplitude against the
//! whole-file peak (`global_peak`). On long conversational recordings a
//! single loud event (click, laugh) depresses every frame's relative
//! intensity, forcing quiet voiced frames unvoiced. This module provides a
//! robust alternative: norming standards derived from **frame-level** robust
//! statistics over speech-active frames.
//!
//! Frame-level (not sample-level) is the crux: a 0.5 s, 40x-amplitude burst
//! is a small fraction of speech samples but carries ~99% of the energy, so a
//! sample-level std or high percentile still lands inside the burst. A median
//! or percentile over per-frame statistics does not — a short event is only a
//! few frames.
//!
//! The algorithm and all constants are normative and shared across the
//! praatfan package family (praatfan, praatfan-rust, praatfan-gpl) so that
//! standards computed by one package can be passed to any other. The
//! canonical reference implementation lives in praatfan-core-clean
//! (`DECISIONS-speech-reference-normalization.md` there pins the algorithm).

/// Default analysis frame length in seconds.
pub const DEFAULT_FRAME_S: f64 = 0.05;
/// Default hop between frames in seconds.
pub const DEFAULT_HOP_S: f64 = 0.01;
/// Default speech-mask floor below the 95th-percentile frame level, in dB.
pub const DEFAULT_SPEECH_FLOOR_DB: f64 = 30.0;
/// Default percentile (across frames) of per-frame peak |x| used as the
/// reference peak. The contamination cliff is `(100 - p)%` of speech-masked
/// time; the default p=75 puts it at an effectively-unreachable 25%. Raising
/// it toward 90–95 gives a "loud-speech peak" closer to Praat's original
/// global-peak semantics.
pub const DEFAULT_REFERENCE_PERCENTILE: f64 = 75.0;

/// The percentile of frame dB levels that anchors the speech mask.
/// Fixed constant of the cross-package contract, not a parameter.
const MASK_ANCHOR_PERCENTILE: f64 = 95.0;

/// Small epsilon added inside the RMS / log to match the reference
/// implementation's float behavior bit-for-bit (avoids log(0)).
const EPS: f64 = 1e-20;

/// Result of [`estimate_speech_reference`].
///
/// `speech_mask` / `frame_times` are on the framing grid; the scalar
/// standards are frame-level robust statistics over the speech-masked frames
/// (burst- and silence-robust).
#[derive(Debug, Clone)]
pub struct SpeechReference {
    /// Per-frame "this looks like speech" flags (hop grid).
    pub speech_mask: Vec<bool>,
    /// Frame centers in seconds.
    pub frame_times: Vec<f64>,
    /// Median per-frame mean over speech frames (DC reference).
    pub mean: f64,
    /// Median per-frame RMS over speech frames (the z-norm scale). Never <= 0
    /// (falls back to 1.0).
    pub std: f64,
    /// `reference_percentile`-ile of per-frame peak |x| over speech frames — a
    /// typical-speech peak, the Praat-`global_peak` replacement.
    pub reference_peak: f64,
    /// Fraction of frames in the speech mask.
    pub speech_fraction: f64,
}

/// Linear-interpolation percentile (Hyndman & Fan type 7, NumPy's default).
///
/// `a` must be sorted ascending and non-empty; `q` in [0, 100].
fn percentile_sorted(a: &[f64], q: f64) -> f64 {
    let n = a.len();
    debug_assert!(n > 0);
    if n == 1 {
        return a[0];
    }
    let h = (n - 1) as f64 * q / 100.0;
    let l = h.floor() as usize;
    if l + 1 >= n {
        return a[n - 1];
    }
    a[l] + (h - l as f64) * (a[l + 1] - a[l])
}

/// Median (matches NumPy `median`): type-7 percentile at 50.
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    percentile_sorted(&v, 50.0)
}

/// Estimate speech-referenced norming standards for `samples`.
///
/// 1. Frame the signal (`frame_s` windows, `hop_s` hop, full frames only,
///    starting at sample 0); per-frame RMS.
/// 2. `speech_mask` = frames whose level in dB is within `speech_floor_db` of
///    the 95th-percentile frame level. If that leaves no frames, the mask
///    falls back to all-true.
/// 3. Standards over the speech-masked frames: `std` = median per-frame RMS,
///    `mean` = median per-frame mean, `reference_peak` =
///    `reference_percentile`-ile of per-frame peak |x|.
///
/// Degenerate inputs (all-zero or shorter than one frame) fall back to a
/// single all-true frame over whatever samples exist.
pub fn estimate_speech_reference(
    samples: &[f64],
    sample_rate: f64,
    frame_s: f64,
    hop_s: f64,
    speech_floor_db: f64,
    reference_percentile: f64,
) -> SpeechReference {
    let n = samples.len();

    // Empty signal: no frames, neutral standards (matches DECISIONS §1.1 and
    // the praatfan-core-clean reference impl).
    if n == 0 {
        return SpeechReference {
            speech_mask: Vec::new(),
            frame_times: Vec::new(),
            mean: 0.0,
            std: 1.0,
            reference_peak: 0.0,
            speech_fraction: 0.0,
        };
    }

    let frame = ((frame_s * sample_rate).round() as usize).max(1);
    let hop = ((hop_s * sample_rate).round() as usize).max(1);

    // Too short to frame — treat everything as one speech frame.
    if n < frame {
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        let reference_peak = samples.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        return SpeechReference {
            speech_mask: vec![true],
            frame_times: vec![n as f64 / (2.0 * sample_rate)],
            mean,
            std: if std > 0.0 { std } else { 1.0 },
            reference_peak,
            speech_fraction: 1.0,
        };
    }

    let n_frames = 1 + (n - frame) / hop;

    let mut frame_times = Vec::with_capacity(n_frames);
    let mut rms = Vec::with_capacity(n_frames);
    let mut log_rms_db = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop;
        let f = &samples[start..start + frame];
        let mean_sq: f64 = f.iter().map(|&x| x * x).sum::<f64>() / frame as f64;
        let r = (mean_sq + EPS).sqrt();
        rms.push(r);
        log_rms_db.push(20.0 * (r + EPS).log10());
        frame_times.push((start as f64 + frame as f64 / 2.0) / sample_rate);
    }

    let mut sorted_db = log_rms_db.clone();
    sorted_db.sort_unstable_by(f64::total_cmp);
    let ceiling_db = percentile_sorted(&sorted_db, MASK_ANCHOR_PERCENTILE);
    let threshold_db = ceiling_db - speech_floor_db;
    let mut speech_mask: Vec<bool> = log_rms_db.iter().map(|&d| d >= threshold_db).collect();
    if !speech_mask.iter().any(|&m| m) {
        speech_mask = vec![true; n_frames];
    }

    // Frame-level robust standards over the speech-masked frames.
    let mut masked_rms = Vec::new();
    let mut masked_means = Vec::new();
    let mut masked_peaks = Vec::new();
    for (i, &masked) in speech_mask.iter().enumerate() {
        if !masked {
            continue;
        }
        let start = i * hop;
        let f = &samples[start..start + frame];
        masked_rms.push(rms[i]);
        masked_means.push(f.iter().sum::<f64>() / frame as f64);
        masked_peaks.push(f.iter().fold(0.0_f64, |m, &x| m.max(x.abs())));
    }

    let std = median(&masked_rms);
    let mean = median(&masked_means);
    masked_peaks.sort_unstable_by(f64::total_cmp);
    let reference_peak = percentile_sorted(&masked_peaks, reference_percentile);
    let speech_fraction =
        speech_mask.iter().filter(|&&m| m).count() as f64 / n_frames as f64;

    SpeechReference {
        speech_mask,
        frame_times,
        mean,
        std: if std > 0.0 { std } else { 1.0 },
        reference_peak,
        speech_fraction,
    }
}

/// [`estimate_speech_reference`] with the contract's default parameters.
pub fn estimate_speech_reference_default(samples: &[f64], sample_rate: f64) -> SpeechReference {
    estimate_speech_reference(
        samples,
        sample_rate,
        DEFAULT_FRAME_S,
        DEFAULT_HOP_S,
        DEFAULT_SPEECH_FLOOR_DB,
        DEFAULT_REFERENCE_PERCENTILE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_type7() {
        // np.percentile([1, 2, 3, 4], 75) == 3.25
        assert_eq!(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 75.0), 3.25);
        // np.percentile([1, 2, 3, 4], 100) == 4
        assert_eq!(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 100.0), 4.0);
        // np.percentile([1, 2, 3, 4], 0) == 1
        assert_eq!(percentile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.0), 1.0);
        // np.median([1, 2, 3, 4]) == 2.5
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        // np.median([5, 1, 3]) == 3
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
    }

    #[test]
    fn all_zero_signal_falls_back_to_all_true() {
        // Long all-zero signal: every frame is "speech" (mask fallback),
        // frame peaks are 0, so reference_peak is 0; std falls back to 1.0.
        let samples = vec![0.0; 16000];
        let r = estimate_speech_reference_default(&samples, 16000.0);
        assert!(r.speech_mask.iter().all(|&m| m));
        assert_eq!(r.reference_peak, 0.0);
        // RMS carries the +1e-20 epsilon → median RMS = sqrt(1e-20) = 1e-10
        // (> 0, so no fallback to 1.0). Matches the reference implementation.
        assert_eq!(r.std, 1e-10);
        assert_eq!(r.mean, 0.0);
        assert_eq!(r.speech_fraction, 1.0);
        assert_eq!(r.speech_mask.len(), r.frame_times.len());
    }

    #[test]
    fn empty_signal() {
        // N == 0 → no frames, neutral standards (DECISIONS §1.1).
        let r = estimate_speech_reference_default(&[], 16000.0);
        assert!(r.speech_mask.is_empty());
        assert!(r.frame_times.is_empty());
        assert_eq!(r.mean, 0.0);
        assert_eq!(r.std, 1.0);
        assert_eq!(r.reference_peak, 0.0);
        assert_eq!(r.speech_fraction, 0.0);
    }

    #[test]
    fn short_signal_single_frame() {
        // N < n_frame (800 at 16 kHz): one frame; std = population std,
        // reference_peak = max|x|.
        let samples = vec![0.5; 100];
        let r = estimate_speech_reference_default(&samples, 16000.0);
        assert_eq!(r.speech_mask.len(), 1);
        assert!(r.speech_mask[0]);
        assert_eq!(r.frame_times[0], 100.0 / (2.0 * 16000.0));
        assert!((r.reference_peak - 0.5).abs() < 1e-15);
        assert!((r.mean - 0.5).abs() < 1e-15);
        // constant signal → population std is 0 → falls back to 1.0
        assert_eq!(r.std, 1.0);
    }

    #[test]
    fn framing_counts() {
        // N = 16000, frame = 800, hop = 160 → 1 + (16000-800)/160 = 96
        let samples = vec![0.1; 16000];
        let r = estimate_speech_reference_default(&samples, 16000.0);
        assert_eq!(r.speech_mask.len(), 96);
        assert_eq!(r.frame_times[0], (0.0 + 400.0) / 16000.0);
        assert_eq!(r.frame_times[1], (160.0 + 400.0) / 16000.0);
    }

    #[test]
    fn reference_peak_is_per_frame_not_sample() {
        // A constant tone: every frame's peak |x| ≈ amplitude, so the 75th
        // percentile of per-frame peaks is ≈ amplitude.
        let sr = 16000.0;
        let samples: Vec<f64> = (0..16000 * 2)
            .map(|i| 0.1 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / sr).sin())
            .collect();
        let r = estimate_speech_reference_default(&samples, sr);
        // per-frame peak of a 120 Hz tone over a 50 ms window ≈ amplitude
        assert!((r.reference_peak - 0.1).abs() < 0.005, "ref={}", r.reference_peak);
        // std = median per-frame RMS ≈ amplitude / sqrt(2)
        assert!((r.std - 0.1 / 2.0_f64.sqrt()).abs() < 0.005, "std={}", r.std);
    }

    #[test]
    fn burst_has_bounded_leverage() {
        // Quiet tone everywhere + 0.5 s loud burst: reference within 5% of
        // the burst-free value (SPEC: scale and peak unchanged to <5%).
        let sr = 16000usize;
        let n = sr * 60;
        let speech: Vec<f64> = (0..n)
            .map(|i| 0.02 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / sr as f64).sin())
            .collect();
        let mut burst = speech.clone();
        for s in &mut burst[sr * 30..sr * 30 + sr / 2] {
            *s = 0.9;
        }
        let r_speech = estimate_speech_reference_default(&speech, sr as f64);
        let r_burst = estimate_speech_reference_default(&burst, sr as f64);
        assert!(r_speech.reference_peak > 0.0);
        let rel_peak =
            (r_burst.reference_peak - r_speech.reference_peak).abs() / r_speech.reference_peak;
        let rel_std = (r_burst.std - r_speech.std).abs() / r_speech.std;
        assert!(rel_peak < 0.05, "peak moved {rel_peak}");
        assert!(rel_std < 0.05, "std moved {rel_std}");
    }
}
