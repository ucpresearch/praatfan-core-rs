#!/usr/bin/env python3
"""
Generate ground truth values from parselmouth for testing praatfan-core-rs.

This script analyzes audio files using parselmouth (Python bindings for Praat)
and outputs JSON files with the expected values for comparison testing.
"""

import json
import os
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call


def analyze_file(audio_path: str) -> dict:
    """Analyze an audio file and return all analysis results."""
    print(f"Analyzing: {audio_path}")

    # Load sound
    snd = parselmouth.Sound(audio_path)

    result = {
        "file": os.path.basename(audio_path),
        "sound": {
            "duration": snd.duration,
            "sample_rate": snd.sampling_frequency,
            "num_samples": snd.n_samples,
            "num_channels": snd.n_channels,
        },
        "pitch": {},
        "intensity": {},
        "formant": {},
        "harmonicity": {},
        "spectrum": {},
    }

    # Pitch analysis
    try:
        pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)

        # Get pitch values at regular intervals
        times = []
        values = []
        time_step = 0.01
        t = 0.0
        while t <= snd.duration:
            times.append(t)
            try:
                val = call(pitch, "Get value at time", t, "Hertz", "Linear")
                values.append(val if val == val else None)  # NaN check
            except:
                values.append(None)
            t += time_step

        result["pitch"] = {
            "time_step": time_step,
            "pitch_floor": 75.0,
            "pitch_ceiling": 600.0,
            "times": times,
            "values": values,
            "mean": call(pitch, "Get mean", 0, 0, "Hertz") if any(v for v in values if v) else None,
            "min": call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic") if any(v for v in values if v) else None,
            "max": call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic") if any(v for v in values if v) else None,
        }
    except Exception as e:
        print(f"  Pitch analysis failed: {e}")

    # Intensity analysis
    try:
        intensity = call(snd, "To Intensity", 100.0, 0.0)

        times = []
        values = []
        time_step = 0.01
        t = 0.0
        while t <= snd.duration:
            times.append(t)
            try:
                val = call(intensity, "Get value at time", t, "Cubic")
                values.append(val if val == val else None)
            except:
                values.append(None)
            t += time_step

        result["intensity"] = {
            "min_pitch": 100.0,
            "time_step": time_step,
            "times": times,
            "values": values,
            "mean": call(intensity, "Get mean", 0, 0, "energy"),
            "min": call(intensity, "Get minimum", 0, 0, "Parabolic"),
            "max": call(intensity, "Get maximum", 0, 0, "Parabolic"),
        }
    except Exception as e:
        print(f"  Intensity analysis failed: {e}")

    # Formant analysis
    # Note: Use explicit time_step=0.01 to match test parameters
    # (time_step=0.0 means Praat uses default = window_length/4 = 0.00625)
    try:
        formant = call(snd, "To Formant (burg)", 0.01, 5, 5500.0, 0.025, 50.0)

        times = []
        f1_values = []
        f2_values = []
        f3_values = []
        b1_values = []
        time_step = 0.01
        t = 0.0
        while t <= snd.duration:
            times.append(t)
            try:
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
                b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
                f1_values.append(f1 if f1 == f1 else None)
                f2_values.append(f2 if f2 == f2 else None)
                f3_values.append(f3 if f3 == f3 else None)
                b1_values.append(b1 if b1 == b1 else None)
            except:
                f1_values.append(None)
                f2_values.append(None)
                f3_values.append(None)
                b1_values.append(None)
            t += time_step

        result["formant"] = {
            "max_num_formants": 5,
            "max_formant_hz": 5500.0,
            "window_length": 0.025,
            "time_step": time_step,
            "times": times,
            "f1": f1_values,
            "f2": f2_values,
            "f3": f3_values,
            "b1": b1_values,
        }
    except Exception as e:
        print(f"  Formant analysis failed: {e}")

    # Harmonicity analysis
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)

        times = []
        values = []
        time_step = 0.01
        t = 0.0
        while t <= snd.duration:
            times.append(t)
            try:
                val = call(harmonicity, "Get value at time", t, "Cubic")
                values.append(val if val == val else None)
            except:
                values.append(None)
            t += time_step

        result["harmonicity"] = {
            "time_step": 0.01,
            "min_pitch": 75.0,
            "times": times,
            "values": values,
            "mean": call(harmonicity, "Get mean", 0, 0),
        }
    except Exception as e:
        print(f"  Harmonicity analysis failed: {e}")

    # Spectrum analysis (single frame at center)
    try:
        # Extract center portion
        center_time = snd.duration / 2
        extract_duration = 0.05
        start = max(0, center_time - extract_duration / 2)
        end = min(snd.duration, center_time + extract_duration / 2)

        snd_extract = call(snd, "Extract part", start, end, "Hanning", 1, "no")
        spectrum = call(snd_extract, "To Spectrum", "yes")

        result["spectrum"] = {
            "center_of_gravity": call(spectrum, "Get centre of gravity", 2),
            "standard_deviation": call(spectrum, "Get standard deviation", 2),
            "skewness": call(spectrum, "Get skewness", 2),
            "kurtosis": call(spectrum, "Get kurtosis", 2),
        }
    except Exception as e:
        print(f"  Spectrum analysis failed: {e}")

    return result


def main():
    # Find test fixtures
    script_dir = Path(__file__).parent
    fixtures_dir = script_dir.parent / "tests" / "fixtures"
    output_dir = script_dir.parent / "tests" / "ground_truth"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all WAV files (parselmouth works best with WAV)
    wav_files = list(fixtures_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {fixtures_dir}")
        return 1

    all_results = {}

    for wav_file in sorted(wav_files):
        try:
            result = analyze_file(str(wav_file))
            all_results[wav_file.stem] = result

            # Also save individual file
            output_file = output_dir / f"{wav_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {output_file}")
        except Exception as e:
            print(f"  Error analyzing {wav_file}: {e}")

    # Save combined results
    combined_file = output_dir / "all_ground_truth.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results: {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
