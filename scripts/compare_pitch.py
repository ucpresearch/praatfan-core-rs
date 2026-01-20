#!/usr/bin/env python3
"""Compare pitch extraction between Praat/parselmouth and praatfan-core-rs."""

import json
import subprocess
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call

def get_praat_pitch(audio_path: str, time_step: float, pitch_floor: float, pitch_ceiling: float) -> dict:
    """Get pitch values from Praat/parselmouth."""
    snd = parselmouth.Sound(audio_path)

    # Use default parameters matching Praat's Sound_to_Pitch
    pitch = call(snd, "To Pitch", time_step, pitch_floor, pitch_ceiling)

    n_frames = call(pitch, "Get number of frames")
    start_time = call(pitch, "Get time from frame number", 1)
    time_step_actual = call(pitch, "Get time step")

    frames = []
    for i in range(1, n_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")

        frames.append({
            "time": t,
            "frequency": f0 if f0 is not None else 0.0,
            "strength": 0.0,  # Not easily accessible
            "voiced": f0 is not None and f0 > 0
        })

    return {
        "n_frames": n_frames,
        "start_time": start_time,
        "time_step": time_step_actual,
        "ceiling": pitch_ceiling,
        "frames": frames
    }

def get_rust_pitch(audio_path: str, time_step: float, pitch_floor: float, pitch_ceiling: float) -> dict:
    """Get pitch values from Rust implementation."""
    # Run the pitch example binary
    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "pitch_json", "--",
         audio_path, str(time_step), str(pitch_floor), str(pitch_ceiling)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    if result.returncode != 0:
        print(f"Error running Rust: {result.stderr}")
        return None

    return json.loads(result.stdout)

def compare_pitch(praat_data: dict, rust_data: dict) -> dict:
    """Compare pitch results."""
    praat_frames = praat_data["frames"]
    rust_frames = rust_data["frames"]

    n_praat = len(praat_frames)
    n_rust = len(rust_frames)

    print(f"Frame count: Praat={n_praat}, Rust={n_rust}")
    print(f"Start time: Praat={praat_data['start_time']:.6f}, Rust={rust_data['start_time']:.6f}")
    print(f"Time step: Praat={praat_data['time_step']:.6f}, Rust={rust_data['time_step']:.6f}")

    n_compare = min(n_praat, n_rust)

    matches = 0
    voicing_mismatches = 0
    freq_errors = []

    for i in range(n_compare):
        pf = praat_frames[i]
        rf = rust_frames[i]

        praat_voiced = pf["voiced"]
        rust_voiced = rf["voiced"]

        if praat_voiced != rust_voiced:
            voicing_mismatches += 1
            continue

        if praat_voiced and rust_voiced:
            error = abs(pf["frequency"] - rf["frequency"])
            freq_errors.append(error)
            if error < 1.0:  # Within 1 Hz
                matches += 1

    n_voiced = len(freq_errors)
    if n_voiced > 0:
        mean_error = sum(freq_errors) / n_voiced
        max_error = max(freq_errors)
        within_1hz = sum(1 for e in freq_errors if e < 1.0)
        within_5hz = sum(1 for e in freq_errors if e < 5.0)
        pct_1hz = 100 * within_1hz / n_voiced
        pct_5hz = 100 * within_5hz / n_voiced
    else:
        mean_error = max_error = 0
        pct_1hz = pct_5hz = 100

    print(f"\nComparison (n={n_compare} frames):")
    print(f"  Voicing mismatches: {voicing_mismatches}")
    print(f"  Voiced frames compared: {n_voiced}")
    if n_voiced > 0:
        print(f"  Mean F0 error: {mean_error:.4f} Hz")
        print(f"  Max F0 error: {max_error:.4f} Hz")
        print(f"  Within 1 Hz: {within_1hz}/{n_voiced} ({pct_1hz:.1f}%)")
        print(f"  Within 5 Hz: {within_5hz}/{n_voiced} ({pct_5hz:.1f}%)")

    # Show first few mismatches
    print("\nFirst 10 frame comparisons:")
    for i in range(min(10, n_compare)):
        pf = praat_frames[i]
        rf = rust_frames[i]
        voicing_match = "✓" if pf["voiced"] == rf["voiced"] else "✗"
        if pf["voiced"] and rf["voiced"]:
            error = abs(pf["frequency"] - rf["frequency"])
            freq_match = "✓" if error < 1.0 else f"Δ{error:.2f}"
            print(f"  Frame {i:3d}: t={pf['time']:.4f} Praat={pf['frequency']:.2f}Hz Rust={rf['frequency']:.2f}Hz {voicing_match} {freq_match}")
        else:
            praat_str = f"{pf['frequency']:.2f}Hz" if pf["voiced"] else "unvoiced"
            rust_str = f"{rf['frequency']:.2f}Hz" if rf["voiced"] else "unvoiced"
            print(f"  Frame {i:3d}: t={pf['time']:.4f} Praat={praat_str} Rust={rust_str} {voicing_match}")

    return {
        "n_frames_praat": n_praat,
        "n_frames_rust": n_rust,
        "voicing_mismatches": voicing_mismatches,
        "n_voiced": n_voiced,
        "mean_error_hz": mean_error,
        "max_error_hz": max_error,
        "pct_within_1hz": pct_1hz,
        "pct_within_5hz": pct_5hz
    }

def main():
    if len(sys.argv) < 2:
        audio_path = "tests/fixtures/one_two_three_four_five.wav"
    else:
        audio_path = sys.argv[1]

    # Default pitch parameters
    time_step = 0.0  # Auto
    pitch_floor = 75.0
    pitch_ceiling = 600.0

    print(f"Audio file: {audio_path}")
    print(f"Parameters: time_step={time_step}, floor={pitch_floor}, ceiling={pitch_ceiling}")
    print()

    # Get Praat results
    print("Running Praat/parselmouth...")
    praat_data = get_praat_pitch(audio_path, time_step, pitch_floor, pitch_ceiling)

    # Get Rust results
    print("Running Rust implementation...")
    rust_data = get_rust_pitch(audio_path, time_step, pitch_floor, pitch_ceiling)

    if rust_data is None:
        print("Failed to get Rust results")
        return 1

    # Compare
    print("\n" + "="*60)
    compare_pitch(praat_data, rust_data)

    return 0

if __name__ == "__main__":
    sys.exit(main())
