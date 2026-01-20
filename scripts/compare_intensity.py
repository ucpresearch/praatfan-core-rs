#!/usr/bin/env python3
"""Compare intensity analysis between Praat (parselmouth) and praat-core-rs.

Usage:
    python scripts/compare_intensity.py path/to/audio.wav [--min-pitch 100] [--time-step 0.01]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import numpy as np


def get_praat_intensity(audio_path: str, min_pitch: float, time_step: float,
                        subtract_mean: bool = True) -> dict:
    """Extract intensity using Praat/parselmouth."""
    snd = parselmouth.Sound(audio_path)
    intensity = call(snd, "To Intensity", min_pitch, time_step, "yes" if subtract_mean else "no")

    n_frames = call(intensity, "Get number of frames")
    times = []
    values = []

    for i in range(1, n_frames + 1):
        t = call(intensity, "Get time from frame number", i)
        v = call(intensity, "Get value at time", t, "Cubic")
        times.append(t)
        values.append(v if not np.isnan(v) else None)

    return {
        "sample_rate": snd.sampling_frequency,
        "duration": snd.duration,
        "n_samples": snd.n_samples,
        "intensity": {
            "times": times,
            "values": values,
            "time_step": time_step,
            "min_pitch": min_pitch,
        }
    }


def get_rust_intensity(audio_path: str, min_pitch: float, time_step: float) -> dict:
    """Extract intensity using praat-core-rs."""
    project_root = Path(__file__).parent.parent
    rust_binary = project_root / "target" / "release" / "examples" / "intensity_json"

    if not rust_binary.exists():
        rust_binary = project_root / "target" / "debug" / "examples" / "intensity_json"

    if not rust_binary.exists():
        raise FileNotFoundError(
            "intensity_json example not found. Run: cargo build --release --example intensity_json"
        )

    result = subprocess.run(
        [str(rust_binary), audio_path, str(min_pitch), str(time_step)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust intensity extraction failed: {result.stderr}")

    return json.loads(result.stdout)


def compare_intensity(praat_data: dict, rust_data: dict) -> dict:
    """Compare intensity values between Praat and Rust."""
    praat_times = praat_data["intensity"]["times"]
    praat_values = praat_data["intensity"]["values"]
    rust_times = rust_data["intensity"]["times"]
    rust_values = rust_data["intensity"]["values"]

    # Match frames by time (within tolerance)
    errors = []
    time_tolerance = 0.0001  # 0.1 ms

    for i, pt in enumerate(praat_times):
        pv = praat_values[i]
        if pv is None:
            continue

        # Find matching rust frame
        rv = None
        for j, rt in enumerate(rust_times):
            if abs(pt - rt) < time_tolerance:
                rv = rust_values[j]
                break

        if rv is not None:
            err = abs(pv - rv)
            errors.append({
                "time": pt,
                "praat": pv,
                "rust": rv,
                "error": err
            })

    if not errors:
        return {
            "total_points": 0,
            "within_0_1db": 0,
            "within_1db": 0,
            "max_error": 0.0,
            "mean_error": 0.0,
            "worst_errors": []
        }

    errors_db = [e["error"] for e in errors]
    errors.sort(key=lambda e: -e["error"])

    return {
        "total_points": len(errors),
        "within_0_1db": sum(1 for e in errors_db if e <= 0.1),
        "within_1db": sum(1 for e in errors_db if e <= 1.0),
        "max_error": max(errors_db),
        "mean_error": sum(errors_db) / len(errors_db),
        "worst_errors": errors[:5]
    }


def main():
    parser = argparse.ArgumentParser(description="Compare intensity analysis: Praat vs praat-core-rs")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--min-pitch", type=float, default=100.0, help="Minimum pitch (default: 100)")
    parser.add_argument("--time-step", type=float, default=0.0, help="Time step (default: 0.0 = auto)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error information")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {audio_path.name}")
    print(f"Parameters: min_pitch={args.min_pitch}, time_step={args.time_step}")
    print()

    # Get Praat intensity
    print("Extracting intensity with Praat...", end=" ", flush=True)
    praat_data = get_praat_intensity(str(audio_path), args.min_pitch, args.time_step)
    print(f"done ({len(praat_data['intensity']['times'])} frames)")

    # Get Rust intensity
    print("Extracting intensity with praat-core-rs...", end=" ", flush=True)
    try:
        rust_data = get_rust_intensity(str(audio_path), args.min_pitch, args.time_step)
        print(f"done ({len(rust_data['intensity']['times'])} frames)")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo build the required Rust example, run:")
        print("  cargo build --release --example intensity_json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Intensity Comparison Results")
    print("=" * 70)

    # Show frame count comparison
    praat_frames = len(praat_data["intensity"]["times"])
    rust_frames = len(rust_data["intensity"]["times"])
    print(f"Praat frames: {praat_frames}, Rust frames: {rust_frames}")

    if praat_frames > 0 and rust_frames > 0:
        print(f"Praat time range: {praat_data['intensity']['times'][0]:.4f} - {praat_data['intensity']['times'][-1]:.4f}")
        print(f"Rust time range:  {rust_data['intensity']['times'][0]:.4f} - {rust_data['intensity']['times'][-1]:.4f}")
    print()

    stats = compare_intensity(praat_data, rust_data)

    if stats["total_points"] == 0:
        print("No matching data points found!")
        return

    pct_0_1db = 100.0 * stats["within_0_1db"] / stats["total_points"]
    pct_1db = 100.0 * stats["within_1db"] / stats["total_points"]

    print(f"Matched frames: {stats['total_points']}")
    print(f"Within 0.1 dB: {stats['within_0_1db']:3d}/{stats['total_points']} ({pct_0_1db:5.1f}%)")
    print(f"Within 1.0 dB: {stats['within_1db']:3d}/{stats['total_points']} ({pct_1db:5.1f}%)")
    print(f"Max error: {stats['max_error']:.2f} dB")
    print(f"Mean error: {stats['mean_error']:.2f} dB")

    if args.verbose and stats["worst_errors"]:
        print(f"\nWorst errors:")
        for err in stats["worst_errors"]:
            print(f"  t={err['time']:.3f}s: praat={err['praat']:.2f}, "
                  f"rust={err['rust']:.2f}, error={err['error']:.2f} dB")


if __name__ == "__main__":
    main()
