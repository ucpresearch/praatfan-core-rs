#!/usr/bin/env python3
"""Compare harmonicity analysis between Praat (parselmouth) and praat-core-rs.

Usage:
    python scripts/compare_harmonicity.py path/to/audio.wav [options]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import numpy as np


def get_praat_harmonicity(audio_path: str, time_step: float, min_pitch: float,
                          silence_threshold: float, periods_per_window: float,
                          method: str = "ac") -> dict:
    """Extract harmonicity using Praat/parselmouth.

    method: "ac" for autocorrelation, "cc" for cross-correlation
    """
    snd = parselmouth.Sound(audio_path)
    if method == "cc":
        harmonicity = call(snd, "To Harmonicity (cc)", time_step, min_pitch,
                           silence_threshold, periods_per_window)
    else:
        harmonicity = call(snd, "To Harmonicity (ac)", time_step, min_pitch,
                           silence_threshold, periods_per_window)

    n_frames = harmonicity.nx
    t1 = harmonicity.x1
    dt = harmonicity.dx

    # Get values at each frame
    values = []
    for i in range(1, n_frames + 1):
        t = t1 + (i - 1) * dt
        v = call(harmonicity, "Get value at time", t, "Cubic")
        values.append(v if not np.isnan(v) else None)

    # Statistics
    mean = call(harmonicity, "Get mean", 0.0, 0.0)
    minimum = call(harmonicity, "Get minimum", 0.0, 0.0, "Parabolic")
    maximum = call(harmonicity, "Get maximum", 0.0, 0.0, "Parabolic")

    return {
        "sample_rate": snd.sampling_frequency,
        "duration": snd.duration,
        "harmonicity": {
            "n_frames": n_frames,
            "time_step": dt,
            "first_time": t1,
            "values": values,
            "mean": mean if not np.isnan(mean) else None,
            "min": minimum if not np.isnan(minimum) else None,
            "max": maximum if not np.isnan(maximum) else None,
        }
    }


def get_rust_harmonicity(audio_path: str, time_step: float, min_pitch: float,
                         silence_threshold: float, periods_per_window: float,
                         method: str = "cc") -> dict:
    """Extract harmonicity using praat-core-rs."""
    project_root = Path(__file__).parent.parent
    rust_binary = project_root / "target" / "release" / "examples" / "harmonicity_json"

    if not rust_binary.exists():
        rust_binary = project_root / "target" / "debug" / "examples" / "harmonicity_json"

    if not rust_binary.exists():
        raise FileNotFoundError(
            "harmonicity_json example not found. Run: cargo build --release --example harmonicity_json"
        )

    result = subprocess.run(
        [str(rust_binary), audio_path, str(time_step), str(min_pitch),
         str(silence_threshold), str(periods_per_window), method],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust harmonicity extraction failed: {result.stderr}")

    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Compare harmonicity analysis: Praat vs praat-core-rs")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--time-step", type=float, default=0.01, help="Time step (default: 0.01)")
    parser.add_argument("--min-pitch", type=float, default=75.0, help="Min pitch (default: 75)")
    parser.add_argument("--silence-threshold", type=float, default=0.1, help="Silence threshold (default: 0.1)")
    parser.add_argument("--periods-per-window", type=float, default=None, help="Periods per window (default: 3.0 for ac, 1.0 for cc)")
    parser.add_argument("--method", choices=["ac", "cc"], default="ac", help="Method: ac (autocorrelation) or cc (cross-correlation)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    # Set default periods_per_window based on method
    if args.periods_per_window is None:
        args.periods_per_window = 3.0 if args.method == "ac" else 1.0

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {audio_path.name}")
    print(f"Method: {args.method.upper()}")
    print(f"Parameters: time_step={args.time_step}s, min_pitch={args.min_pitch}Hz, "
          f"silence_threshold={args.silence_threshold}, periods_per_window={args.periods_per_window}")
    print()

    # Get Praat harmonicity
    print(f"Extracting harmonicity with Praat ({args.method})...", end=" ", flush=True)
    praat_data = get_praat_harmonicity(str(audio_path), args.time_step, args.min_pitch,
                                       args.silence_threshold, args.periods_per_window,
                                       method=args.method)
    ph = praat_data["harmonicity"]
    print(f"done ({ph['n_frames']} frames)")

    # Get Rust harmonicity
    print(f"Extracting harmonicity with praat-core-rs ({args.method})...", end=" ", flush=True)
    try:
        rust_data = get_rust_harmonicity(str(audio_path), args.time_step, args.min_pitch,
                                         args.silence_threshold, args.periods_per_window,
                                         method=args.method)
        # Convert Rust output format to match Praat format
        rh = {
            "n_frames": rust_data["n_frames"],
            "time_step": rust_data["time_step"],
            "first_time": rust_data["start_time"],
            "values": [v["hnr"] if v["voiced"] else None for v in rust_data["values"]],
            "mean": None,  # Not computed by rust
            "min": None,
            "max": None
        }
        print(f"done ({rh['n_frames']} frames)")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo build the required Rust example, run:")
        print("  cargo build --release --example harmonicity_json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Harmonicity Comparison Results")
    print("=" * 70)

    # Compare dimensions
    print(f"\nDimensions:")
    print(f"  Praat:  {ph['n_frames']} frames")
    print(f"  Rust:   {rh['n_frames']} frames")

    print(f"\nTiming:")
    print(f"  Praat:  first_time={ph['first_time']:.6f}s, time_step={ph['time_step']:.6f}s")
    print(f"  Rust:   first_time={rh['first_time']:.6f}s, time_step={rh['time_step']:.6f}s")

    print(f"\nStatistics:")
    print(f"  Praat:  mean={ph['mean']}, min={ph['min']}, max={ph['max']}")
    print(f"  Rust:   mean={rh['mean']}, min={rh['min']}, max={rh['max']}")

    # Compare values at sample frames
    print(f"\nHNR values at sample frames:")
    matches = 0
    total = 0

    # Sample frames evenly across the signal
    n_samples = min(20, min(ph['n_frames'], rh['n_frames']))
    sample_frames = [int(i * ph['n_frames'] / n_samples) for i in range(n_samples)]

    def is_unvoiced(v):
        """Check if value represents unvoiced (None or <= -199)"""
        if v is None:
            return True
        return v <= -199.0

    for frame in sample_frames:
        if frame >= len(ph['values']) or frame >= len(rh['values']):
            continue

        praat_v = ph['values'][frame]
        rust_v = rh['values'][frame]

        praat_unvoiced = is_unvoiced(praat_v)
        rust_unvoiced = is_unvoiced(rust_v)

        if praat_unvoiced and rust_unvoiced:
            # Both unvoiced - match
            if args.verbose:
                t = ph['first_time'] + frame * ph['time_step']
                print(f"  t={t:.3f}s: both unvoiced ✓")
            matches += 1
            total += 1
        elif praat_unvoiced or rust_unvoiced:
            # One voiced, one unvoiced - mismatch
            t = ph['first_time'] + frame * ph['time_step']
            print(f"  t={t:.3f}s: praat={praat_v}, rust={rust_v} ✗")
            total += 1
        else:
            total += 1
            error = abs(praat_v - rust_v)
            # For HNR, consider 1 dB tolerance
            if error < 1.0:
                matches += 1
                status = "✓"
            else:
                status = "✗"
            if args.verbose or error >= 1.0:
                t = ph['first_time'] + frame * ph['time_step']
                print(f"  t={t:.3f}s: praat={praat_v:.2f} dB, rust={rust_v:.2f} dB, diff={error:.2f} dB {status}")

    if total > 0:
        print(f"\nSummary: {matches}/{total} values within 1 dB ({100*matches/total:.1f}%)")
    else:
        print("\nNo valid values to compare")


if __name__ == "__main__":
    main()
