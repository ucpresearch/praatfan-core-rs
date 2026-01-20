#!/usr/bin/env python3
"""Compare spectrogram analysis between Praat (parselmouth) and praatfan-core-rs.

Usage:
    python scripts/compare_spectrogram.py path/to/audio.wav [options]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import numpy as np


def get_praat_spectrogram(audio_path: str, window_length: float, max_freq: float,
                          time_step: float, freq_step: float) -> dict:
    """Extract spectrogram using Praat/parselmouth."""
    snd = parselmouth.Sound(audio_path)
    # "Gaussian" window is the default
    spectrogram = call(snd, "To Spectrogram", window_length, max_freq, time_step, freq_step, "Gaussian")

    # Access properties directly
    n_times = spectrogram.nx
    n_freqs = spectrogram.ny
    t1 = spectrogram.x1
    dt = spectrogram.dx
    f1 = spectrogram.y1
    df = spectrogram.dy

    # Sample some power values at specific time/frequency points
    test_points = []
    for t_frac in [0.25, 0.5, 0.75]:
        t = snd.duration * t_frac
        frame = int(round((t - t1) / dt) + 1)
        if 1 <= frame <= n_times:
            actual_t = t1 + (frame - 1) * dt
            # Sample at various frequency bins across the spectrum
            for bin_idx in [5, 10, 20, 40, 64, 100, 150, 200]:
                if bin_idx < n_freqs:
                    actual_f = f1 + bin_idx * df
                    power = call(spectrogram, "Get power at", actual_t, actual_f)
                    test_points.append({
                        "time": actual_t,
                        "freq": actual_f,
                        "power": power if not np.isnan(power) else None
                    })

    return {
        "sample_rate": snd.sampling_frequency,
        "duration": snd.duration,
        "spectrogram": {
            "n_times": n_times,
            "n_freqs": n_freqs,
            "time_step": dt,
            "freq_step": df,
            "first_time": t1,
            "first_freq": f1,
            "test_points": test_points,
        }
    }


def get_rust_spectrogram(audio_path: str, window_length: float, max_freq: float,
                         time_step: float, freq_step: float) -> dict:
    """Extract spectrogram using praatfan-core-rs."""
    project_root = Path(__file__).parent.parent
    rust_binary = project_root / "target" / "release" / "examples" / "spectrogram_json"

    if not rust_binary.exists():
        rust_binary = project_root / "target" / "debug" / "examples" / "spectrogram_json"

    if not rust_binary.exists():
        raise FileNotFoundError(
            "spectrogram_json example not found. Run: cargo build --release --example spectrogram_json"
        )

    result = subprocess.run(
        [str(rust_binary), audio_path, str(window_length), str(max_freq),
         str(time_step), str(freq_step)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust spectrogram extraction failed: {result.stderr}")

    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Compare spectrogram analysis: Praat vs praatfan-core-rs")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--window-length", type=float, default=0.005, help="Window length (default: 0.005)")
    parser.add_argument("--max-freq", type=float, default=5000.0, help="Max frequency (default: 5000)")
    parser.add_argument("--time-step", type=float, default=0.002, help="Time step (default: 0.002)")
    parser.add_argument("--freq-step", type=float, default=20.0, help="Frequency step (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {audio_path.name}")
    print(f"Parameters: window={args.window_length}s, max_freq={args.max_freq}Hz, "
          f"time_step={args.time_step}s, freq_step={args.freq_step}Hz")
    print()

    # Get Praat spectrogram
    print("Extracting spectrogram with Praat...", end=" ", flush=True)
    praat_data = get_praat_spectrogram(str(audio_path), args.window_length, args.max_freq,
                                        args.time_step, args.freq_step)
    ps = praat_data["spectrogram"]
    print(f"done ({ps['n_times']} times × {ps['n_freqs']} freqs)")

    # Get Rust spectrogram
    print("Extracting spectrogram with praatfan-core-rs...", end=" ", flush=True)
    try:
        rust_data = get_rust_spectrogram(str(audio_path), args.window_length, args.max_freq,
                                          args.time_step, args.freq_step)
        rs = rust_data["spectrogram"]
        print(f"done ({rs['n_times']} times × {rs['n_freqs']} freqs)")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo build the required Rust example, run:")
        print("  cargo build --release --example spectrogram_json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Spectrogram Comparison Results")
    print("=" * 70)

    # Compare dimensions
    print(f"\nDimensions:")
    print(f"  Praat:  {ps['n_times']} frames × {ps['n_freqs']} frequency bins")
    print(f"  Rust:   {rs['n_times']} frames × {rs['n_freqs']} frequency bins")

    print(f"\nTiming:")
    print(f"  Praat:  first_time={ps['first_time']:.6f}s, time_step={ps['time_step']:.6f}s")
    print(f"  Rust:   first_time={rs['first_time']:.6f}s, time_step={rs['time_step']:.6f}s")

    print(f"\nFrequency:")
    print(f"  Praat:  first_freq={ps['first_freq']:.2f}Hz, freq_step={ps['freq_step']:.2f}Hz")
    print(f"  Rust:   first_freq={rs['first_freq']:.2f}Hz, freq_step={rs['freq_step']:.2f}Hz")

    # Compare test points
    print(f"\nPower values at sample points:")
    praat_points = {(p["time"], p["freq"]): p["power"] for p in ps["test_points"]}
    rust_points = {(p["time"], p["freq"]): p["power"] for p in rs["test_points"]}

    matches = 0
    total = 0
    for (t, f), praat_power in praat_points.items():
        if praat_power is None:
            continue
        # Find closest rust point
        rust_power = None
        for (rt, rf), rp in rust_points.items():
            if abs(rt - t) < 0.001 and abs(rf - f) < 1.0:
                rust_power = rp
                break

        if rust_power is not None:
            total += 1
            if praat_power > 0 and rust_power > 0:
                error_pct = abs(praat_power - rust_power) / praat_power * 100
                if error_pct < 5.0:
                    matches += 1
                status = "✓" if error_pct < 5.0 else "✗"
                if args.verbose or error_pct >= 5.0:
                    print(f"  t={t:.3f}s, f={f:.0f}Hz: praat={praat_power:.2e}, rust={rust_power:.2e}, error={error_pct:.1f}% {status}")
            elif praat_power == 0 and rust_power == 0:
                matches += 1
            else:
                if args.verbose:
                    print(f"  t={t:.3f}s, f={f:.0f}Hz: praat={praat_power:.2e}, rust={rust_power:.2e} ✗")

    if total > 0:
        print(f"\nSummary: {matches}/{total} power values within 5% ({100*matches/total:.1f}%)")
    else:
        print("\nNo matching points found for comparison")


if __name__ == "__main__":
    main()
