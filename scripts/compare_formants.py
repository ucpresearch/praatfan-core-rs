#!/usr/bin/env python3
"""Compare formant analysis between Praat (parselmouth) and praat-core-rs.

Usage:
    python scripts/compare_formants.py path/to/audio.wav [--time-step 0.01] [--max-formants 5] [--max-formant-hz 5500]

Supported formats:
    - WAV (all sample rates, bit depths, mono/stereo): Full support
    - FLAC: Full support
    - MP3: Works but may show timing differences due to different decoder padding
    - OGG: Only supported by praat-core-rs (Praat doesn't support OGG natively)

Note on MP3 files:
    MP3 decoders handle encoder delay differently. Symphonia (used by praat-core-rs)
    and Praat's internal decoder may produce slightly different sample counts and
    timing. For accurate comparison, use lossless formats (WAV, FLAC).

Requirements:
    - parselmouth (pip install praat-parselmouth)
    - praat-core-rs must be built (cargo build --release --example formant_json)
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import numpy as np


def get_praat_formants(audio_path: str, time_step: float, max_formants: int,
                       max_formant_hz: float, window_length: float = 0.025,
                       pre_emphasis: float = 50.0) -> dict:
    """Extract formants using Praat/parselmouth."""
    snd = parselmouth.Sound(audio_path)
    formant = call(snd, "To Formant (burg)", time_step, max_formants,
                   max_formant_hz, window_length, pre_emphasis)

    # Get time range
    t_start = call(formant, "Get start time")
    t_end = call(formant, "Get end time")
    n_frames = call(formant, "Get number of frames")

    times = []
    f1, f2, f3, f4, f5 = [], [], [], [], []
    b1, b2, b3, b4, b5 = [], [], [], [], []

    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        times.append(t)

        for fn, (flist, blist) in enumerate([(f1, b1), (f2, b2), (f3, b3), (f4, b4), (f5, b5)], 1):
            try:
                freq = call(formant, "Get value at time", fn, t, "Hertz", "Linear")
                bw = call(formant, "Get bandwidth at time", fn, t, "Hertz", "Linear")
                flist.append(freq if not np.isnan(freq) else None)
                blist.append(bw if not np.isnan(bw) else None)
            except:
                flist.append(None)
                blist.append(None)

    return {
        "sample_rate": snd.sampling_frequency,
        "duration": snd.duration,
        "n_samples": snd.n_samples,
        "n_channels": snd.n_channels,
        "formant": {
            "times": times,
            "time_step": time_step,
            "max_num_formants": max_formants,
            "max_formant_hz": max_formant_hz,
            "window_length": window_length,
            "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
            "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5,
        }
    }


def get_rust_formants(audio_path: str, time_step: float, max_formants: int,
                      max_formant_hz: float, window_length: float = 0.025,
                      pre_emphasis: float = 50.0) -> dict:
    """Extract formants using praat-core-rs."""
    # Run the Rust example with JSON output
    project_root = Path(__file__).parent.parent
    rust_binary = project_root / "target" / "release" / "examples" / "formant_json"

    if not rust_binary.exists():
        # Try debug build
        rust_binary = project_root / "target" / "debug" / "examples" / "formant_json"

    if not rust_binary.exists():
        raise FileNotFoundError(
            "formant_json example not found. Run: cargo build --release --example formant_json"
        )

    result = subprocess.run(
        [str(rust_binary), audio_path,
         str(time_step), str(max_formants), str(max_formant_hz),
         str(window_length), str(pre_emphasis)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust formant extraction failed: {result.stderr}")

    return json.loads(result.stdout)


def compare_formants(praat_data: dict, rust_data: dict, formant_num: int) -> dict:
    """Compare formant values between Praat and Rust."""
    times = praat_data["formant"]["times"]
    praat_values = praat_data["formant"][f"f{formant_num}"]
    rust_values = rust_data["formant"][f"f{formant_num}"]

    errors = []
    for i, t in enumerate(times):
        pv = praat_values[i] if i < len(praat_values) else None
        rv = rust_values[i] if i < len(rust_values) else None

        if pv is not None and rv is not None:
            err = abs(pv - rv)
            errors.append({
                "time": t,
                "praat": pv,
                "rust": rv,
                "error": err
            })

    if not errors:
        return {
            "formant": formant_num,
            "total_points": 0,
            "within_1hz": 0,
            "within_5hz": 0,
            "max_error": 0.0,
            "mean_error": 0.0,
            "worst_errors": []
        }

    errors_hz = [e["error"] for e in errors]
    errors.sort(key=lambda e: -e["error"])  # Sort by descending error

    return {
        "formant": formant_num,
        "total_points": len(errors),
        "within_1hz": sum(1 for e in errors_hz if e <= 1.0),
        "within_5hz": sum(1 for e in errors_hz if e <= 5.0),
        "max_error": max(errors_hz),
        "mean_error": sum(errors_hz) / len(errors_hz),
        "worst_errors": errors[:5]  # Top 5 worst
    }


def main():
    parser = argparse.ArgumentParser(description="Compare formant analysis: Praat vs praat-core-rs")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--time-step", type=float, default=0.01, help="Time step (default: 0.01)")
    parser.add_argument("--max-formants", type=int, default=5, help="Max number of formants (default: 5)")
    parser.add_argument("--max-formant-hz", type=float, default=5500.0, help="Max formant frequency (default: 5500)")
    parser.add_argument("--window-length", type=float, default=0.025, help="Window length (default: 0.025)")
    parser.add_argument("--pre-emphasis", type=float, default=50.0, help="Pre-emphasis from Hz (default: 50)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error information")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Warn about lossy format limitations
    ext = audio_path.suffix.lower()
    if ext == ".ogg":
        print("Warning: OGG files are only supported by praat-core-rs, not Praat.", file=sys.stderr)
        print("         Comparison will fail on the Praat side.", file=sys.stderr)
        print()
    elif ext == ".mp3":
        print("Warning: MP3 files may show large differences due to decoder timing differences.", file=sys.stderr)
        print("         For accurate comparison, use lossless formats (WAV, FLAC).", file=sys.stderr)
        print()

    print(f"Analyzing: {audio_path.name}")
    print(f"Parameters: time_step={args.time_step}, max_formants={args.max_formants}, "
          f"max_formant_hz={args.max_formant_hz}, window_length={args.window_length}")
    print()

    # Get Praat formants
    print("Extracting formants with Praat...", end=" ", flush=True)
    praat_data = get_praat_formants(
        str(audio_path), args.time_step, args.max_formants,
        args.max_formant_hz, args.window_length, args.pre_emphasis
    )
    print(f"done ({praat_data['n_samples']} samples, {praat_data['duration']:.3f}s, "
          f"{praat_data['n_channels']} ch)")

    # Get Rust formants
    print("Extracting formants with praat-core-rs...", end=" ", flush=True)
    try:
        rust_data = get_rust_formants(
            str(audio_path), args.time_step, args.max_formants,
            args.max_formant_hz, args.window_length, args.pre_emphasis
        )
        print("done")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo build the required Rust example, run:")
        print("  cargo build --release --example formant_json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Formant Comparison Results")
    print("=" * 70)

    results = []
    for fn in range(1, min(4, args.max_formants + 1)):  # F1, F2, F3
        stats = compare_formants(praat_data, rust_data, fn)
        results.append(stats)

        if stats["total_points"] == 0:
            print(f"F{fn}: No data points")
            continue

        pct_1hz = 100.0 * stats["within_1hz"] / stats["total_points"]
        pct_5hz = 100.0 * stats["within_5hz"] / stats["total_points"]

        print(f"F{fn}: {stats['within_1hz']:3d}/{stats['total_points']} within 1 Hz ({pct_1hz:5.1f}%), "
              f"{stats['within_5hz']:3d}/{stats['total_points']} within 5 Hz ({pct_5hz:5.1f}%), "
              f"max: {stats['max_error']:.1f} Hz, mean: {stats['mean_error']:.2f} Hz")

        if args.verbose and stats["worst_errors"]:
            print(f"    Worst errors:")
            for err in stats["worst_errors"]:
                print(f"      t={err['time']:.3f}s: praat={err['praat']:.1f}, "
                      f"rust={err['rust']:.1f}, error={err['error']:.1f} Hz")

    if args.json:
        print()
        print("JSON output:")
        print(json.dumps({
            "file": str(audio_path),
            "praat": praat_data,
            "rust": rust_data,
            "comparison": results
        }, indent=2))


if __name__ == "__main__":
    main()
