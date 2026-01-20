#!/usr/bin/env python3
"""Compare spectrum analysis between Praat (parselmouth) and praatfan-core-rs.

Usage:
    python scripts/compare_spectrum.py path/to/audio.wav
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import call
import numpy as np


def get_praat_spectrum(audio_path: str) -> dict:
    """Extract spectrum using Praat/parselmouth."""
    snd = parselmouth.Sound(audio_path)
    spectrum = call(snd, "To Spectrum", "yes")  # yes = fast (power of 2)

    # Get spectral moments
    cog = call(spectrum, "Get centre of gravity", 2.0)
    std_dev = call(spectrum, "Get standard deviation", 2.0)
    skewness = call(spectrum, "Get skewness", 2.0)
    kurtosis = call(spectrum, "Get kurtosis", 2.0)

    # Get band energy (total)
    total_energy = call(spectrum, "Get band energy", 0.0, 0.0)

    # Get band energy for specific bands
    nyquist = snd.sampling_frequency / 2.0
    low_energy = call(spectrum, "Get band energy", 0.0, 1000.0)
    mid_energy = call(spectrum, "Get band energy", 1000.0, 4000.0)
    high_energy = call(spectrum, "Get band energy", 4000.0, nyquist)

    return {
        "sample_rate": snd.sampling_frequency,
        "duration": snd.duration,
        "n_samples": snd.n_samples,
        "spectrum": {
            "center_of_gravity": cog if not np.isnan(cog) else None,
            "standard_deviation": std_dev if not np.isnan(std_dev) else None,
            "skewness": skewness if not np.isnan(skewness) else None,
            "kurtosis": kurtosis if not np.isnan(kurtosis) else None,
            "total_energy": total_energy if not np.isnan(total_energy) else None,
            "low_energy": low_energy if not np.isnan(low_energy) else None,
            "mid_energy": mid_energy if not np.isnan(mid_energy) else None,
            "high_energy": high_energy if not np.isnan(high_energy) else None,
        }
    }


def get_rust_spectrum(audio_path: str) -> dict:
    """Extract spectrum using praatfan-core-rs."""
    project_root = Path(__file__).parent.parent
    rust_binary = project_root / "target" / "release" / "examples" / "spectrum_json"

    if not rust_binary.exists():
        rust_binary = project_root / "target" / "debug" / "examples" / "spectrum_json"

    if not rust_binary.exists():
        raise FileNotFoundError(
            "spectrum_json example not found. Run: cargo build --release --example spectrum_json"
        )

    result = subprocess.run(
        [str(rust_binary), audio_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust spectrum extraction failed: {result.stderr}")

    return json.loads(result.stdout)


def compare_value(name: str, praat_val, rust_val, tolerance_pct: float = 1.0) -> bool:
    """Compare two values and print results."""
    if praat_val is None and rust_val is None:
        print(f"  {name}: both undefined ✓")
        return True
    if praat_val is None or rust_val is None:
        print(f"  {name}: praat={praat_val}, rust={rust_val} ✗")
        return False

    if praat_val == 0:
        error_pct = abs(rust_val) * 100
    else:
        error_pct = abs(praat_val - rust_val) / abs(praat_val) * 100

    ok = error_pct <= tolerance_pct
    status = "✓" if ok else "✗"
    print(f"  {name}: praat={praat_val:.6g}, rust={rust_val:.6g}, error={error_pct:.2f}% {status}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare spectrum analysis: Praat vs praatfan-core-rs")
    parser.add_argument("audio_file", help="Path to audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {audio_path.name}")
    print()

    # Get Praat spectrum
    print("Extracting spectrum with Praat...", end=" ", flush=True)
    praat_data = get_praat_spectrum(str(audio_path))
    print("done")

    # Get Rust spectrum
    print("Extracting spectrum with praatfan-core-rs...", end=" ", flush=True)
    try:
        rust_data = get_rust_spectrum(str(audio_path))
        print("done")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo build the required Rust example, run:")
        print("  cargo build --release --example spectrum_json")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Spectrum Comparison Results")
    print("=" * 70)

    ps = praat_data["spectrum"]
    rs = rust_data["spectrum"]

    print("\nSpectral moments (power=2.0):")
    all_ok = True
    all_ok &= compare_value("Center of gravity", ps["center_of_gravity"], rs["center_of_gravity"])
    all_ok &= compare_value("Standard deviation", ps["standard_deviation"], rs["standard_deviation"])
    all_ok &= compare_value("Skewness", ps["skewness"], rs["skewness"])
    all_ok &= compare_value("Kurtosis", ps["kurtosis"], rs["kurtosis"])

    print("\nBand energies:")
    all_ok &= compare_value("Total energy", ps["total_energy"], rs["total_energy"])
    all_ok &= compare_value("Low (0-1000 Hz)", ps["low_energy"], rs["low_energy"])
    all_ok &= compare_value("Mid (1000-4000 Hz)", ps["mid_energy"], rs["mid_energy"])
    all_ok &= compare_value("High (4000-Nyquist)", ps["high_energy"], rs["high_energy"])

    print()
    if all_ok:
        print("All comparisons within tolerance ✓")
    else:
        print("Some comparisons outside tolerance ✗")


if __name__ == "__main__":
    main()
