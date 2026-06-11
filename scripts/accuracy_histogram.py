#!/usr/bin/env python3
"""Generate accuracy histogram for all praatfan-core-rs statistics.

Compares praatfan-core-rs against Praat/parselmouth and generates
a histogram visualization of the accuracy across all modules.

Usage:
    python scripts/accuracy_histogram.py [--audio path/to/audio.wav]

Requirements:
    - parselmouth (pip install praat-parselmouth)
    - matplotlib (pip install matplotlib)
    - praatfan-core-rs examples built (cargo build --release --examples)

Output:
    Saves histogram to scripts/accuracy_histogram.png
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "tests" / "fixtures" / "one_two_three_four_five.wav"


@dataclass
class AccuracyStats:
    """Accuracy statistics for a single metric."""
    name: str
    total_points: int = 0
    within_tolerance: int = 0
    tolerance: float = 0.0
    tolerance_unit: str = ""
    mean_error: float = 0.0
    max_error: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        if self.total_points == 0:
            return 100.0
        return 100.0 * self.within_tolerance / self.total_points


def run_rust_example(name: str, *args) -> Optional[dict]:
    """Run a Rust example and return parsed JSON output."""
    binary = PROJECT_ROOT / "target" / "release" / "examples" / name
    if not binary.exists():
        binary = PROJECT_ROOT / "target" / "debug" / "examples" / name
    if not binary.exists():
        return None

    result = subprocess.run(
        [str(binary)] + [str(a) for a in args],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Warning: {name} failed: {result.stderr[:100]}")
        return None

    return json.loads(result.stdout)


def collect_formant_accuracy(audio_path: str) -> list[AccuracyStats]:
    """Collect formant accuracy statistics for F1, F2, F3."""
    time_step, max_formants, max_formant_hz = 0.01, 5, 5500.0
    window_length, pre_emphasis = 0.025, 50.0

    # Get Praat formants
    snd = parselmouth.Sound(audio_path)
    formant = call(snd, "To Formant (burg)", time_step, max_formants,
                   max_formant_hz, window_length, pre_emphasis)
    n_frames = call(formant, "Get number of frames")

    praat_data = {"f1": [], "f2": [], "f3": [], "times": []}
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        praat_data["times"].append(t)
        for fn in [1, 2, 3]:
            freq = call(formant, "Get value at time", fn, t, "Hertz", "Linear")
            praat_data[f"f{fn}"].append(freq if not np.isnan(freq) else None)

    # Get Rust formants
    rust_data = run_rust_example("formant_json", audio_path, time_step, max_formants,
                                  max_formant_hz, window_length, pre_emphasis)
    if rust_data is None:
        return []

    results = []
    for fn in [1, 2, 3]:
        stats = AccuracyStats(name=f"F{fn}", tolerance=1.0, tolerance_unit="Hz")
        praat_vals = praat_data[f"f{fn}"]
        rust_vals = rust_data["formant"][f"f{fn}"]

        for i in range(min(len(praat_vals), len(rust_vals))):
            pv, rv = praat_vals[i], rust_vals[i]
            if pv is not None and rv is not None:
                err = abs(pv - rv)
                stats.errors.append(err)
                stats.total_points += 1
                if err <= stats.tolerance:
                    stats.within_tolerance += 1

        if stats.errors:
            stats.mean_error = np.mean(stats.errors)
            stats.max_error = np.max(stats.errors)
        results.append(stats)

    return results


def collect_intensity_accuracy(audio_path: str) -> list[AccuracyStats]:
    """Collect intensity accuracy statistics."""
    min_pitch, time_step = 100.0, 0.0

    # Get Praat intensity
    snd = parselmouth.Sound(audio_path)
    intensity = call(snd, "To Intensity", min_pitch, time_step, "yes")
    n_frames = call(intensity, "Get number of frames")

    praat_times, praat_vals = [], []
    for i in range(1, n_frames + 1):
        t = call(intensity, "Get time from frame number", i)
        v = call(intensity, "Get value at time", t, "Cubic")
        praat_times.append(t)
        praat_vals.append(v if not np.isnan(v) else None)

    # Get Rust intensity
    rust_data = run_rust_example("intensity_json", audio_path, min_pitch, time_step)
    if rust_data is None:
        return []

    rust_times = rust_data["intensity"]["times"]
    rust_vals = rust_data["intensity"]["values"]

    stats = AccuracyStats(name="Intensity", tolerance=0.1, tolerance_unit="dB")

    # Match by time
    for i, pt in enumerate(praat_times):
        pv = praat_vals[i]
        if pv is None:
            continue
        for j, rt in enumerate(rust_times):
            if abs(pt - rt) < 0.0001:
                rv = rust_vals[j]
                if rv is not None:
                    err = abs(pv - rv)
                    stats.errors.append(err)
                    stats.total_points += 1
                    if err <= stats.tolerance:
                        stats.within_tolerance += 1
                break

    if stats.errors:
        stats.mean_error = np.mean(stats.errors)
        stats.max_error = np.max(stats.errors)

    return [stats]


def collect_pitch_accuracy(audio_path: str) -> list[AccuracyStats]:
    """Collect pitch (F0) accuracy statistics."""
    time_step, pitch_floor, pitch_ceiling = 0.0, 75.0, 600.0

    # Get Praat pitch
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", time_step, pitch_floor, pitch_ceiling)
    n_frames = call(pitch, "Get number of frames")

    praat_frames = []
    for i in range(1, n_frames + 1):
        t = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        voiced = f0 is not None and f0 > 0
        praat_frames.append({"time": t, "freq": f0 if voiced else 0.0, "voiced": voiced})

    # Get Rust pitch
    rust_data = run_rust_example("pitch_json", audio_path, time_step, pitch_floor, pitch_ceiling)
    if rust_data is None:
        return []

    rust_frames = rust_data["frames"]

    stats = AccuracyStats(name="Pitch (F0)", tolerance=1.0, tolerance_unit="Hz")
    voicing_stats = AccuracyStats(name="Voicing", tolerance=0, tolerance_unit="match")

    n_compare = min(len(praat_frames), len(rust_frames))
    for i in range(n_compare):
        pf, rf = praat_frames[i], rust_frames[i]

        voicing_stats.total_points += 1
        if pf["voiced"] == rf["voiced"]:
            voicing_stats.within_tolerance += 1

        if pf["voiced"] and rf["voiced"]:
            err = abs(pf["freq"] - rf["frequency"])
            stats.errors.append(err)
            stats.total_points += 1
            if err <= stats.tolerance:
                stats.within_tolerance += 1

    if stats.errors:
        stats.mean_error = np.mean(stats.errors)
        stats.max_error = np.max(stats.errors)

    return [stats, voicing_stats]


def collect_spectrum_accuracy(audio_path: str) -> list[AccuracyStats]:
    """Collect spectrum accuracy statistics."""
    # Get Praat spectrum
    snd = parselmouth.Sound(audio_path)
    spectrum = call(snd, "To Spectrum", "yes")

    praat_vals = {
        "cog": call(spectrum, "Get centre of gravity", 2.0),
        "std": call(spectrum, "Get standard deviation", 2.0),
        "skew": call(spectrum, "Get skewness", 2.0),
        "kurt": call(spectrum, "Get kurtosis", 2.0),
        "total_energy": call(spectrum, "Get band energy", 0.0, 0.0),
    }

    # Get Rust spectrum
    rust_data = run_rust_example("spectrum_json", audio_path)
    if rust_data is None:
        return []

    rs = rust_data["spectrum"]
    rust_vals = {
        "cog": rs["center_of_gravity"],
        "std": rs["standard_deviation"],
        "skew": rs["skewness"],
        "kurt": rs["kurtosis"],
        "total_energy": rs["total_energy"],
    }

    results = []
    metric_names = {
        "cog": "Spectrum CoG",
        "std": "Spectrum StdDev",
        "skew": "Spectrum Skew",
        "kurt": "Spectrum Kurt",
        "total_energy": "Total Energy"
    }

    for key, name in metric_names.items():
        pv, rv = praat_vals[key], rust_vals[key]
        if pv is None or rv is None or np.isnan(pv) or np.isnan(rv):
            continue

        stats = AccuracyStats(name=name, tolerance=1.0, tolerance_unit="%")

        # Calculate relative error
        if abs(pv) > 1e-10:
            rel_err = abs(pv - rv) / abs(pv) * 100.0
        else:
            rel_err = abs(rv) * 100.0

        stats.errors.append(rel_err)
        stats.total_points = 1
        stats.within_tolerance = 1 if rel_err <= stats.tolerance else 0
        stats.mean_error = rel_err
        stats.max_error = rel_err
        results.append(stats)

    return results


def collect_harmonicity_accuracy(audio_path: str, method: str = "ac") -> list[AccuracyStats]:
    """Collect harmonicity (HNR) accuracy statistics."""
    time_step, min_pitch = 0.01, 75.0
    silence_threshold = 0.1
    periods_per_window = 3.0 if method == "ac" else 1.0

    # Get Praat harmonicity
    snd = parselmouth.Sound(audio_path)
    if method == "cc":
        harmonicity = call(snd, "To Harmonicity (cc)", time_step, min_pitch,
                           silence_threshold, periods_per_window)
    else:
        harmonicity = call(snd, "To Harmonicity (ac)", time_step, min_pitch,
                           silence_threshold, periods_per_window)

    n_frames = harmonicity.nx
    t1, dt = harmonicity.x1, harmonicity.dx

    praat_vals = []
    for i in range(1, n_frames + 1):
        t = t1 + (i - 1) * dt
        v = call(harmonicity, "Get value at time", t, "Cubic")
        praat_vals.append(v if not np.isnan(v) else None)

    # Get Rust harmonicity
    rust_data = run_rust_example("harmonicity_json", audio_path, time_step, min_pitch,
                                  silence_threshold, periods_per_window, method)
    if rust_data is None:
        return []

    rust_vals = [v["hnr"] if v["voiced"] else None for v in rust_data["values"]]

    method_name = "AC" if method == "ac" else "CC"
    stats = AccuracyStats(name=f"HNR ({method_name})", tolerance=1.0, tolerance_unit="dB")

    def is_unvoiced(v):
        return v is None or v <= -199.0

    n_compare = min(len(praat_vals), len(rust_vals))
    for i in range(n_compare):
        pv, rv = praat_vals[i], rust_vals[i]

        if is_unvoiced(pv) and is_unvoiced(rv):
            continue  # Both unvoiced, skip
        if is_unvoiced(pv) or is_unvoiced(rv):
            stats.total_points += 1
            stats.errors.append(100.0)  # Voicing mismatch
            continue

        err = abs(pv - rv)
        stats.errors.append(err)
        stats.total_points += 1
        if err <= stats.tolerance:
            stats.within_tolerance += 1

    if stats.errors:
        stats.mean_error = np.mean(stats.errors)
        stats.max_error = np.max(stats.errors)

    return [stats]


def collect_all_accuracy(audio_path: str) -> list[AccuracyStats]:
    """Collect accuracy statistics for all modules."""
    all_stats = []

    print("Collecting accuracy statistics...")

    print("  Formants (F1, F2, F3)...", end=" ", flush=True)
    formant_stats = collect_formant_accuracy(audio_path)
    all_stats.extend(formant_stats)
    print(f"done ({len(formant_stats)} metrics)")

    print("  Intensity...", end=" ", flush=True)
    intensity_stats = collect_intensity_accuracy(audio_path)
    all_stats.extend(intensity_stats)
    print(f"done ({len(intensity_stats)} metrics)")

    print("  Pitch (F0)...", end=" ", flush=True)
    pitch_stats = collect_pitch_accuracy(audio_path)
    all_stats.extend(pitch_stats)
    print(f"done ({len(pitch_stats)} metrics)")

    print("  Spectrum moments...", end=" ", flush=True)
    spectrum_stats = collect_spectrum_accuracy(audio_path)
    all_stats.extend(spectrum_stats)
    print(f"done ({len(spectrum_stats)} metrics)")

    print("  Harmonicity (AC)...", end=" ", flush=True)
    hnr_ac_stats = collect_harmonicity_accuracy(audio_path, method="ac")
    all_stats.extend(hnr_ac_stats)
    print(f"done ({len(hnr_ac_stats)} metrics)")

    print("  Harmonicity (CC)...", end=" ", flush=True)
    hnr_cc_stats = collect_harmonicity_accuracy(audio_path, method="cc")
    all_stats.extend(hnr_cc_stats)
    print(f"done ({len(hnr_cc_stats)} metrics)")

    return all_stats


def print_summary(stats: list[AccuracyStats]):
    """Print summary table of accuracy statistics."""
    print()
    print("=" * 80)
    print("Accuracy Summary: praatfan-core-rs vs Praat/parselmouth")
    print("=" * 80)
    print()
    print(f"{'Metric':<20} {'Accuracy':>10} {'Points':>10} {'Mean Error':>15} {'Max Error':>15}")
    print("-" * 80)

    for s in stats:
        if s.total_points == 0:
            continue
        tol_str = f"{s.tolerance} {s.tolerance_unit}"
        err_str = f"{s.mean_error:.4g} {s.tolerance_unit}"
        max_str = f"{s.max_error:.4g} {s.tolerance_unit}"
        print(f"{s.name:<20} {s.accuracy_pct:>9.1f}% {s.total_points:>10} {err_str:>15} {max_str:>15}")

    print("-" * 80)

    # Overall summary
    total_points = sum(s.total_points for s in stats)
    total_within = sum(s.within_tolerance for s in stats)
    overall_pct = 100.0 * total_within / total_points if total_points > 0 else 0
    print(f"{'OVERALL':<20} {overall_pct:>9.1f}% {total_points:>10}")
    print()


def plot_histogram(stats: list[AccuracyStats], output_path: Path):
    """Generate and save histogram visualization."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping histogram plot")
        return

    # Filter out metrics with no data
    valid_stats = [s for s in stats if s.total_points > 0]
    if not valid_stats:
        print("No valid data to plot")
        return

    names = [s.name for s in valid_stats]
    accuracies = [s.accuracy_pct for s in valid_stats]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of accuracy per metric
    colors = ['#2ecc71' if a >= 99 else '#f1c40f' if a >= 90 else '#e74c3c' for a in accuracies]
    bars = ax1.barh(names, accuracies, color=colors)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Accuracy by Metric\n(within tolerance)')
    ax1.axvline(x=100, color='#2ecc71', linestyle='--', alpha=0.5, label='100%')
    ax1.axvline(x=95, color='#f1c40f', linestyle='--', alpha=0.5, label='95%')

    # Add percentage labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{acc:.1f}%', va='center', fontsize=9)

    # Histogram of all individual errors
    all_errors = []
    for s in valid_stats:
        all_errors.extend(s.errors)

    if all_errors:
        # Clip extreme outliers for visualization
        clipped_errors = np.clip(all_errors, 0, np.percentile(all_errors, 99))
        ax2.hist(clipped_errors, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of All Errors\n(n={len(all_errors)}, clipped at 99th percentile)')

        # Add statistics annotation
        median_err = np.median(all_errors)
        p95_err = np.percentile(all_errors, 95)
        stats_text = f'Median: {median_err:.4g}\n95th %ile: {p95_err:.4g}'
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, va='top', ha='right',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('praatfan-core-rs Accuracy vs Praat/parselmouth\n(one_two_three_four_five.wav)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate accuracy histogram for praatfan-core-rs")
    parser.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO),
                        help=f"Path to audio file (default: {DEFAULT_AUDIO.name})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for histogram (default: scripts/accuracy_histogram.png)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path(__file__).parent / "accuracy_histogram.png"

    print(f"Audio file: {audio_path.name}")
    print()

    # Check if Rust examples are built
    examples = ["formant_json", "intensity_json", "pitch_json", "spectrum_json", "harmonicity_json"]
    missing = []
    for ex in examples:
        if not (PROJECT_ROOT / "target" / "release" / "examples" / ex).exists():
            if not (PROJECT_ROOT / "target" / "debug" / "examples" / ex).exists():
                missing.append(ex)

    if missing:
        print(f"Warning: Missing Rust examples: {', '.join(missing)}")
        print("Run: cargo build --release --examples")
        print()

    # Collect all accuracy statistics
    all_stats = collect_all_accuracy(str(audio_path))

    # Print summary
    print_summary(all_stats)

    # Generate histogram
    plot_histogram(all_stats, output_path)


if __name__ == "__main__":
    main()
