#!/usr/bin/env python3
"""
Acoustic Feature Extraction using praatfan-core

This example demonstrates how to extract comprehensive acoustic features
from audio files using praatfan-core, matching the functionality of
parselmouth/Praat. It's designed as a drop-in replacement for the
acoustic extraction in the ozen project.

Features extracted:
    - Pitch (F0): Fundamental frequency
    - Formants (F1-F4): Vocal tract resonances
    - Formant bandwidths (B1-B4)
    - Intensity: Sound pressure level in dB
    - HNR: Harmonics-to-noise ratio
    - Spectral moments: CoG, std, skewness, kurtosis
    - Spectral tilt and nasal-related measures

Usage:
    python acoustic_extraction.py audio.wav [--output features.csv]
    python acoustic_extraction.py audio.wav --json  # JSON output
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Import praatfan_gpl - the Rust-based Praat reimplementation
from praatfan_gpl import Sound, Pitch, Formant, Intensity, Harmonicity, Spectrum


@dataclass
class AcousticFeatures:
    """
    Container for time-aligned acoustic features.

    All arrays are aligned to the same time axis (self.times).
    NaN values indicate frames where a feature could not be computed.
    """
    times: np.ndarray
    f0: np.ndarray
    intensity: np.ndarray
    hnr: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    b3: np.ndarray
    b4: np.ndarray
    cog: np.ndarray
    spectral_std: np.ndarray
    skewness: np.ndarray
    kurtosis: np.ndarray
    nasal_murmur_ratio: np.ndarray
    spectral_tilt: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary with lists (JSON-serializable)."""
        return {
            'times': self.times.tolist(),
            'f0': [None if np.isnan(x) else x for x in self.f0],
            'intensity': [None if np.isnan(x) else x for x in self.intensity],
            'hnr': [None if np.isnan(x) else x for x in self.hnr],
            'formants': {
                'F1': [None if np.isnan(x) else x for x in self.f1],
                'F2': [None if np.isnan(x) else x for x in self.f2],
                'F3': [None if np.isnan(x) else x for x in self.f3],
                'F4': [None if np.isnan(x) else x for x in self.f4],
            },
            'bandwidths': {
                'B1': [None if np.isnan(x) else x for x in self.b1],
                'B2': [None if np.isnan(x) else x for x in self.b2],
                'B3': [None if np.isnan(x) else x for x in self.b3],
                'B4': [None if np.isnan(x) else x for x in self.b4],
            },
            'cog': [None if np.isnan(x) else x for x in self.cog],
            'spectral_std': [None if np.isnan(x) else x for x in self.spectral_std],
            'skewness': [None if np.isnan(x) else x for x in self.skewness],
            'kurtosis': [None if np.isnan(x) else x for x in self.kurtosis],
            'nasal_murmur_ratio': [None if np.isnan(x) else x for x in self.nasal_murmur_ratio],
            'spectral_tilt': [None if np.isnan(x) else x for x in self.spectral_tilt],
        }


def extract_features(
    audio_path: str | Path,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    max_formant: float = 5500.0,
    window_duration: float = 0.025,
    progress_callback: Optional[callable] = None
) -> AcousticFeatures:
    """
    Extract acoustic features from an audio file.

    This function provides the same features as ozen's acoustic.py but uses
    praatfan-core instead of parselmouth.

    Args:
        audio_path: Path to audio file (WAV, FLAC, MP3, OGG)
        time_step: Analysis time step in seconds (default 0.01 = 10ms)
        pitch_floor: Minimum F0 for pitch analysis (Hz)
        pitch_ceiling: Maximum F0 for pitch analysis (Hz)
        max_formant: Maximum formant frequency (Hz)
        window_duration: Window size for spectral analysis
        progress_callback: Optional callback(progress: float) for updates

    Returns:
        AcousticFeatures with all extracted measurements
    """
    # Load audio
    snd = Sound.from_file(str(audio_path))
    duration = snd.duration

    # Create analysis objects (computed once, then queried)
    pitch = snd.to_pitch(time_step, pitch_floor, pitch_ceiling)
    intensity = snd.to_intensity(pitch_floor, time_step)
    formants = snd.to_formant_burg(time_step, 5, max_formant, 0.025, 50.0)
    harmonicity = snd.to_harmonicity_cc(time_step, pitch_floor, 0.1, 1.0)

    # Generate time points
    times = np.arange(0, duration, time_step)
    n_frames = len(times)

    # Initialize arrays
    f0_vals = np.full(n_frames, np.nan)
    intensity_vals = np.full(n_frames, np.nan)
    hnr_vals = np.full(n_frames, np.nan)
    f1_vals = np.full(n_frames, np.nan)
    f2_vals = np.full(n_frames, np.nan)
    f3_vals = np.full(n_frames, np.nan)
    f4_vals = np.full(n_frames, np.nan)
    b1_vals = np.full(n_frames, np.nan)
    b2_vals = np.full(n_frames, np.nan)
    b3_vals = np.full(n_frames, np.nan)
    b4_vals = np.full(n_frames, np.nan)
    cog_vals = np.full(n_frames, np.nan)
    std_vals = np.full(n_frames, np.nan)
    skew_vals = np.full(n_frames, np.nan)
    kurt_vals = np.full(n_frames, np.nan)
    nasal_vals = np.full(n_frames, np.nan)
    tilt_vals = np.full(n_frames, np.nan)

    # Extract features at each time point
    for i, t in enumerate(times):
        if progress_callback and i % 100 == 0:
            progress_callback(i / n_frames)

        # Basic features from pre-computed analysis objects
        f0_val = pitch.get_value_at_time(t, "hertz", "linear")
        f0_vals[i] = f0_val if f0_val is not None else np.nan

        int_val = intensity.get_value_at_time(t, "cubic")
        intensity_vals[i] = int_val if int_val is not None else np.nan

        hnr_val = harmonicity.get_value_at_time(t, "linear")
        hnr_vals[i] = hnr_val if hnr_val is not None else np.nan

        # Formants and bandwidths
        f1_vals[i] = formants.get_value_at_time(1, t, "hertz", "linear") or np.nan
        f2_vals[i] = formants.get_value_at_time(2, t, "hertz", "linear") or np.nan
        f3_vals[i] = formants.get_value_at_time(3, t, "hertz", "linear") or np.nan
        f4_vals[i] = formants.get_value_at_time(4, t, "hertz", "linear") or np.nan
        b1_vals[i] = formants.get_bandwidth_at_time(1, t, "hertz", "linear") or np.nan
        b2_vals[i] = formants.get_bandwidth_at_time(2, t, "hertz", "linear") or np.nan
        b3_vals[i] = formants.get_bandwidth_at_time(3, t, "hertz", "linear") or np.nan
        b4_vals[i] = formants.get_bandwidth_at_time(4, t, "hertz", "linear") or np.nan

        # Spectral moments from short-time spectrum
        t_start = max(0, t - window_duration / 2)
        t_end = min(duration, t + window_duration / 2)

        try:
            segment = snd.extract_part(t_start, t_end, "rectangular", 1.0, False)
            spectrum = segment.to_spectrum(True)

            cog_vals[i] = spectrum.get_center_of_gravity(2)
            std_vals[i] = spectrum.get_standard_deviation(2)
            skew_vals[i] = spectrum.get_skewness(2)
            kurt_vals[i] = spectrum.get_kurtosis(2)

            # Nasal-related features
            low_freq_energy = spectrum.get_band_energy(0, 500)
            total_energy = spectrum.get_band_energy(0, 5000)
            nasal_vals[i] = low_freq_energy / total_energy if total_energy > 0 else np.nan

            band_low = spectrum.get_band_energy(0, 500)
            band_high = spectrum.get_band_energy(2000, 4000)
            if band_low > 0 and band_high > 0:
                tilt_vals[i] = 10 * np.log10(band_low) - 10 * np.log10(band_high)
            else:
                tilt_vals[i] = np.nan
        except Exception:
            # Segment too short or other error
            pass

    if progress_callback:
        progress_callback(1.0)

    return AcousticFeatures(
        times=times,
        f0=f0_vals,
        intensity=intensity_vals,
        hnr=hnr_vals,
        f1=f1_vals,
        f2=f2_vals,
        f3=f3_vals,
        f4=f4_vals,
        b1=b1_vals,
        b2=b2_vals,
        b3=b3_vals,
        b4=b4_vals,
        cog=cog_vals,
        spectral_std=std_vals,
        skewness=skew_vals,
        kurtosis=kurt_vals,
        nasal_murmur_ratio=nasal_vals,
        spectral_tilt=tilt_vals,
    )


def extract_features_fast(
    audio_path: str | Path,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    max_formant: float = 5500.0,
) -> AcousticFeatures:
    """
    Fast feature extraction using bulk array methods.

    This version is much faster than extract_features() because it uses
    praatfan-core's bulk array methods instead of querying each time point.
    However, it doesn't compute per-frame spectral moments.

    Args:
        audio_path: Path to audio file
        time_step: Analysis time step in seconds
        pitch_floor: Minimum F0 (Hz)
        pitch_ceiling: Maximum F0 (Hz)
        max_formant: Maximum formant frequency (Hz)

    Returns:
        AcousticFeatures (spectral moments will be NaN)
    """
    # Load audio
    snd = Sound.from_file(str(audio_path))

    # Create analysis objects
    pitch = snd.to_pitch(time_step, pitch_floor, pitch_ceiling)
    intensity = snd.to_intensity(pitch_floor, time_step)
    formants = snd.to_formant_burg(time_step, 5, max_formant, 0.025, 50.0)
    harmonicity = snd.to_harmonicity_cc(time_step, pitch_floor, 0.1, 1.0)

    # Get bulk arrays (much faster than per-frame queries)
    times = pitch.times()
    f0_vals = pitch.values()  # NaN for unvoiced

    # Intensity and HNR arrays (need to interpolate to pitch times)
    n_frames = len(times)
    intensity_vals = np.array([
        intensity.get_value_at_time(t, "cubic") or np.nan
        for t in times
    ])
    hnr_vals = np.array([
        harmonicity.get_value_at_time(t, "linear") or np.nan
        for t in times
    ])

    # Formant arrays
    f1_vals = formants.formant_values(1)
    f2_vals = formants.formant_values(2)
    f3_vals = formants.formant_values(3)
    f4_vals = formants.formant_values(4)
    b1_vals = formants.bandwidth_values(1)
    b2_vals = formants.bandwidth_values(2)
    b3_vals = formants.bandwidth_values(3)
    b4_vals = formants.bandwidth_values(4)

    # Spectral moments not computed in fast mode
    empty = np.full(n_frames, np.nan)

    return AcousticFeatures(
        times=times,
        f0=f0_vals,
        intensity=intensity_vals,
        hnr=hnr_vals,
        f1=f1_vals,
        f2=f2_vals,
        f3=f3_vals,
        f4=f4_vals,
        b1=b1_vals,
        b2=b2_vals,
        b3=b3_vals,
        b4=b4_vals,
        cog=empty.copy(),
        spectral_std=empty.copy(),
        skewness=empty.copy(),
        kurtosis=empty.copy(),
        nasal_murmur_ratio=empty.copy(),
        spectral_tilt=empty.copy(),
    )


def save_csv(features: AcousticFeatures, output_path: str | Path) -> None:
    """Save features to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'time', 'f0', 'intensity', 'hnr',
            'F1', 'F2', 'F3', 'F4',
            'B1', 'B2', 'B3', 'B4',
            'cog', 'spectral_std', 'skewness', 'kurtosis',
            'nasal_murmur_ratio', 'spectral_tilt'
        ])

        for i in range(len(features.times)):
            def fmt(v):
                return '' if np.isnan(v) else f'{v:.4f}'

            writer.writerow([
                f'{features.times[i]:.4f}',
                fmt(features.f0[i]),
                fmt(features.intensity[i]),
                fmt(features.hnr[i]),
                fmt(features.f1[i]),
                fmt(features.f2[i]),
                fmt(features.f3[i]),
                fmt(features.f4[i]),
                fmt(features.b1[i]),
                fmt(features.b2[i]),
                fmt(features.b3[i]),
                fmt(features.b4[i]),
                fmt(features.cog[i]),
                fmt(features.spectral_std[i]),
                fmt(features.skewness[i]),
                fmt(features.kurtosis[i]),
                fmt(features.nasal_murmur_ratio[i]),
                fmt(features.spectral_tilt[i]),
            ])


def main():
    parser = argparse.ArgumentParser(
        description='Extract acoustic features from audio files using praatfan-core'
    )
    parser.add_argument('audio', help='Path to audio file')
    parser.add_argument('--output', '-o', help='Output file path (CSV or JSON)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode (skip per-frame spectral moments)')
    parser.add_argument('--time-step', type=float, default=0.01,
                        help='Analysis time step in seconds (default: 0.01)')
    parser.add_argument('--pitch-floor', type=float, default=75.0,
                        help='Minimum pitch in Hz (default: 75)')
    parser.add_argument('--pitch-ceiling', type=float, default=600.0,
                        help='Maximum pitch in Hz (default: 600)')
    parser.add_argument('--max-formant', type=float, default=5500.0,
                        help='Maximum formant frequency in Hz (default: 5500)')
    args = parser.parse_args()

    # Check input file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Extract features
    print(f"Extracting features from {audio_path}...", file=sys.stderr)

    if args.fast:
        features = extract_features_fast(
            audio_path,
            time_step=args.time_step,
            pitch_floor=args.pitch_floor,
            pitch_ceiling=args.pitch_ceiling,
            max_formant=args.max_formant,
        )
    else:
        def progress(p):
            print(f"\rProgress: {p*100:.1f}%", end='', file=sys.stderr)

        features = extract_features(
            audio_path,
            time_step=args.time_step,
            pitch_floor=args.pitch_floor,
            pitch_ceiling=args.pitch_ceiling,
            max_formant=args.max_formant,
            progress_callback=progress,
        )
        print(file=sys.stderr)  # Newline after progress

    print(f"Extracted {len(features.times)} frames", file=sys.stderr)

    # Output
    if args.json or (args.output and args.output.endswith('.json')):
        output = json.dumps(features.to_dict(), indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(output)
    elif args.output:
        save_csv(features, args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        # Print summary to stdout
        print(f"\nAcoustic Features Summary:")
        print(f"  Duration: {features.times[-1]:.3f}s")
        print(f"  Frames: {len(features.times)}")

        voiced = ~np.isnan(features.f0)
        if np.any(voiced):
            print(f"\n  Pitch (F0):")
            print(f"    Mean: {np.nanmean(features.f0):.1f} Hz")
            print(f"    Range: {np.nanmin(features.f0):.1f} - {np.nanmax(features.f0):.1f} Hz")
            print(f"    Voiced frames: {np.sum(voiced)} ({100*np.mean(voiced):.1f}%)")

        print(f"\n  Intensity:")
        print(f"    Mean: {np.nanmean(features.intensity):.1f} dB")
        print(f"    Range: {np.nanmin(features.intensity):.1f} - {np.nanmax(features.intensity):.1f} dB")

        print(f"\n  HNR:")
        print(f"    Mean: {np.nanmean(features.hnr):.1f} dB")

        print(f"\n  Formants (mean values):")
        print(f"    F1: {np.nanmean(features.f1):.0f} Hz")
        print(f"    F2: {np.nanmean(features.f2):.0f} Hz")
        print(f"    F3: {np.nanmean(features.f3):.0f} Hz")
        print(f"    F4: {np.nanmean(features.f4):.0f} Hz")

        if not np.all(np.isnan(features.cog)):
            print(f"\n  Spectral moments:")
            print(f"    CoG: {np.nanmean(features.cog):.0f} Hz")
            print(f"    Std: {np.nanmean(features.spectral_std):.0f} Hz")


if __name__ == '__main__':
    main()
