#!/usr/bin/env python3
"""
Acoustic analysis script using praatfan_core.

Extracts pitch, intensity, formants, harmonicity, and spectral properties
from an audio file and outputs as TSV or JSON.

Usage:
    python analyze.py audio.wav                    # Output TSV to stdout
    python analyze.py audio.wav -o results.tsv    # Save to file
    python analyze.py audio.wav --json            # Output JSON
    python analyze.py audio.wav --json -o out.json
"""

import argparse
import json
import sys
from praatfan_core import Sound


def analyze(audio_path, time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0,
            max_formant=5500.0):
    """Run acoustic analysis on audio file."""
    sound = Sound.from_file(audio_path)

    # Compute analyses
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor,
                          pitch_ceiling=pitch_ceiling)
    intensity = sound.to_intensity(min_pitch=pitch_floor, time_step=time_step)
    formant = sound.to_formant_burg(time_step=time_step, max_num_formants=5,
                                    max_formant_hz=max_formant,
                                    window_length=0.025, pre_emphasis_from=50.0)
    harmonicity = sound.to_harmonicity_cc(time_step=time_step,
                                          min_pitch=pitch_floor,
                                          silence_threshold=0.1,
                                          periods_per_window=1.0)
    spectrum = sound.to_spectrum(fast=True)

    # Get time-aligned values
    times = pitch.times()
    f0_values = pitch.values()

    def safe_round(val, decimals):
        """Round value, handling None and NaN."""
        if val is None:
            return None
        if val != val:  # NaN check
            return None
        return round(val, decimals)

    # Window duration for per-frame CoG computation
    window_duration = 0.025
    duration = sound.duration

    frames = []
    for i, t in enumerate(times):
        f0 = f0_values[i]

        # Compute per-frame CoG by extracting a short segment
        cog = None
        t_start = max(0, t - window_duration / 2)
        t_end = min(duration, t + window_duration / 2)
        if t_end - t_start >= 0.01:  # Need minimum segment length
            try:
                segment = sound.extract_part(t_start, t_end, 'hanning', 1.0, False)
                seg_spectrum = segment.to_spectrum(fast=True)
                cog = seg_spectrum.get_center_of_gravity(2.0)
            except Exception:
                pass

        frame = {
            'time': round(t, 4),
            'f0': safe_round(f0, 2),
            'intensity': safe_round(intensity.get_value_at_time(t, 'cubic'), 2),
            'hnr': safe_round(harmonicity.get_value_at_time(t, 'linear'), 2),
            'F1': safe_round(formant.get_value_at_time(1, t, 'hertz', 'linear'), 1),
            'F2': safe_round(formant.get_value_at_time(2, t, 'hertz', 'linear'), 1),
            'F3': safe_round(formant.get_value_at_time(3, t, 'hertz', 'linear'), 1),
            'B1': safe_round(formant.get_bandwidth_at_time(1, t, 'hertz', 'linear'), 1),
            'B2': safe_round(formant.get_bandwidth_at_time(2, t, 'hertz', 'linear'), 1),
            'B3': safe_round(formant.get_bandwidth_at_time(3, t, 'hertz', 'linear'), 1),
            'CoG': safe_round(cog, 1),
        }
        frames.append(frame)

    return {
        'filename': audio_path,
        'duration': round(sound.duration, 3),
        'sample_rate': sound.sample_rate,
        'global_cog': round(spectrum.get_center_of_gravity(2.0), 1),
        'params': {
            'time_step': time_step,
            'pitch_floor': pitch_floor,
            'pitch_ceiling': pitch_ceiling,
            'max_formant': max_formant,
        },
        'frames': frames,
    }


def to_tsv(data):
    """Convert analysis data to TSV string."""
    headers = ['time', 'f0', 'intensity', 'hnr', 'F1', 'F2', 'F3', 'B1', 'B2', 'B3', 'CoG']
    lines = ['\t'.join(headers)]

    for f in data['frames']:
        row = [
            str(f['time']),
            str(f['f0']) if f['f0'] is not None else '',
            str(f['intensity']) if f['intensity'] is not None else '',
            str(f['hnr']) if f['hnr'] is not None else '',
            str(f['F1']) if f['F1'] is not None else '',
            str(f['F2']) if f['F2'] is not None else '',
            str(f['F3']) if f['F3'] is not None else '',
            str(f['B1']) if f['B1'] is not None else '',
            str(f['B2']) if f['B2'] is not None else '',
            str(f['B3']) if f['B3'] is not None else '',
            str(f['CoG']) if f['CoG'] is not None else '',
        ]
        lines.append('\t'.join(row))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract acoustic features from an audio file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python analyze.py speech.wav                  # TSV to stdout
    python analyze.py speech.wav -o features.tsv # Save TSV
    python analyze.py speech.wav --json          # JSON to stdout
    python analyze.py speech.wav --json -o f.json
        '''
    )
    parser.add_argument('audio', help='Path to audio file (WAV, FLAC, MP3)')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of TSV')
    parser.add_argument('--time-step', type=float, default=0.01,
                        help='Time step between frames (default: 0.01)')
    parser.add_argument('--pitch-floor', type=float, default=75.0,
                        help='Minimum pitch in Hz (default: 75)')
    parser.add_argument('--pitch-ceiling', type=float, default=600.0,
                        help='Maximum pitch in Hz (default: 600)')
    parser.add_argument('--max-formant', type=float, default=5500.0,
                        help='Maximum formant frequency in Hz (default: 5500)')

    args = parser.parse_args()

    # Run analysis
    data = analyze(
        args.audio,
        time_step=args.time_step,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        max_formant=args.max_formant,
    )

    # Format output
    if args.json:
        output = json.dumps(data, indent=2)
    else:
        output = to_tsv(data)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Wrote {len(data['frames'])} frames to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
