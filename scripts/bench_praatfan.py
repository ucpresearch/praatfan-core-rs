#!/usr/bin/env python3
"""Benchmark praatfan_gpl on a full audio file — equivalent to praatextract02_praatfan.py.

Usage: python scripts/bench_praatfan.py ~/Downloads/tomsdiner-mono.flac
"""

import time
import sys
import numpy as np
from praatfan_gpl import Sound

wav_path = sys.argv[1]

# Load audio
t0 = time.time()
snd = Sound.from_file(wav_path)
all_samples = np.array(snd.samples())
sr = snd.sample_rate
total_duration = snd.duration
t_load = time.time() - t0
print(f"Audio: {wav_path}")
print(f"Duration: {total_duration:.1f}s  SR: {sr:.0f} Hz  Samples: {len(all_samples)}")
print(f"Load: {t_load:.2f}s")

# Use the full file as a single chunk (no chunking overhead)
chunk = snd

# --- Bulk analysis (timed individually) ---

t0 = time.time()
pitch = chunk.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
t_pitch = time.time() - t0
print(f"Pitch AC:        {t_pitch:.2f}s")

t0 = time.time()
intensity = chunk.to_intensity(min_pitch=75, time_step=0.01)
t_intensity = time.time() - t0
print(f"Intensity:       {t_intensity:.2f}s")

t0 = time.time()
formants = chunk.to_formant_burg(
    time_step=0.005, max_num_formants=5,
    max_formant_hz=5500, window_length=0.025, pre_emphasis_from=50)
t_formant = time.time() - t0
print(f"Formant Burg:    {t_formant:.2f}s")

t0 = time.time()
harmonicity = chunk.to_harmonicity_cc(
    time_step=0.01, min_pitch=75,
    silence_threshold=0.1, periods_per_window=1.0)
t_hnr = time.time() - t0
print(f"Harmonicity CC:  {t_hnr:.2f}s")

t_bulk = t_pitch + t_intensity + t_formant + t_hnr
print(f"Bulk total:      {t_bulk:.2f}s")

# --- Per-frame loop (spectrum queries) ---

output_step = 0.01
times = np.arange(0, total_duration, output_step)
n_frames = len(times)

t0 = time.time()
for t in times:
    f0 = pitch.get_value_at_time(t, "Hertz", "linear")
    intens = intensity.get_value_at_time(t, "cubic")
    hnr = harmonicity.get_value_at_time(t, "cubic")

    f1 = formants.get_value_at_time(1, t, "Hertz", "linear")
    f2 = formants.get_value_at_time(2, t, "Hertz", "linear")
    f3 = formants.get_value_at_time(3, t, "Hertz", "linear")
    f4 = formants.get_value_at_time(4, t, "Hertz", "linear")
    bw1 = formants.get_bandwidth_at_time(1, t, "Hertz", "linear")
    bw2 = formants.get_bandwidth_at_time(2, t, "Hertz", "linear")
    bw3 = formants.get_bandwidth_at_time(3, t, "Hertz", "linear")

    # CoG calculation
    window_duration = 0.025
    t_start = max(0, t - window_duration / 2)
    t_end_seg = min(total_duration, t + window_duration / 2)
    i_s = max(0, int(round(t_start * sr)))
    i_e = min(len(all_samples), int(round(t_end_seg * sr)))
    segment = Sound(all_samples[i_s:i_e].copy(), sample_rate=sr)
    spectrum = segment.to_spectrum(fast=True)
    cog = spectrum.get_center_of_gravity(2)
    std_dev = spectrum.get_standard_deviation(2)
    skewness = spectrum.get_skewness(2)
    kurtosis = spectrum.get_kurtosis(2)

    low_freq_energy = spectrum.get_band_energy(0, 500)
    total_energy = spectrum.get_band_energy(0, 5000)

t_loop = time.time() - t0
print(f"Per-frame loop:  {t_loop:.2f}s  ({n_frames} frames)")

t_total = t_bulk + t_loop + t_load
print(f"\nTOTAL:           {t_total:.2f}s")
