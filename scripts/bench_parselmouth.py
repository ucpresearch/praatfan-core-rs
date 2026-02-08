#!/usr/bin/env python3
"""Benchmark parselmouth on a full audio file — equivalent to praatextract02_praatfan.py.

Usage: python scripts/bench_parselmouth.py ~/Downloads/tomsdiner-mono.flac
"""

import time
import sys
import numpy as np
import parselmouth
from parselmouth.praat import call as pm_call

wav_path = sys.argv[1]

# Load audio
t0 = time.time()
snd = parselmouth.Sound(wav_path)
all_samples = snd.values[0]
sr = snd.sampling_frequency
total_duration = snd.duration
t_load = time.time() - t0
print(f"Audio: {wav_path}")
print(f"Duration: {total_duration:.1f}s  SR: {sr:.0f} Hz  Samples: {len(all_samples)}")
print(f"Load: {t_load:.2f}s")

chunk = snd

# --- Bulk analysis (timed individually) ---

t0 = time.time()
pitch = pm_call(chunk, "To Pitch", 0.01, 75, 600)
t_pitch = time.time() - t0
print(f"Pitch AC:        {t_pitch:.2f}s")

t0 = time.time()
intensity = pm_call(chunk, "To Intensity", 75, 0.01, True)
t_intensity = time.time() - t0
print(f"Intensity:       {t_intensity:.2f}s")

t0 = time.time()
formants = pm_call(chunk, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)
t_formant = time.time() - t0
print(f"Formant Burg:    {t_formant:.2f}s")

t0 = time.time()
harmonicity = pm_call(chunk, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
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
    f0 = pm_call(pitch, "Get value at time", float(t), "Hertz", "Linear")
    intens = pm_call(intensity, "Get value at time", float(t), "cubic")
    hnr_val = pm_call(harmonicity, "Get value at time", float(t), "cubic")

    f1 = pm_call(formants, "Get value at time", 1, float(t), "Hertz", "Linear")
    f2 = pm_call(formants, "Get value at time", 2, float(t), "Hertz", "Linear")
    f3 = pm_call(formants, "Get value at time", 3, float(t), "Hertz", "Linear")
    f4 = pm_call(formants, "Get value at time", 4, float(t), "Hertz", "Linear")
    bw1 = pm_call(formants, "Get bandwidth at time", 1, float(t), "Hertz", "Linear")
    bw2 = pm_call(formants, "Get bandwidth at time", 2, float(t), "Hertz", "Linear")
    bw3 = pm_call(formants, "Get bandwidth at time", 3, float(t), "Hertz", "Linear")

    # CoG calculation
    window_duration = 0.025
    t_start = max(0, t - window_duration / 2)
    t_end_seg = min(total_duration, t + window_duration / 2)
    segment = pm_call(chunk, "Extract part", float(t_start), float(t_end_seg),
                       "rectangular", 1.0, True)
    spectrum = pm_call(segment, "To Spectrum", True)
    cog = pm_call(spectrum, "Get centre of gravity", 2)
    std_dev = pm_call(spectrum, "Get standard deviation", 2)
    skewness = pm_call(spectrum, "Get skewness", 2)
    kurtosis = pm_call(spectrum, "Get kurtosis", 2)

    low_freq_energy = pm_call(spectrum, "Get band energy", 0, 500)
    total_energy = pm_call(spectrum, "Get band energy", 0, 5000)

t_loop = time.time() - t0
print(f"Per-frame loop:  {t_loop:.2f}s  ({n_frames} frames)")

t_total = t_bulk + t_loop + t_load
print(f"\nTOTAL:           {t_total:.2f}s")
