#!/usr/bin/env python3
"""
Verify formant extraction matches Praat exactly.

This script demonstrates the correct algorithm for Praat-compatible formant extraction.
Key insight: the window_length parameter (0.025) is actually halfdt_window,
so the actual window is 0.05 seconds = 550 samples at 11000 Hz.

Usage:
    source ~/local/scr/commonpip/bin/activate
    python scripts/verify_formant_match.py
"""
import parselmouth
from parselmouth.praat import call
import numpy as np
import math

sound = parselmouth.Sound("tests/fixtures/one_two_three_four_five.wav")

# Get Praat's formant
formant_direct = call(sound, "To Formant (burg)", 0.01, 5, 5500.0, 0.025, 50.0)
praat_x1 = call(formant_direct, "Get time from frame number", 1)
praat_nframes = call(formant_direct, "Get number of frames")

print(f"Praat: {praat_nframes} frames, x1={praat_x1}")

# Resample and pre-emphasize
resampled = call(sound, "Resample", 11000.0, 50)
samples_raw = resampled.values[0].copy()

cutoff = 50.0
sample_rate = 11000.0
dx = 1.0 / sample_rate
emphasisFactor = math.exp(-2 * math.pi * cutoff * dx)

samples = samples_raw.copy()
for i in range(len(samples) - 1, 0, -1):
    samples[i] -= emphasisFactor * samples[i - 1]

nyquist = sample_rate / 2.0

# CORRECT parameters:
# The window_length parameter (0.025) is actually halfdt_window!
halfdt_window = 0.025
dt_window = 2.0 * halfdt_window  # 0.05 seconds
nsamp_window = int(math.floor(dt_window / dx))  # 550 samples!
halfnsamp_window = nsamp_window // 2  # 275

print(f"\nCorrected window parameters:")
print(f"  dt_window = {dt_window}")
print(f"  nsamp_window = {nsamp_window}")
print(f"  halfnsamp_window = {halfnsamp_window}")

# Create Gaussian window for nsamp_window samples
imid = 0.5 * (nsamp_window + 1)  # 275.5
edge = math.exp(-12.0)
window = np.zeros(nsamp_window)
for i in range(1, nsamp_window + 1):
    window[i-1] = (math.exp(-48.0 * (i - imid)**2 / (nsamp_window + 1)**2) - edge) / (1.0 - edge)

print(f"  Window peak at index {np.argmax(window)} (expected {int(imid-1)})")

x1_sound = resampled.x1

def sampled_xToLowIndex_0based(x):
    return int(math.floor((x - x1_sound) / dx))

def burg_praat_style(samples, order):
    """Praat's VECburg algorithm from dwsys/NUM2.cpp"""
    n = len(samples)
    m = order
    a = np.zeros(m)
    b1 = np.zeros(n)
    b2 = np.zeros(n)
    b1[0] = samples[0]
    b2[n-2] = samples[n-1]
    for j in range(1, n-1):
        b1[j] = samples[j]
        b2[j-1] = samples[j]
    p = np.sum(samples ** 2)
    xms = p / n
    aa = np.zeros(m)
    for i in range(m):
        num = den = 0.0
        for j in range(n - i - 1):
            num += b1[j] * b2[j]
            den += b1[j] * b1[j] + b2[j] * b2[j]
        if den <= 0:
            return a, 0.0
        a[i] = 2.0 * num / den
        xms *= 1.0 - a[i] * a[i]
        for j in range(i):
            a[j] = aa[j] - a[i] * aa[i - j - 1]
        if i < m - 1:
            for j in range(i + 1):
                aa[j] = a[j]
            for j in range(n - i - 2):
                b1[j] -= aa[i] * b2[j]
                b2[j] = b2[j + 1] - aa[i] * b1[j + 1]
    return a, xms

# Test frame 20
t = praat_x1 + 19 * 0.01

f1_p = call(formant_direct, "Get value at time", 1, t, "Hertz", "Linear")
f2_p = call(formant_direct, "Get value at time", 2, t, "Hertz", "Linear")

print(f"\nFrame 20 at t={t:.4f}")
print(f"Praat: F1={f1_p:.2f}, F2={f2_p:.2f}")

# Extract samples with corrected window
leftSample = sampled_xToLowIndex_0based(t)
rightSample = leftSample + 1
startSample = rightSample - halfnsamp_window  # 275 samples to the left
endSample = leftSample + halfnsamp_window      # 275 samples to the right

print(f"\nSample extraction:")
print(f"  leftSample={leftSample}, rightSample={rightSample}")
print(f"  startSample={startSample}, endSample={endSample}")
print(f"  Expected length = {endSample - startSample + 1} (should be ~{nsamp_window})")

raw_frame = samples[startSample:endSample + 1]
actualFrameLength = len(raw_frame)
print(f"  Actual length = {actualFrameLength}")

# Apply window (only first actualFrameLength elements)
windowed = raw_frame * window[:actualFrameLength]

# Run Burg
coeffs, _ = burg_praat_style(windowed, 10)

# Find roots
poly = np.zeros(11)
poly[0] = 1.0
for i in range(10):
    poly[i + 1] = -coeffs[i]

roots = np.roots(poly)

# Extract formants
formants = []
for root in roots:
    if root.imag >= 0:
        freq = abs(np.angle(root)) * nyquist / np.pi
        if 50 <= freq <= nyquist - 50:
            bw = -np.log(abs(root)) * sample_rate / np.pi
            formants.append((freq, bw))
formants.sort(key=lambda x: x[0])

print(f"\nOur implementation: F1={formants[0][0]:.2f}, F2={formants[1][0]:.2f}")
print(f"Difference: F1={formants[0][0]-f1_p:+.2f}, F2={formants[1][0]-f2_p:+.2f}")

# Test multiple frames
print(f"\n\n=== Testing multiple frames ===")
print(f"{'Frame':<8} {'Time':<12} {'Praat F1':<12} {'Our F1':<12} {'Diff F1':<12} {'Praat F2':<12} {'Our F2':<12} {'Diff F2':<12}")
print("-" * 104)

total_f1_err = 0
total_f2_err = 0
n_valid = 0

for frame_idx in range(1, praat_nframes + 1):
    t = praat_x1 + (frame_idx - 1) * 0.01

    f1_p = call(formant_direct, "Get value at time", 1, t, "Hertz", "Linear")
    f2_p = call(formant_direct, "Get value at time", 2, t, "Hertz", "Linear")

    if f1_p is None or f2_p is None:
        continue

    leftSample = sampled_xToLowIndex_0based(t)
    rightSample = leftSample + 1
    startSample = rightSample - halfnsamp_window
    endSample = leftSample + halfnsamp_window

    if startSample < 0 or endSample >= len(samples):
        continue

    raw_frame = samples[startSample:endSample + 1]
    actualFrameLength = len(raw_frame)
    windowed = raw_frame * window[:actualFrameLength]

    coeffs, _ = burg_praat_style(windowed, 10)
    poly = np.zeros(11)
    poly[0] = 1.0
    for i in range(10):
        poly[i + 1] = -coeffs[i]

    roots = np.roots(poly)
    formants = []
    for root in roots:
        if root.imag >= 0:
            freq = abs(np.angle(root)) * nyquist / np.pi
            if 50 <= freq <= nyquist - 50:
                formants.append(freq)
    formants.sort()

    if len(formants) >= 2:
        f1_diff = formants[0] - f1_p
        f2_diff = formants[1] - f2_p
        total_f1_err += abs(f1_diff)
        total_f2_err += abs(f2_diff)
        n_valid += 1

        if frame_idx <= 5 or frame_idx % 20 == 0 or frame_idx >= praat_nframes - 3:
            print(f"{frame_idx:<8} {t:<12.4f} {f1_p:<12.1f} {formants[0]:<12.1f} {f1_diff:<+12.1f} {f2_p:<12.1f} {formants[1]:<12.1f} {f2_diff:<+12.1f}")

print("-" * 104)
if n_valid > 0:
    print(f"\nMean absolute error over {n_valid} voiced frames:")
    print(f"  F1: {total_f1_err / n_valid:.2f} Hz")
    print(f"  F2: {total_f2_err / n_valid:.2f} Hz")
