#!/usr/bin/env python3
"""Comprehensive accuracy comparison: praatfan-core-rs vs Praat (parselmouth).

Runs all acoustic measures on a given audio file and reports error distributions
with mean, median, 95th percentile, 99th percentile, and max errors.

Usage:
    python scripts/full_comparison.py path/to/audio.flac
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call as pm_call


PROJ = Path(__file__).parent.parent
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/one_two_three_four_five.wav"


def run_rust(example, args):
    """Run a Rust example binary and return parsed JSON."""
    binary = PROJ / "target" / "release" / "examples" / example
    result = subprocess.run(
        [str(binary)] + [str(a) for a in args],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"{example} failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


def stats(errors, label, unit):
    """Print distribution statistics for a list of absolute errors."""
    if not errors:
        print(f"  {label}: no data")
        return {}
    s = sorted(errors)
    n = len(s)
    mean = np.mean(s)
    median = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    p99 = s[min(int(n * 0.99), n - 1)]
    mx = s[-1]
    within_tol = {
        "Hz": [(0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (5.0, "5")],
        "dB": [(0.001, "0.001"), (0.01, "0.01"), (0.1, "0.1"), (0.5, "0.5"), (1.0, "1"), (5.0, "5")],
    }.get(unit, [(0.01, "0.01"), (0.1, "0.1"), (1.0, "1")])

    print(f"  {label} ({n} frames):")
    print(f"    Mean={mean:.6f}  Median={median:.6f}  P95={p95:.6f}  P99={p99:.6f}  Max={mx:.6f} {unit}")
    for tol, tol_s in within_tol:
        cnt = sum(1 for e in s if e < tol)
        print(f"    Within {tol_s:>5s} {unit}: {cnt}/{n} ({100*cnt/n:.1f}%)")

    return {"n": n, "mean": mean, "median": median, "p95": p95, "p99": p99, "max": mx}


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Load audio
# ============================================================
print(f"Audio: {AUDIO}")
pm_snd = parselmouth.Sound(AUDIO)
print(f"Duration: {pm_snd.duration:.3f}s  Sample rate: {pm_snd.sampling_frequency:.0f} Hz  Channels: {pm_snd.n_channels}")
print(f"Samples: {pm_snd.n_samples}")

all_results = {}
t_total = time.time()

# ============================================================
section("1. PITCH (AC)")
# ============================================================
t0 = time.time()
rust_pitch = run_rust("pitch_json", [AUDIO, "0", "75", "600"])
t_rust = time.time() - t0

t0 = time.time()
pm_pitch = pm_call(pm_snd, "To Pitch", 0, 75, 600)
pm_nf = pm_call(pm_pitch, "Get number of frames")
t_praat = time.time() - t0

rust_frames = rust_pitch["frames"]
print(f"  Frames: Rust={len(rust_frames)}  Praat={pm_nf}  (Rust {t_rust:.1f}s, Praat {t_praat:.1f}s)")

# Vectorized pitch extraction
pm_f0_all = np.array([pm_call(pm_pitch, "Get value in frame", i + 1, "Hertz") for i in range(pm_nf)])

n = min(len(rust_frames), pm_nf)
pitch_errs = []
voicing_mm = 0
for i in range(n):
    rf = rust_frames[i]
    pm_f0 = pm_f0_all[i]
    r_voiced = rf["voiced"]
    p_voiced = not np.isnan(pm_f0) and pm_f0 > 0
    if r_voiced != p_voiced:
        voicing_mm += 1
    elif r_voiced and p_voiced:
        pitch_errs.append(abs(rf["frequency"] - pm_f0))

print(f"  Voicing mismatches: {voicing_mm}/{n}")
all_results["Pitch F0"] = stats(pitch_errs, "F0 error", "Hz")
all_results["Pitch F0"]["vmm"] = voicing_mm
all_results["Pitch F0"]["unit"] = "Hz"

# ============================================================
section("2. FORMANT (Burg)")
# ============================================================
t0 = time.time()
rust_fmt = run_rust("formant_json", [AUDIO, "0.005", "5", "5500", "0.025", "50"])
t_rust = time.time() - t0

t0 = time.time()
pm_fmt = pm_call(pm_snd, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)
pm_fmt_nf = pm_call(pm_fmt, "Get number of frames")
t_praat = time.time() - t0

rust_fmt_data = rust_fmt["formant"]
rust_fmt_times = rust_fmt_data["times"]
rust_nf = len(rust_fmt_times)
print(f"  Frames: Rust={rust_nf}  Praat={pm_fmt_nf}  (Rust {t_rust:.1f}s, Praat {t_praat:.1f}s)")

# Extract all Praat formant values at once using To Matrix
for fn in [1, 2, 3]:
    pm_mat = pm_call(pm_fmt, "To Matrix", fn)
    pm_fn_all = pm_mat.values[0]  # shape (n_frames,)

    rust_fn_vals = rust_fmt_data[f"f{fn}"]
    nf = min(len(rust_fn_vals), len(pm_fn_all))

    fmt_errs = []
    for i in range(nf):
        r_val = rust_fn_vals[i]
        pm_val = pm_fn_all[i]
        if r_val is not None and not np.isnan(r_val) and r_val > 0 and \
           not np.isnan(pm_val) and pm_val > 0:
            fmt_errs.append(abs(r_val - pm_val))

    r = stats(fmt_errs, f"F{fn} error", "Hz")
    r["unit"] = "Hz"
    all_results[f"Formant F{fn}"] = r

# ============================================================
section("3. HARMONICITY (CC)")
# ============================================================
t0 = time.time()
rust_hcc = run_rust("harmonicity_json", [AUDIO, "0.01", "75", "0.1", "1.0", "cc"])
t_rust = time.time() - t0

t0 = time.time()
pm_hcc = pm_call(pm_snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
pm_hcc_nf = pm_hcc.nx
t_praat = time.time() - t0

# Vectorized: use .values array
pm_hcc_all = pm_hcc.values[0]  # shape (n_frames,)

rust_hcc_vals = rust_hcc["values"]
print(f"  Frames: Rust={len(rust_hcc_vals)}  Praat={pm_hcc_nf}  (Rust {t_rust:.1f}s, Praat {t_praat:.1f}s)")

hcc_errs = []
hcc_vmm = 0
nf = min(len(rust_hcc_vals), pm_hcc_nf)
for i in range(nf):
    rv = rust_hcc_vals[i]
    pm_v = pm_hcc_all[i]
    r_ok = rv["voiced"]
    p_ok = not np.isnan(pm_v) and pm_v > -200
    if r_ok != p_ok:
        hcc_vmm += 1
    elif r_ok and p_ok:
        hcc_errs.append(abs(rv["hnr"] - pm_v))

print(f"  Voicing mismatches: {hcc_vmm}/{nf}")
r = stats(hcc_errs, "HNR error (CC)", "dB")
r["vmm"] = hcc_vmm
r["unit"] = "dB"
all_results["Harmonicity CC"] = r

# ============================================================
section("4. HARMONICITY (AC)")
# ============================================================
t0 = time.time()
rust_hac = run_rust("harmonicity_json", [AUDIO, "0.01", "75", "0.1", "4.5", "ac"])
t_rust = time.time() - t0

t0 = time.time()
pm_hac = pm_call(pm_snd, "To Harmonicity (ac)", 0.01, 75, 0.1, 4.5)
pm_hac_nf = pm_hac.nx
t_praat = time.time() - t0

# Vectorized
pm_hac_all = pm_hac.values[0]

rust_hac_vals = rust_hac["values"]
print(f"  Frames: Rust={len(rust_hac_vals)}  Praat={pm_hac_nf}  (Rust {t_rust:.1f}s, Praat {t_praat:.1f}s)")

hac_errs = []
hac_vmm = 0
nf = min(len(rust_hac_vals), pm_hac_nf)
for i in range(nf):
    rv = rust_hac_vals[i]
    pm_v = pm_hac_all[i]
    r_ok = rv["voiced"]
    p_ok = not np.isnan(pm_v) and pm_v > -200
    if r_ok != p_ok:
        hac_vmm += 1
    elif r_ok and p_ok:
        hac_errs.append(abs(rv["hnr"] - pm_v))

print(f"  Voicing mismatches: {hac_vmm}/{nf}")
r = stats(hac_errs, "HNR error (AC)", "dB")
r["vmm"] = hac_vmm
r["unit"] = "dB"
all_results["Harmonicity AC"] = r

# ============================================================
section("5. INTENSITY")
# ============================================================
t0 = time.time()
rust_int = run_rust("intensity_json", [AUDIO, "100", "0.01"])
t_rust = time.time() - t0

t0 = time.time()
pm_int = pm_call(pm_snd, "To Intensity", 100, 0.01, True)
pm_int_nf = pm_int.nx
t_praat = time.time() - t0

# Vectorized
pm_int_all = pm_int.values[0]

rust_int_data = rust_int["intensity"]
rust_int_vals = rust_int_data["values"]
print(f"  Frames: Rust={len(rust_int_vals)}  Praat={pm_int_nf}  (Rust {t_rust:.1f}s, Praat {t_praat:.1f}s)")

int_errs = []
nf = min(len(rust_int_vals), pm_int_nf)
for i in range(nf):
    rv = rust_int_vals[i]
    pm_v = pm_int_all[i]
    if rv is not None and not np.isnan(rv) and not np.isnan(pm_v):
        int_errs.append(abs(rv - pm_v))

r = stats(int_errs, "Intensity error", "dB")
r["unit"] = "dB"
all_results["Intensity"] = r

# ============================================================
section("SUMMARY TABLE")
# ============================================================
t_elapsed = time.time() - t_total
print()
hdr = f"  {'Measure':20s} {'Frames':>8s} {'Mean':>12s} {'Median':>12s} {'P95':>12s} {'P99':>12s} {'Max':>12s} {'Unit':>4s}"
print(hdr)
print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*4}")

for label in ["Pitch F0", "Formant F1", "Formant F2", "Formant F3",
              "Harmonicity CC", "Harmonicity AC", "Intensity"]:
    r = all_results.get(label, {})
    if not r or "n" not in r:
        print(f"  {label:20s} {'---':>8s}")
        continue
    n_str = str(r["n"])
    if "vmm" in r and r["vmm"] > 0:
        n_str += f"+{r['vmm']}v"
    unit = r.get("unit", "")
    print(f"  {label:20s} {n_str:>8s} {r['mean']:12.6f} {r['median']:12.6f} {r['p95']:12.6f} {r['p99']:12.6f} {r['max']:12.6f} {unit:>4s}")

print()
print(f"Total time: {t_elapsed:.1f}s")
print("vmm = voicing mismatches (frames where one is voiced and the other unvoiced)")
print("Done.")
