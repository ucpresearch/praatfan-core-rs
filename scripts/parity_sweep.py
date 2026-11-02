#!/usr/bin/env python3
"""Parameter-sweep parity check: praatfan_gpl (Python bindings) vs parselmouth.

The per-module compare_*.py scripts test default parameters on the speech
fixtures. This one sweeps signals (sample rates, short/silent/noisy/tonal
sounds) x parameters x analyses, feeding the *same* float64 samples to both
engines, and prints one line per case that is not an exact match.

Usage:
    python scripts/parity_sweep.py [--only pitch,formant,...] [--all] [--tol 1e-6]

Needs a venv with parselmouth 0.4.7 and a dev build of praatfan_gpl.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

import praatfan_gpl as pf

PROJ = Path(__file__).resolve().parent.parent
FIX = PROJ / "tests" / "fixtures" / "one_two_three_four_five.wav"


# ----------------------------------------------------------------- signals

def signals():
    base = parselmouth.Sound(str(FIX))
    out = {}
    x, sr = base.values[0].copy(), base.sampling_frequency
    out["speech24k"] = (x, sr)
    for rate in (8000, 11025, 16000, 44100):
        r = base.resample(rate)
        out[f"speech{rate}"] = (r.values[0].copy(), rate)
    out["short80ms"] = (x[int(0.30 * sr):int(0.38 * sr)].copy(), sr)
    out["short30ms"] = (x[int(0.30 * sr):int(0.33 * sr)].copy(), sr)
    rng = np.random.default_rng(1)
    out["noise16k"] = (0.1 * rng.standard_normal(8000), 16000.0)
    t = np.arange(int(0.6 * 16000)) / 16000.0
    f0 = 140 + 30 * np.sin(2 * np.pi * 3 * t)
    ph = 2 * np.pi * np.cumsum(f0) / 16000.0
    out["harm16k"] = (sum(0.3 / k * np.sin(k * ph) for k in range(1, 12)), 16000.0)
    out["silence16k"] = (np.zeros(4800), 16000.0)
    sil = np.zeros(int(0.2 * sr))
    out["speech+silence"] = (np.concatenate([sil, x, sil]), sr)
    out["clipped"] = (np.clip(4 * x, -0.5, 0.5), sr)
    out["dc_offset"] = (x + 0.2, sr)
    return out


# ----------------------------------------------------------------- compare

class Result:
    def __init__(self, name):
        self.name = name
        self.issues = []
        self.maxerr = 0.0

    def add(self, msg):
        self.issues.append(msg)


def cmp_arrays(res, label, a, b, tol, rel=False):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        res.add(f"{label}: shape praat={a.shape} ours={b.shape}")
        return
    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    if (na != nb).any():
        idx = np.nonzero(na != nb)[0]
        res.add(f"{label}: {len(idx)} undefined mismatches (first idx {idx[:5].tolist()})")
    m = ~(na | nb)
    if m.any():
        d = np.abs(a[m] - b[m])
        if rel:
            d = d / np.maximum(np.abs(a[m]), 1e-300)
        e = float(d.max())
        res.maxerr = max(res.maxerr, e)
        if e > tol:
            i = int(np.argmax(d))
            res.add(f"{label}: max {'rel ' if rel else ''}err {e:.3g} "
                    f"(praat={a[m][i]:.9g} ours={b[m][i]:.9g}); "
                    f"{int((d > tol).sum())}/{m.sum()} > tol")


def cmp_grid(res, label, p_n, p_t1, p_dt, o_n, o_t1, o_dt):
    if p_n != o_n:
        res.add(f"{label}: nframes praat={p_n} ours={o_n}")
    if abs(p_t1 - o_t1) > 1e-9:
        res.add(f"{label}: t1 praat={p_t1:.12g} ours={o_t1:.12g}")
    if abs(p_dt - o_dt) > 1e-12:
        res.add(f"{label}: dt praat={p_dt:.12g} ours={o_dt:.12g}")


def both(res, fp, fo):
    """Run both sides. Returns (p, o) or None if either raised (recorded)."""
    pe = oe = None
    try:
        p = fp()
    except Exception as e:  # noqa: BLE001
        pe = e
    try:
        o = fo()
    except Exception as e:  # noqa: BLE001
        oe = e
    if pe and oe:
        return None  # both refuse: fine
    if pe:
        res.add(f"praat raises ({str(pe).splitlines()[0][:80]}) but ours returns")
        return None
    if oe:
        res.add(f"ours raises ({type(oe).__name__}: {str(oe)[:100]}) but praat returns")
        return None
    return p, o


# ----------------------------------------------------------------- analyses

def pitch_cases():
    for ts, fl, ce in [(0.0, 75, 600), (0.005, 50, 800), (0.01, 100, 300), (0.02, 60, 500)]:
        yield "ac", dict(time_step=ts, floor=fl, ceiling=ce)
        yield "cc", dict(time_step=ts, floor=fl, ceiling=ce)
    yield "ac", dict(time_step=0.01, floor=75, ceiling=600, silence=0.0, voicing=0.3,
                     octave=0.05, jump=0.6, vuv=0.3)
    yield "cc", dict(time_step=0.01, floor=75, ceiling=600, silence=0.1, voicing=0.6,
                     octave=0.0, jump=0.0, vuv=0.0)


def run_pitch(res, x, sr, method, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    kw = dict(silence=0.03, voicing=0.45, octave=0.01, jump=0.35, vuv=0.14)
    kw.update(c)
    ts = c["time_step"] if c["time_step"] > 0 else None
    if method == "ac":
        fp = lambda: ps.to_pitch_ac(time_step=ts, pitch_floor=kw["floor"], pitch_ceiling=kw["ceiling"],
                                    silence_threshold=kw["silence"], voicing_threshold=kw["voicing"],
                                    octave_cost=kw["octave"], octave_jump_cost=kw["jump"],
                                    voiced_unvoiced_cost=kw["vuv"])
        fo = lambda: os_.to_pitch(kw["time_step"], kw["floor"], kw["ceiling"], kw["voicing"],
                                  kw["silence"], kw["octave"], kw["jump"], kw["vuv"])
    else:
        fp = lambda: ps.to_pitch_cc(time_step=ts, pitch_floor=kw["floor"], pitch_ceiling=kw["ceiling"],
                                    very_accurate=True,
                                    silence_threshold=kw["silence"], voicing_threshold=kw["voicing"],
                                    octave_cost=kw["octave"], octave_jump_cost=kw["jump"],
                                    voiced_unvoiced_cost=kw["vuv"])
        fo = lambda: os_.to_pitch_cc(kw["time_step"], kw["floor"], kw["ceiling"], kw["voicing"],
                                     kw["silence"], kw["octave"], kw["jump"], kw["vuv"])
    r = both(res, fp, fo)
    if not r:
        return
    p, o = r
    cmp_grid(res, "grid", p.n_frames, p.x1, p.dx, o.num_frames, o.start_time, o.time_step)
    if p.n_frames != o.num_frames:
        return
    pv = p.selected_array["frequency"].astype(float)
    pv[pv == 0] = np.nan
    ov = np.asarray(o.values(), dtype=float)
    ov[ov == 0] = np.nan
    cmp_arrays(res, "f0", pv, ov, tol)
    # queries: units x interpolation on an off-grid time grid
    times = np.linspace(p.xmin, p.xmax, 97)[1:-1]
    for unit_p, unit_o in [("Hertz", "hertz"), ("semitones re 100 Hz", "semitones_re_100hz"),
                           ("mel", "mel"), ("ERB", "erb")]:
        for ip, io in [("linear", "linear"), ("nearest", "nearest")]:
            a = [call(p, "Get value at time", t, unit_p, ip) for t in times]
            b = [o.get_value_at_time(t, unit_o, io) for t in times]
            b = [np.nan if v is None else v for v in b]
            cmp_arrays(res, f"q[{unit_o},{io}]", a, b, tol)
    d = p.xmax - p.xmin
    ranges = [(0, 0), (p.xmin + 0.13 * d, p.xmin + 0.71 * d), (p.xmin + 0.4 * d, p.xmin + 0.4 * d + 0.3 * p.dx)]
    units = [("Hertz", "hertz"), ("semitones re 100 Hz", "semitones_re_100hz"), ("mel", "mel"), ("ERB", "erb")]
    checks = [("count", call(p, "Count voiced frames"), o.count_voiced_frames())]
    for (a0, a1) in ranges:
        for up, uo in units:
            checks.append((f"mean[{uo},{a0:.2f}]", call(p, "Get mean", a0, a1, up), o.get_mean(a0, a1, uo)))
            # Praat's stdev takes reference-free units ("semitones")
            sd_unit = "semitones" if up.startswith("semitones") else up
            checks.append((f"sd[{uo},{a0:.2f}]", call(p, "Get standard deviation", a0, a1, sd_unit), o.get_standard_deviation(a0, a1, uo)))
            for q in (0.1, 0.5, 0.9):
                checks.append((f"q{q}[{uo},{a0:.2f}]", call(p, "Get quantile", a0, a1, q, up), o.get_quantile(a0, a1, q, uo)))
            for ip in ("None", "Parabolic"):
                checks.append((f"min[{uo},{ip},{a0:.2f}]", call(p, "Get minimum", a0, a1, up, ip), o.get_minimum(a0, a1, uo, ip.lower())))
                checks.append((f"max[{uo},{ip},{a0:.2f}]", call(p, "Get maximum", a0, a1, up, ip), o.get_maximum(a0, a1, uo, ip.lower())))
                checks.append((f"tmin[{uo},{ip},{a0:.2f}]", call(p, "Get time of minimum", a0, a1, up, ip), o.get_time_of_minimum(a0, a1, uo, ip.lower())))
    for name, a, b in checks:
        a = np.nan if a is None else a
        cmp_arrays(res, name, [a], [np.nan if b is None else b], max(tol, 1e-6))


def formant_cases():
    for nf in (5, 4, 5.5, 3):
        yield dict(ts=0.0, nf=nf, ceil=5500, win=0.025, pre=50)
    for ceil in (5000, 8000, 3000):
        yield dict(ts=0.01, nf=5, ceil=ceil, win=0.025, pre=50)
    for win in (0.005, 0.05, 0.1, 0.2):
        yield dict(ts=0.01, nf=5, ceil=5500, win=win, pre=50)
    yield dict(ts=0.01, nf=5, ceil=5500, win=0.025, pre=0)
    yield dict(ts=0.01, nf=5, ceil=5500, win=0.025, pre=200)
    yield dict(ts=0.003, nf=5, ceil=5500, win=0.025, pre=50)


def run_formant(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    fp = lambda: call(ps, "To Formant (burg)", c["ts"], c["nf"], c["ceil"], c["win"], c["pre"])
    fo = lambda: os_.to_formant_burg(c["ts"], c["nf"], c["ceil"], c["win"], c["pre"])
    r = both(res, fp, fo)
    if not r:
        return
    p, o = r
    cmp_grid(res, "grid", p.n_frames, p.x1, p.dx, o.num_frames, o.start_time, o.time_step)
    if p.n_frames != o.num_frames:
        return
    n = p.n_frames
    pn = [call(p, "Get number of formants", i + 1) for i in range(n)]
    pcount = [call(p, "Get number of formants", i + 1) for i in range(n)]
    ocount = [sum(1 for k in range(1, 11) if k <= o.max_num_formants and
                  np.isfinite(o.formant_values(k)[i] if o.formant_values(k)[i] is not None else np.nan))
              for i in range(n)] if False else None
    for k in range(1, int(math.ceil(c["nf"])) + 1):
        pf_ = np.array(call(p, "To Matrix", k).values[0], float)
        pf_[pf_ == 0] = np.nan
        times = [p.get_time_from_frame_number(i + 1) for i in range(n)]
        pb = np.array([p.get_bandwidth_at_time(k, t) if k <= pcount[i] else np.nan
                       for i, t in enumerate(times)], float)
        of = np.array([np.nan if v is None else v for v in o.formant_values(k)], float)
        ob = np.array([np.nan if v is None else v for v in o.bandwidth_values(k)], float)
        of[of == 0] = np.nan
        ob[~np.isfinite(of)] = np.nan
        cmp_arrays(res, f"F{k}", pf_, of, tol)
        cmp_arrays(res, f"B{k}", pb, ob, max(tol, 1e-6), rel=True)
    d = p.xmax - p.xmin
    checks = []
    for (a0, a1) in [(0, 0), (p.xmin + 0.13 * d, p.xmin + 0.71 * d)]:
        for k in (1, 2, 3):
            for up, uo in (("hertz", "hertz"), ("bark", "bark")):
                checks.append((f"mean F{k}[{uo},{a0:.2f}]", call(p, "Get mean", k, a0, a1, up), o.get_mean(k, a0, a1, uo)))
                checks.append((f"sd F{k}[{uo},{a0:.2f}]", call(p, "Get standard deviation", k, a0, a1, up), o.get_standard_deviation(k, a0, a1, uo)))
                for q in (0.1, 0.5, 0.9):
                    checks.append((f"q{q} F{k}[{uo}]", call(p, "Get quantile", k, a0, a1, up, q), o.get_quantile(k, a0, a1, q, uo)))
                    checks.append((f"qbw{q} F{k}[{uo}]", call(p, "Get quantile of bandwidth", k, a0, a1, up, q), o.get_quantile_of_bandwidth(k, a0, a1, q, uo)))
                for ip in ("None", "Parabolic"):
                    checks.append((f"min F{k}[{uo},{ip}]", call(p, "Get minimum", k, a0, a1, up, ip), o.get_minimum(k, a0, a1, uo, ip.lower())))
                    checks.append((f"max F{k}[{uo},{ip}]", call(p, "Get maximum", k, a0, a1, up, ip), o.get_maximum(k, a0, a1, uo, ip.lower())))
                    checks.append((f"tmax F{k}[{uo},{ip}]", call(p, "Get time of maximum", k, a0, a1, up, ip), o.get_time_of_maximum(k, a0, a1, uo, ip.lower())))
    for name, a, b in checks:
        a = np.nan if a is None else a
        cmp_arrays(res, name, [a], [np.nan if b is None else b], max(tol, 1e-6))


def run_intensity(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    fp = lambda: ps.to_intensity(minimum_pitch=c["mp"], time_step=c["ts"] or None, subtract_mean=True)
    fo = lambda: os_.to_intensity(c["mp"], c["ts"])
    r = both(res, fp, fo)
    if not r:
        return
    p, o = r
    cmp_grid(res, "grid", p.n_frames, p.x1, p.dx, o.num_frames, o.start_time, o.time_step)
    if p.n_frames != o.num_frames:
        return
    cmp_arrays(res, "dB", p.values[0], o.values(), tol)
    times = np.linspace(p.xmin, p.xmax, 61)[1:-1]
    for ip in ("linear", "cubic", "nearest"):
        a = [call(p, "Get value at time", t, ip) for t in times]
        b = [o.get_value_at_time(t, ip) for t in times]
        cmp_arrays(res, f"q[{ip}]", a, [np.nan if v is None else v for v in b], tol)
    d = p.xmax - p.xmin
    ranges = [(0, 0), (p.xmin + 0.13 * d, p.xmin + 0.71 * d), (p.xmin + 0.4 * d, p.xmin + 0.4 * d + 0.3 * p.dx)]
    checks = [("mean(energy)", call(p, "Get mean", 0, 0, "energy"), o.mean()),
              ("min", call(p, "Get minimum", 0, 0, "Parabolic"), o.min()),
              ("max", call(p, "Get maximum", 0, 0, "Parabolic"), o.max())]
    for (a0, a1) in ranges:
        for m in ("energy", "sones", "dB"):
            checks.append((f"mean[{m},{a0:.3f}]", call(p, "Get mean", a0, a1, m), o.get_average(a0, a1, m)))
        for ip in ("None", "Parabolic", "Cubic", "Sinc70"):
            checks.append((f"min[{ip},{a0:.3f}]", call(p, "Get minimum", a0, a1, ip), o.get_minimum(a0, a1, ip.lower())))
            checks.append((f"max[{ip},{a0:.3f}]", call(p, "Get maximum", a0, a1, ip), o.get_maximum(a0, a1, ip.lower())))
            checks.append((f"tmax[{ip},{a0:.3f}]", call(p, "Get time of maximum", a0, a1, ip), o.get_time_of_maximum(a0, a1, ip.lower())))
    for name, a, b in checks:
        # time-of-extremum: Brent stops at 1e-10 in index units
        cmp_arrays(res, name, [a], [np.nan if b is None else b], 1e-8 if name.startswith("t") else tol)


def run_hnr(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    if c["m"] == "ac":
        fp = lambda: ps.to_harmonicity_ac(c["ts"], c["mp"], c["sil"], c["ppw"])
        fo = lambda: os_.to_harmonicity_ac(c["ts"], c["mp"], c["sil"], c["ppw"])
    else:
        fp = lambda: ps.to_harmonicity_cc(c["ts"], c["mp"], c["sil"], c["ppw"])
        fo = lambda: os_.to_harmonicity_cc(c["ts"], c["mp"], c["sil"], c["ppw"])
    r = both(res, fp, fo)
    if not r:
        return
    p, o = r
    cmp_grid(res, "grid", p.n_frames, p.x1, p.dx, o.num_frames, o.start_time, o.time_step)
    if p.n_frames != o.num_frames:
        return
    cmp_arrays(res, "hnr", p.values[0], o.values(), max(tol, 1e-6))
    hnr_ok = np.nanmax(np.abs(p.values[0] - np.asarray(o.values()))) <= 1e-6 if p.n_frames else True
    if not hnr_ok:
        return  # frame differences already reported; aggregates would just echo them
    times = np.linspace(p.xmin, p.xmax, 61)[1:-1]
    for ip in ("linear", "cubic", "nearest"):
        a = [call(p, "Get value at time", t, ip) for t in times]
        b = [o.get_value_at_time(t, ip) for t in times]
        cmp_arrays(res, f"q[{ip}]", a, [np.nan if v is None else v for v in b], 1e-6)
    d = p.xmax - p.xmin
    checks = [("mean", call(p, "Get mean", 0, 0), o.mean()),
              ("min", call(p, "Get minimum", 0, 0, "Parabolic"), o.min()),
              ("max", call(p, "Get maximum", 0, 0, "Parabolic"), o.max()),
              ("sd", call(p, "Get standard deviation", 0, 0), o.get_standard_deviation(0, 0))]
    for (a0, a1) in [(p.xmin + 0.13 * d, p.xmin + 0.71 * d)]:
        checks.append(("mean[r]", call(p, "Get mean", a0, a1), o.get_mean(a0, a1)))
        checks.append(("sd[r]", call(p, "Get standard deviation", a0, a1), o.get_standard_deviation(a0, a1)))
        for ip in ("None", "Parabolic", "Sinc70"):
            checks.append((f"min[{ip}]", call(p, "Get minimum", a0, a1, ip), o.get_minimum(a0, a1, ip.lower())))
            checks.append((f"max[{ip}]", call(p, "Get maximum", a0, a1, ip), o.get_maximum(a0, a1, ip.lower())))
    for name, a, b in checks:
        cmp_arrays(res, name, [a], [np.nan if b is None else b], 1e-6)


def run_spectrum(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    r = both(res, lambda: ps.to_spectrum(fast=c["fast"]), lambda: os_.to_spectrum(c["fast"]))
    if not r:
        return
    p, o = r
    if p.n_bins != o.num_bins:
        res.add(f"nbins praat={p.n_bins} ours={o.num_bins}")
        return
    for pw in (1.0, 2.0, 0.5):
        cmp_arrays(res, f"cog p={pw}", [p.get_center_of_gravity(pw)], [o.get_center_of_gravity(pw)], 1e-9, rel=True)
        cmp_arrays(res, f"sd p={pw}", [p.get_standard_deviation(pw)], [o.get_standard_deviation(pw)], 1e-9, rel=True)
        cmp_arrays(res, f"skew p={pw}", [p.get_skewness(pw)], [o.get_skewness(pw)], 1e-9, rel=True)
        cmp_arrays(res, f"kurt p={pw}", [p.get_kurtosis(pw)], [o.get_kurtosis(pw)], 1e-9, rel=True)
    for lo, hi in [(0, 0), (0, 500), (500, 4000), (1000, 1e9), (123.4, 567.8)]:
        cmp_arrays(res, f"band {lo}-{hi}", [p.get_band_energy(lo, hi)], [o.get_band_energy(lo, hi)], 1e-9, rel=True)


def run_spectrogram(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    shape_p = {"gaussian": parselmouth.SpectralAnalysisWindowShape.GAUSSIAN,
               "hanning": parselmouth.SpectralAnalysisWindowShape.HANNING,
               "hamming": parselmouth.SpectralAnalysisWindowShape.HAMMING,
               "square": parselmouth.SpectralAnalysisWindowShape.SQUARE,
               "bartlett": parselmouth.SpectralAnalysisWindowShape.BARTLETT,
               "welch": parselmouth.SpectralAnalysisWindowShape.WELCH}[c["shape"]]
    shape_o = {"square": "rectangular", "bartlett": "triangular"}.get(c["shape"], c["shape"])
    fp = lambda: ps.to_spectrogram(c["win"], c["fmax"], c["ts"], c["fs"], shape_p)
    fo = lambda: os_.to_spectrogram(c["win"], c["fmax"], c["ts"], c["fs"], shape_o)
    r = both(res, fp, fo)
    if not r:
        return
    p, o = r
    cmp_grid(res, "tgrid", p.n_frames, p.x1, p.dx, o.num_frames, o.start_time, o.time_step)
    cmp_grid(res, "fgrid", p.ny, p.y1, p.dy, o.num_freq_bins, o.freq_min, o.freq_step)
    if p.n_frames != o.num_frames or p.ny != o.num_freq_bins:
        return
    pv = p.values
    ov = np.asarray(o.values())
    if ov.shape != pv.shape and ov.T.shape == pv.shape:
        ov = ov.T
    cmp_arrays(res, "power", pv, ov, 1e-9, rel=True)


def _inplace(s, meth, *a):
    getattr(s, meth)(*a)
    return s


def run_sound(res, x, sr, c, tol):
    ps, os_ = parselmouth.Sound(x, sr), pf.Sound(x, sr)
    k = c["op"]
    if k == "resample":
        r = both(res, lambda: ps.resample(c["rate"], 50), lambda: os_.resample(c["rate"]))
        if r:
            p, o = r
            cmp_grid(res, "grid", p.n_samples, p.x1, p.dx, o.num_samples, o.x1, 1 / o.sample_rate)
            if p.n_samples == o.num_samples:
                cmp_arrays(res, "samples", p.values[0], o.samples(), 1e-12)
    elif k == "pre":
        r = both(res, lambda: _inplace(ps.copy(), "pre_emphasize", c["f"]), lambda: os_.pre_emphasis(c["f"]))
        if r:
            cmp_arrays(res, "samples", r[0].values[0], r[1].samples(), 1e-15)
    elif k == "de":
        r = both(res, lambda: _inplace(ps.copy(), "de_emphasize", c["f"]), lambda: os_.de_emphasis(c["f"]))
        if r:
            cmp_arrays(res, "samples", r[0].values[0], r[1].samples(), 1e-12)
    elif k == "stats":
        cmp_arrays(res, "rms", [call(ps, "Get root-mean-square", 0, 0)], [os_.rms()], 1e-14, rel=True)
        cmp_arrays(res, "peak", [call(ps, "Get absolute extremum", 0, 0, "None")], [os_.peak()], 1e-14, rel=True)
        times = np.linspace(ps.xmin, ps.xmax, 41)[1:-1]
        cmp_arrays(res, "value@t", [call(ps, "Get value at time", 0, t, "Sinc70") for t in times],
                   [os_.get_value_at_time(t) for t in times], 1e-12)
    elif k == "extract":
        d = len(x) / sr
        t0, t1 = 0.23 * d, 0.61 * d
        shp_p = {"rectangular": parselmouth.WindowShape.RECTANGULAR, "hanning": parselmouth.WindowShape.HANNING,
                 "hamming": parselmouth.WindowShape.HAMMING, "gaussian": parselmouth.WindowShape.GAUSSIAN1,
                 "triangular": parselmouth.WindowShape.TRIANGULAR}[c["shape"]]
        r = both(res, lambda: ps.extract_part(t0, t1, shp_p, c["rw"], c["keep"]),
                 lambda: os_.extract_part(t0, t1, c["shape"], c["rw"], c["keep"]))
        if r:
            p, o = r
            cmp_grid(res, "grid", p.n_samples, p.x1, p.dx, o.num_samples, o.x1, 1 / o.sample_rate)
            if p.n_samples == o.num_samples:
                cmp_arrays(res, "samples", p.values[0], o.samples(), 1e-14)
            cmp_arrays(res, "xmax", [p.xmax], [o.end_time], 1e-15)
            # analyses of the extracted part must sit on Praat's grid too
            pr, orr = p.resample(11000, 50), o.resample(11000.0)
            cmp_grid(res, "resampled part", pr.n_samples, pr.x1, pr.dx, orr.num_samples, orr.x1, 1 / orr.sample_rate)
            if p.n_samples >= 500:
                pf_ = call(p, "To Formant (burg)", 0.005, 5, 5500, 0.025, 50)
                of_ = o.to_formant_burg(0.005, 5, 5500, 0.025, 50)
                cmp_grid(res, "formant on part", pf_.n_frames, pf_.x1, pf_.dx, of_.num_frames, of_.start_time, of_.time_step)


def all_cases():
    for m, c in pitch_cases():
        yield "pitch", f"{m} {c}", lambda res, x, sr, m=m, c=c, tol=1e-6: run_pitch(res, x, sr, m, c, tol)
    for c in formant_cases():
        yield "formant", str(c), lambda res, x, sr, c=c: run_formant(res, x, sr, c, 1e-6)
    for mp in (50, 100, 300):
        for ts in (0.0, 0.01):
            c = dict(mp=mp, ts=ts)
            yield "intensity", str(c), lambda res, x, sr, c=c: run_intensity(res, x, sr, c, 1e-9)
    for m in ("ac", "cc"):
        for mp, ppw in [(75, 4.5 if m == "ac" else 1.0), (100, 3.0 if m == "ac" else 2.0), (60, 6.0 if m == "ac" else 1.5)]:
            for sil in (0.1, 0.0):
                c = dict(m=m, ts=0.01, mp=mp, ppw=ppw, sil=sil)
                yield "harmonicity", str(c), lambda res, x, sr, c=c: run_hnr(res, x, sr, c, 1e-6)
    for fast in (True, False):
        yield "spectrum", f"fast={fast}", lambda res, x, sr, c=dict(fast=fast): run_spectrum(res, x, sr, c, 0)
    for shape in ("gaussian", "hanning", "hamming", "square", "bartlett", "welch"):
        c = dict(win=0.005, fmax=5000, ts=0.002, fs=20, shape=shape)
        yield "spectrogram", str(c), lambda res, x, sr, c=c: run_spectrogram(res, x, sr, c, 0)
    for c in [dict(win=0.03, fmax=8000, ts=0.0, fs=0.0, shape="gaussian"),
              dict(win=0.005, fmax=0.0, ts=0.01, fs=50, shape="gaussian")]:
        yield "spectrogram", str(c), lambda res, x, sr, c=c: run_spectrogram(res, x, sr, c, 0)
    for rate in (8000, 10000, 22050, 48000):
        yield "sound", f"resample {rate}", lambda res, x, sr, c=dict(op="resample", rate=rate): run_sound(res, x, sr, c, 0)
    for f in (50, 1000):
        yield "sound", f"pre {f}", lambda res, x, sr, c=dict(op="pre", f=f): run_sound(res, x, sr, c, 0)
        yield "sound", f"de {f}", lambda res, x, sr, c=dict(op="de", f=f): run_sound(res, x, sr, c, 0)
    yield "sound", "stats", lambda res, x, sr: run_sound(res, x, sr, dict(op="stats"), 0)
    for shape in ("rectangular", "hanning", "hamming", "gaussian", "triangular"):
        for keep in (True, False):
            c = dict(op="extract", shape=shape, rw=1.0, keep=keep)
            yield "sound", f"extract {shape} keep={keep}", lambda res, x, sr, c=c: run_sound(res, x, sr, c, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--signals", default="")
    ap.add_argument("--all", action="store_true", help="also print passing cases")
    a = ap.parse_args()
    only = set(a.only.split(",")) - {""}
    sigs = signals()
    if a.signals:
        sigs = {k: v for k, v in sigs.items() if k in a.signals.split(",")}
    nbad = ntot = 0
    for group, desc, fn in all_cases():
        if only and group not in only:
            continue
        for sname, (x, sr) in sigs.items():
            res = Result(f"{group} {desc} [{sname}]")
            try:
                fn(res, np.ascontiguousarray(x, dtype=float), float(sr))
            except Exception as e:  # noqa: BLE001
                res.add("harness error: " + "".join(traceback.format_exception_only(e)).strip()[:200])
            ntot += 1
            if res.issues:
                nbad += 1
                print(f"FAIL {res.name}")
                for i in res.issues:
                    print(f"     - {i}")
            elif a.all:
                print(f"ok   {res.name}  maxerr={res.maxerr:.2g}")
            sys.stdout.flush()
    print(f"\n{ntot - nbad}/{ntot} cases exact; {nbad} with differences")


if __name__ == "__main__":
    main()
