#!/usr/bin/env python3
"""Black-box parity of the pipe's `spectrogram` analysis against parselmouth.

Runs praatfan-gpl-pipe with a spectrogram request and compares the full grid
(times, freqs) and every power value with Praat's "To Spectrogram".

Usage:
    cargo build --release --features pipe --bin praatfan-gpl-pipe
    python scripts/compare_pipe_spectrogram.py audio.wav [--channel N]
"""
import argparse
import base64
import json
import subprocess
import sys

import numpy as np
import parselmouth
from parselmouth.praat import call

PIPE = "target/release/praatfan-gpl-pipe"

CASES = [
    dict(window_length=0.005, max_frequency=5000.0, time_step=0.002, frequency_step=20.0, window_shape="gaussian"),
    dict(window_length=0.025, max_frequency=5000.0, time_step=0.002, frequency_step=20.0, window_shape="gaussian"),
    dict(window_length=0.005, max_frequency=5000.0, time_step=0.002, frequency_step=20.0, window_shape="hanning"),
    dict(window_length=0.025, max_frequency=8000.0, time_step=0.005, frequency_step=10.0, window_shape="hanning"),
    # Clamped to Nyquist by the pipe (Praat itself rejects <= 0); tiny steps
    # exercise the oversampling floors.
    dict(window_length=0.005, max_frequency=0.0, time_step=0.0001, frequency_step=1.0, window_shape="gaussian"),
    dict(window_length=0.005, max_frequency=1e9, time_step=0.002, frequency_step=20.0, window_shape="hanning"),
]


def run_pipe(path, channel, params):
    req = {"wav_path": path, "analyses": [dict(type="spectrogram", **params)]}
    if channel is not None:
        req["channel"] = channel
    out = subprocess.run([PIPE], input=json.dumps(req), capture_output=True, text=True, check=True)
    resp = json.loads(out.stdout)
    if not resp["ok"]:
        sys.exit(f"pipe error: {resp['error']}")
    r = resp["results"][0]
    power = np.frombuffer(base64.b64decode(r["power_b64"]), dtype="<f8")
    return r, power.reshape(r["n_times"], r["n_freqs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--channel", type=int)
    args = ap.parse_args()

    snd = parselmouth.Sound(args.audio)
    if args.channel is not None:
        snd = snd.extract_channel(args.channel + 1)

    worst = 0.0
    for params in CASES:
        r, power = run_pipe(args.audio, args.channel, params)
        shape = {"gaussian": "Gaussian", "hanning": "Hanning (sine-squared)"}[params["window_shape"]]
        nyquist = 0.5 * snd.sampling_frequency
        max_f = params["max_frequency"]
        max_f = nyquist if max_f <= 0 or max_f > nyquist else max_f
        spec = call(snd, "To Spectrogram", params["window_length"], max_f,
                    params["time_step"], params["frequency_step"], shape)
        praat = spec.values.T  # parselmouth: (n_freqs, n_times)
        times, freqs = np.array(r["times"]), np.array(r["freqs"])
        grid_ok = (praat.shape == power.shape
                   and np.allclose(times, spec.xs(), rtol=0, atol=1e-12)
                   and np.allclose(freqs, spec.ys(), rtol=0, atol=1e-9)
                   and abs(r["time_step"] - spec.dx) < 1e-15
                   and abs(r["frequency_step"] - spec.dy) < 1e-9)
        scale = np.max(np.abs(praat)) or 1.0
        rel = float(np.max(np.abs(power - praat)) / scale) if grid_ok else float("inf")
        worst = max(worst, rel)
        print(f"{params}: grid {'OK' if grid_ok else 'MISMATCH'} {power.shape} "
              f"vs {praat.shape}, max |diff|/max = {rel:.2e}")
    print(f"worst relative error: {worst:.2e}")
    sys.exit(0 if worst < 1e-10 else 1)


if __name__ == "__main__":
    main()
