"""Smoke-test an installed praatfan_gpl wheel on the platform it was built for.

Checks, on a deterministic synthetic signal:
  * the package imports and reports the expected version (if given);
  * every multi-threaded analysis is bit-identical between a 1-thread run and
    an all-cores run (each in a fresh interpreter, since RAYON_NUM_THREADS is
    read when the thread pool is first built);
  * on platforms with fork(), forked children don't deadlock after the parent
    has run an analysis.

Bits are compared within one platform only: exp/log/sin may round differently
across platforms, so cross-platform bit equality is not expected.

Usage: python smoke_test_wheel.py [expected_version]
"""

import hashlib
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np


def _sound():
    import praatfan_gpl

    sr = 16000.0
    t = np.arange(int(4.0 * sr)) / sr
    rng = np.random.default_rng(12345)
    # Voiced stretches with a gliding F0, separated by noise and silence.
    f0 = 110.0 + 30.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    gate = (np.sin(2 * np.pi * 0.75 * t) > -0.3).astype(float)
    x = 0.2 * voiced * gate + 0.01 * rng.standard_normal(t.size)
    x[int(1.5 * sr):int(1.7 * sr)] = 0.0
    return praatfan_gpl.Sound(x, sr)


def digest():
    """Hash of every analysis's output bits."""
    snd = _sound()
    h = hashlib.sha256()

    def add(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(name.encode())
        h.update(a.tobytes())

    add("resample", snd.resample(11000.0).samples())
    add("pitch_ac", snd.to_pitch(0.01, 75.0, 600.0).values())
    add("pitch_cc", snd.to_pitch_cc(0.01, 75.0, 600.0).values())
    add("hnr_ac", snd.to_harmonicity_ac(0.01, 75.0, 0.1, 4.5).values())
    add("hnr_cc", snd.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0).values())
    add("intensity", snd.to_intensity(100.0, 0.0).values())
    f = snd.to_formant_burg(0.0, 5, 5500.0, 0.025, 50.0)
    for k in range(1, 6):
        add(f"formant_f{k}", f.formant_values(k))
        add(f"formant_b{k}", f.bandwidth_values(k))
    for i, fm in enumerate(snd.to_formant_burg_multi(0.01, 5, [4500.0, 5500.0], 0.025, 50.0)):
        add(f"multi{i}", fm.formant_values(2))
    fp = snd.to_formant_path_burg(0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4)
    fp.path_finder(0.5, 0.5, 0.5, 0.5, 5.0, 0.035, [3, 3, 3, 3], 1.25)
    ex = fp.extract_formant()
    for k in range(1, 4):
        add(f"path_f{k}", ex.formant_values(k))
    add("spectrogram", snd.to_spectrogram(0.005, 5000.0, 0.005, 20.0, "gaussian").values())
    return h.hexdigest()


def _digest_in_subprocess(threads):
    env = dict(os.environ)
    if threads is None:
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["RAYON_NUM_THREADS"] = str(threads)
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--digest"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if out.returncode != 0:
        sys.exit(f"digest run failed (threads={threads}):\n{out.stderr}")
    return out.stdout.strip()


def _child(_):
    return digest()


def main():
    import praatfan_gpl

    expected = sys.argv[1] if len(sys.argv) > 1 else None
    print("praatfan_gpl", praatfan_gpl.__version__, "on", sys.platform, os.cpu_count(), "cpus")
    if expected and praatfan_gpl.__version__ != expected:
        sys.exit(f"version mismatch: installed {praatfan_gpl.__version__}, expected {expected}")

    serial = _digest_in_subprocess(1)
    for threads in (2, None):
        par = _digest_in_subprocess(threads)
        label = threads or "all"
        if par != serial:
            sys.exit(f"FAIL: output differs between 1 and {label} threads")
        print(f"bit-identical: 1 thread == {label} threads ({serial[:16]})")

    if "fork" in mp.get_all_start_methods() and sys.platform.startswith("linux"):
        parent = digest()  # builds the thread pool in the parent
        with mp.get_context("fork").Pool(2) as pool:
            children = pool.map_async(_child, range(2)).get(timeout=300)
        if any(c != parent for c in children):
            sys.exit("FAIL: forked children produced different output")
        print("fork: children ran without deadlock and match the parent")

    print("OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--digest"]:
        print(digest())
    else:
        main()
