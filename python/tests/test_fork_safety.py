"""Parallel analyses must keep working in forked children.

rayon's global pool doesn't survive fork(); praatfan_gpl uses a pool keyed on
the process id instead. This runs an analysis in the parent (spinning up the
pool), then forks a multiprocessing Pool that runs the same analysis. A
regression shows up as a hang, so the children are given a timeout.

Run: python python/tests/test_fork_safety.py   (or via pytest)
"""

import multiprocessing as mp

import numpy as np

import praatfan_gpl


def _signal():
    sr = 16000.0
    t = np.arange(int(2.0 * sr)) / sr
    return np.sin(2 * np.pi * 120 * t) * 0.3 + 0.01 * np.sin(2 * np.pi * 2300 * t), sr


def _analyse(_):
    samples, sr = _signal()
    snd = praatfan_gpl.Sound(samples, sr)
    pitch = snd.to_pitch(0.01, 75.0, 600.0)
    hnr = snd.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0)
    formant = snd.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0)
    return (
        np.asarray(pitch.values()).tobytes(),
        np.asarray(hnr.values()).tobytes(),
        np.asarray(formant.formant_values(1)).tobytes(),
    )


def test_fork_after_parallel_call():
    parent = _analyse(None)  # starts the thread pool in the parent
    ctx = mp.get_context("fork")
    with ctx.Pool(3) as pool:
        result = pool.map_async(_analyse, range(6))
        children = result.get(timeout=120)  # a hang here is the bug
    assert all(c == parent for c in children)


if __name__ == "__main__":
    test_fork_after_parallel_call()
    print("ok")
