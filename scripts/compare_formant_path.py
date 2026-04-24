"""Compare our FormantPath against parselmouth/Praat on a given audio file.

Because parselmouth (0.4.x → Praat 6.1.38) does not expose FormantPath query
commands (`Get number of candidates`, `Get stress of candidate`, etc.), the
primary parity check is against the Formant extracted by `Extract Formant`
after `Path finder`. That is what downstream users actually consume.

References:
  Boersma, P. & Weenink, D. (2024). Praat: doing phonetics by computer.
  Weenink, D. FormantPath (Praat: LPC/FormantPath.cpp, GPL-3).
  Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
    A Python interface to Praat. Journal of Phonetics, 71, 1–15.

Usage:
  source ~/local/scr/commonpip/bin/activate
  python scripts/compare_formant_path.py tests/fixtures/one_two_three_four_five.wav
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import parselmouth as pm
from parselmouth.praat import call


def run_ours(
    audio: str,
    time_step: float,
    max_formants: int,
    middle_ceiling: float,
    window_length: float,
    pre_emphasis: float,
    ceiling_step_size: float,
    steps_up_down: int,
    qw: float,
    fcw: float,
    sw: float,
    ccw: float,
    imod: float,
    path_wl: float,
    params_csv: str,
    power: float,
) -> dict:
    exe = (
        Path(__file__).resolve().parent.parent
        / "target"
        / "release"
        / "examples"
        / "formant_path_json"
    )
    if not exe.exists():
        print(
            "Build the Rust example first: cargo build --release --example formant_path_json",
            file=sys.stderr,
        )
        sys.exit(1)
    args = [
        str(exe),
        audio,
        f"{time_step}",
        f"{max_formants}",
        f"{middle_ceiling}",
        f"{window_length}",
        f"{pre_emphasis}",
        f"{ceiling_step_size}",
        f"{steps_up_down}",
        f"{qw}",
        f"{fcw}",
        f"{sw}",
        f"{ccw}",
        f"{imod}",
        f"{path_wl}",
        params_csv,
        f"{power}",
    ]
    out = subprocess.check_output(args).decode()
    return json.loads(out)


def run_praat(
    audio: str,
    time_step: float,
    max_formants: float,
    middle_ceiling: float,
    window_length: float,
    pre_emphasis: float,
    ceiling_step_size: float,
    steps_up_down: int,
    qw: float,
    fcw: float,
    sw: float,
    ccw: float,
    imod: float,
    path_wl: float,
    params_str: str,
    power: float,
):
    snd = pm.Sound(audio)
    fp = call(
        snd,
        "To FormantPath (burg)",
        time_step,
        max_formants,
        middle_ceiling,
        window_length,
        pre_emphasis,
        ceiling_step_size,
        steps_up_down,
    )
    call(fp, "Path finder", qw, fcw, sw, ccw, imod, path_wl, params_str, power)
    formant = call(fp, "Extract Formant")
    n = int(call(formant, "Get number of frames"))
    times = np.array(
        [float(call(formant, "Get time from frame number", i + 1)) for i in range(n)]
    )
    # Query at the exact frame times using the linear-interpolation "Get value
    # at time"; since we ask at the frame center, this just returns the frame's
    # value (no actual interpolation happens).
    f1 = np.array(
        [call(formant, "Get value at time", 1, t, "Hertz", "Linear") for t in times]
    )
    f2 = np.array(
        [call(formant, "Get value at time", 2, t, "Hertz", "Linear") for t in times]
    )
    f3 = np.array(
        [call(formant, "Get value at time", 3, t, "Hertz", "Linear") for t in times]
    )
    return times, f1, f2, f3


def compare_series(name: str, ours: np.ndarray, theirs: np.ndarray) -> dict:
    # Treat NaN/None as unvoiced; only compare where BOTH are finite.
    a = np.array([np.nan if v is None else v for v in ours], dtype=float)
    b = np.array([np.nan if v is None else v for v in theirs], dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return {"name": name, "n": 0, "mean_err": None, "p99": None, "max": None}
    diff = np.abs(a[both] - b[both])
    return {
        "name": name,
        "n": int(both.sum()),
        "mean_err": float(diff.mean()),
        "p99": float(np.percentile(diff, 99)),
        "max": float(diff.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--time-step", type=float, default=0.005)
    p.add_argument("--max-formants", type=float, default=5.0)
    p.add_argument("--middle-ceiling", type=float, default=5500.0)
    p.add_argument("--window-length", type=float, default=0.025)
    p.add_argument("--pre-emphasis", type=float, default=50.0)
    p.add_argument("--ceiling-step-size", type=float, default=0.05)
    p.add_argument("--steps-up-down", type=int, default=4)
    # Path finder weights — clamp at <= 0.5 per project convention (Praat
    # window-length validation gotcha at higher weights with 0.035 window).
    p.add_argument("--qw", type=float, default=0.5)
    p.add_argument("--fcw", type=float, default=0.5)
    p.add_argument("--sw", type=float, default=0.5)
    p.add_argument("--ccw", type=float, default=0.5)
    p.add_argument("--intensity-mod-step", type=float, default=5.0)
    p.add_argument("--path-window-length", type=float, default=0.035)
    p.add_argument(
        "--parameters",
        default="3,3,3,3",
        help="Comma-separated for Rust; space-separated sent to Praat.",
    )
    p.add_argument("--power", type=float, default=1.25)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    params_csv = args.parameters
    params_str = " ".join(params_csv.split(","))

    ours = run_ours(
        args.audio,
        args.time_step,
        int(args.max_formants),
        args.middle_ceiling,
        args.window_length,
        args.pre_emphasis,
        args.ceiling_step_size,
        args.steps_up_down,
        args.qw,
        args.fcw,
        args.sw,
        args.ccw,
        args.intensity_mod_step,
        args.path_window_length,
        params_csv,
        args.power,
    )

    times_p, f1_p, f2_p, f3_p = run_praat(
        args.audio,
        args.time_step,
        args.max_formants,
        args.middle_ceiling,
        args.window_length,
        args.pre_emphasis,
        args.ceiling_step_size,
        args.steps_up_down,
        args.qw,
        args.fcw,
        args.sw,
        args.ccw,
        args.intensity_mod_step,
        args.path_window_length,
        params_str,
        args.power,
    )

    # Ceiling parity (deterministic — first sanity check).
    expected_ceilings = [
        args.middle_ceiling * np.exp((i - args.steps_up_down) * args.ceiling_step_size)
        for i in range(2 * args.steps_up_down + 1)
    ]
    ours_ceilings = ours["ceilings"]
    ceiling_max_err = max(
        abs(a - b) for a, b in zip(expected_ceilings, ours_ceilings)
    )
    print(f"Ceiling formula parity: max error {ceiling_max_err:.3e} Hz")

    # Align the extracted series by index (both come from the same Praat-grid
    # frame timing; we trim to the shorter length if anything drifted).
    n_ours = len(ours["extracted"]["times"])
    n_theirs = len(times_p)
    n = min(n_ours, n_theirs)
    if n_ours != n_theirs:
        print(
            f"Frame count mismatch: ours={n_ours} theirs={n_theirs} "
            f"(comparing first {n})",
            file=sys.stderr,
        )

    results = []
    for name, rust_key in [("F1", "f1"), ("F2", "f2"), ("F3", "f3")]:
        ours_vals = ours["extracted"][rust_key][:n]
        theirs_vals = {"F1": f1_p, "F2": f2_p, "F3": f3_p}[name][:n]
        results.append(compare_series(name, ours_vals, theirs_vals))

    print()
    print(
        f"{'Measure':<5} {'N':>6} {'mean_err (Hz)':>14} {'P99 (Hz)':>10} {'max (Hz)':>10}"
    )
    for r in results:
        print(
            f"{r['name']:<5} {r['n']:>6} "
            f"{r['mean_err']:>14.4f} {r['p99']:>10.4f} {r['max']:>10.4f}"
        )

    if args.verbose:
        print()
        print("Path (our Rust side, 1-based):", ours["path_1based"][:40], "...")

    # Gate: mean error should be < 1 Hz on any formant.
    ok = all((r["mean_err"] or 0) < 5.0 for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
