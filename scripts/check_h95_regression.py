"""Re-run the h95 parity benchmark with the CURRENT build and compare against
the pinned baseline at `tests/baselines/h95_formantpath_parity.csv`.

Usage:
    source ~/local/scr/commonpip/bin/activate
    python scripts/check_h95_regression.py             # full run (~30 min)
    python scripts/check_h95_regression.py --limit 60  # quick smoke check

The baseline was generated once (pre-faer, pre-Burg-tightening) and pins a
known-good set of `(filename, praat_f1/2/3, ours_f1/2/3)` triples. This
script computes the SAME rows with the current build and reports:

  * Tokens where the new build changed its answer by > tol Hz vs the old
    build ("regression candidates" — could be improvements or regressions).
  * Whether the |praat - ours| magnitudes shrank (good) or grew (regression).

Exits non-zero if any token grew its |praat - ours| by more than --fail-tol
Hz — use that as a CI gate.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import parselmouth as pm
from parselmouth.praat import call
from praatfan_gpl import Sound

REPO = Path(__file__).resolve().parent.parent
H95 = Path.home() / "local/scr/hillenbrand/h95"
BASELINE = REPO / "tests/baselines/h95_formantpath_parity.csv"


def speaker_group_and_ceiling(name: str):
    c = name[0]
    if c == "m":
        return "men", 5000.0
    if c == "w":
        return "women", 5500.0
    return "kids", 6500.0


def load_baseline():
    rows = []
    with BASELINE.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def analyze_ours(wav_path: Path, ceiling: float, t_mid: float):
    s = Sound.from_file(str(wav_path))
    fp = s.to_formant_path_burg(0.005, 5, ceiling, 0.025, 50.0, 0.05, 4)
    fp.path_finder(0.5, 0.5, 0.5, 0.5, 5.0, 0.035, [3, 3, 3, 3], 1.25)
    f = fp.extract_formant()
    return (
        f.get_value_at_time(1, t_mid, "hertz", "linear"),
        f.get_value_at_time(2, t_mid, "hertz", "linear"),
        f.get_value_at_time(3, t_mid, "hertz", "linear"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max tokens (0=all)")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="Hz threshold for 'changed' vs old build")
    ap.add_argument("--fail-tol", type=float, default=1.0,
                    help="non-zero exit when any token's |praat-ours| grew "
                         "by more than this many Hz")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not BASELINE.exists():
        print(f"ERROR: baseline missing at {BASELINE}", file=sys.stderr)
        print("Generate it with: python scripts/generate_h95_baseline.py",
              file=sys.stderr)
        return 2

    rows = load_baseline()
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"Checking {len(rows)} baseline rows against current build")

    changed = 0
    improved = 0
    regressed = 0
    regressions = []  # (name, formant, old_diff, new_diff)
    for i, row in enumerate(rows, 1):
        name = row["filename"]
        group, ceiling = speaker_group_and_ceiling(name)
        wav = H95 / group / f"{name}.wav"
        if not wav.exists():
            continue
        t_mid = float(row["t_mid_s"])
        try:
            u1, u2, u3 = analyze_ours(wav, ceiling, t_mid)
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if None in (u1, u2, u3):
            continue
        old = [float(row["ours_f1"]), float(row["ours_f2"]), float(row["ours_f3"])]
        new = [u1, u2, u3]
        praat = [float(row["praat_f1"]), float(row["praat_f2"]), float(row["praat_f3"])]
        for k, name_ in enumerate(("F1", "F2", "F3")):
            delta = new[k] - old[k]
            if abs(delta) > args.tol:
                changed += 1
                old_diff = abs(old[k] - praat[k])
                new_diff = abs(new[k] - praat[k])
                if new_diff < old_diff:
                    improved += 1
                elif new_diff > old_diff + args.fail_tol:
                    regressed += 1
                    regressions.append((name, name_, old_diff, new_diff, old[k], new[k], praat[k]))
        if args.verbose and i % 100 == 0:
            print(f"  [{i}/{len(rows)}] changed={changed} improved={improved} regressed={regressed}")

    print(f"\nChanged values: {changed}")
    print(f"  improved (closer to Praat): {improved}")
    print(f"  regressed by > {args.fail_tol} Hz: {regressed}")
    if regressions:
        print(f"\nRegressions:")
        for (name, f, old_d, new_d, old_v, new_v, p) in regressions[:20]:
            print(f"  {name} {f}: |praat-ours| went {old_d:.3f} -> {new_d:.3f}  "
                  f"(old={old_v:.2f} new={new_v:.2f} praat={p:.2f})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
