"""Generate the h95 FormantPath parity baseline CSV.

For every Hillenbrand 1995 token where we have (a) ground-truth F1/F2/F3 at
50% from vowdata.dat and (b) vowel boundaries from timedata.dat, run:

  * Praat 6.1.38's `Sound: To FormantPath (burg)` + `Path finder` +
    `Extract Formant`, query at the vowel midpoint.
  * Our praatfan-gpl FormantPath with matching parameters.

Write every (filename, group, ceiling, t_mid, gt, praat, ours, diff) row to
`tests/baselines/h95_formantpath_parity.csv`. This file is the pinned
ground truth for FormantPath parity — future numerical-precision work
(faer root-finding, tightened Burg, LAPACK) is expected to SHRINK the
|praat - ours| columns and MUST NOT regress any token outside a small
tolerance.

Note: Praat's runtime dominates (~1 s per token via `parselmouth.praat.call`
round-trips). Full run takes ~30 minutes on 1500+ tokens.

Usage (from repo root):
    source ~/local/scr/commonpip/bin/activate
    python scripts/generate_h95_baseline.py
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth as pm
from parselmouth.praat import call

from praatfan_gpl import Sound

REPO = Path(__file__).resolve().parent.parent
H95 = Path.home() / "local/scr/hillenbrand/h95"
OUT = REPO / "tests/baselines/h95_formantpath_parity.csv"


def parse_vowdata():
    out = {}
    with open(H95 / "vowdata.dat") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 16:
                continue
            name = parts[0]
            if name[0] not in "mwbg":
                continue
            try:
                f1_50 = int(parts[10]); f2_50 = int(parts[11]); f3_50 = int(parts[12])
            except ValueError:
                continue
            if 0 in (f1_50, f2_50, f3_50):
                continue
            out[name] = (f1_50, f2_50, f3_50)
    return out


def parse_timedata():
    out = {}
    with open(H95 / "timedata.dat") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            if parts[0] in ("Start", "File"):
                continue
            try:
                start = float(parts[1]) / 1000.0
                end = float(parts[2]) / 1000.0
            except ValueError:
                continue
            out[parts[0]] = (start, end)
    return out


def speaker_group_and_ceiling(name: str):
    c = name[0]
    if c == "m":
        return "men", 5000.0
    if c == "w":
        return "women", 5500.0
    # b=boy, g=girl
    return "kids", 6500.0


def analyze_praat(wav_path: Path, ceiling: float, t_mid: float):
    s = pm.Sound(str(wav_path))
    fp = call(s, "To FormantPath (burg)",
              0.005, 5.0, ceiling, 0.025, 50.0, 0.05, 4)
    call(fp, "Path finder",
         0.5, 0.5, 0.5, 0.5, 5.0, 0.035, "3 3 3 3", 1.25)
    f = call(fp, "Extract Formant")

    def q(k):
        v = call(f, "Get value at time", k, t_mid, "Hertz", "Linear")
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    return q(1), q(2), q(3)


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
    vowdata = parse_vowdata()
    timedata = parse_timedata()
    names = sorted(set(vowdata) & set(timedata))
    print(f"Processing {len(names)} h95 tokens with both vowdata and timedata",
          file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "group", "ceiling_hz", "t_mid_s",
        "gt_f1", "gt_f2", "gt_f3",
        "praat_f1", "praat_f2", "praat_f3",
        "ours_f1", "ours_f2", "ours_f3",
        "diff_f1", "diff_f2", "diff_f3",
    ]

    t0 = time.time()
    written = 0
    skipped = 0
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)

        for i, name in enumerate(names, 1):
            group, ceiling = speaker_group_and_ceiling(name)
            wav = H95 / group / f"{name}.wav"
            if not wav.exists():
                skipped += 1
                continue
            v_start, v_end = timedata[name]
            t_mid = 0.5 * (v_start + v_end)
            try:
                pm_f = analyze_praat(wav, ceiling, t_mid)
                us_f = analyze_ours(wav, ceiling, t_mid)
            except Exception as e:
                print(f"{name}: {type(e).__name__}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if None in pm_f or None in us_f:
                skipped += 1
                continue
            g1, g2, g3 = vowdata[name]
            p1, p2, p3 = pm_f
            u1, u2, u3 = us_f
            writer.writerow([
                name, group, f"{ceiling:.4f}", f"{t_mid:.6f}",
                g1, g2, g3,
                f"{p1:.6f}", f"{p2:.6f}", f"{p3:.6f}",
                f"{u1:.6f}", f"{u2:.6f}", f"{u3:.6f}",
                f"{u1 - p1:+.6f}", f"{u2 - p2:+.6f}", f"{u3 - p3:+.6f}",
            ])
            written += 1
            if i % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (len(names) - i)
                print(f"  [{i}/{len(names)}] written={written} skipped={skipped} "
                      f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                      file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  written={written}  skipped={skipped}",
          file=sys.stderr)
    print(f"Baseline saved to {OUT}")


if __name__ == "__main__":
    main()
