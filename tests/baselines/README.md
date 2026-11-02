# Baselines

Pinned outputs used as regression-test fixtures. Each file here captures one
specific numerical snapshot of FormantPath behaviour at a specific moment in
the project's numerical-precision history; newer code is expected to MATCH
the `praat_*` columns more closely and MUST NOT regress against the `ours_*`
columns.

## `h95_formantpath_parity.csv`

Per-token `(filename, group, ceiling_hz, t_mid_s, gt_f1/2/3, praat_f1/2/3,
ours_f1/2/3, diff_f1/2/3)` for every Hillenbrand-1995 (h95) token where
both `vowdata.dat` and `timedata.dat` have entries.

- `gt_*` — Hillenbrand's hand-corrected F1/F2/F3 at 50% of vowel duration
  (from `vowdata.dat` cols 11/12/13).
- `praat_*` — Praat 6.1.38's `FormantPath(burg) → Path finder(0.5/0.5/0.5/
  0.5/5.0/0.035/"3 3 3 3"/1.25) → Extract Formant`, queried at `t_mid_s`.
- `ours_*` — same protocol via `praatfan-gpl`'s `to_formant_path_burg +
  path_finder + extract_formant`.
- `diff_*` = `ours_* - praat_*`. Pinned at this baseline's build.

**Ceilings**: men 5000 Hz, women 5500 Hz, kids 6500 Hz (Praat convention).

### How the baseline was produced

```bash
source /path/to/venv-with-parselmouth/bin/activate
python scripts/generate_h95_baseline.py
```

### Regenerating / comparing

Run the regression checker against the pinned baseline after any change
that could affect FormantPath numerics (resample, Burg LPC, root finding,
stress computation, Viterbi):

```bash
source /path/to/venv-with-parselmouth/bin/activate
python scripts/check_h95_regression.py                  # full run
python scripts/check_h95_regression.py --limit 60       # quick smoke
python scripts/check_h95_regression.py --fail-tol 1.0   # CI gate
```

The checker re-runs our FormantPath on every baseline row's `(filename,
t_mid)`, compares the new `ours_*` values to the old `ours_*` in the
baseline, classifies each changed value as improvement or regression
relative to `praat_*`, and exits non-zero on any regression > `--fail-tol`.

### Provenance of this baseline

Re-pinned 2026-11-01 for 0.1.15 (see `STATE.md`): all 1609 tokens agree
with parselmouth 0.4.7 within 5e-7 Hz on F1-F3. (The first baseline, saved
before the switch to `faer` root-finding, had medians of 0.01-0.03 Hz and
outliers up to ~31 Hz.)

Any future build should match or beat these numbers on every token.
