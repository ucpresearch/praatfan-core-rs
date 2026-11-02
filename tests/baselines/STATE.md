# FormantPath Praat-parity state snapshot

Re-pinned 2026-11-01 for 0.1.15 after the parameter-sweep fixes
(resampled-sound `x1`, `Sound_into_LPC` framing, Praat's
`Sampled_shortTermAnalysis` grid; see CHANGELOG). The baseline CSV
`h95_formantpath_parity.csv` in this directory was regenerated at this
state; `check_h95_regression.py` against it will see zero changes on a
matching build.

## Current parity vs parselmouth 0.4.x (Praat 6.1.38)

Hillenbrand 1995, 1609 vowel tokens, F1/F2/F3 at 50% of vowel duration,
default weights (0.5 / 0.5 / 0.5 / 0.5 / 5.0 / 0.035 / "3 3 3 3" / 1.25):

| | median | mean | P99 | max | <= 0.01 Hz | <= 1 Hz | <= 5 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|
| all 4827 values | 0.0000 | 0.0000 | 0.000 | 5e-7 | **100%** | 100% | 100% |
| (2026-04-23 pin) | 0.019 | 0.130 | 1.22 | 50.8 | 36.8% | 98.82% | 99.63% |

All 1609 tokens agree with Praat to within 5e-7 Hz on F1-F3 (the CSV's
rounding). The last three (w32aw, m09ah, m40ei) were path-finder bugs, found
by recovering Praat's per-frame path from `Extract Formant` (every frame
matches exactly one of our bit-exact candidates) and comparing Praat's
`To Matrix (stress)` / `(qsums)`:

- frames are NaN-padded to the maximum number of formants; the path finder
  and Q-sums used that padded length instead of Praat's `numberOfFormants`
  (frames with 3 real formants were scored as 4);
- FormantModeler tracks with fewer valid points than parameters were made
  "undefined" (stress NaN, so the stress term dropped out); Praat keeps them,
  with zero parameters, chi-squared against the zero model and negative
  degrees of freedom; points with an unusable bandwidth stay valid with
  weight 1.

There is no remaining FormantPath parity gap on this corpus.

## Historical context (2026-04-23 session)

| change | commit | net h95 impact |
|---|---|---:|
| faer for root-finding | `501fefe` | 0 (machine-precision equivalent) |
| Kahan-Neumaier Burg | `501fefe` | 0 (below measurement resolution) |
| FFTPACK f64 via speexdsp-rs (fixed bugs) | `501fefe` | 0 (not the bottleneck) |
| Sinc window-depth 6.1.38 formula | `501fefe` | sample-level: bit-exact; formant-level: nil |
| **Viterbi transition-cost track cap** | `d751da7` | **F2 mean 14×, max 35×, F3 P99 29×** |

The one-line track-cap fix was the entire session's parity win.
Everything else was code-quality hygiene that did not move formant
numbers at 1-Hz resolution.
