# FormantPath Praat-parity state snapshot

Pinned after the 2026-04-23 precision pass (commits `501fefe` and
`d751da7`). The baseline CSV `h95_formantpath_parity.csv` in this
directory was regenerated at this state; `check_h95_regression.py`
against it will see zero changes on a matching build.

## Current parity vs parselmouth 0.4.x (Praat 6.1.38)

Measured on the Hillenbrand 1995 dataset, 1609 vowel tokens, F1/F2/F3
at 50% of vowel duration, default weights (0.5 / 0.5 / 0.5 / 0.5 / 5.0 /
0.035 / "3 3 3 3" / 1.25):

| formant | median | mean | P90 | P99 | max | ≤0.01 Hz | ≤1 Hz | ≤5 Hz | ≤50 Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **F1** | 0.008 | 0.063 | 0.052 | 0.63 | 31.9 | 54.1% | **99.3%** | 99.9% | 100% |
| **F2** | 0.026 | 0.140 | 0.14 | 1.02 | 50.8 | 28.9% | **98.9%** | 99.6% | 100% |
| **F3** | 0.031 | 0.186 | 0.17 | 1.62 | 42.7 | 27.2% | **98.3%** | 99.3% | 100% |

All 4827 values (1609 × 3 formants):
- 99.2% within 1 Hz
- 99.6% within 5 Hz
- 100% within 51 Hz
- median diff 0.02 Hz (bit-accurate on half of tokens)

## Top 10 remaining F1 outliers

The biggest parity gaps are now all Viterbi tie-break disagreements on
specific tokens, not per-candidate formant errors.

| token | Praat F1 | Ours F1 | Δ | group |
|---|---:|---:|---:|---|
| **w32aw** | 748.69 | 716.80 | **31.89** | women |
| m45aw | 714.43 | 707.39 | 7.03 | men |
| m48ah | 694.46 | 690.00 | 4.46 | men |
| b01eh | 719.44 | 723.85 | 4.41 | kids |
| m18uh | 623.45 | 619.59 | 3.85 | men |
| b07eh | 908.50 | 905.44 | 3.06 | kids |
| w46uh | 699.09 | 701.98 | 2.89 | women |
| w13aw | 743.49 | 746.23 | 2.74 | women |
| m18oa | 476.41 | 474.11 | 2.30 | men |
| m41ei | 451.08 | 449.63 | 1.45 | men |

### Next candidate for investigation: `w32aw`

- **What it is**: Women speaker 32, vowel /aw/ (as in "hod"). Female
  voice. Low back vowel, typical F1 around 750 Hz.
- **Symptom**: our path picks a candidate whose F1 is ~32 Hz lower
  than Praat's.
- **Hypothesis based on `w32iy` investigation**: likely another
  Viterbi-level disagreement driven by a subtle difference in one of
  the transition / static cost terms on a specific frame. The
  per-candidate formants almost certainly agree with Praat to
  sub-hertz precision (they did for every token we traced); the
  divergence is in the Viterbi DP's candidate selection.
- **Suggested starting point**: apply the same debug protocol that
  found the track-cap bug:
  1. Trace frame-by-frame around the vowel midpoint to confirm the
     path diverges in a specific window (as with w32iy frames 36–47).
  2. Run the Python Viterbi port with identical inputs to our Rust.
     If they agree, the disagreement is inside Praat's internals
     we can't see directly (unobservable).
     If they disagree, there's another Rust-specific bug.
  3. Print `delta`/`psi` at the divergence frames via
     `PRAATFAN_DEBUG_VITERBI=1` (instrument path_finder similarly to
     commit `d751da7`'s temporary dump).

## Historical context (today's session)

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
