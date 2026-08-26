# Changelog

Notable changes to `praatfan-core-rs` (the crate) and `praatfan_gpl` (its
Python bindings).

## [Unreleased]

### Added

- `tests/test_harmonicity_saturation.rs` — a guard against the AC-harmonicity
  saturation bug that was found in the sibling clean-room package
  (`praatfan` / `praatfan_rust`, praatfan-core-clean).  There, the correlation
  was clamped to `[1e-10, 1 - 1e-10]` before `10*log10(r/(1-r))`, so every
  correlation at or above `1 - 1e-10` returned exactly 99.99999964022885 dB —
  62% of voiced frames on Buckeye conversational speech at
  `periods_per_window = 1.0`.

  **`praatfan_gpl` was not affected**, and no behaviour changed in this repo.
  It already does what Praat does at both places that matter: the pitch frame
  routine reflects an over-1 correlation around 1 (`r -> 1/r`, Praat's "high
  values due to short windows are to be reflected around 1"), and it bounds the
  correlation-maximum search by `brent_ixmax = interpolation_depth *
  nsamp_window` so the search never reaches the lags where the window
  autocorrelation has decayed to nothing.  `Harmonicity::strength_to_hnr`
  already reported Praat's own boundary markers (-200 dB unvoiced, -150 dB at
  `r <= 1e-15`, +150 dB at `r > 1 - 1e-15`) rather than a clamp.

  The new test pins that property — noise must not read as harmonic, a pure
  tone must read high but finite and unclamped, and a mixed signal's high
  values must not cluster on a single number — so a future change to the pitch
  strength path cannot reintroduce the clamp behaviour here.

### Verified

Measured against parselmouth / Praat 6.1.38 on a 10 s Buckeye excerpt
(`s0101a.wav`, `time_step=0.01`, `pitch_floor=75`, `silence_threshold=0.1`,
`periods_per_window=4.5` — Praat's own default, and its enforced minimum for
"To Harmonicity (ac)..." is 3.0):

- 399 of 989 frames bit-exact with Praat;
- Pearson r = 0.9935 over the remaining frames;
- no repeated saturation constant at any `periods_per_window` tested
  (1.0 and 4.5).

The residual disagreement is concentrated on frames that legitimately approach
`r = 1` (the calibration tone at the head of the Buckeye file): Praat reports
126.70 dB there and this crate 52.85 dB.  Both mean "perfectly periodic"; the
gap is the precision of the sinc interpolation of a degenerate correlation, not
a clamp.

### Note on `periods_per_window` for the AC method

Measured against parselmouth / Praat 6.1.38, because a claim that "Praat's
default for To Harmonicity (ac) is 1.0" is in circulation and is wrong:

- Praat's `Sound: To Harmonicity (ac)...` form defaults `periods per window`
  to **4.5** and *refuses* anything below **3.0** ("Number of periods per
  window must be at least 3.0") at every duration.
- parselmouth's Python method `Sound.to_harmonicity_ac(...)` defaults it to
  **1.0** and calls Praat's C function directly, bypassing that check.  Any
  measurement showing "Praat accepting 1.0" is measuring parselmouth's binding,
  not Praat.
- 1.0 *is* the correct Praat default for the **cc** method.

This bears on the "praatfan_gpl returns zero frames on files under ~120 ms"
report.  That threshold is Praat's own: the AC method doubles the window
internally (`dt_window = 2 * periods_per_window / pitch_floor` = 0.12 s at 4.5
periods and a 75 Hz floor), and real Praat *raises* on shorter input ("To
analyse this Sound, 'minimum pitch' must not be less than ...") rather than
analysing it.  `praatfan_gpl`'s deviation is only how it signals that refusal —
an empty result instead of an exception.  At durations Praat accepts, frame
counts match Praat exactly (150 Hz tone, 16 kHz, `time_step = 0.005`: 1 / 7 /
17 / 37 frames at 120 / 150 / 200 / 300 ms).
