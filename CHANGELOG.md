# Changelog

Notable changes to `praatfan-core-rs` (the crate) and `praatfan_gpl` (its
Python bindings).

## [0.1.10] - 2026-08-26

Adds the JSON pipe binary; everything else is documentation and a regression
guard.  **No analysis output changes** — every value this release produces is
bit-identical to 0.1.9.

### Added

- **`praatfan-gpl-pipe`** — a JSON stdin/stdout batch-analysis binary, behind a
  new `pipe` cargo feature (optional `serde`/`serde_json` with
  `float_roundtrip`; implies `file-io`, so default, WASM and Python builds are
  unchanged).  One request loads the audio once and runs any number of
  analyses: `pitch_ac`, `pitch_cc`, `formant_burg`, `intensity`,
  `harmonicity_ac`, `harmonicity_cc`, `spectral_moments`, `band_energy`.

  ```bash
  cargo build --release --features pipe --bin praatfan-gpl-pipe
  # or: cargo install --path . --features pipe
  ```

  The wire protocol is **identical** to praatfan-core-clean's
  `praatfan-open-pipe`, so the two binaries are swappable without workflow
  changes.  The values differ — that one is the clean-room engine, this one is
  the Praat-bit-accurate engine.  Three deliberate contract points: an omitted
  `channel` on a multi-channel file is a hard error rather than a mixdown
  (matching the clean pipe's mono enforcement); `pitch_cc` runs
  `PitchMethod::FccAccurate`, this crate's only CC method; absent
  formants/moments serialize as JSON `null`, never bare `NaN`.  Unknown
  analysis types and typo'd parameter keys are hard errors.

  Release CI gains a 6-platform matrix job attaching
  `praatfan-gpl-pipe-<os>-<arch>` binaries to GitHub releases.

- `DIVERGENCES.md` — a standing record of every place where this crate is
  knowingly *not* bit-accurate with Praat, and of the places where the sibling
  clean-room package (`praatfan` / `praatfan_rust`, MIT, praatfan-core-clean)
  has deliberately chosen different behaviour that this crate does **not**
  adopt.  Written after reviewing praatfan-core-clean 0.1.10, whose release
  changes the AC-harmonicity boundary representation.

  Nothing in this repo changed.  The six entries, each verified against Praat
  6.1.38 source and parselmouth output rather than asserted:

  1. **HNR at `r → 1`.**  Praat reports `+150.0` dB for `r > 1 - 1e-15`; we do
     the same; praatfan 0.1.10 now reports `NaN`.  Not adopted — `NaN` turns
     every downstream `numpy` reduction over a contour into `NaN` for callers
     who were getting a finite number from Praat.
  2. **HNR at `r → 0`.**  Praat reports `-150.0` dB for `r <= 1e-15`; we do the
     same; praatfan 0.1.10 folds this into its `-200.0` unvoiced marker.  Not
     adopted, same reason.

     Both markers were measured to be unreachable on speech: 0 frames at or
     above 150 dB on `one_two_three_four_five.wav` at
     `periods_per_window` 1.0, 3.0 and 4.5.
  3. **The clamp.**  praatfan 0.1.10's headline fix removes
     `r.clamp(1e-10, 1 - 1e-10)`.  Not applicable — we never had it, as pinned
     by `tests/test_harmonicity_saturation.rs` in `8bd4bfb`.
  4. **AC `periods_per_window < 3.0`.**  praatfan 0.1.10 raises a
     `FutureWarning` and intends an error.  Not adopted: our lag search is
     bounded by Praat's own `brent_ixmax`, so low `ppw` is a noisier estimate
     here rather than the ill-conditioned regime that motivated the warning
     there.  Measured against parselmouth on the standard fixture (voiced
     frames, AC): max abs diff **0.014 dB / Pearson 1.000000** at ppw 4.5
     (Praat's default), **5.11 dB / 0.996556** at 3.0 (Praat's minimum),
     **11.05 dB / 0.988150** at 1.0 (which Praat's command layer refuses).
     Use 4.5.
  5. **Interpolating across marker frames** — the one entry that is a genuine
     pre-existing parity gap *here*, not just a difference to note.  Praat's
     `Harmonicity: Get value at time...` is a plain `Vector_getValueAtX`, so it
     blends the `-200` sentinel into neighbouring readings: across the
     boundary at t = 0.391 s (12.361 dB) → t = 0.401 s (-200 dB), parselmouth
     returns -35.6 / -106.4 / -170.9 dB at the quarter points.  We treat the
     sentinel as undefined and hold the defined neighbour (12.361 dB at all
     three).  praatfan 0.1.10 excludes markers too but returns the *nearest*
     frame, a third answer again.  We keep ours, but callers needing
     byte-parity with Praat at arbitrary times should read `values()` and
     interpolate themselves; frame-level access already matches Praat exactly.
  6. **HNR bandwidth on 48 kHz input.**  praatfan-core-clean's synthetic
     ground-truth work indicates Praat's HNR is effectively band-limited and
     under-responds to aperiodic energy near Nyquist.  We match Praat, so we
     inherit that.  Resample to 16–24 kHz when HNR is the measurement of
     interest on wideband recordings.

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

### Changed

- Release assets for the pipe binary follow the shared cross-package naming
  convention `praatfan-gpl-pipe-<os>-<arch>[.exe]`, using Rust target
  spellings (`x86_64` / `aarch64`, not `x64` / `arm64`).  The v0.1.9 assets
  were re-uploaded under the new names.
- `Harmonicity::from_sound_ac`'s documentation no longer says
  `periods_per_window` is "typically 1.0".  That is parselmouth's default, not
  Praat's — see the note at the end of this entry.  1.0 remains correct for
  `from_sound_cc`.

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
