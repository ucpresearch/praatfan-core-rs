# Changelog

Notable changes to `praatfan-core-rs` (the crate) and `praatfan_gpl` (its
Python bindings).

## [0.1.15] - unreleased

Bug-fix release: AC harmonicity now matches Praat exactly, pitch and CC
harmonicity lose their last residual error, and a parameter sweep against
parselmouth (`scripts/parity_sweep.py`: 13 signals, 8-44.1 kHz, short, silent,
noisy, DC-offset; every analysis and query over a grid of parameters) found
and closed a series of gaps away from the default parameters. Query helpers
now follow Praat too (*Changed*); everything else is additive (*Added*).

### Fixed (parameter-sweep gaps)

- **Formant (burg)**: no more parameter clamping (window length was clamped
  to 0.01-0.1 s, ceiling to >= 1000 Hz and <= Nyquist, formants to <= 10), so
  e.g. a 0.005 s or 0.2 s window gave a different frame grid. The sound is
  now resampled to 2 x ceiling *upwards* too, as Praat does (an 8 kHz file
  with a 5500 Hz ceiling is analysed at 11 kHz; it used to be analysed at
  4000 Hz). Sounds shorter than the window are analysed as one frame, as in
  Praat (they returned no frames). Roots are kept on Praat's
  `50 <= f <= nyquist - 50` rule with no bandwidth test.
- **Resampled sounds lost Praat's `x1`.** Praat keeps `xmin`/`xmax` and
  centres the new samples (`x1 = (xmin + xmax - (n-1)/fs) / 2`); we assumed
  `x1 = xmin + dx/2`, which shifted every formant frame by up to a quarter
  sample whenever duration x 2 x ceiling is not an integer (most files at
  11025/44100 Hz, the Hillenbrand 16 kHz files at 11 kHz). `Sound` now keeps
  Praat's `x1` (exposed as `Sound.x1` in Python), computed with Praat's exact
  arithmetic. This also fixed the real-stereo FLAC fixture's one 2.2 Hz
  formant frame.
- **FormantPath candidate framing** now follows `Sound_into_LPC` exactly
  (frame start from the window *duration*, `round(duration x fs)` samples,
  ties-up rounding), and the shared frame grid uses Praat's
  `Sampled_shortTermAnalysis` expression on the middle-ceiling sound (the
  standalone-Formant `t1` formula is algebraically equal but can differ by
  an ulp, which moves frames centred on a sample tie). With the `x1` fix, Hillenbrand 1995 FormantPath parity
  (4827 values, framing fixes alone): within 0.01 Hz 36.8% -> 99.8%,
  within 1 Hz 98.82% -> 99.83%, P99 1.22 -> 0.000 Hz,
  median 0.019 -> 0.0000 Hz.
- **FormantPath path finder**: frame formant counts are Praat's real counts,
  not the NaN-padded length (frequency-change and Q terms were divided by 4
  where Praat divides by 3), and the stress model keeps under-determined
  tracks as Praat does (zero-model chi-squared, negative degrees of
  freedom) instead of making the whole stress undefined. With the framing
  fixes above, **all 1609 Hillenbrand tokens now match Praat on F1-F3 within
  5e-7 Hz** (max error was 50.8 Hz on w32aw).
- **Resample**: an exact doubling now uses Praat's FFT upsampler
  (`Sound_upsample`) instead of sinc interpolation (was off by up to 0.046);
  sinc interpolation takes Praat's cubic branch where the depth is clipped to
  2 at the edges (the second sample from each end was off by up to 7e-6).
  All directions now agree with Praat to <= 5e-13.
- **CC harmonicity / CC pitch**:
  - the silence gate measured the local peak over the whole window instead of
    +/- half a longest period, so with `periods_per_window != 1` frames Praat
    calls silent came out voiced (HNR CC off by ~210 dB on several frames per
    file at ppw 1.5-2);
  - the correlation is now computed as Praat does, by direct summation with a
    running energy term (compensated sums for Praat's `long double`), not by
    FFT. FFT rounding ripple made fake maxima on constant stretches (digital
    silence, DC offset), giving +48..+85 dB where Praat has -200 dB. CC pitch
    is ~2.6x slower and CC HNR ~1.2x (10-minute 24 kHz file, 20 threads:
    1.06 -> 2.8 s and 2.8 -> 3.3 s).
- **AC pitch / AC HNR frames were one sample short**: the frame copy treated
  Praat's inclusive `endSample` as exclusive, dropping the last (small-weight)
  sample. Strengths were off by up to 1e-5 and F0 by up to 3e-5 Hz (7e-4 Hz
  on 8 kHz input). Now strengths agree to ~1e-15 and F0 to ~3e-8 Hz, with
  many frames bit-identical.
- **Pitch peak refinement in Praat's coordinates**: the first-pass sinc
  strength and the Brent refinement now run on the 1-based index of the
  `-brent_ixmax..=brent_ixmax` slice, as Praat does (Brent's tolerance scales
  with |x|), and Brent uses Praat's machine epsilon 2^-53 (`dlamch`), not
  `f64::EPSILON`. Most CC frames are now bit-identical to Praat. Gaussian
  windows (AC_GAUSS, formant) use Praat's operation order.
- **Pitch with `silence_threshold = 0`**: the unvoiced candidate's strength
  was 0 instead of Praat's `voicing_threshold`, flipping voicing on ~25% of
  frames (F0 off by up to 289 Hz on octave errors).
- **Pitch/HNR on an all-zero sound** returned no frames; Praat returns the full
  frame grid, all unvoiced (HNR -200 dB).
- **Pitch and Formant `get_value_at_time`** follow Praat's
  `Sampled_getValueAtX`: frames are converted to the unit *before*
  interpolating (semitone/mel/ERB queries were off by up to 1 unit), the
  nearer frame decides definedness, and a missing farther frame returns the
  nearer value.
- **Unit formulas** are Praat's: ERB was Glasberg-Moore instead of
  `11.17 ln((f+312)/(f+14680)) + 43` (off by ~0.5), Bark used 600 Hz instead
  of 650 Hz, semitones `(12/ln 2) ln(f/ref)`. Formant bandwidths in Bark are
  the Bark width of `[f - b/2, f + b/2]`, as in Praat.
- **Intensity**: frame centres round as Praat does (`(x - x1)/dx + 1`, ties
  up), fixing 0.001-0.01 dB on one or two frames per file at 8/16 kHz; frame
  means use Praat's long-double `NUMmean`, so constant stretches give exactly
  -300 dB (were about -201 dB). `get_value_at_time` follows
  `Vector_getValueAtX` (Praat's nearest/linear/cubic/sinc kernels).
- **Spectrum**: FFT via the vendored FFTPACK (as Praat; was rustfft), and
  centre of gravity / moments ported exactly (long-double sums, `m3/m2^1.5`,
  `m4/m2^2 - 3`); an all-zero spectrum gives NaN (Praat: undefined), not 0.
  All moments and band energies now agree to <= 5e-13 relative.
- **Spectrogram**: FFT via FFTPACK, as Praat.
- **`Sound.extract_part`** ports `Sound_extractPart`: Praat's sample range
  (one sample fewer in general), zero-filled virtual samples, the
  `relative_width` margin, and Praat's window formulas over the whole part
  (`gaussian` = Praat Gaussian1, `kaiser` = Kaiser1).

### Changed (Praat compliance; observable)

- **`Intensity.mean()/min()/max()`** now return Praat's `Get mean`
  (energy-averaged, interpolated contour) and `Get minimum/maximum`
  (parabolic) over the whole sound; they were plain dB frame statistics
  (mean off by up to 15 dB on speech). New Praat queries with time ranges:
  `get_average(from, to, averaging)` (energy/sones/dB),
  `get_minimum/get_maximum/get_time_of_minimum/get_time_of_maximum(from, to,
  interpolation)` (none/parabolic/cubic/sinc70/sinc700).
- **`Harmonicity.mean()/standard_deviation()`** follow Praat (sounding
  frames only, as before, now with Praat's arithmetic and edge cases);
  **`min()/max()`** follow Praat's `Get minimum/maximum` (parabolic, and
  -200 frames take part, as in Praat). New `get_mean`,
  `get_standard_deviation`, and the extremum getters with time ranges.
- **`Harmonicity.get_value_at_time`** is now Praat's plain
  `Vector_getValueAtX`: -200 frames are interpolated in, as in Praat
  (DIVERGENCES §5 retired; it used to hold the defined neighbour).
- **`Sound.get_value_at_time`** defaults to Sinc70 interpolation, Praat's
  default (was linear); Python takes `interpolation=`.
- **`Sound.pre_emphasis/de_emphasis`** (Python and WASM) rescale the result
  to a 0.99 peak, as Praat's commands and parselmouth do; Python
  `normalize=False` returns the raw filter. The Rust methods stay raw;
  `Sound::scale_peak` is Praat's `Vector_scale`.
- Pipe: spectrum moments of an all-zero signal are `null` (Praat:
  undefined; were 0); segment extraction follows `extract_part` above.

### Added

- `Formant::from_sound_burg_f` (Rust) and a float `max_num_formants` in
  Python `Sound.to_formant_burg` (e.g. 5.5, as in Praat: LPC order
  `round(2 x n)`).
- `Sound::get_value_at_time_interpolated` (Rust); Python
  `Sound.get_value_at_time(time, interpolation="linear")` now accepts
  `nearest`/`linear`/`cubic`/`sinc70`/`sinc700` (Praat's default is sinc70;
  ours stays linear for compatibility).
- Python `Sound.x1`; `Spectrum::get_central_moment` (Rust); window-shape
  aliases `square`, `bartlett`, `welch` (Praat's spectrogram names).
- `scripts/parity_sweep.py`: the sweep above.

### Fixed

- **AC harmonicity (`To Harmonicity (ac)`) differed from Praat by up to
  5–12 dB on a few frames**, at high HNR (`r` near 1) and short windows:
  5.11 dB at `periods_per_window = 3` on the 22.05 kHz fixture, 12.4 dB on the
  16 kHz one, 1.35 dB on stereo, 11.05 dB at `periods_per_window = 1`. Praat
  refines each pitch candidate on lags `-brent_ixmax..=brent_ixmax` of the
  correlation only, and its sinc interpolator clips its depth to that slice;
  we passed the whole buffer, so the AC_GAUSS method's SINC700 used a deeper
  kernel and moved `r` by ~1e-4. Now max |ΔHNR| vs parselmouth is 6e-11 dB at
  4.5, 6e-9 dB at 3.0 and 1e-6 dB at 1.0, with identical voicing, and the
  16 kHz/44.1 kHz/48 kHz/stereo fixtures are exact too. (The same applies to
  the multichannel path and to the latent AC_HANNING case where a candidate
  above `0.3/dx` is refined with SINC700.)
- **Pitch and CC harmonicity used Praat 6.2's sinc window depth**
  (`maxDepth + 0.5`) instead of 6.1.38's asymmetric depths, which is what
  parselmouth 0.4.x ships. Pitch max error vs parselmouth 0.0018 -> 0.0000 Hz
  (mono and stereo); CC HNR max 0.0019 -> 0.0000 dB.

### Documentation

- DIVERGENCES §4: the AC-HNR gap was this bug, not the parameter and not the
  FFT; table re-measured. §1–2 table re-measured (maxima now identical to
  parselmouth). DEVELOPMENT.md accuracy summary updated (CC HNR and stereo
  pitch were also listed below their measured accuracy).

## [0.1.14] - 2026-11-01

**Last-bit output changes** across most analyses, with parity against
parselmouth unchanged (formants F1–F3 159/159 within 0.0 Hz, pitch 104/104
within 0.0018 Hz, intensity/spectrum/spectrogram exact, AC/CC HNR as before,
h95 FormantPath 1609/1609 rows unchanged). The Python and pipe APIs are
unchanged.

### Changed

- **Host-independent output.** All transcendental maths goes through the
  pure-Rust `libm` crate instead of the platform C library. Previously the
  last bits depended on which code path glibc picked for the CPU (FMA/AVX2),
  so the same wheel or binary could give slightly different numbers on
  different machines. Output is now byte-identical across those paths at
  every tested sample rate (8–48 kHz). A `clippy.toml` keeps std
  transcendentals out of the library.
- AC pitch and AC HNR compute their autocorrelation with the vendored
  FFTPACK real FFT (as Praat does) instead of rustfft, whose twiddle tables
  depend on the host libm. AC pitch is ~17–21% slower and AC HNR ~7–9%
  slower (on a 10-minute file at 20 threads: +0.06 s and +0.16 s).
- WASM and native results are now closer, about 1e-12 relative for pitch
  (was ~1e-9). They are still not bit-identical (spectrogram power ~1e-11).

### Fixed

- Two general-radix bugs in the vendored FFTPACK (from the c2rust
  translation): the forward transform panicked at lengths such as 9, 18 and
  27, and the inverse was wrong for any length with an odd factor. Every
  current caller uses power-of-two lengths, so no output was affected. A new
  test checks every length from 1 to 700 against a direct DFT.

### Documentation

- DIVERGENCES §4: the AC-HNR differences at low `periods_per_window` are
  **not** caused by the FFT, as previously stated. They are unchanged with
  FFTPACK, and their cause is still unknown. (Found and fixed in 0.1.15.)

## [0.1.13] - 2026-10-19

First PyPI release since 0.1.11 (0.1.12 was GitHub-only), so the PyPI wheels
also pick up everything listed under 0.1.12.

### Changed

- `desphere` (NIST SPHERE fallback decoder) upgraded from v0.1.0 to v0.1.1
  (decoder robustness fixes). Ordinary formats are unaffected.

## [0.1.12] - 2026-10-16

Pipe and WASM additions for the R package. **No analysis output changes**
for any existing entry point; the Python API is unchanged (this version is
not published to PyPI).

### Added

- **Pipe `spectrogram` analysis**: Praat's "Sound: To Spectrogram"
  defaults, `window_shape` `"gaussian"`/`"hanning"`, `dtype`
  `"f64le"`/`"f32le"`, optional `start_time`/`end_time` (filter the returned
  frames of the full-sound grid). The result carries `times`, `freqs`, the
  steps actually used, and base64 power (time-major, linear Pa²/Hz). Matches
  parselmouth to ~1e-15 relative with an identical grid.
  `scripts/compare_pipe_spectrogram.py` checks it. The schema is shared with
  `praatfan-open-pipe`; only the shape is shared, not values.
- Pipe response `"warnings"` (omitted when empty).
- In-memory pipe transports: `pipe::handle_request_bytes` and
  `pipe::handle_request_samples`, plus `Sound::from_bytes_channels`. New Cargo
  features: `decode` (decoders without filesystem access) and `pipe-core`
  (the pipe without file I/O, builds for wasm32).
- **`praatfan-gpl-wasm-v8.zip`** release asset: a `--target no-modules`
  bundle exporting `version()`, `pipe_request_bytes`, and
  `pipe_request_samples`, for embedded V8 hosts (R's V8 package,
  mini-racer). Needs V8 ≥ 9.6. Built and checked by `wasm/v8/build.sh`:
  - builder paths stripped;
  - relaxed SIMD rejected;
  - bare-`node:vm` smoke test against the native pipe;
  - praatfan-core-clean's response-shape checker.
- **`SHA256SUMS`** release asset covering every other asset.
- `LICENSE` (GPL-3.0) at the repository root.
- `DIVERGENCES.md` §7: the pipe spectrogram's two interface departures from
  Praat (a too-short sound gives an empty result, `max_frequency <= 0` means
  Nyquist).

### Changed

- `Spectrogram::from_sound` on a sound shorter than the window still reports
  the frequency grid, with zero frames. Previously it reported no bins.
- CI pins `wasm-pack` 0.14.0 `--locked`; 0.15.0 fails to install.

## [0.1.11] - 2026-09-23

Multi-threaded frame analysis.  **No analysis output changes**: every value
is bit-identical to 0.1.10 at any thread count, and serial builds (including
WASM) are bit-identical to 0.1.10's.

### Changed

- **Per-frame work runs in parallel** under the `parallel` feature (on for the
  Python bindings and, now, the `pipe` feature; ignored on `wasm32`):
  pitch AC/CC (and therefore HNR AC/CC), formant frames, formant ceilings
  (multi-ceiling and FormantPath), intensity, spectrogram, and the sinc step
  of resampling (the FFT low-pass stays serial).  Frames are mapped in order
  with unchanged per-frame arithmetic.  Thread count honours
  `RAYON_NUM_THREADS`.  On a 623 s, 16 kHz file with 20 threads: pitch AC
  1.7 → 0.3 s, pitch CC 6.6 → 1.1 s, HNR AC 20 → 2.7 s, formant 7.0 → 2.3 s,
  FormantPath (9 ceilings) 92 → 17 s.
- **Python: every `Sound.to_*` analysis releases the GIL** (previously only
  some did).
- The FFT wrapper reuses rustfft scratch (`process_with_scratch`) instead of
  allocating on every transform.

### Fixed

- **Deadlock in forked children.**  Earlier releases ran pitch/HNR on rayon's
  global pool, which does not survive `fork()`: after one analysis in the
  parent, any analysis in a Python `multiprocessing` fork child (the Linux
  default) hung forever.  The crate now owns a pool tagged with the process
  id and rebuilds it in a forked child.  A caller's own rayon pool is still
  used when the call is made from inside it.

### Added

- `examples/bitdump.rs` (exact-bit dump of every analysis, for serial vs
  parallel comparison), `tests/test_parallel_bit_identity.rs` (pools of
  1/2/3/8 threads), `python/tests/test_fork_safety.py`.

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
