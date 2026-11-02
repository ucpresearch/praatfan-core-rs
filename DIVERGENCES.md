# DIVERGENCES.md — where `praatfan_gpl` differs from Praat, and why

`praatfan-core-rs` targets **bit-accurate parity with Praat**, reached through
parselmouth (Praat 6.1.38) as the oracle. That is the whole design contract:
we are a Rust reimplementation of what parselmouth exposes, plus capabilities
parselmouth lacks (`FormantPath`, speech-referenced normalization, the JSON
pipe, NIST SPHERE input).

This file records the places where we are **knowingly not bit-accurate**, and
the places where the sibling clean-room package
[`praatfan` / `praatfan_rust`](../praatfan-core-clean) (MIT) has deliberately
chosen *different* behaviour from Praat. Those clean-room choices are often
defensible on the merits — but adopting them here would break our contract, so
they are documented rather than applied.

Last reviewed: 2026-11-01 (§1–2 and §4 re-measured after the 0.1.15 AC-HNR
fix); 2026-10-16 (§7 added); §1–6 against praatfan-core-clean at 0.1.10.

---

## Summary table

| # | Topic | Praat 6.1.38 | `praatfan_gpl` (this repo) | `praatfan` 0.1.10 (clean) | Adopted here? |
|---|-------|--------------|----------------------------|---------------------------|---------------|
| 1 | HNR at `r → 1` | `+150.0` dB when `r > 1 - 1e-15` | same | `NaN` | **No** |
| 2 | HNR at `r → 0` | `-150.0` dB when `r <= 1e-15` | same | `-200.0` dB | **No** |
| 3 | HNR clamp | none | none | removed in 0.1.10 (was `[1e-10, 1-1e-10]`) | n/a — never had it |
| 4 | AC `periods_per_window < 3` | command layer refuses; C function does not | accepted silently | `FutureWarning`, error later | **No** — documented instead |
| 5 | `Harmonicity: Get value at time` across a marker | interpolates the marker in | same as Praat (since 0.1.15; held the defined neighbour before) | returns the nearest frame | n/a — we match Praat |
| 6 | HNR bandwidth on wideband (48 kHz) input | effectively band-limited | same as Praat | broadband | **No** |
| 7 | Pipe `spectrogram`: too-short sound, `max_frequency <= 0` | errors | empty result / Nyquist (pipe only) | same as ours (shared spec) | **Yes** — interface choice, values unaffected |
| 8 | Residual gaps found by `scripts/parity_sweep.py` | — | numerical floors + unported query semantics | n/a | documented, see §8 |

---

## 1–2. HNR boundary markers at `r → 1` and `r → 0`

**Praat** (`fon/Sound_to_Harmonicity.cpp`, identical in the `_ac` and `_cc`
functions):

```cpp
if (pitch -> frames [i]. candidates [1]. frequency == 0.0) {
    thy z [1] [i] = -200.0;
} else {
    const double r = pitch -> frames [i]. candidates [1]. strength;
    thy z [1] [i] = ( r <= 1e-15 ? -150.0 : r > 1.0 - 1e-15 ? 150.0 : 10.0 * log10 (r / (1.0 - r)) );
}
```

`Harmonicity::strength_to_hnr` in `src/harmonicity.rs` reproduces this exactly,
including both marker constants.

**praatfan 0.1.10** replaced the markers with a two-branch policy: `r <= 0`
returns `-200.0`, and `1 - r <= f64::EPSILON` returns `NaN`. Its rationale
(recorded in that repo's `BUG-hnr-ac-saturation.md`) is that HNR is genuinely
unbounded as `r → 1`, so `NaN` is the honest float for "undefined", and it
composes correctly downstream because threshold comparisons reject `NaN`.

**Why we do not adopt it.** The argument is sound as *design*, but the markers
are Praat's observable output, and consumers written against Praat treat
`-200` as the unvoiced sentinel and nothing else as special. Substituting
`NaN` would silently change every `numpy` reduction over a contour
(`mean` → `nan`, `max` → `nan`) for callers who were getting a finite number
from Praat. `praatfan_gpl` exists precisely so a caller can swap it for
parselmouth without a numerical diff.

**How much this actually costs, measured.** The markers are nearly unreachable
on real signals, because we implement Praat's reflection of over-1
correlations (`Sound_to_Pitch.cpp`: *"High values due to short windows are to
be reflected around 1"*, `r → 1/r`, applied at all eight of our frame-strength
sites). Hitting `+150` requires `r ∈ (1 - 1e-15, 1]`. On
`tests/fixtures/one_two_three_four_five.wav`, AC method, `time_step=0.01`,
`floor=75`, `silence_threshold=0.1` (measured with 0.1.15; before 0.1.15 our
maxima at 1.0 and 3.0 differed, see §4):

| `periods_per_window` | voiced frames | frames `>= 150` | max HNR (ours / parselmouth) |
|---|---|---|---|
| 1.0 | 78 | 0 | 39.99 / 39.99 dB |
| 3.0 | 102 | 0 | 40.25 / 40.25 dB |
| 4.5 | 104 | 0 | 29.55 / 29.55 dB |

Praat's own boundary constants were never reached. The divergence is therefore
almost entirely theoretical for speech, and real only for synthetic material
(pure tones, calibration tones) — where Praat also reports its own large finite
number rather than a clamp.

### 3. The clamp bug does not exist here

praatfan 0.1.10's headline fix was removing `r.clamp(1e-10, 1.0 - 1e-10)`,
which collapsed every `r >= 1 - 1e-10` onto exactly `99.99999964022885` dB —
62 % of voiced frames on Buckeye speech at `periods_per_window = 1.0`.

**`praatfan_gpl` never had that clamp and was never affected**; this was
established and pinned by `tests/test_harmonicity_saturation.rs` in commit
`8bd4bfb` (see the CHANGELOG entry). Two Praat behaviours we already had are
what prevent it:

- the `r → 1/r` reflection above, so `r > 1` cannot reach `strength_to_hnr`;
- the lag-search bound. `Sound_to_Pitch.cpp` searches maxima over
  `for (i = 2; i < maximumLag && i < brent_ixmax; i ++)`, and for the AC method
  `interpolation_depth = 0.25` with the window doubled internally, so
  `brent_ixmax = 0.25 * nsamp_window`. The search never enters the region where
  the window autocorrelation `r_w(tau)` has decayed toward zero and the
  normalizing quotient becomes ill-conditioned. `src/pitch.rs` reproduces both
  bounds (`maximum_lag`, `brent_ixmax`) exactly.

That second point is also the reason we do not need §4 as a correctness fix.

---

## 4. `periods_per_window < 3.0` for the AC method

**Praat** enforces this in the *command* layer, not the library. From
`fon/praat_Sound.cpp`:

```cpp
POSITIVE (periodsPerWindow, U"Periods per window", U"4.5")
...
Melder_require (periodsPerWindow >= 3.0,
    U"The number of periods per window must be at least 3.0.");
```

The CC command a few lines below defaults to `1.0` and has **no** such check.
`Sound_to_Harmonicity_ac()` itself accepts any value — which is why
parselmouth, calling the C function directly, defaults
`Sound.to_harmonicity_ac(...)` to `periods_per_window=1.0` and accepts it.

So "Praat's AC default is 1.0" is false; it is parselmouth's default, and it
bypasses a check the Praat UI applies. (The claim is in circulation; the
CHANGELOG records the same correction.)

**praatfan 0.1.10** raises a `FutureWarning` below 3.0 and intends to make it
an error, because its lag search *does* reach the ill-conditioned region and
produced correlations up to ~35.9 there.

**Why we do not adopt it.** Our search is bounded (§3), so low `ppw` is not a
correctness hazard here — it is merely a noisier estimate, exactly as it is in
parselmouth. Adding a warning would fire on every existing parselmouth-shaped
call site while changing no value, and making it an error would break the
substitutability that is the point of this package. Our Rust and Python APIs
take `periods_per_window` as a **required** argument with no default, so we do
not propagate parselmouth's questionable 1.0 either.

**What callers should know.** Since 0.1.15 we match parselmouth at every
`periods_per_window`, including values Praat's command layer refuses. Same
fixture and settings as above, voiced frames only:

| `periods_per_window` | max abs diff vs parselmouth, 0.1.15 | before 0.1.15 |
|---|---|---|
| 4.5 (Praat's default) | **6e-11 dB** | 0.014 dB |
| 3.0 (Praat's minimum) | **6e-9 dB** | 5.11 dB |
| 1.0 (Praat refuses) | **1e-6 dB** | 11.05 dB |

Pearson r is 1.000000 at all three, and voicing agrees frame for frame. The
16 kHz, 44.1 kHz, 48 kHz and stereo fixtures are also exact (max 0.0000 dB at
3.0; they were off by up to 12.4 dB at 16 kHz).

**The earlier gap was our bug, not the parameter.** Praat's second pass hands
`NUM_interpolate_sinc` / `NUMimproveExtremum` only lags `-brent_ixmax ..
brent_ixmax` of the correlation (`Sound_to_Pitch.cpp`: `r [offset + 1]`, size
`2*brent_ixmax + 1`), and the interpolator clips its depth to that slice's
edges. We passed the whole buffer (lags out to `nsamp_window`), so SINC700 used
a deeper kernel. That moved `r` by about 1e-4, and near `r = 1`
`10*log10(r/(1-r))` turned that into several dB. Only AC_GAUSS is affected
(its `brent_ixmax` is a quarter of the window), and short windows most. Earlier
versions of this file blamed the FFT; that was wrong (0.1.14 swapped the FFT
for Praat's FFTPACK and nothing moved).

**Still use 4.5 for the AC method**, for the estimate's sake rather than for
parity: it is Praat's own default, and shorter windows give noisier HNR in
Praat and here alike.

---

## 5. Interpolating `Harmonicity` across marker frames

> **Retired in 0.1.15** (user decision, 2026-11-01: maximal Praat
> compliance). `get_value_at_time` is now Praat's `Vector_getValueAtX`,
> markers included; use `get_value_at_frame` (None for -200) to detect
> unvoiced frames. The text below is the earlier rationale, kept for history.

This one is a divergence we **already** carried before 0.1.10, in a third
direction from both Praat and the clean package. It is recorded here so it is
a known deviation rather than an accident.

**Praat.** `Harmonicity: Get value at time...` is
`Vector_getValueAtX (me, time, 1, interpolation)`
(`fon/praat_uvafon_init.cpp`) — a plain `Vector` interpolation with no
awareness that `-200` is a sentinel. Measured on
`one_two_three_four_five.wav` at a voiced→unvoiced boundary (frame 33 =
12.361 dB at t = 0.3910, frame 34 = −200 dB at t = 0.4010), parselmouth's
`get_value`:

| t | parselmouth |
|---|---|
| 0.39350 | −35.632 |
| 0.39600 | −106.408 |
| 0.39850 | −170.890 |

Those intermediate values are arithmetic, not measurements. (Praat *does*
exclude `-200` from `Get mean`, `Get standard deviation` and `Get quantile`,
via `Harmonicity_getSoundingValues` — so Praat's own aggregate queries treat
it as a sentinel while its point query does not.)

**Here.** `Harmonicity::get_value_at_time` maps every value `<= -199` to `NaN`
and calls `Interpolation::interpolate_with_undefined`, which returns the
*defined* neighbour when exactly one of the two is undefined — so it holds the
last real reading flat across the gap:

| t | `praatfan_gpl` |
|---|---|
| 0.39350 | 12.361 |
| 0.39600 | 12.361 |
| 0.39850 | 12.361 |

**praatfan 0.1.10** now excludes markers too, but returns the frame the query
sits *nearest* — so it would report 12.361 at t = 0.3935 and `-200` at
t = 0.3985 — and degrades Catmull-Rom to the two central taps when any of the
four is a marker (its negative outer weights can otherwise cancel and blow up
the quotient).

**Assessment.** All three behaviours differ, none is obviously right, and
Praat's is the one that is hardest to defend on the merits — it reports a
blend of a measurement and a sentinel as though it were dB. We keep ours: it
never invents a value between a reading and a sentinel, and unlike the
nearest-frame rule it never hands back `-200` for a time at which a real
reading exists nearby.

**This is nonetheless a parity gap.** Anything comparing our
`get_value_at_time` against parselmouth's `get_value` near a voicing boundary
will disagree, by up to ~180 dB. Frame-level access
(`values()`, `get_value_at_frame`) is unaffected and matches Praat exactly.
Callers needing byte-parity with Praat at arbitrary times should read the
frames and interpolate themselves.

---

## 6. HNR bandwidth on wideband input

praatfan-core-clean investigated a 48 kHz fixture where its AC output sat at
Pearson 0.638 against parselmouth, and concluded (with synthetic ground truth
at known harmonic-to-noise ratios) that the disagreement is **definitional**:
its estimator responds to aperiodic energy across the full band, while Praat's
is effectively band-limited and barely responds to noise near Nyquist —
14.96 dB where truth was 9.54 dB. Its measurements put the clean package
closer to truth in every cell, and it reports the broadband reading as the one
consistent with Boersma (1993) as published.

**We match Praat, including here.** That is the contract, and on this axis the
contract means we inherit a reading that the published algorithm arguably does
not support on wideband material.

**Practical guidance.** The disagreement is a monotone function of how much
12–14 kHz energy a frame carries, so it is confined to material with real
high-frequency aperiodic content (sibilant frication) analysed at its native
high rate. If HNR is the measurement of interest on 48 kHz recordings,
**resample to 16–24 kHz first** — the clean repo measured Pearson 0.953 at
24 kHz and 0.969 at 16 kHz on the same content, i.e. the two definitions
converge once the disputed band is gone. That is also good practice for the
pitch floor/ceiling this analysis implies.

We take no position on *how* Praat's implementation comes to be band-limited;
that question belongs to this repo's Praat-source-informed side and not to the
clean-room record.

---

## 7. Pipe `spectrogram` edge cases

The pipe's `spectrogram` analysis (0.1.12) follows a request/response spec
shared with `praatfan-open-pipe`. That spec departs from Praat's
`Sound: To Spectrogram` command in two edge cases. Both concern the
interface, not the values:

- **Sound shorter than the analysis window.** Praat refuses ("Your sound is
  too short"). The pipe returns `ok: true` with `n_times: 0`, `times: []`,
  `power_b64: ""`, and still reports `freqs`/`n_freqs`. The library's
  `Spectrogram::from_sound` behaves the same way (zero frames, full frequency
  grid).
- **`max_frequency <= 0`.** Praat rejects it ("must be greater than 0"). The
  pipe treats it as Nyquist. Above Nyquist, Praat and the pipe both clamp to
  Nyquist.

All other requests give Praat's grid exactly and Praat's values to ~1e-15
relative (`scripts/compare_pipe_spectrogram.py`).

**Not a shared value contract.** The clean engine's spectrogram is a plain
STFT by design, so its grid and values differ from ours for the same request.
Only the response shape is common. Anyone who needs Praat's exact
spectrogram should use this engine.

---

## 8. What the parameter sweep still finds (0.1.15)

`scripts/parity_sweep.py` runs every analysis and query over a parameter
grid on 13 signals (the speech fixture at 8/11.025/16/24/44.1 kHz, 30 and
80 ms excerpts, white noise, a vibrato harmonic tone, silence, silence-padded
speech, clipped speech, speech + 0.2 DC) against parselmouth 0.4.7. Measured
2026-11-01 after the 0.1.15 sweep fixes (CHANGELOG). What is left falls in
two groups.

**Numerical floors** (inputs where the last bits decide the answer; not
fixable without reproducing x87 `long double` and glibc's `exp`/`sin`
bit for bit):

- Pitch F0: most frames bit-identical; the rest agree to ~1e-8 Hz, except
  one or two frames per file on very flat, high-correlation peaks (up to
  ~1e-5 Hz, ~6e-8 relative). Strengths agree to ~1e-15 everywhere, so the
  correlation is right; Brent's stopping point inside its ~sqrt(eps) bracket
  is steered by last-bit `sin`/`cos` differences. Measured: the `libm` crate
  differs from glibc 2.43 in the last bit on ~1.6% of the sinc's arguments
  (glibc itself is not correctly rounded: 0.07%), and glibc picks FMA or
  non-FMA paths per CPU, so Praat's own last bits here depend on the host.
- Frames that contain only numerical noise: digital silence after
  resampling (FFT ringing ~1e-13), or a constant DC stretch (centring
  residue ~1e-16). Formants of such frames are noise-determined in Praat as
  in ours; formant frames inside 0.2 s of zero padding differ by up to
  248 Hz.
- Spectrogram bins 15-20 orders of magnitude below the frame's peak
  (relative error up to 5e-7; absolute ~1e-33).

FormantPath on Hillenbrand 1995 has no remaining gap (all 1609 tokens
within 5e-7 Hz on F1-F3, since 0.1.15).

**Interface choice:** for sounds too short for the analysis, Praat refuses
("shorter than window length", "minimum pitch must not be less than ...");
pitch, HNR and intensity here return an empty object instead (kept by user
decision, 2026-11-01). The query helpers that used to differ (Intensity and
Harmonicity mean/min/max, Harmonicity point queries, Sound value at time,
pre-emphasis normalization) follow Praat since 0.1.15.

## Related documents

- `CHANGELOG.md` — the `8bd4bfb` entry, with the Buckeye measurements and the
  `periods_per_window` correction.
- `PRAAT.md` — inconsistencies *within* Praat itself (e.g. `Sound → To Formant
  (burg)` vs `Sound → To LPC (burg) → To Formant`).
- `CLEANROOM.md` — why this repo may read Praat's source and the sibling
  package may not.
- `tests/test_harmonicity_saturation.rs` — the regression guard for §3.
- `../praatfan-core-clean/BUG-hnr-ac-saturation.md` — the clean-room package's
  own account of §1–4 and §6.
