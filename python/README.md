# praatfan-gpl

Praat-compatible acoustic analysis in Python, powered by Rust.

This package provides exact reimplementations of Praat's acoustic analysis algorithms, designed to produce bit-accurate output matching Praat/parselmouth. The
remaining last-bit differences, on inputs that contain only numerical noise,
are listed in [DIVERGENCES.md](../DIVERGENCES.md) §8.

## Installation

```bash
pip install praatfan-gpl
```

### Building from source

Requires Rust and maturin:

```bash
cd python
pip install maturin
maturin develop --release
```

## Quick Start

```python
import praatfan_gpl as pc

# Load an audio file
sound = pc.Sound.from_file("speech.wav")

# Or create from numpy array
import numpy as np
samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
sound = pc.Sound(samples, 44100.0)

# Compute pitch (F0)
pitch = sound.to_pitch(0.01, 75.0, 600.0)
print(f"Pitch at 0.5s: {pitch.get_value_at_time(0.5, 'hertz', 'linear')}")

# Get all pitch values as numpy array
f0_values = pitch.values()  # NaN for unvoiced frames
times = pitch.times()

# Compute formants
formant = sound.to_formant_burg(
    time_step=0.01,
    max_num_formants=5,
    max_formant_hz=5500.0,  # Use 5000 for female speakers
    window_length=0.025,
    pre_emphasis_from=50.0
)

# Get F1, F2 at specific time
f1 = formant.get_value_at_time(1, 0.5, "hertz", "linear")
f2 = formant.get_value_at_time(2, 0.5, "hertz", "linear")

# Get all F1 values
f1_values = formant.formant_values(1)  # numpy array

# Compute intensity
intensity = sound.to_intensity(100.0, 0.01)
db_values = intensity.values()

# Compute spectrum (single-frame FFT)
spectrum = sound.to_spectrum(fast=True)
cog = spectrum.get_center_of_gravity(2.0)
std = spectrum.get_standard_deviation(2.0)

# Compute spectrogram
spectrogram = sound.to_spectrogram(
    effective_analysis_width=0.005,
    max_frequency=5000.0,
    time_step=0.002,
    frequency_step=20.0,
    window_shape="gaussian"
)
spec_data = spectrogram.values()  # 2D numpy array [freq, time]

# Compute harmonicity (HNR)
hnr = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 1.0)
# Or cross-correlation method:
hnr_cc = sound.to_harmonicity_cc(0.01, 75.0, 0.1, 1.0)
```

## API Reference

### Sound

```python
# Loading and creating
Sound.from_file(path)                    # WAV, FLAC, MP3, OGG, NIST SPHERE (incl. shorten)
Sound(samples, sample_rate)              # From a numpy array
Sound.create_tone(frequency, duration, sample_rate, amplitude, phase)
Sound.create_silence(duration, sample_rate)

# Properties
sound.sample_rate                        # Sample rate in Hz
sound.duration                           # Duration in seconds
sound.num_samples                        # Number of samples
sound.start_time, sound.end_time         # Time domain (Praat's xmin, xmax)
sound.x1                                 # Time of the first sample (Praat's x1)

# Methods
sound.samples()                          # Samples as a numpy array
sound.get_value_at_time(time, interpolation="sinc70")  # Praat's default
sound.extract_part(start, end, window_shape, relative_width, preserve_times)
                                         # all required, e.g. "rectangular", 1.0, False
sound.pre_emphasis(from_frequency=50.0, normalize=True)  # as Praat: rescaled to 0.99 peak
sound.de_emphasis(from_frequency=50.0, normalize=True)
sound.resample(new_sample_rate)          # Praat's Resample
sound.rms(), sound.peak()                # Amplitude measures

# Analysis
sound.to_pitch(dt, floor, ceil, voicing_threshold=0.45, silence_threshold=0.03,
               octave_cost=0.01, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14)
sound.to_pitch_cc(...)                   # same arguments, cross-correlation method
sound.to_formant_burg(dt, max_num_formants, max_formant_hz, window_length, pre_emphasis_from)
                                         # max_num_formants may be fractional (e.g. 5.5)
sound.to_formant_burg_multi(dt, max_num_formants, [ceilings...], window_length, pre_emphasis_from)
sound.to_formant_path_burg(dt, max_num_formants, middle_ceiling, window_length,
                           pre_emphasis_from, ceiling_step_size, number_of_steps_up_down)
sound.to_intensity(min_pitch, dt)        # Intensity contour
sound.to_spectrum(fast)                  # Single-frame FFT
sound.to_spectrogram(window_length, max_frequency, dt, frequency_step, window_shape)
sound.to_harmonicity_ac(dt, min_pitch, silence_threshold, periods_per_window)
sound.to_harmonicity_cc(dt, min_pitch, silence_threshold, periods_per_window)
sound.to_point_process_periodic_cc(pitch_floor=75.0, pitch_ceiling=600.0)  # glottal pulses
sound.to_point_process_cc(pitch)         # pulses from an existing Pitch

# Speech-referenced variants (robust on long recordings — see below)
sound.to_pitch_ac_referenced(dt, floor, ceil, *, reference_peak=None)
sound.to_pitch_cc_referenced(dt, floor, ceil, *, reference_peak=None)
sound.to_harmonicity_ac_referenced(dt, min_pitch, silence, ppw, *, reference_peak=None)
sound.to_harmonicity_cc_referenced(dt, min_pitch, silence, ppw, *, reference_peak=None)
```

### Speech-referenced normalization (long recordings)

Praat references each frame's amplitude against the **whole-file peak**. On
utterance-length audio that peak is the speech peak, which is correct. On long
conversational recordings (e.g. 10-minute Buckeye files) the peak is set by
whatever is loudest anywhere — a click, laugh, or interviewer burst — so every
quiet voiced frame's relative intensity drops and gets misclassified as silent,
which then NaN-gates downstream formants and HNR.

The `*_referenced` methods substitute a robust amplitude reference for the
whole-file peak. The plain `to_pitch` / `to_harmonicity_*` methods are
unchanged and remain bit-exact with Praat.

```python
import praatfan_gpl as pg

# Estimate one reference per recording, then share it across pitch and HNR.
# Standards are frame-level robust statistics over speech-active frames, so a
# short loud burst cannot dominate them:
ref = pg.estimate_speech_reference(samples, sample_rate)
# keyword-only options: frame_s=0.05, hop_s=0.01, speech_floor_db=30.0,
# reference_percentile=75.0
ref.reference_peak    # 75th-pct of per-frame peak |x| — a typical-speech peak
ref.std               # median per-frame RMS (z-norm scale)
ref.mean              # median per-frame mean (DC reference)
ref.speech_mask       # per-frame bool, hop_s=0.01 grid
ref.frame_times       # frame centers (s)
ref.speech_fraction   # fraction of frames in the speech mask

pitch = sound.to_pitch_ac_referenced(0.01, 75.0, 600.0,
                                     reference_peak=ref.reference_peak)
hnr   = sound.to_harmonicity_cc_referenced(0.01, 75.0, 0.1, 1.0,
                                           reference_peak=ref.reference_peak)

# Or pass reference_peak=None (the default) to have each call estimate the
# reference internally.
```

The estimator definition and constants are shared across the praatfan package
family (`praatfan`, `praatfan-rust`, `praatfan-gpl`), so a `reference_peak`
computed by one package is interchangeable with the others.

Ranged queries on every analysis object take `from_time`, `to_time`
(default `0.0, 0.0`, meaning the whole object, as in Praat's dialogs) and
return `None` where Praat says `--undefined--`.

### Pitch

```python
pitch.get_value_at_time(time, unit, interp)  # Query pitch (None = unvoiced)
pitch.get_mean(t0, t1, "hertz"), pitch.get_standard_deviation(t0, t1, "hertz")
pitch.get_quantile(t0, t1, 0.5, "hertz")          # Praat "Get quantile"
pitch.get_minimum(t0, t1, "hertz", "parabolic")   # get_maximum, get_time_of_minimum/maximum
pitch.count_voiced_frames(), pitch.mean(), pitch.min(), pitch.max()
pitch.values()                               # All values (NaN = unvoiced)
pitch.strengths(), pitch.raw_ac_strengths()  # Winner's correlation per frame
pitch.times(), pitch.get_time_from_frame(i)  # Frame times (i is 0-based)
pitch.num_frames, pitch.time_step, pitch.start_time
pitch.xmin, pitch.xmax                       # Time domain
pitch.pitch_floor, pitch.pitch_ceiling      # Analysis parameters
```

### Formant

```python
formant.get_value_at_time(n, time, unit, interp)      # Fn frequency
formant.get_mean(n, t0, t1, "hertz"), formant.get_standard_deviation(n, t0, t1, "hertz")
formant.get_quantile(n, t0, t1, 0.5, "hertz")     # also get_quantile_of_bandwidth
formant.get_minimum(n, t0, t1, "hertz", "parabolic")  # get_maximum, get_time_of_*
formant.get_bandwidth_at_time(n, time, unit, interp)  # Fn bandwidth
formant.formant_values(n)                             # All Fn values
formant.bandwidth_values(n)                           # All Bn values
formant.times(), formant.get_time_from_frame(i)       # Frame times (i is 0-based)
formant.num_frames, formant.time_step, formant.start_time, formant.max_num_formants
formant.xmin, formant.xmax                            # Time domain
```

Formant numbers `n` are 1-based, as in Praat.

### FormantPath

```python
fp = sound.to_formant_path_burg(0.005, 5, 5500.0, 0.025, 50.0, 0.05, 4)
fp.num_candidates, fp.ceilings()          # Candidate ceilings (Hz)
fp.candidate(i)                           # Formant object of candidate i
fp.path_finder(1.0, 1.0, 1.0, 1.0, 5.0, 0.035, [3, 3, 3, 3], 1.25)  # Praat's "Path finder..."
fp.path_finder(..., minimum_final_cost=False, single_static_cost=False)
                                          # keyword-only corrections, off = Praat
fp.path(), fp.set_path(t_min, t_max, candidate)
fp.extract_formant()                      # Formant along the chosen path
fp.get_stress_of_candidate(t_min, t_max, from_formant, to_formant, parameters, power, candidate)
fp.get_stress_of_candidates(t_min, t_max, from_formant, to_formant, parameters, power)
fp.num_frames, fp.time_step, fp.start_time
```

### Intensity

```python
intensity.get_value_at_time(time, interp)  # Query dB
intensity.values()                          # All values
intensity.times()                           # Frame times
intensity.num_frames, intensity.time_step, intensity.start_time, intensity.xmin, intensity.xmax
intensity.min(), intensity.max(), intensity.mean()  # Praat "Get minimum/maximum/mean" defaults
intensity.get_average(t0, t1, "energy")     # also "sones", "dB"
intensity.get_minimum(t0, t1, "parabolic") # get_maximum, get_time_of_minimum/maximum
```

### Spectrum

```python
spectrum.get_band_energy(f_min, f_max)    # Energy in band (Pa² s)
spectrum.get_center_of_gravity(power)     # Spectral centroid
spectrum.get_standard_deviation(power)    # Spectral spread
spectrum.get_skewness(power)              # Spectral skewness
spectrum.get_kurtosis(power)              # Spectral kurtosis
spectrum.get_total_energy()
spectrum.num_bins, spectrum.df, spectrum.max_frequency
```

### Spectrogram

```python
spectrogram.values()                       # 2D array [freq, time], Pa²/Hz
spectrogram.get_time_from_frame(i)        # Frame time (i is 0-based)
spectrogram.get_frequency_from_bin(i)     # Bin frequency (i is 0-based)
spectrogram.num_frames, spectrogram.num_freq_bins
spectrogram.time_step, spectrogram.freq_step, spectrogram.start_time
spectrogram.freq_min, spectrogram.freq_max
```

### Harmonicity

```python
harmonicity.get_value_at_time(time, interp)  # Query HNR (dB); interpolates -200 frames in, as Praat
harmonicity.values()                          # All values (-200 = unvoiced, as Praat)
harmonicity.times()                           # Frame times
harmonicity.min(), harmonicity.max(), harmonicity.mean()  # Praat semantics (mean skips -200)
harmonicity.get_mean(t0, t1), harmonicity.get_standard_deviation(t0, t1)
harmonicity.get_minimum(t0, t1, "parabolic")  # get_maximum, get_time_of_minimum/maximum
harmonicity.num_frames, harmonicity.time_step, harmonicity.start_time, harmonicity.xmin, harmonicity.xmax
```

### Voice quality (PointProcess)

Defaults are those of Praat's dialogs.

```python
pulses = sound.to_point_process_periodic_cc(75.0, 600.0)   # or sound.to_point_process_cc(pitch)
pulses.times(), pulses.num_points, pulses.xmin, pulses.xmax
PointProcess(times, xmin, xmax)                # build one from pulse times

# Shared optional arguments: from_time=0, to_time=0, period_floor=0.0001,
# period_ceiling=0.02, max_period_factor=1.3 (+ max_amplitude_factor=1.6 for shimmer)
pulses.get_number_of_periods(), pulses.get_mean_period(), pulses.get_stdev_period()
pulses.get_jitter_local(), pulses.get_jitter_local_absolute()
pulses.get_jitter_rap(), pulses.get_jitter_ppq5(), pulses.get_jitter_ddp()
pulses.get_shimmer_local(sound)          # also _local_db, _apq3, _apq5, _apq11, _dda
pulses.get_shimmer(sound)                # dict: local, local_db, apq3, apq5, apq11, dda

praatfan_gpl.voice_report(sound, pitch, pulses, from_time=0.0, to_time=0.0,
    pitch_floor=75.0, pitch_ceiling=600.0, max_period_factor=1.3,
    max_amplitude_factor=1.6, silence_threshold=0.03, voicing_threshold=0.45)
    # dict of every number in Praat's Voice report: fractions (not %),
    # periods in seconds, "shimmer" a nested dict
```

### Other

```python
praatfan_gpl.math_backend()      # "libm" (default) or "system"; see below
praatfan_gpl.compat              # parselmouth-style layer: compat.parselmouth.Sound, call
```

## Unit Options

All option strings are case-insensitive.

- **Pitch units**: `"hertz"` (`"hz"`), `"mel"`, `"erb"`, `"semitones"`
  (`"st"`, re 1 Hz), `"semitones_re_100hz"` (`"st100"`),
  `"semitones_re_200hz"` (`"st200"`), `"semitones_re_440hz"` (`"st440"`)
- **Formant units**: `"hertz"` (`"hz"`), `"bark"`, `"mel"`, `"erb"`
- **Interpolation for `get_value_at_time`**: `"nearest"` (`"none"`),
  `"linear"`, `"cubic"`, `"sinc70"`, `"sinc700"`. Pitch and Formant follow
  Praat, which only has nearest and linear there: every other choice is linear.
- **Interpolation for minimum/maximum queries**: Pitch and Formant
  `"parabolic"` (default) or `"none"`; Intensity and Harmonicity also
  `"cubic"`, `"sinc70"`, `"sinc700"`.
- **Window shapes**: `"rectangular"` (`"rect"`, `"square"`), `"triangular"`
  (`"tri"`, `"bartlett"`), `"parabolic"` (`"para"`, `"welch"`), `"hanning"`
  (`"hann"`), `"hamming"`, `"gaussian"` (`"gauss"`), and `"kaiser"`
  (Praat's Kaiser1 in `extract_part`; Praat's spectrogram has no Kaiser
  window, and `to_spectrogram` treats it as rectangular).
- **Intensity averaging**: `"energy"` (default), `"sones"`, `"dB"`.

## Threading

Pitch, harmonicity, intensity, formant (including FormantPath), spectrogram
and resampling spread their frames across all CPU cores. Results are
bit-identical to a single-threaded run, whatever the thread count. Every
`Sound.to_*` analysis (and `resample`, `estimate_speech_reference`,
`FormantPath.path_finder`) releases the GIL while it runs, so other Python
threads keep going.

- **Thread count:** each call uses every core unless you set
  `RAYON_NUM_THREADS` (e.g. `RAYON_NUM_THREADS=1`) before importing the
  package. If you already run files in parallel across processes
  (`multiprocessing`, joblib, a job array), set it to avoid oversubscribing:
  N processes × all cores each is slower than either alone.
- **fork():** safe. A child forked after the parent has run an analysis
  (Python `multiprocessing`'s default on Linux) builds a fresh thread pool
  on its first call instead of deadlocking on the parent's.

## Comparison with parselmouth

praatfan-gpl aims for bit-accurate compatibility with Praat/parselmouth:

```python
import parselmouth
import praatfan_gpl as pc
import numpy as np

sound_pm = parselmouth.Sound("speech.wav")
sound_pc = pc.Sound.from_file("speech.wav")

# Formant comparison
formant_pm = sound_pm.to_formant_burg(0.01, 5, 5500, 0.025, 50)
formant_pc = sound_pc.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0)

for t in np.arange(0, sound_pm.duration, 0.01):
    f1_pm = formant_pm.get_value_at_time(1, t)
    f1_pc = formant_pc.get_value_at_time(1, t, "hertz", "linear")
    if f1_pc is not None and not np.isnan(f1_pm):  # parselmouth gives NaN, we give None
        assert abs(f1_pm - f1_pc) < 1.0  # Within 1 Hz
```

### Last-bit agreement: the `system-libm` build (opt-in)

The published wheels compute `sin`, `cos`, `exp`, `log`, ... with a pure-Rust
math library, so results are identical on every platform. Praat uses the C
library of the machine it runs on, and its last bits follow that library
(and, under glibc, the CPU). The two differ in the last bit on a small
fraction of arguments, which is invisible in normal speech but decides the
answer on frames that contain only numerical noise (resampled digital
silence, very flat pitch peaks).

If you need the same last bits as the parselmouth on *your* machine, build
with the `system-libm` feature:

```bash
cd python
maturin develop --release --features system-libm
python -c "import praatfan_gpl; print(praatfan_gpl.math_backend())"   # system
```

Results then depend on the host library, exactly like Praat's. Tested once
(2026-11-02) on Linux x86-64, glibc 2.43, against parselmouth 0.4.7: every
pitch and spectrogram case of the parameter sweep matched exactly; formants
matched except ~35 noise-only frames on silence-padded input (<= 0.05 Hz)
and two cases at <= 1e-5 Hz (see DIVERGENCES.md §8). macOS and Windows are
untested. This is a convenience, not a guarantee: a different C library or
glibc version may give different last bits, as it would for Praat.

## License

GPL-3.0 (same as Praat).

## Citing

If you use praatfan-gpl in published work, please cite **both praatfan-gpl
and Praat**. praatfan-gpl reimplements Praat's analyses from Praat's source
code, and was made to match Praat as exposed by parselmouth 0.4.7, which
bundles Praat 6.1.38. Its results are verified against that version.

- Boersma, P. & Weenink, D. *Praat: doing phonetics by computer*
  [Computer program]. <https://www.praat.org>
- Cohen Priva, Uriel (2026). *Praatfan-gpl: a reimplementation in Rust of a
  number of Praat functions.* <https://github.com/ucpresearch/praatfan-core-rs>

```bibtex
@misc{praatfan-gpl,
  author       = {Cohen Priva, Uriel},
  title        = {Praatfan-gpl: a reimplementation in {Rust} of a number of {Praat} functions},
  year         = {2026},
  howpublished = {\url{https://github.com/ucpresearch/praatfan-core-rs}},
  note         = {Version 0.1.15}
}
```

Please also cite Parselmouth and, if you use `FormantPath`, Weenink (2015):

- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
  A Python interface to Praat. *Journal of Phonetics, 71*, 1–15.
  <https://doi.org/10.1016/j.wocn.2018.07.001>
- Weenink, D. (2015). "Improved formant frequency measurements of short
  segments." *Proceedings of ICPhS 2015*, Glasgow.
