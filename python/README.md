# praatfan-gpl

Praat-compatible acoustic analysis in Python, powered by Rust.

This package provides exact reimplementations of Praat's acoustic analysis algorithms, designed to produce bit-accurate output matching Praat/parselmouth.

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
# Loading
Sound.from_file(path)                    # Load from WAV, MP3, FLAC, OGG
Sound(samples, sample_rate)              # From numpy array

# Properties
sound.sample_rate                        # Sample rate in Hz
sound.duration                           # Duration in seconds
sound.num_samples                        # Number of samples
sound.start_time, sound.end_time         # Time bounds

# Methods
sound.samples()                          # Get samples as numpy array
sound.get_value_at_time(time)            # Linear interpolation
sound.extract_part(start, end, ...)      # Extract segment
sound.pre_emphasis(from_freq)            # High-pass filter
sound.rms(), sound.peak()                # Amplitude measures

# Analysis
sound.to_pitch(dt, floor, ceil)          # Pitch extraction
sound.to_formant_burg(...)               # Formant analysis
sound.to_intensity(min_pitch, dt)        # Intensity contour
sound.to_spectrum(fast)                  # Single-frame FFT
sound.to_spectrogram(...)                # Time-frequency analysis
sound.to_harmonicity_ac/cc(...)          # HNR analysis
```

### Pitch

```python
pitch.get_value_at_time(time, unit, interp)  # Query pitch
pitch.values()                               # All values (NaN=unvoiced)
pitch.times()                                # Frame times
pitch.num_frames, pitch.time_step           # Frame info
pitch.pitch_floor, pitch.pitch_ceiling      # Analysis parameters
```

### Formant

```python
formant.get_value_at_time(n, time, unit, interp)      # Fn frequency
formant.get_bandwidth_at_time(n, time, unit, interp)  # Fn bandwidth
formant.formant_values(n)                             # All Fn values
formant.bandwidth_values(n)                           # All Bn values
formant.times()                                       # Frame times
```

### Intensity

```python
intensity.get_value_at_time(time, interp)  # Query dB
intensity.values()                          # All values
intensity.times()                           # Frame times
intensity.min(), intensity.max(), intensity.mean()
```

### Spectrum

```python
spectrum.get_band_energy(f_min, f_max)    # Energy in band (Pa² s)
spectrum.get_center_of_gravity(power)     # Spectral centroid
spectrum.get_standard_deviation(power)    # Spectral spread
spectrum.get_skewness(power)              # Spectral skewness
spectrum.get_kurtosis(power)              # Spectral kurtosis
spectrum.num_bins, spectrum.df            # Frequency info
```

### Spectrogram

```python
spectrogram.values()                       # 2D array [freq, time]
spectrogram.get_time_from_frame(i)        # Frame time
spectrogram.get_frequency_from_bin(i)     # Bin frequency
spectrogram.num_frames, spectrogram.num_freq_bins
spectrogram.time_step, spectrogram.freq_step
```

### Harmonicity

```python
harmonicity.get_value_at_time(time, interp)  # Query HNR (dB)
harmonicity.values()                          # All values
harmonicity.times()                           # Frame times
harmonicity.min(), harmonicity.max(), harmonicity.mean()
```

## Unit Options

- **Pitch units**: `"hertz"`, `"mel"`, `"semitones"`, `"erb"`
- **Frequency units**: `"hertz"`, `"bark"`, `"mel"`, `"erb"`
- **Interpolation**: `"nearest"`, `"linear"`, `"cubic"`
- **Window shapes**: `"gaussian"`, `"hanning"`, `"hamming"`, `"rectangular"`

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
    if f1_pm is not None and f1_pc is not None:
        assert abs(f1_pm - f1_pc) < 1.0  # Within 1 Hz
```

## License

GPL-3.0 (same as Praat)
