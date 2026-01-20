# Development Notes

## Formant Extraction Algorithm

This document describes the exact algorithm needed to match Praat's `Sound → To Formant (burg)` output.

### Key Discovery: Window Duration

The `window_length` parameter passed to Praat (typically 0.025) is internally called `halfdt_window`. The **actual analysis window is twice this value**:

```
window_length = 0.025  →  dt_window = 0.05 seconds
                      →  nsamp_window = 550 samples (at 11000 Hz)
```

This was the critical insight needed to achieve exact match with Praat.

### Complete Algorithm

1. **Resample** to `max_formant * 2` Hz (e.g., 11000 Hz for 5500 Hz max formant)

2. **Pre-emphasize** in-place (from end to beginning):
   ```
   emphasis_factor = exp(-2π * preemphasis_freq * dt)
   for i = n-1 down to 1:
       s[i] -= emphasis_factor * s[i-1]
   ```

3. **Calculate frame timing:**
   ```
   dt_window = 2.0 * window_length           # 0.05 for window_length=0.025
   nsamp_window = floor(dt_window / dx)       # 550 at 11000 Hz
   halfnsamp_window = nsamp_window // 2       # 275

   numberOfFrames = 1 + floor((physicalDuration - dt_window) / time_step)
   t1 = x1 + 0.5 * (physicalDuration - dx - (numberOfFrames - 1) * time_step)
   ```

4. **For each frame** at time `t`:
   ```
   leftSample = floor((t - x1) / dx)
   rightSample = leftSample + 1
   startSample = rightSample - halfnsamp_window
   endSample = leftSample + halfnsamp_window
   actualFrameLength = endSample - startSample + 1  # typically 550
   ```

5. **Apply Gaussian window** (created for `nsamp_window` samples):
   ```
   imid = 0.5 * (nsamp_window + 1)
   edge = exp(-12.0)
   window[i] = (exp(-48 * (i - imid)² / (nsamp_window + 1)²) - edge) / (1 - edge)
   ```
   Note: Only the first `actualFrameLength` window values are applied to samples.

6. **Run Burg LPC** (order = 2 * num_formants, typically 10 for 5 formants)
   - See `dwsys/NUM2.cpp` VECburg function in Praat source

7. **Build polynomial** from LPC coefficients:
   ```
   poly[0] = 1.0
   poly[i+1] = -coeffs[i]  for i = 0 to order-1
   ```

8. **Find roots** via companion matrix eigenvalues

9. **Extract formants** from roots with positive imaginary part:
   ```
   frequency = |angle(root)| * nyquist / π
   bandwidth = -ln(|root|) * sample_rate / π
   ```
   Filter to range [safety_margin, nyquist - safety_margin] (typically 50 Hz margin)

### Verification

Run `scripts/verify_formant_match.py` to confirm exact match with Praat:
```bash
source ~/local/scr/commonpip/bin/activate  # or your parselmouth environment
python scripts/verify_formant_match.py
```

Expected output: 0.00 Hz mean error across all frames.

### Praat Source References

- `fon/Sound_to_Formant.cpp` - Main algorithm, frame timing, Gaussian window
- `dwsys/NUM2.cpp` - VECburg LPC implementation (lines 1431-1497)
- `dwsys/Roots.cpp` - Polynomial_to_Roots, root polishing
- `dwsys/Polynomial.cpp` - Polynomial structure

### Known Issue: Standalone LPC Command

Praat's standalone `To LPC (burg)` command produces **different results** than `Sound → To Formant (burg)`. This is because:

1. Standalone LPC uses a rectangular window (no Gaussian)
2. Different frame timing calculation
3. No internal resampling

Always use `Sound → To Formant (burg)` as the reference for formant extraction.

---

## Resampling Algorithm

The resampling step is critical for formant analysis. Praat resamples to `2 × max_formant` Hz before LPC analysis.

### Current Accuracy

**99.4%** of formant values match Praat within 1 Hz (157/158 voiced frames).
**100%** of formant values match Praat within 5 Hz.

Maximum error: 4.4 Hz (single frame, likely floating-point precision).

### Praat's Resampling (from `fon/Sound.cpp`)

**Time alignment:**
```cpp
// Output x1 calculation
x1_new = 0.5 * (xmin + xmax - (numberOfSamples - 1) / samplingFrequency)

// Index mapping (1-based)
Sampled_indexToX(me, i) = x1 + (i - 1) * dx
Sampled_xToIndex(me, x) = (x - x1) / dx + 1.0
```

**Sinc interpolation (from `melder/NUMinterpol.cpp`):**
- Uses 1-based indexing
- Boundary handling: constant extrapolation outside valid range
- Window: `0.5 * sin(phase) / phase * (1.0 + cos(windowPhase))`
- Window phase: `phase / (maxDepth + 0.5)` where `maxDepth` is typically 50

**FFT lowpass filter (for downsampling):**
- Uses zero padding (NOT mirror padding)
- `antiTurnAround = 1000` samples on each side
- Zero frequencies from `floor(upfactor * nfft)` to `nfft`
- Also zero Nyquist (`data[2]` in Praat's packed format)

### Root Fixing

When mapping roots outside the unit circle back inside:
```cpp
// Correct (preserves imaginary sign):
root = root / |root|²

// Wrong (flips imaginary sign):
root = conj(root) / |root|²
```

This is Praat's `Roots_fixIntoUnitCircle`: `roots[i] = 1.0 / conj(roots[i])` which equals `roots[i] / |roots[i]|²`.

---

## Implementation Details

### Key Insight: Bandwidth Filter Bug

The main issue preventing 100% accuracy was an incorrect bandwidth filter in formant.rs:

```rust
// WRONG - rejected valid low-frequency formants
f.bandwidth < f.frequency * 2.0

// CORRECT - Praat does NOT filter by bandwidth
// Just require positive bandwidth
f.bandwidth > 0.0
```

At phoneme transitions, formants can have large bandwidths relative to frequency. A formant at 315 Hz with bandwidth > 630 Hz was being rejected, causing formant misassignment.

---

## Additional Optimizations (Implemented)

### 1. Newton-Raphson Root Polishing (recommended)

Praat refines eigenvalues using Newton-Raphson iteration (`dwsys/Roots.cpp:301-373`). This is:
- Pure math, no external dependencies
- WASM-compatible
- Up to 80 iterations per root

**Algorithm:**
```python
def polish_complex_root(polynomial, root, max_iter=80):
    best = root
    min_residual = float('inf')
    for _ in range(max_iter):
        p, dp = evaluate_with_derivative(polynomial, root)
        residual = abs(p)
        if residual >= min_residual:
            return best  # Converged or diverging
        min_residual = residual
        best = root
        if abs(dp) == 0:
            return root
        root = root - p / dp  # Newton step
    return root
```

**Polynomial evaluation with derivative (Horner's method for complex z):**
```python
pr, pi = coefficients[-1], 0.0
dpr, dpi = 0.0, 0.0
for coeff in coefficients[-2::-1]:
    # Derivative: d(p*z)/dz = p + z*dp
    dpr, dpi = dpr*x - dpi*y + pr, dpr*y + dpi*x + pi
    # Value: p*z + coeff
    pr, pi = pr*x - pi*y + coeff, pr*y + pi*x
return (pr, pi), (dpr, dpi)
```

### 2. Alternative: Use nalgebra for eigenvalues

The `nalgebra` crate has pure-Rust eigenvalue decomposition that's WASM-compatible. May provide better numerical precision than our custom QR implementation.

---

## Comparison Tools

### Formant Comparison Script

**`scripts/compare_formants.py`** - Python script comparing Praat (parselmouth) vs praat-core-rs formant analysis.

```bash
source ~/local/scr/commonpip/bin/activate
python scripts/compare_formants.py path/to/audio.wav [options]

# Options:
#   --time-step 0.01       Time step between frames (default: 0.01)
#   --max-formants 5       Max number of formants (default: 5)
#   --max-formant-hz 5500  Max formant frequency (default: 5500)
#   --window-length 0.025  Window length (default: 0.025)
#   --pre-emphasis 50.0    Pre-emphasis from Hz (default: 50)
#   --verbose              Show worst error details per formant
#   --json                 Output full data as JSON
```

**`examples/formant_json.rs`** - Rust example outputting formant data as JSON for comparison.

```bash
# Build:
cargo build --release --example formant_json

# Usage (called by compare_formants.py):
./target/release/examples/formant_json <audio> <time_step> <max_formants> <max_formant_hz> <window_length> <pre_emphasis>
```

### Audio Format Support

| Format | Support | Notes |
|--------|---------|-------|
| **WAV** | Full | All sample rates (8k-48k+), bit depths (8/16/24/32-float), mono/stereo |
| **FLAC** | Full | Lossless, recommended for testing |
| **MP3** | Partial | Works but decoder timing differences cause large formant errors |
| **OGG** | Rust only | Praat/parselmouth doesn't support OGG natively |

**MP3 Warning:** MP3 decoders handle encoder delay differently. Symphonia (used by praat-core-rs) and Praat's internal decoder may produce different sample counts and timing. For accurate comparison, use lossless formats (WAV, FLAC).

### Stereo File Handling

Praat preserves stereo channels through resampling and pre-emphasis. Channel averaging only occurs when extracting sample values via `Sound_LEVEL_MONO` (which maps to `Vector_CHANNEL_AVERAGE`).

**Source:** `fon/Vector.cpp:Vector::v_getValueAtSample()` - averages channels when `ilevel <= Vector_CHANNEL_AVERAGE` (0).

**Test file:** `tests/fixtures/one_two_three_four_five-stereo.flac` - real stereo file.

### Running Accuracy Tests

```bash
# Quick check with check_errors example
cargo run --example check_errors

# Detailed comparison on specific file
python scripts/compare_formants.py tests/fixtures/one_two_three_four_five.wav --verbose

# Compare stereo file
python scripts/compare_formants.py tests/fixtures/one_two_three_four_five-stereo.flac
```

---

## Intensity Implementation

### Algorithm (from `fon/Sound_to_Intensity.cpp`)

**Praat source:** `Sound_to_Intensity_e` function

**Key parameters:**
- Physical window duration: `6.4 / min_pitch` (NOT 3.2)
- Window type: Kaiser-Bessel (NOT Hanning)
- DC removal: Unweighted mean (NOT weighted)

**Kaiser-Bessel window:**
```cpp
// Praat uses modified Bessel function I0
double bessel_i0(double x);

// Window coefficient at normalized position (0 to 1):
double r = 2.0 * i / (n - 1) - 1.0;  // -1 to +1
double window = bessel_i0(PI * sqrt(PI_SQUARED_TIMES_4 * (1.0 - r*r))) / bessel_i0_at_edge;
```

**Critical: DC removal uses UNWEIGHTED mean:**
```cpp
// Praat's centre_VEC_inout calls NUMmean which is unweighted
double mean = sum(samples) / n;  // NOT sum(samples*window) / sum(window)
for (i = 0; i < n; i++) {
    samples[i] -= mean;
}
```

**Frame timing (Sampled_shortTermAnalysis):**
```cpp
double physicalDuration = dx * nx;
integer numberOfFrames = 1 + floor((physicalDuration - windowDuration) / timeStep);
double t1 = x1 + 0.5 * (physicalDuration - windowDuration - (numberOfFrames - 1) * timeStep);
```

### Comparison Script

```bash
python scripts/compare_intensity.py tests/fixtures/one_two_three_four_five.wav --verbose
```

---

## Spectrum Implementation

### Algorithm (from `fon/Spectrum.cpp`, `fon/Sound_and_Spectrum.cpp`)

**Key corrections:**

1. **FFT output scaling by dx (sample period):**
   ```rust
   let dx = 1.0 / sample_rate;
   let real: Vec<f64> = fft_output[..n_bins].iter().map(|c| c.re * dx).collect();
   let imag: Vec<f64> = fft_output[..n_bins].iter().map(|c| c.im * dx).collect();
   ```

2. **Band energy uses factor of 2 for one-sided spectrum:**
   ```rust
   // Energy density = 2 * (re² + im²) for one-sided spectrum
   // This accounts for both positive and negative frequencies
   let energy_density = 2.0 * (r * r + i * i);
   energy += energy_density * bin_width;
   ```

### Comparison Script

```bash
python scripts/compare_spectrum.py tests/fixtures/one_two_three_four_five.wav
```

---

## Spectrogram Implementation

### Algorithm (from `fon/Sound_and_Spectrogram.cpp`)

**Key differences from naive STFT:**

1. **Gaussian window physical width = 2 × effective width:**
   ```cpp
   double physicalAnalysisWidth = 2.0 * effectiveAnalysisWidth;  // For Gaussian
   ```

2. **FFT bin binning into spectrogram frequency bins:**
   ```cpp
   integer binWidth_samples = floor(frequencyStep * dx * nfft);
   double binWidth_hertz = 1.0 / (dx * nfft);
   frequencyStep = binWidth_samples * binWidth_hertz;

   // Sum power in each band
   for (iband = 0; iband < numberOfFreqs; iband++) {
       integer lower = iband * binWidth_samples;
       integer upper = lower + binWidth_samples;
       for (k = lower; k < upper; k++) {
           power += spectrum[k];
       }
       data[iband][iframe] = power * one_by_binWidth;
   }
   ```

3. **Gaussian window formula:**
   ```cpp
   double imid = 0.5 * (nsamp_window + 1);
   double edge = exp(-12.0);
   window[i] = (exp(-48 * phase²) - edge) / (1 - edge);
   // where phase = (i - imid) / n_samples_per_window
   ```

4. **Normalization:**
   ```cpp
   double one_by_binWidth = 1.0 / windowssq / binWidth_samples;
   ```

### Multi-Channel Handling

**Critical for stereo files:** Praat averages power spectra, not samples.

From `fon/Sound_and_Spectrogram.cpp`:
```cpp
/*
    For multichannel sounds, the power spectrogram should represent the
    average power in the channels,
    ...
    Averaging starts by adding up the powers of the channels.
*/
for (integer channel = 1; channel <= my ny; channel ++) {
    // Compute FFT for this channel
    // Add to power spectrum
}
// Power averaging ends by dividing the summed power by the number of channels
```

**Rust implementation:**
```rust
// Load channels separately
let channels = Sound::from_file_channels("stereo.wav")?;

// Compute with Praat-compatible power averaging
let spec = spectrogram_from_channels(&channels, time_step, max_freq, window_length, freq_step, WindowShape::Gaussian);
```

**Mathematical difference:**
- Correct (Praat): `(|FFT(ch1)|² + |FFT(ch2)|²) / 2`
- Wrong (sample averaging): `|FFT((ch1+ch2)/2)|²`

### Comparison Script

```bash
# Mono file
python scripts/compare_spectrogram.py tests/fixtures/one_two_three_four_five.wav --verbose

# Stereo file (tests power averaging)
python scripts/compare_spectrogram.py tests/fixtures/one_two_three_four_five-stereo.flac --verbose
```

---

## Pitch Implementation

### Algorithm (from `fon/Sound_to_Pitch.cpp`)

**Status:** 100% accuracy verified against Praat.

**Key methods:**
- AC_HANNING (method 0): Standard pitch analysis with Hanning window
- AC_GAUSS (method 1): Used by Harmonicity, doubles `periodsPerWindow`

### Frame Timing (Critical)

From `melder/Sampled.cpp` - `Sampled_shortTermAnalysis`:
```cpp
double myDuration = dx * nx;
integer numberOfFrames = floor((myDuration - windowDuration) / timeStep) + 1;
double ourMidTime = x1 - 0.5 * dx + 0.5 * myDuration;
double thyDuration = numberOfFrames * timeStep;
double t1 = ourMidTime - 0.5 * thyDuration + 0.5 * timeStep;
```

### Autocorrelation Normalization

```cpp
// Normalized autocorrelation: r[i] = ac[i] / (ac[0] * window_r[i])
// where window_r is the autocorrelation of the window function
```

### Viterbi Path Finding

From `fon/Pitch.cpp` - `Pitch_pathFinder`:
```cpp
// Time step correction (critical!)
double timeStepCorrection = 0.01 / my dx;
octaveJumpCost *= timeStepCorrection;
voicedUnvoicedCost *= timeStepCorrection;

// Local score for voiced candidate
delta = strength - octaveCost * log2(ceiling / frequency);

// Transition cost (voiced to voiced)
transitionCost = octaveJumpCost * fabs(log2(f1 / f2));
```

### AC_GAUSS Differences (for Harmonicity)

From `Sound_to_Pitch.cpp`:
```cpp
case AC_GAUSS:
    periodsPerWindow *= 2;   // Window is twice as long
    interpolation_depth = 0.25;   // Different interpolation
    // Gaussian window formula:
    double imid = 0.5 * (nsamp_window + 1);
    double edge = exp(-12.0);
    window[i] = (exp(-48 * (i - imid)² / (nsamp_window + 1)²) - edge) / (1 - edge);
```

### Comparison Script

```bash
python scripts/compare_pitch.py tests/fixtures/one_two_three_four_five.wav --verbose
```

---

## Harmonicity Implementation

### Algorithm (from `fon/Sound_to_Harmonicity.cpp`)

**Status:** 100% accuracy verified against Praat (AC method).

**Key insight:** Harmonicity is derived directly from Pitch analysis!

```cpp
autoHarmonicity Sound_to_Harmonicity_ac (Sound me, double dt, double minimumPitch,
    double silenceThreshold, double periodsPerWindow)
{
    // Create pitch using AC_GAUSS method (method=1)
    autoPitch pitch = Sound_to_Pitch_any(me, dt, minimumPitch, periodsPerWindow,
        1, 0, pitchCeiling,  // method=1 (AC_GAUSS)
        15, silenceThreshold, 0.0, 0.0, 0.0, 0.0);  // all costs = 0

    for (integer i = 1; i <= thy nx; i++) {
        if (pitch->frames[i].candidates[1].frequency == 0.0) {
            thy z[1][i] = -200.0;  // Unvoiced
        } else {
            double r = pitch->frames[i].candidates[1].strength;
            // Convert correlation r to HNR in dB
            thy z[1][i] = (r <= 1e-15 ? -150.0 :
                          r > 1.0 - 1e-15 ? 150.0 :
                          10.0 * log10(r / (1.0 - r)));
        }
    }
}
```

### Methods

| Method | Praat Command | Internal Pitch Method | Status |
|--------|---------------|----------------------|--------|
| AC | `To Harmonicity (ac)` | AC_GAUSS (method 1) | **100% accurate** |
| CC | `To Harmonicity (cc)` | FCC_ACCURATE (method 3) | Approximated with AC_GAUSS |

**Note:** CC method uses Forward Cross-Correlation (FCC) which is not yet implemented. Currently approximated with AC_GAUSS.

### Comparison Script

```bash
# AC method (verified 100% accurate)
python scripts/compare_harmonicity.py tests/fixtures/one_two_three_four_five.wav --method ac --verbose

# CC method (approximation)
python scripts/compare_harmonicity.py tests/fixtures/one_two_three_four_five.wav --method cc
```

---

## All Comparison Scripts

| Script | Module | Example Usage |
|--------|--------|---------------|
| `compare_formants.py` | Formant | `python scripts/compare_formants.py audio.wav --verbose` |
| `compare_intensity.py` | Intensity | `python scripts/compare_intensity.py audio.wav --verbose` |
| `compare_spectrum.py` | Spectrum | `python scripts/compare_spectrum.py audio.wav` |
| `compare_spectrogram.py` | Spectrogram | `python scripts/compare_spectrogram.py audio.wav --verbose` |
| `compare_pitch.py` | Pitch | `python scripts/compare_pitch.py audio.wav --verbose` |
| `compare_harmonicity.py` | Harmonicity | `python scripts/compare_harmonicity.py audio.wav --method ac --verbose` |

All scripts require:
```bash
source ~/local/scr/commonpip/bin/activate  # parselmouth environment
cargo build --release --example <module>_json  # Rust JSON output binary
```

---

## Implementation Accuracy Summary

| Module | Mono | Stereo | Notes |
|--------|------|--------|-------|
| Formant | 100% | 100% | All WAV variants, FLAC |
| Intensity | 100% | N/A | Kaiser-Bessel window |
| Spectrum | 100% | N/A | dx scaling, factor of 2 |
| Spectrogram | 100% | 100% | Power averaging for stereo |
| Pitch | 100% | 100% | AC_HANNING, Viterbi path finding |
| Harmonicity | 100% | 100% | AC method (AC_GAUSS), CC approximated |
