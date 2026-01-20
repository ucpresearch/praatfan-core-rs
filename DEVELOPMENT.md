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
