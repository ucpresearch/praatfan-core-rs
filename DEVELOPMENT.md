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
