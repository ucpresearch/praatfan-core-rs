# PRAAT.md - Praat Behavior Notes

This file documents observed behaviors and inconsistencies in Praat that are relevant for reimplementation.

## Internal Inconsistency: Sound_to_Formant vs To LPC (burg)

**Observation Date:** 2025-01-19

### The Problem

When extracting formants in Praat, there are two code paths:

1. **Direct path**: `Sound → To Formant (burg)`
   - Uses internal `burg()` function in `Sound_to_Formant.cpp`

2. **Two-step path**: `Sound → Resample → Pre-emphasis → To LPC (burg) → To Formant`
   - Uses `VECburg()` function in `LPC/Sound_and_LPC.cpp`

These produce **different results** even when given identical input samples.

### Evidence

Using the same pre-processed windowed samples at t=0.20 on `one_two_three_four_five.wav`:

| Path | F1 (Hz) | F2 (Hz) |
|------|---------|---------|
| Sound → Formant (burg) direct | 315.5 | 925.8 |
| Sound → Resample → PreEmph → LPC → Formant | 482.6 | 1122.4 |

**Difference:** F1 differs by ~167 Hz, F2 differs by ~197 Hz

### Possible Causes

1. **Window function differences**: `Sound_to_Formant.cpp` applies its own Gaussian window internally with a specific formula, while `To LPC (burg)` uses a different windowing approach.

2. **Burg algorithm implementation**: The internal `burg()` in Sound_to_Formant.cpp may differ from `VECburg()` in `Sound_and_LPC.cpp`.

3. **Sample indexing**: Different handling of sample boundaries or frame centering.

### Implications for praat-core-rs

To match Praat's `Sound → To Formant (burg)` output, we must replicate the internal implementation in `Sound_to_Formant.cpp`, NOT the standalone `To LPC (burg)` command.

The relevant source files in Praat:
- `fon/Sound_to_Formant.cpp` - The correct reference implementation
- `LPC/Sound_and_LPC.cpp` - NOT the same, produces different results

### Verification Script

To verify this inconsistency yourself with any WAV file:

```bash
#!/bin/bash
# Save as: verify_praat_inconsistency.sh
# Usage: ./verify_praat_inconsistency.sh your_audio.wav

WAV_FILE="${1:-tests/fixtures/one_two_three_four_five.wav}"
TIME="0.20"

python3 << EOF
import parselmouth
from parselmouth.praat import call
import numpy as np

sound = parselmouth.Sound("$WAV_FILE")
t = $TIME

# Path 1: Direct Sound -> Formant (burg)
formant_direct = call(sound, "To Formant (burg)", 0.01, 5, 5500.0, 0.025, 50.0)
f1_direct = call(formant_direct, "Get value at time", 1, t, "Hertz", "Linear")
f2_direct = call(formant_direct, "Get value at time", 2, t, "Hertz", "Linear")

# Path 2: Sound -> Resample -> PreEmph -> LPC -> Formant
resampled = call(sound, "Resample", 11000.0, 50)
emphasized = call(resampled, "Filter (pre-emphasis)", 50.0)
lpc = call(emphasized, "To LPC (burg)", 12, 0.025, 0.01, 50.0)
formant_via_lpc = call(lpc, "To Formant")
f1_lpc = call(formant_via_lpc, "Get value at time", 1, t, "Hertz", "Linear")
f2_lpc = call(formant_via_lpc, "Get value at time", 2, t, "Hertz", "Linear")

print(f"File: $WAV_FILE at t={t}")
print(f"Path 1 (Sound -> Formant):     F1={f1_direct:.1f} Hz, F2={f2_direct:.1f} Hz")
print(f"Path 2 (Sound -> LPC -> Form): F1={f1_lpc:.1f} Hz, F2={f2_lpc:.1f} Hz")
print(f"Difference:                    F1={f1_lpc-f1_direct:+.1f} Hz, F2={f2_lpc-f2_direct:+.1f} Hz")
EOF
```

### Additional Finding: Standard Burg also differs

Even using a well-tested reference implementation (spectrum library's arburg), the formants
don't match Praat's Sound → Formant output, suggesting Praat's internal implementation has
unique characteristics beyond standard Burg:

| Method | F1 (Hz) | F2 (Hz) |
|--------|---------|---------|
| Praat Sound → Formant | 301.2 | 943.0 |
| spectrum library arburg | 334.0 | 1123.4 |
| Our Rust implementation | 334.0 | 1123.4 |

The consistent ~33 Hz F1 offset and ~180 Hz F2 offset across all non-Praat implementations
suggests Praat's internal `burg()` function has implementation-specific behavior not documented
in the source code.

### TODO

- [ ] Extract and compare actual LPC coefficients from both paths
- [ ] Identify exact implementation differences in Burg's algorithm
- [ ] Verify window function formulas match Sound_to_Formant.cpp exactly
- [ ] Investigate if Praat applies any undocumented post-processing to formants
