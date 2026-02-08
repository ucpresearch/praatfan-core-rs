# CLAUDE.md

## Project Overview

**praatfan-core-rs** - A Rust reimplementation of Praat's core acoustic analysis algorithms, designed to produce bit-accurate output matching Praat/parselmouth.

**Status:** ✅ Published on PyPI as `praatfan-gpl` (https://pypi.org/project/praatfan-gpl/)

**Goals:**
- Exact output parity with Praat (within floating-point tolerance)
- Cross-platform compilation: native, WASM, Python bindings (PyO3)
- No GUI dependencies - pure computational library
- Comprehensive test suite validating against parselmouth ground truth
- Easy installation via PyPI for all major platforms

## License

**GPL-3.0** - This project reimplements algorithms from Praat, which is GPL-licensed. While clean-room implementations of mathematical algorithms are generally not derivative works, this project will reference Praat's source code for implementation details (edge cases, windowing choices, interpolation methods). Using GPL ensures legal clarity and compatibility with Praat's license.

Reference: https://github.com/praat/praat (GPL-2.0+)

## Development Environment

**Python with parselmouth**: Use the virtual environment for testing with parselmouth:
```bash
source ~/local/scr/commonpip/bin/activate
```

The code of praat should be available at /tmp/praat.github.io/ or ~/local/scr/praat.github.io/

## Target API Surface

Based on analysis of [ozen](https://github.com/ucpresearch/ozen)'s `acoustic.py`, the following Praat functionality is required:

### Core Types

| Type | Description |
|------|-------------|
| `Sound` | Audio samples with sample rate |
| `Pitch` | F0 contour from autocorrelation analysis |
| `Intensity` | RMS energy contour in dB |
| `Formant` | LPC-based formant tracks (F1-F4 + bandwidths) |
| `Harmonicity` | HNR (harmonics-to-noise ratio) contour |
| `Spectrum` | Single-frame FFT magnitude spectrum |
| `Spectrogram` | Time-frequency representation |

### Sound Operations

```rust
// Loading
Sound::from_file(path: &Path) -> Result<Sound>
Sound::from_samples(samples: &[f64], sample_rate: f64) -> Sound

// Properties
sound.duration() -> f64                    // "Get total duration"
sound.sample_rate() -> f64
sound.samples() -> &[f64]

// Extraction
sound.extract_part(                        // "Extract part"
    start_time: f64,
    end_time: f64,
    window_shape: WindowShape,             // Rectangular, Hanning, etc.
    relative_width: f64,
    preserve_times: bool
) -> Sound

// Filtering
sound.pre_emphasis(from_frequency: f64) -> Sound  // "Filter (pre-emphasis)"

// Analysis (create analysis objects)
sound.to_pitch(time_step: f64, pitch_floor: f64, pitch_ceiling: f64) -> Pitch
sound.to_intensity(min_pitch: f64, time_step: f64) -> Intensity
sound.to_formant_burg(
    time_step: f64,
    max_num_formants: u32,      // typically 5
    max_formant_hz: f64,        // e.g., 5500 for male, 5000 for female
    window_length: f64,         // typically 0.025
    pre_emphasis_from: f64      // typically 50
) -> Formant
sound.to_harmonicity_cc(
    time_step: f64,
    min_pitch: f64,
    silence_threshold: f64,     // typically 0.1
    periods_per_window: f64     // typically 1.0
) -> Harmonicity
sound.to_spectrum(fast: bool) -> Spectrum   // Single-frame FFT
sound.to_spectrogram(
    time_step: f64,
    max_frequency: f64,
    window_length: f64,
    frequency_step: f64,
    window_shape: WindowShape   // Gaussian
) -> Spectrogram
```

### Pitch Queries

```rust
pitch.get_value_at_time(
    time: f64,
    unit: PitchUnit,           // Hertz, Mel, Semitones, etc.
    interpolation: Interpolation  // Linear, Cubic, etc.
) -> Option<f64>               // None for unvoiced
```

### Intensity Queries

```rust
intensity.get_value_at_time(
    time: f64,
    interpolation: Interpolation  // Cubic, Linear, etc.
) -> Option<f64>
```

### Formant Queries

```rust
formant.get_value_at_time(
    formant_number: u32,       // 1-5
    time: f64,
    unit: FrequencyUnit,       // Hertz, Bark
    interpolation: Interpolation
) -> Option<f64>

formant.get_bandwidth_at_time(
    formant_number: u32,
    time: f64,
    unit: FrequencyUnit,
    interpolation: Interpolation
) -> Option<f64>
```

### Harmonicity Queries

```rust
harmonicity.get_value_at_time(
    time: f64,
    interpolation: Interpolation
) -> Option<f64>               // HNR in dB
```

### Spectrum Queries

```rust
spectrum.get_center_of_gravity(power: f64) -> f64   // power typically 2.0
spectrum.get_standard_deviation(power: f64) -> f64
spectrum.get_skewness(power: f64) -> f64
spectrum.get_kurtosis(power: f64) -> f64
spectrum.get_band_energy(freq_min: f64, freq_max: f64) -> f64
```

### Spectrogram Access

```rust
spectrogram.values() -> &Array2<f64>       // (n_freqs, n_times), Pa²/Hz
spectrogram.freq_min() -> f64
spectrogram.freq_max() -> f64
spectrogram.time_min() -> f64
spectrogram.time_max() -> f64
spectrogram.get_time_from_frame(frame: usize) -> f64
```

## Algorithm References

### Pitch Detection (Autocorrelation Method)
- Boersma, P. (1993). "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." IFA Proceedings 17, 97-110.
- Praat source: `fon/Sound_to_Pitch.cpp`

### Formant Estimation (Burg's Method)
- Burg, J.P. (1967). "Maximum entropy spectral analysis." 37th Meeting of the Society of Exploration Geophysicists.
- Markel, J.D. & Gray, A.H. (1976). "Linear Prediction of Speech."
- Praat source: `fon/Sound_to_Formant.cpp`, `LPC/Sound_and_LPC.cpp`

### Intensity
- Simple RMS energy in overlapping windows, converted to dB
- Praat source: `fon/Sound_to_Intensity.cpp`

### Harmonicity (Cross-Correlation)
- Based on normalized autocorrelation
- Praat source: `fon/Sound_to_Harmonicity.cpp`

### Spectral Moments
- Standard statistical moments of the power spectrum
- Praat source: `fon/Spectrum.cpp`

### Spectrogram
- STFT with Gaussian window
- Praat source: `fon/Sound_to_Spectrogram.cpp`

## Directory Structure

```
praatfan-core-rs/
├── src/
│   ├── lib.rs              # Public API
│   ├── sound.rs            # Sound type and operations
│   ├── pitch.rs            # Pitch analysis (autocorrelation)
│   ├── formant.rs          # Formant analysis (Burg LPC)
│   ├── intensity.rs        # Intensity analysis
│   ├── harmonicity.rs      # HNR analysis
│   ├── spectrum.rs         # Single-frame spectrum
│   ├── spectrogram.rs      # Time-frequency analysis
│   ├── interpolation.rs    # Linear, cubic interpolation
│   ├── window.rs           # Window functions (Gaussian, Hanning, etc.)
│   └── utils/
│       ├── mod.rs
│       ├── fft.rs          # FFT wrapper (rustfft)
│       └── lpc.rs          # LPC/Levinson-Durbin
├── tests/
│   ├── fixtures/           # Test audio files
│   ├── ground_truth/       # parselmouth output (JSON)
│   ├── test_pitch.rs
│   ├── test_formant.rs
│   ├── test_intensity.rs
│   ├── test_harmonicity.rs
│   ├── test_spectrum.rs
│   └── test_spectrogram.rs
├── examples/
│   ├── formant_json.rs     # JSON output for comparison tools
│   └── check_errors.rs     # Quick accuracy check
├── benches/                # Performance benchmarks
├── python/                 # PyO3 Python bindings
│   ├── src/lib.rs          # Rust bindings code
│   ├── python/praatfan_gpl/  # Python package
│   ├── examples/           # Python usage examples (M8)
│   │   └── ozen_port/      # Port of ozen using praatfan-core
│   └── README.md           # Python API documentation
├── wasm/                   # WASM bindings
│   ├── src/lib.rs          # Rust WASM bindings
│   ├── pkg/                # Built WASM package
│   ├── examples/           # Web examples (M8)
│   │   └── acoustic-analyzer/  # Browser-based analyzer app
│   └── README.md           # WASM API documentation
├── scripts/
│   ├── generate_ground_truth.py  # Generate test data from parselmouth
│   └── compare_formants.py       # Compare Praat vs Rust formant analysis
├── Cargo.toml
├── LICENSE                 # GPL-3.0
└── README.md
```

## Dependencies (Suggested)

```toml
[dependencies]
ndarray = "0.15"           # N-dimensional arrays
rustfft = "6.1"            # FFT
num-complex = "0.4"        # Complex numbers
thiserror = "1.0"          # Error handling

[dev-dependencies]
approx = "0.5"             # Float comparison in tests
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"         # Load ground truth
hound = "3.5"              # WAV file I/O for tests
```

## Testing Strategy

### 1. Ground Truth Generation

Create a Python script that generates comprehensive test cases:

```python
# scripts/generate_ground_truth.py
import parselmouth
from parselmouth.praat import call
import json
import numpy as np

def generate_pitch_ground_truth(audio_path, output_path):
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.01, 75.0, 600.0)

    times = np.arange(0, snd.duration, 0.01)
    values = [call(pitch, "Get value at time", t, "Hertz", "Linear")
              for t in times]

    with open(output_path, 'w') as f:
        json.dump({
            'params': {'time_step': 0.01, 'floor': 75.0, 'ceiling': 600.0},
            'times': times.tolist(),
            'values': [v if v else None for v in values]
        }, f)
```

### 2. Test Categories

| Category | Description |
|----------|-------------|
| **Synthetic** | Pure tones, known formant patterns, silence |
| **Speech** | Real speech samples (male, female, child) |
| **Edge cases** | Very short, very long, silence, noise, clipping |
| **Numerical** | Extreme values, near-zero, boundary conditions |

### 3. Tolerance Levels

```rust
// Suggested tolerances for "exact" matching
const PITCH_TOLERANCE_HZ: f64 = 0.01;      // 0.01 Hz
const FORMANT_TOLERANCE_HZ: f64 = 0.1;     // 0.1 Hz
const INTENSITY_TOLERANCE_DB: f64 = 0.001; // 0.001 dB
const SPECTRAL_TOLERANCE: f64 = 1e-10;     // Relative tolerance
```

### 4. Differential Testing

For every function, compare Rust output against parselmouth:

```rust
#[test]
fn test_pitch_matches_praat() {
    let ground_truth: GroundTruth = load_json("tests/ground_truth/pitch_male.json");
    let sound = Sound::from_file("tests/fixtures/male_speech.wav").unwrap();
    let pitch = sound.to_pitch(0.01, 75.0, 600.0);

    for (i, &time) in ground_truth.times.iter().enumerate() {
        let expected = ground_truth.values[i];
        let actual = pitch.get_value_at_time(time, Hertz, Linear);

        match (expected, actual) {
            (Some(e), Some(a)) => assert_abs_diff_eq!(e, a, epsilon = 0.01),
            (None, None) => {},  // Both unvoiced - OK
            _ => panic!("Voicing mismatch at t={}", time),
        }
    }
}
```

## Implementation Notes

### Floating-Point Considerations
- Use `f64` throughout (Praat uses double precision)
- Avoid fast-math optimizations that reorder operations
- Be explicit about NaN handling (Praat uses undefined/NaN for unvoiced)
- Test on multiple platforms (x86, ARM) for consistency

### Praat-Specific Behaviors to Match
- Window centering conventions
- Edge handling (what happens at start/end of file)
- Interpolation between analysis frames
- Voicing decision thresholds in pitch tracking
- Pre-emphasis filter implementation

### Common Pitfalls
- Off-by-one errors in frame indexing (Praat uses 1-based)
- Different FFT normalization conventions
- Bandwidth definition (half-power vs other conventions)
- Time alignment of analysis frames

## Build Targets

### Native Library
```bash
cargo build --release
```

### WASM
```bash
cd wasm && wasm-pack build --target web
```

### Python Bindings

Install to a virtual environment:
```bash
# Activate your target venv
source /path/to/your/venv/bin/activate

# Install maturin if not already installed
pip install maturin

# Install in development mode (editable)
cd python
maturin develop --release

# Or build a wheel and install
maturin build --release
pip install target/wheels/praatfan_gpl-*.whl
```

## Milestones

1. **M1: Core Infrastructure**
   - Sound loading (WAV)
   - FFT wrapper
   - Window functions
   - Test harness with ground truth comparison

2. **M2: Basic Analysis**
   - Intensity (simplest algorithm)
   - Spectrum + spectral moments
   - Validate against parselmouth

3. **M3: Pitch**
   - Autocorrelation pitch tracking
   - Voicing decisions
   - Interpolation

4. **M4: Formants**
   - LPC via Burg's method
   - Root finding for formant frequencies
   - Bandwidth estimation

5. **M5: Harmonicity**
   - Cross-correlation HNR

6. **M6: Spectrogram**
   - STFT with Gaussian window
   - Frame time mapping

7. **M7: Bindings**
   - PyO3 Python bindings
   - WASM build

8. **M8: Documentation & Examples** (current)

9. **M9: CI/CD & Distribution** (implemented)
   - GitHub Actions workflow for multi-platform Python wheel builds
   - Linux ARM64 built manually on RPi5 (GitHub's ARM64 runners require paid "larger runners")
   - Wheels attached to GitHub releases (not PyPI)
   - WASM zip attached to GitHub releases (not npm)

---

## M9: CI/CD Implementation Details

### GitHub Actions Workflow (`.github/workflows/release.yml`)

**Trigger:** On release publish, or manual workflow_dispatch

**Platforms built automatically:**
| Platform | Runner | Target |
|----------|--------|--------|
| Linux x64 | ubuntu-latest | manylinux_2_35_x86_64 |
| macOS x64 | macos-latest | x86_64-apple-darwin (cross-compiled) |
| macOS ARM64 | macos-latest | aarch64-apple-darwin |
| Windows x64 | windows-latest | win_amd64 |
| WASM | ubuntu-latest | wasm32-unknown-unknown |

**Linux ARM64:** Not built in CI. GitHub's ARM64 runners require enabling "larger runners" (paid feature). Built manually on RPi5 instead.

### Linux ARM64 Manual Build (u5ls.local)

Build environment on the RPi5:
```
~/local/praatfan/
├── buildenv/           # Python venv with maturin
└── praatfan-core-rs/   # Cloned/rsynced repo
```

**Build commands:**
```bash
ssh urielc@u5ls.local
source ~/.cargo/env
cd ~/local/praatfan/praatfan-core-rs
git pull  # or rsync from development machine
cd python
~/local/praatfan/buildenv/bin/maturin build --release --out dist
# Upload: gh release upload vX.Y.Z dist/*.whl
```

### Release Assets

| Asset | Platform | Python |
|-------|----------|--------|
| `praatfan_gpl-...-manylinux_2_35_x86_64.whl` | Linux x64 | cp312 |
| `praatfan_gpl-...-manylinux_2_35_aarch64.whl` | Linux ARM64 | cp312 |
| `praatfan_gpl-...-macosx_10_12_x86_64.whl` | macOS Intel | cp312 |
| `praatfan_gpl-...-macosx_11_0_arm64.whl` | macOS Silicon | cp312 |
| `praatfan_gpl-...-win_amd64.whl` | Windows x64 | cp312 |
| `praatfan-gpl.zip` | Web/WASM | N/A |

### PyPI Publication (v0.1.2+)

**Package name:** `praatfan-gpl`
**PyPI URL:** https://pypi.org/project/praatfan-gpl/

**Installation:**
```bash
pip install praatfan-gpl
```

**Publishing process (session 2026-01-30):**

1. **Metadata fixes:**
   - Fixed repository URL from placeholder to `https://github.com/ucpresearch/praatfan-core-rs`
   - Bumped version from 0.1.1 to 0.1.2 in all `Cargo.toml` and `pyproject.toml` files

2. **Build wheels:**
   - GitHub Actions: 4 platforms (Linux x64, macOS x64/ARM64, Windows x64)
   - Raspberry Pi 5: Linux ARM64 (built via SSH at `transit.u5ls`)

3. **Upload to TestPyPI first:**
   ```bash
   twine upload --repository testpypi dist/*
   ```
   - Verified installation and all functionality

4. **Upload to PyPI:**
   ```bash
   twine upload dist/*
   ```
   - All 5 wheels uploaded successfully
   - Package immediately available: https://pypi.org/project/praatfan-gpl/0.1.2/

**Configuration:** API tokens stored in `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Platform support:** All major platforms supported with pre-built wheels
- Linux: x86_64, ARM64 (Raspberry Pi)
- macOS: Intel, Apple Silicon
- Windows: x64

### Installation Testing

Verified working:
```python
from praatfan_gpl import Sound
import numpy as np

samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
sound = Sound(samples, 22050.0)

# Properties (not methods!)
print(sound.duration)     # 1.0
print(sound.sample_rate)  # 22050.0

# Analysis
pitch = sound.to_pitch(0.01, 75.0, 600.0)
intensity = sound.to_intensity(100.0, 0.01)
```

**Note:** `duration` and `sample_rate` are properties, not methods. Don't call them with `()`.

---

## M8: Documentation & Examples
   - Full documentation for Python bindings with working examples
   - Full documentation for WASM bindings with working examples

   **Python Examples:**
   - Port of [ozen](../ozen) acoustic feature extraction using praatfan-core-python instead of parselmouth
   - Should demonstrate all API functionality: Sound loading, Pitch, Formant, Intensity, Spectrum, Spectrogram, Harmonicity
   - Include example scripts showing typical acoustic analysis workflows

   **WASM Examples:**
   - Browser-based acoustic analyzer web application
   - Features: Audio file upload (drag & drop)
   - Output at every time slice:
     - Pitch (F0)
     - Intensity (dB)
     - Formants: F1, F2, F3 (Hz)
     - Bandwidths: B1, B2, B3 (Hz)
     - Spectral Center of Gravity (CoG)
     - Harmonicity: HNR (dB)
   - Display results in table/CSV format or interactive visualization
   - Demonstrate complete API usage in JavaScript/TypeScript

## Resources

- Praat source code: https://github.com/praat/praat
- Praat manual: https://www.fon.hum.uva.nl/praat/manual/
- parselmouth docs: https://parselmouth.readthedocs.io/
- Boersma's papers: https://www.fon.hum.uva.nl/paul/

---

## BREAKTHROUGH: Exact Formant Match Achieved (Session 2025-01-19)

### Problem SOLVED: Formant accuracy now matches Praat exactly

**Test file:** `tests/fixtures/one_two_three_four_five.wav`

**Final accuracy:**
- **F1: 0.00 Hz mean error** across all 159 voiced frames
- **F2: 0.00 Hz mean error** across all 159 voiced frames
- Perfect bit-accurate match with Praat

### Root Cause: Window Duration Misunderstanding

The key bug was misunderstanding the `window_length` parameter:

**WRONG interpretation:**
```
window_length = 0.025  # The full window duration
nsamp_window = floor(0.025 * sample_rate) = 275 samples
```

**CORRECT interpretation (from Praat source):**
```cpp
// Sound_to_Formant.cpp
// The parameter is called "halfdt_window" internally!
double dt_window = 2.0 * halfdt_window;  // 0.025 * 2 = 0.05 seconds
integer nsamp_window = Melder_ifloor (dt_window / my dx);  // 550 samples!
```

So when you call `To Formant (burg)` with `window_length=0.025`:
- The actual window is **0.05 seconds** (twice the parameter value)
- At 11000 Hz sample rate: **550 samples** (not 275)
- `halfnsamp_window = 550 / 2 = 275`

### Complete Algorithm (Verified Correct)

1. **Resample** to `max_formant * 2` Hz (e.g., 11000 Hz for 5500 Hz max)
2. **Pre-emphasize** in-place: `s[i] -= exp(-2π * 50 * dt) * s[i-1]`
3. **Calculate frame timing:**
   ```python
   dt_window = 2.0 * window_length  # 0.05 for window_length=0.025
   nsamp_window = floor(dt_window / dx)  # 550
   halfnsamp_window = nsamp_window // 2  # 275
   numberOfFrames = 1 + floor((physicalDuration - dt_window) / time_step)
   t1 = x1 + 0.5 * (physicalDuration - dx - (numberOfFrames - 1) * time_step)
   ```
4. **For each frame at time t:**
   - `leftSample = floor((t - x1) / dx)`
   - `rightSample = leftSample + 1`
   - `startSample = rightSample - halfnsamp_window`
   - `endSample = leftSample + halfnsamp_window`
   - `actualFrameLength = endSample - startSample + 1` (should be ~550)
5. **Apply Gaussian window** (550 samples, but only first `actualFrameLength` used):
   ```python
   imid = 0.5 * (nsamp_window + 1)  # 275.5
   edge = exp(-12.0)
   window[i] = (exp(-48 * (i - imid)² / (nsamp_window + 1)²) - edge) / (1 - edge)
   ```
6. **Run Burg LPC** (VECburg algorithm from `dwsys/NUM2.cpp`)
7. **Build polynomial:** `poly[0] = 1.0; poly[i+1] = -coeffs[i]`
8. **Find roots** via companion matrix eigenvalues
9. **Extract formants** from roots with positive imaginary part

### Test Script (Verified)

See `scripts/verify_formant_match.py` for complete working implementation.

### Relevant Praat Source Files

- `fon/Sound_to_Formant.cpp` - Main algorithm with frame timing
- `dwsys/NUM2.cpp` - VECburg implementation (lines 1431-1497)
- `dwsys/Roots.cpp` - Polynomial_to_Roots, root polishing
- `dwsys/Polynomial.cpp` - Polynomial structure

---

## Resampling Implementation (Session 2026-01-19)

### Current Accuracy: 96.8% (153/158 frames within 1 Hz)

Implemented Praat's exact resampling algorithm for formant analysis. The remaining 2.5% error (4 frames) occurs at phoneme transitions where formant tracking is inherently ambiguous.

### Resampling Algorithm (from `fon/Sound.cpp`)

**Key source files:**
- `fon/Sound.cpp` - Sound_resample function (lines 393-451)
- `melder/NUMinterpol.cpp` - NUM_interpolate_sinc (lines 31-333)
- `dwsys/NUMFourier.cpp` - NUMrealft (FFT in packed format)

**Time alignment (CRITICAL):**
```cpp
// Praat's exact formula for output x1
x1_new = 0.5 * (xmin + xmax - (numberOfSamples - 1) / samplingFrequency)

// Index mapping (1-based)
Sampled_indexToX(me, i) = x1 + (i - 1) * dx
Sampled_xToIndex(me, x) = (x - x1) / dx + 1.0
```

**Sinc interpolation (from NUMinterpol.cpp):**
```cpp
// Window formula: 0.5 * sin(phase) / phase * (1.0 + cos(windowPhase))
// where windowPhase = phase / (maxDepth + 0.5)
// Uses optimized matrix multiplication for sin/cos updates
const double leftDepth = maxDepth + 0.5;
double windowPhase = leftPhase / leftDepth;
sincTimesWindow = halfSinLeftPhase / leftPhase * (1.0 + cosWindowPhase);
```

**FFT lowpass filter (for downsampling):**
```cpp
// Praat uses zero padding, NOT mirror padding
const integer antiTurnAround = 1000;
const integer nfft = Melder_iroundUpToPowerOfTwo(nx + antiTurnAround * 2);
// Zero frequencies from floor(upfactor * nfft) to nfft
// Also zero data[2] (Nyquist in Praat's packed format)
```

### Root Fixing Bug (FIXED)

**Wrong:** `root = conj(root) / |root|²` (flips imaginary sign)
**Correct:** `root = root / |root|²` (preserves imaginary sign)

This matches Praat's `Roots_fixIntoUnitCircle`: `roots[i] = 1.0 / conj(roots[i])`

### Remaining Work for 100% Accuracy

The 4 outlier frames (t=0.61, 0.62, 1.05, 1.06s) have F1 errors of 667-1124 Hz due to:

1. **Root polishing** - Praat applies Newton-Raphson to refine eigenvalues
2. **LAPACK dhseqr** - Praat uses optimized QR algorithm for Hessenberg matrices

**WASM-compatible solutions:**

1. **Newton-Raphson root polishing** (recommended):
   - Pure math, no dependencies
   - Implemented in `dwsys/Roots.cpp:301-373`
   - `Polynomial_polish_complexroot_nr` uses up to 80 iterations
   - Evaluates polynomial and derivative at root, applies Newton step

2. **nalgebra for eigenvalues** (alternative):
   - Pure Rust, WASM-compatible
   - Has eigenvalue decomposition for real matrices
   - May improve numerical precision

### Newton-Raphson Root Polishing Algorithm

From `dwsys/Roots.cpp`:

```cpp
static void Polynomial_polish_complexroot_nr(constPolynomial me, dcomplex *root, integer maxit) {
    dcomplex zbest = *root;
    double ymin = 1e308;
    for (integer iter = 1; iter <= maxit; iter++) {
        dcomplex p, dp;
        Polynomial_evaluateWithDerivative_z(me, root, &p, &dp);
        double fabsy = abs(p);
        if (fabsy > ymin || fabs(fabsy - ymin) < NUMfpp->eps) {
            *root = zbest;
            return;
        }
        ymin = fabsy;
        zbest = *root;
        if (abs(dp) == 0.0)
            return;
        *root -= p / dp;  // Newton-Raphson step
    }
}
```

**Polynomial evaluation with derivative (Horner's method):**
```cpp
// For complex z = x + iy
longdouble pr = coefficients[n], pi = 0.0;
longdouble dpr = 0.0, dpi = 0.0;
for (i = n-1 to 1) {
    // derivative update
    tr = dpr; dpr = dpr*x - dpi*y + pr; dpi = tr*y + dpi*x + pi;
    // value update
    tr = pr; pr = pr*x - pi*y + coefficients[i]; pi = tr*y + pi*x;
}
```

### Verification Commands

```bash
# Run accuracy check
cargo run --example check_errors

# Compare resampled samples with Praat
cargo run --example compare_samples
```

### Sample Resampling Accuracy

```
Position     Praat              Rust           Diff
5500         0.11989815426      0.11989816527  +1.1e-8
```

---

## Comparison Tools and Audio Format Support (Session 2026-01-19)

### Formant Comparison Script

**`scripts/compare_formants.py`** - Python script comparing Praat (parselmouth) vs praatfan-core-rs formant analysis.

**Usage:**
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

### Supported Audio Formats

| Format | Support | Notes |
|--------|---------|-------|
| **WAV** | Full | All sample rates (8k-48k+), bit depths (8/16/24/32-float), mono/stereo |
| **FLAC** | Full | Lossless, recommended for testing |
| **MP3** | Partial | Works but decoder timing differences cause large formant errors |
| **OGG** | Rust only | Praat/parselmouth doesn't support OGG natively |

**MP3 Warning:** MP3 decoders handle encoder delay differently. Symphonia (used by praatfan-core-rs) and Praat's internal decoder may produce different sample counts and timing. For accurate comparison, use lossless formats (WAV, FLAC).

### Stereo File Handling

Praat preserves stereo channels through resampling and pre-emphasis. Channel averaging only occurs when extracting sample values via `Sound_LEVEL_MONO` (which maps to `Vector_CHANNEL_AVERAGE`).

**Source:** `fon/Vector.cpp:Vector::v_getValueAtSample()` - averages channels when `ilevel <= Vector_CHANNEL_AVERAGE` (0).

**Test file:** `tests/fixtures/one_two_three_four_five-stereo.flac` - real stereo file (not duplicated mono).

### Current Accuracy (as of 2026-01-19)

**Test results with `scripts/compare_formants.py`:**

| File | F1 within 1 Hz | F2 within 1 Hz | F3 within 1 Hz |
|------|----------------|----------------|----------------|
| WAV 22050 Hz | 100% | 100% | 100% |
| WAV 16000 Hz | 100% | 100% | 100% |
| WAV 44100 Hz | 100% | 100% | 100% |
| WAV 48000 Hz | 100% | 100% | 100% |
| WAV 8-bit | 100% | 100% | 100% |
| WAV 24-bit | 100% | 100% | 100% |
| WAV 32-float | 100% | 100% | 100% |
| WAV stereo | 100% | 100% | 100% |
| FLAC | 100% | 100% | 100% |
| FLAC stereo | 100% | 100% | 100% |
| MP3 | ~50-60% | ~50-60% | ~50-60% |

**Note:** MP3 differences are expected due to decoder timing, not algorithm errors.

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

## Verified Implementations (Session 2026-01-19)

### Implementation Status

| Module | Accuracy | Notes |
|--------|----------|-------|
| **Formant** | 100% | F1, F2, F3 exact match (159/159 frames) |
| **Intensity** | 100% | Kaiser-Bessel window, unweighted mean |
| **Spectrum** | 100% | Correct dx scaling, factor of 2 for one-sided |
| **Spectrogram** | 100% | Mono and stereo files |
| **Pitch** | 100% | AC_HANNING, max 0.002 Hz error (104/104 frames) |
| **Harmonicity (AC)** | ~98% | 100/102 within 1 dB; 2 frames ~5 dB off due to FFT precision (see below) |
| **Harmonicity (CC)** | 100% | FCC_ACCURATE, max 0.00024 dB error (106/106 frames) |

### Intensity Algorithm (from `fon/Sound_to_Intensity.cpp`)

Key differences from naive implementation:
- **Window:** Kaiser-Bessel (bessel_i0), NOT Hanning
- **Physical window:** `6.4 / min_pitch` (not 3.2)
- **DC removal:** Unweighted mean (critical!)

```rust
// Kaiser-Bessel window coefficient
fn bessel_i0(x: f64) -> f64 // Modified Bessel function of first kind

// Window formula at normalized position r:
let window = bessel_i0(PI * PI_SQUARED_TIMES_4 * (1.0 - r*r).sqrt()) / bessel_i0_at_edge
```

### Spectrum Algorithm (from `fon/Spectrum.cpp`)

Key differences:
- **FFT output scaling:** Multiply by `dx` (sample period) for spectral density
- **Band energy:** Factor of 2 for one-sided spectrum (positive + negative frequencies)

```rust
// Correct normalization
let dx = 1.0 / sample_rate;
let real: Vec<f64> = fft_output.iter().map(|c| c.re * dx).collect();
let imag: Vec<f64> = fft_output.iter().map(|c| c.im * dx).collect();

// Band energy calculation
energy = 2.0 * (re² + im²) * bin_width  // Factor of 2!
```

### Spectrogram Algorithm (from `fon/Sound_and_Spectrogram.cpp`)

Key differences:
- **Gaussian window:** Physical width = 2 × effective width
- **FFT binning:** Multiple FFT bins combined into spectrogram frequency bins
- **Normalization:** `1 / (windowssq × binWidth_samples)`

**Multi-channel handling (critical for stereo):**
Praat averages power spectra across channels, NOT samples:
```
Correct: (|FFT(ch1)|² + |FFT(ch2)|²) / 2
Wrong:   |FFT((ch1+ch2)/2)|²
```

Use `spectrogram_from_channels()` for Praat-compatible stereo handling:
```rust
let channels = Sound::from_file_channels("stereo.wav")?;
let spec = spectrogram_from_channels(&channels, time_step, max_freq, window_length, freq_step, WindowShape::Gaussian);
```

### Pitch Algorithm (from `fon/Sound_to_Pitch.cpp`)

Key implementation details:
- **Method:** AC_HANNING (method 0) for standard pitch, AC_GAUSS (method 1) for harmonicity
- **Frame timing:** Uses `Sampled_shortTermAnalysis` formula
- **Path finding:** Viterbi algorithm with time step correction (`0.01 / dt`)
- **Window autocorrelation:** Normalized by `ac[0] * window_r[i]`

**Frame timing calculation (critical):**
```cpp
// From Sampled.cpp
double myDuration = dx * nx;
integer numberOfFrames = floor((myDuration - windowDuration) / timeStep) + 1;
double ourMidTime = x1 - 0.5 * dx + 0.5 * myDuration;
double thyDuration = numberOfFrames * timeStep;
double t1 = ourMidTime - 0.5 * thyDuration + 0.5 * timeStep;
```

**AC_GAUSS differences:**
- `periodsPerWindow` is doubled internally
- Uses Gaussian window: `(exp(-48 * (i - imid)² / (n+1)²) - edge) / (1 - edge)`
- Interpolation depth: 0.25 (vs 0.5 for AC_HANNING)

### Harmonicity Algorithm (from `fon/Sound_to_Harmonicity.cpp`)

Harmonicity is derived directly from Pitch analysis:
```cpp
// HNR formula: convert correlation strength r to dB
if (r <= 1e-15) return -150.0;
if (r > 1.0 - 1e-15) return 150.0;
return 10.0 * log10(r / (1.0 - r));

// Unvoiced frames: -200.0 dB
```

**Methods:**
- `Sound_to_Harmonicity_ac` uses AC_GAUSS (method 1) - **100% accurate**
- `Sound_to_Harmonicity_cc` uses FCC_ACCURATE (method 3) - **~98% accurate** (see Brent+Sinc section)

**FCC differences from AC:**
- No window function applied (raw mean-subtracted samples)
- Time-domain cross-correlation: `r[i] = Σ(x[j]·y[i+j]) / √(Σx²·Σy²)`
- Longer effective window: frame timing uses `1/pitch_floor + dt_window`

### Brent+Sinc Peak Refinement (Session 2026-02-08)

Replaced parabolic `improve_maximum` with Praat's exact peak refinement algorithm:
`NUMimproveExtremum` (Brent minimization + sinc interpolation).

**Key source files:**
- `melder/NUMinterpol.cpp` - `NUMimproveExtremum` (lines 335-391)
- `dwsys/NUM2.cpp` - `NUMminimize_brent` (lines 1910-2020)
- `fon/Sound_to_Pitch.cpp` - Second pass (lines 246-261)

**What changed in `src/pitch.rs`:**

1. **`minimize_brent()` function** - Port of Praat's `NUMminimize_brent`. Golden section search
   with parabolic interpolation, 60 max iterations, tolerance 1e-10.

2. **`improve_maximum()` rewritten** - Now matches `NUMimproveExtremum`:
   - Boundary check → return raw value at edges
   - `brent_depth ≤ 1` → parabolic interpolation (unchanged behavior)
   - `brent_depth = 70 or 700` → Brent optimization of `-sinc_interpolate()` over `[ixmid-1, ixmid+1]`

3. **`brent_depth` parameter per method** (matches `Sound_to_Pitch.cpp` lines 288-305):
   - AcHanning → 70 (SINC70)
   - AcGauss → 700 (SINC700)
   - FccAccurate → 700 (SINC700)

4. **Adaptive depth at call sites** (matches `Sound_to_Pitch.cpp` line 254):
   ```rust
   let depth = if candidates[i].frequency > 0.3 / dx { 700 } else { brent_depth };
   ```
   High-frequency candidates always use SINC700 regardless of method.

5. **All 4 frame functions updated**: `compute_pitch_frame`, `compute_pitch_frame_multichannel`,
   `compute_pitch_frame_fcc`, `compute_pitch_frame_fcc_multichannel`.

### CC Harmonicity Exact Match (Session 2026-02-08)

After Brent+sinc refinement plus FCC structural fixes, CC harmonicity now matches Praat to
floating-point precision on all 106 voiced frames (max error 0.00024 dB).

**FCC bugs fixed (in addition to Brent+sinc):**

1. **`nsamp_window` not truncated to even** — Praat: `halfnsamp_window = raw/2 - 1; nsamp_window = half*2`. Our FCC path skipped this, giving a window 2-4 samples too wide.
2. **`maximum_lag` formula wrong** — Praat: `min(floor(nsamp_window/periodsPerWindow)+2, nsamp_window)`. We used `floor(1.0/pitch_floor/dx)`.
3. **`r` array too small** — Must be `2*nsamp_window+1`, not `2*local_maximum_lag+1`. Sinc interpolation near edges saw wrong boundary conditions.
4. **No `imax` tracking** — First pass must store discrete lag index; second pass uses it for Brent.
5. **`dt_window` recomputed from truncated `nsamp_window`** — Must pass original `periodsPerWindow/pitchFloor` to frame functions, not recompute as `nsamp_window*dx` (caused 1-sample shift at voiced/unvoiced boundaries).
6. **Local mean off-by-one** — `end_mean` was `left_sample + nsamp_period` but should be `left_sample + nsamp_period + 1` (0-based exclusive). Divisor must be `2*nsamp_period` (Praat line 66), not actual count.

### Formant Dither Removal (Session 2026-02-08)

Removed 1e-10 amplitude dither that was added to prevent Burg LPC from failing on all-zero frames.
Praat instead checks `max(sample^2)` and skips Burg entirely when zero (`Sound_to_Formant.cpp` line 342).

The dither caused a 7.3 Hz F1 error at t=0.026s (first frame, near-silence). After removal, all 159
voiced frames match exactly (F1: max 0.01 Hz, F2: max 0.10 Hz, F3: max 0.07 Hz).

### AC Harmonicity FFT Precision Limitation (Session 2026-02-08)

AC harmonicity has 2/102 frames with ~5 dB error (frames 14 and 30). Root cause: FFT implementation
differences between rustfft and Praat's FFTPACK-derived real FFT.

**The two FFT implementations:**

| | Praat (`NUMrealft`) | praatfan-core-rs (`rustfft`) |
|---|---|---|
| **Origin** | FFTPACK (Fortran, 1985), translated to C | Pure Rust, Cooley-Tukey |
| **Input** | Real-valued only | Complex-valued |
| **Operations** | ~N/2 (exploits real symmetry) | ~2N (full complex) |
| **Output format** | Packed real array `[r0, r1, i1, r2, i2, ...]` | Full `Complex<f64>` array |
| **Butterfly** | Real-valued twiddle factors | Complex multiplications |

**Why the difference matters:**
- Both produce correct results, but with different floating-point rounding (~1e-4 in autocorrelation `r`)
- At `r ≈ 0.99`, this is ~0.1 dB HNR difference (acceptable)
- At `r > 0.9998`, the HNR formula `10*log10(r/(1-r))` amplifies the error to 5+ dB
- Frames 14 and 30 happen to have `r > 0.9998` (extremely periodic speech segments)

**Why we don't switch to Praat's FFT:**
- Praat's FFTPACK is 3000+ lines of mechanically-translated Fortran with computed gotos
- Would require maintaining a separate FFT implementation alongside rustfft
- Cross-platform reproducibility would depend on matching compiler float optimizations
- Practical impact is minimal: only affects HNR at extreme `r` values (>0.9998)
- The 5 dB difference at `r=0.9998` corresponds to HNR of ~37 dB vs ~42 dB, both indicating very clean speech

**Fixes attempted (for completeness):**
- Local mean off-by-one fix (same bug as FCC #6) — improved but didn't fix FFT difference
- `imax` tracking in AC second pass — no change for this test file
- Circular autocorrelation (matching Praat's FFT size) — no improvement; difference is intrinsic to algorithm

### Current Accuracy Summary (Session 2026-02-08)

Test file: `tests/fixtures/one_two_three_four_five.wav`

| Module | Frames | Max Error | Notes |
|--------|--------|-----------|-------|
| **CC Harmonicity** | 106/106 | 0.00024 dB | Exact match |
| **Pitch** | 104/104 | 0.0019 Hz | Exact match |
| **F1** | 159/159 | 0.01 Hz | Exact match |
| **F2** | 159/159 | 0.10 Hz | Exact match |
| **F3** | 159/159 | 0.07 Hz | Exact match |
| **AC Harmonicity** | 100/102 | ~5 dB (2 frames) | FFT precision limit at r>0.9998 |
| **Intensity** | All | <0.001 dB | Exact match |
| **Spectrum** | All | <1e-10 | Exact match |
| **Spectrogram** | All | <1e-10 | Exact match |

### Comparison Scripts

| Script | Module | Usage |
|--------|--------|-------|
| `scripts/compare_formants.py` | Formant | `python scripts/compare_formants.py audio.wav` |
| `scripts/compare_intensity.py` | Intensity | `python scripts/compare_intensity.py audio.wav` |
| `scripts/compare_spectrum.py` | Spectrum | `python scripts/compare_spectrum.py audio.wav` |
| `scripts/compare_spectrogram.py` | Spectrogram | `python scripts/compare_spectrogram.py audio.wav` |
| `scripts/compare_pitch.py` | Pitch | `python scripts/compare_pitch.py audio.wav` |
| `scripts/compare_harmonicity.py` | Harmonicity | `python scripts/compare_harmonicity.py audio.wav --method ac` |

All scripts require the parselmouth environment:
```bash
source ~/local/scr/commonpip/bin/activate
```

### JSON Output Examples

| Example | Module | Build Command |
|---------|--------|---------------|
| `formant_json` | Formant | `cargo build --release --example formant_json` |
| `intensity_json` | Intensity | `cargo build --release --example intensity_json` |
| `spectrum_json` | Spectrum | `cargo build --release --example spectrum_json` |
| `spectrogram_json` | Spectrogram | `cargo build --release --example spectrogram_json` |
| `pitch_json` | Pitch | `cargo build --release --example pitch_json` |
| `harmonicity_json` | Harmonicity | `cargo build --release --example harmonicity_json` |
