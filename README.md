# praatfan-core-rs

A Rust reimplementation of Praat's core acoustic analysis algorithms, designed to produce bit-accurate output matching 
Praat/parselmouth. One advantage of using Rust is that code can be webassembled and then used in a browser. 

## Features

- **Exact output parity** with Praat (within floating-point tolerance)
- **Cross-platform**: Native Rust, Python bindings (PyO3), and WASM
- **No GUI dependencies** - pure computational library

### Supported Analysis Types

| Type | Description |
|------|-------------|
| `Sound` | Audio samples with sample rate |
| `Pitch` | F0 contour from autocorrelation analysis |
| `Intensity` | RMS energy contour in dB |
| `Formant` | LPC-based formant tracks (F1-F4 + bandwidths) |
| `Harmonicity` | HNR (harmonics-to-noise ratio) contour |
| `Spectrum` | Single-frame FFT magnitude spectrum |
| `Spectrogram` | Time-frequency representation |

## Quick Start

```bash
pip install praatfan-gpl
```

```python
from praatfan_gpl import Sound

# Load audio and analyze
sound = Sound.from_file("speech.wav")
pitch = sound.to_pitch(0.01, 75.0, 600.0)
formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0)

# Get F0 at time t=0.5s
f0 = pitch.get_value_at_time(0.5, "Hertz", "Linear")
print(f"F0 at 0.5s: {f0:.1f} Hz")

# Get F1, F2 at time t=0.5s
f1 = formant.get_value_at_time(1, 0.5, "Hertz", "Linear")
f2 = formant.get_value_at_time(2, 0.5, "Hertz", "Linear")
print(f"F1={f1:.0f} Hz, F2={f2:.0f} Hz")
```

See [python/README.md](python/README.md) for full API documentation.

## Installation

### Python (from PyPI)

```bash
pip install praatfan-gpl
```

[![PyPI](https://img.shields.io/pypi/v/praatfan-gpl.svg)](https://pypi.org/project/praatfan-gpl/)

**Platform support**: Linux (x86_64, ARM64), macOS (Intel, Apple Silicon), Windows (x86_64)

### Python (from GitHub Release)

Alternatively, install directly from the [releases page](https://github.com/ucpresearch/praatfan-core-rs/releases):

```bash
# Linux x86_64
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.2/praatfan_gpl-0.1.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Linux ARM64 (e.g., Raspberry Pi 5)
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.2/praatfan_gpl-0.1.2-cp312-cp312-manylinux_2_35_aarch64.whl

# macOS Apple Silicon (M1/M2/M3)
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.2/praatfan_gpl-0.1.2-cp312-cp312-macosx_11_0_arm64.whl

# macOS Intel
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.2/praatfan_gpl-0.1.2-cp312-cp312-macosx_10_12_x86_64.whl

# Windows x86_64
pip install https://github.com/ucpresearch/praatfan-core-rs/releases/download/v0.1.2/praatfan_gpl-0.1.2-cp312-cp312-win_amd64.whl
```

**Note:** These wheels require Python 3.12. For other Python versions, build from source (see below).

### WASM (from GitHub Release)

Download `praatfan-gpl.zip` from the [releases page](https://github.com/ucpresearch/praatfan-core-rs/releases), extract, and copy the `pkg/` directory to your web project.

### WASM (from CDN)

The WASM module is also available via GitHub Pages:

```javascript
import init, { Sound } from 'https://ucpresearch.github.io/praatfan-core-rs/pkg/praatfan_gpl.js';
```

**Live Demo:** https://ucpresearch.github.io/praatfan-core-rs/

### Build from Source

#### Native Rust Library

```bash
cargo build --release
```

#### Python Bindings

```bash
# Activate your target venv
source /path/to/your/venv/bin/activate

# Install maturin if not already installed
pip install maturin

# Install in development mode (editable)
cd python
maturin develop --release
```

Or build and install a wheel:

```bash
cd python
maturin build --release
pip install target/wheels/praatfan_gpl-*.whl
```

#### WASM

```bash
# Build WASM package
cd wasm && wasm-pack build --target web

# The built package is in wasm/pkg/ containing:
#   praatfan_gpl.js      - JavaScript bindings
#   praatfan_gpl_bg.wasm - WebAssembly binary
#   praatfan_gpl.d.ts    - TypeScript definitions
#   package.json         - npm package metadata
```

To create a release zip:

```bash
cd wasm/pkg && zip -r ../../praatfan-gpl.zip .
```

To update the GitHub Pages demo (docs/):

```bash
rm -rf docs/pkg && cp -r wasm/pkg docs/
```

### Package Names

| Build Target | Package Name | Import As |
|--------------|--------------|-----------|
| Python (PyO3) | `praatfan-gpl` | `import praatfan_gpl` |
| WASM | `praatfan-gpl` | `import ... from 'praatfan_gpl.js'` |
| Rust crate | `praatfan-core-rs` | `use praatfan_core::*` |

**Note:** There is also a separate pure-Python clean-room implementation called `praatfan` (without `-core`) in the [praatfan-core-clean](https://github.com/ucpresearch/praatfan-core-clean) repository.

## Usage

### Python

```python
from praatfan_gpl import Sound

# Load audio file
sound = Sound.from_file("audio.wav")

# Pitch analysis
pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
f0_values = pitch.values()
times = pitch.times()

# Formant analysis
formant = sound.to_formant_burg(
    time_step=0.01,
    max_num_formants=5,
    max_formant_hz=5500.0,
    window_length=0.025,
    pre_emphasis_from=50.0
)
f1_values = formant.formant_values(1)
f2_values = formant.formant_values(2)

# Intensity analysis
intensity = sound.to_intensity(min_pitch=100.0, time_step=0.01)
intensity_values = intensity.values()

# Harmonicity (HNR)
hnr = sound.to_harmonicity_ac(
    time_step=0.01,
    min_pitch=75.0,
    silence_threshold=0.1,
    periods_per_window=1.0
)
hnr_values = hnr.values()

# Spectrum
spectrum = sound.to_spectrum(fast=True)
cog = spectrum.get_center_of_gravity(power=2.0)
```

### Command-Line Script

A ready-to-use script is included at `python/examples/analyze.py`:

```bash
# Extract all features to TSV (tab-separated)
python python/examples/analyze.py audio.wav

# Save to file
python python/examples/analyze.py audio.wav -o features.tsv

# Output as JSON
python python/examples/analyze.py audio.wav --json -o features.json

# Custom parameters
python python/examples/analyze.py audio.wav --pitch-floor 100 --max-formant 5000
```

**Output columns:** time, f0, intensity, hnr, F1, F2, F3, B1, B2, B3, CoG

### JavaScript (WASM)

```javascript
import init, { Sound } from './pkg/praatfan_gpl.js';

await init();

// Create Sound from samples
const sound = new Sound(new Float64Array(samples), sampleRate);

// Analysis
const pitch = sound.to_pitch(0.01, 75.0, 600.0);
const formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);
const intensity = sound.to_intensity(100.0, 0.01);

// Get values
const f0Values = pitch.values();
const f1Values = formant.formant_values(1);

// Free memory when done
pitch.free();
formant.free();
intensity.free();
sound.free();
```

## Verification

### Rust vs Praat (parselmouth)

Compare native Rust output against parselmouth ground truth:

```bash
# Requires parselmouth: pip install praat-parselmouth
cargo build --release --examples

# Compare individual analysis types
python scripts/compare_formants.py tests/fixtures/one_two_three_four_five.wav
python scripts/compare_pitch.py tests/fixtures/one_two_three_four_five.wav
python scripts/compare_intensity.py tests/fixtures/one_two_three_four_five.wav
python scripts/compare_harmonicity.py tests/fixtures/one_two_three_four_five.wav --method ac
python scripts/compare_spectrum.py tests/fixtures/one_two_three_four_five.wav
python scripts/compare_spectrogram.py tests/fixtures/one_two_three_four_five.wav
```

### WASM vs Native Rust

Verify WASM output matches native Rust binaries:

```bash
# Build WASM package for Node.js
cd wasm && wasm-pack build --target nodejs && cd ..

# Build Rust examples
cargo build --release --examples

# Run verification
node scripts/verify_wasm.mjs [audio_file]
```

This compares Pitch, Formant, Intensity, Spectrum, and Harmonicity outputs between WASM and native Rust, expecting 100% match.

## Accuracy

Accuracy comparison against Praat/parselmouth using `tests/fixtures/one_two_three_four_five.wav`:

| Metric | Accuracy | Points | Mean Error | Max Error |
|--------|----------|--------|------------|-----------|
| F1 | 99.4% | 159 | 0.047 Hz | 7.3 Hz |
| F2 | 100.0% | 159 | 0.004 Hz | 0.33 Hz |
| F3 | 100.0% | 159 | 0.005 Hz | 0.34 Hz |
| Intensity | 100.0% | 196 | ~0 dB | ~0 dB |
| Pitch (F0) | 100.0% | 104 | 0.047 Hz | 0.66 Hz |
| Voicing | 100.0% | 160 | 0 | 0 |
| Spectrum CoG | 100.0% | 1 | ~0% | ~0% |
| Spectrum StdDev | 100.0% | 1 | ~0% | ~0% |
| Spectrum Skew | 100.0% | 1 | 0% | 0% |
| Spectrum Kurt | 100.0% | 1 | ~0% | ~0% |
| Total Energy | 100.0% | 1 | ~0% | ~0% |
| HNR (AC) | 100.0% | 102 | 0.036 dB | 0.77 dB |
| HNR (CC) | 98.1% | 106 | 0.16 dB | 8.0 dB |
| **OVERALL** | **99.7%** | **1150** | | |

Accuracy is measured as percentage of points within tolerance (1 Hz for frequency metrics, 0.1 dB for intensity, 1 dB for HNR, 1% relative error for spectrum moments).

To regenerate this table:
```bash
# Activate a venv with parselmouth installed (pip install praat-parselmouth matplotlib)
cargo build --release --examples
python scripts/accuracy_histogram.py
```

## License

GPL-3.0 - This project reimplements algorithms from [Praat](https://github.com/praat/praat), which is GPL-licensed.

## References

- [Praat source code](https://github.com/praat/praat)
- [Praat manual](https://www.fon.hum.uva.nl/praat/manual/)
- [parselmouth](https://parselmouth.readthedocs.io/)


## Authors

- Uriel Cohen Priva designed, tested, and vibe-coded
- Claude (Opus 4.5) by Anthropic imeplemented
