# Acoustic Analyzer - Browser Example

A browser-based acoustic analysis tool demonstrating praat-core-wasm.

## Features

- **Drag & drop** audio file upload (WAV, FLAC, MP3, OGG)
- **Real-time analysis** using praat-core-wasm
- **Extracted features:**
  - Pitch (F0) in Hz
  - Intensity in dB
  - HNR (Harmonics-to-Noise Ratio) in dB
  - Formants F1, F2, F3 in Hz
  - Bandwidths B1, B2, B3 in Hz
  - Center of Gravity (CoG)
- **Export** results as CSV or JSON
- **Configurable** analysis parameters

## Running Locally

1. Build the WASM package from the `wasm/` directory:

```bash
cd wasm
wasm-pack build --target web
```

2. Serve this directory with a local HTTP server:

```bash
# Using Python
python -m http.server 8000

# Or using Node.js
npx serve .
```

3. Open http://localhost:8000/examples/acoustic-analyzer/ in your browser.

## How It Works

The example:
1. Uses the Web Audio API to decode uploaded audio files
2. Creates a praat-core-wasm `Sound` object from the decoded samples
3. Computes pitch, intensity, formants, and harmonicity
4. Displays results in an interactive table with summary statistics

## Browser Compatibility

Requires a modern browser with:
- WebAssembly support
- Web Audio API
- ES modules

Tested on: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
