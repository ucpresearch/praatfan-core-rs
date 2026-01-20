# praat-core-wasm

WebAssembly bindings for praat-core-rs, enabling acoustic analysis in the browser.

## Building

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
cd wasm
wasm-pack build --target web

# Or for bundlers (webpack, etc.)
wasm-pack build --target bundler
```

## Usage in JavaScript

### In a Web Application

```html
<script type="module">
import init, { Sound, Pitch, Formant } from './pkg/praat_core_wasm.js';

async function main() {
  await init();

  // Create a sound from an AudioBuffer or Float64Array
  const audioContext = new AudioContext();
  const response = await fetch('speech.wav');
  const arrayBuffer = await response.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Get samples as Float64Array
  const samples = new Float64Array(audioBuffer.getChannelData(0));
  const sampleRate = audioBuffer.sampleRate;

  // Create Sound object
  const sound = new Sound(samples, sampleRate);
  console.log(`Duration: ${sound.duration}s`);

  // Compute pitch
  const pitch = sound.to_pitch(0.01, 75.0, 600.0);
  console.log(`Pitch frames: ${pitch.num_frames}`);

  // Get pitch values as typed array
  const f0Values = pitch.values();  // Float64Array (NaN for unvoiced)
  const times = pitch.times();      // Float64Array

  // Compute formants
  const formant = sound.to_formant_burg(0.01, 5, 5500.0, 0.025, 50.0);
  const f1 = formant.get_value_at_time(1, 0.5, 'hertz', 'linear');
  const f2 = formant.get_value_at_time(2, 0.5, 'hertz', 'linear');

  // Get all F1 values
  const f1Values = formant.formant_values(1);

  // Compute intensity
  const intensity = sound.to_intensity(100.0, 0.01);
  const dbValues = intensity.values();

  // Compute spectrum
  const spectrum = sound.to_spectrum(true);
  const cog = spectrum.get_center_of_gravity(2.0);

  // Compute spectrogram
  const spectrogram = sound.to_spectrogram(0.005, 5000.0, 0.002, 20.0, 'gaussian');
  const specData = spectrogram.values();  // Flat Float64Array [n_freqs × n_times]

  // Compute harmonicity
  const hnr = sound.to_harmonicity_ac(0.01, 75.0, 0.1, 1.0);
  const hnrValues = hnr.values();
}

main();
</script>
```

### With npm/webpack

```javascript
import init, { Sound } from 'praat-core-wasm';

async function analyze(audioBuffer) {
  await init();

  const samples = new Float64Array(audioBuffer.getChannelData(0));
  const sound = new Sound(samples, audioBuffer.sampleRate);

  const pitch = sound.to_pitch(0.01, 75.0, 600.0);
  return {
    duration: sound.duration,
    pitchFrames: pitch.num_frames,
    f0Values: pitch.values(),
    times: pitch.times(),
  };
}
```

## API Reference

### Sound

```javascript
// Constructor
new Sound(samples: Float64Array, sampleRate: number)

// Static methods
Sound.create_tone(frequency, duration, sampleRate, amplitude, phase)
Sound.create_silence(duration, sampleRate)

// Properties
sound.sample_rate      // number
sound.duration         // number (seconds)
sound.num_samples      // number
sound.start_time       // number
sound.end_time         // number

// Methods
sound.samples()                              // Float64Array
sound.get_value_at_time(time)               // number | undefined
sound.pre_emphasis(fromFreq)                 // Sound
sound.de_emphasis(fromFreq)                  // Sound
sound.rms()                                  // number
sound.peak()                                 // number

// Analysis methods
sound.to_pitch(dt, floor, ceiling)           // Pitch
sound.to_formant_burg(dt, n, maxHz, win, pre) // Formant
sound.to_intensity(minPitch, dt)             // Intensity
sound.to_spectrum(fast)                      // Spectrum
sound.to_spectrogram(width, maxF, dt, df, win) // Spectrogram
sound.to_harmonicity_ac(dt, minPitch, sil, periods) // Harmonicity
sound.to_harmonicity_cc(dt, minPitch, sil, periods) // Harmonicity
```

### Pitch

```javascript
pitch.get_value_at_time(time, unit, interp)  // number | undefined
pitch.values()                               // Float64Array (NaN for unvoiced)
pitch.times()                                // Float64Array
pitch.get_time_from_frame(frame)             // number
pitch.num_frames                             // number
pitch.time_step                              // number
pitch.pitch_floor                            // number
pitch.pitch_ceiling                          // number
```

### Formant

```javascript
formant.get_value_at_time(n, time, unit, interp)      // number | undefined
formant.get_bandwidth_at_time(n, time, unit, interp)  // number | undefined
formant.formant_values(n)                             // Float64Array
formant.bandwidth_values(n)                           // Float64Array
formant.times()                                       // Float64Array
formant.num_frames                                    // number
formant.time_step                                     // number
formant.max_num_formants                              // number
```

### Intensity

```javascript
intensity.get_value_at_time(time, interp)    // number | undefined
intensity.values()                           // Float64Array
intensity.times()                            // Float64Array
intensity.min()                              // number | undefined
intensity.max()                              // number | undefined
intensity.mean()                             // number | undefined
intensity.num_frames                         // number
intensity.time_step                          // number
```

### Spectrum

```javascript
spectrum.get_band_energy(fMin, fMax)         // number
spectrum.get_center_of_gravity(power)        // number
spectrum.get_standard_deviation(power)       // number
spectrum.get_skewness(power)                 // number
spectrum.get_kurtosis(power)                 // number
spectrum.get_total_energy()                  // number
spectrum.num_bins                            // number
spectrum.df                                  // number
spectrum.max_frequency                       // number
```

### Spectrogram

```javascript
spectrogram.values()                         // Float64Array (flat, [n_freqs × n_times])
spectrogram.get_time_from_frame(frame)       // number
spectrogram.get_frequency_from_bin(bin)      // number
spectrogram.num_frames                       // number
spectrogram.num_freq_bins                    // number
spectrogram.time_step                        // number
spectrogram.freq_step                        // number
spectrogram.freq_min                         // number
spectrogram.freq_max                         // number
```

### Harmonicity

```javascript
harmonicity.get_value_at_time(time, interp)  // number | undefined
harmonicity.values()                         // Float64Array
harmonicity.times()                          // Float64Array
harmonicity.min()                            // number | undefined
harmonicity.max()                            // number | undefined
harmonicity.mean()                           // number | undefined
harmonicity.num_frames                       // number
harmonicity.time_step                        // number
```

## Parameter Options

- **Units**: `"hertz"`, `"mel"`, `"semitones"`, `"erb"`, `"bark"`
- **Interpolation**: `"nearest"`, `"linear"`, `"cubic"`
- **Window shapes**: `"gaussian"`, `"hanning"`, `"hamming"`, `"rectangular"`

## License

GPL-3.0 (same as Praat)
