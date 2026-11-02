# praatfan-gpl

WebAssembly bindings for praatfan-core-rs, enabling acoustic analysis in the browser.

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

The WASM build is single-threaded (the core crate's `parallel` feature is
ignored on `wasm32`) and always uses the pure-Rust `libm` crate. Its output
is bit-identical to the default native build (measured 2026-11-02 on the
24/16/44.1 kHz fixtures: pitch AC/CC, formants, intensity, HNR, spectrum,
spectrogram). A native build with the opt-in `system-libm` feature differs
from it in the last bits, by design.

If loading the module fails with `CompileError: ... Invalid prefixed opcode
263`, your local `wasm/Cargo.lock` has `pulp` 0.22.2, which emits relaxed-SIMD
instructions. Run `cargo update -p pulp` in `wasm/` and rebuild.

## Embedded-V8 bundle (R, mini-racer)

Releases also carry `praatfan-gpl-wasm-v8.zip`. It is a `--target
no-modules` build that adds the JSON pipe (`version()`,
`pipe_request_bytes(bytes, json)`, `pipe_request_samples(samples,
sample_rate, json)`; only built with `--features pipe`), for hosts with a
bare V8 and no ES modules. The classes below are in it too, but only the pipe
is part of the R package's contract. Build it
with `wasm/v8/build.sh`. The main README covers loading and the V8 ≥ 9.6
requirement.

## Usage in JavaScript

### In a Web Application

```html
<script type="module">
import init, { Sound, Pitch, Formant } from './pkg/praatfan_gpl.js';

async function main() {
  await init();

  // Decode with this library's own decoders (WAV, FLAC, MP3, OGG, NIST
  // SPHERE): one Sound per channel, at the file's own sample rate. Avoid the
  // browser's decodeAudioData for analysis: it resamples to the
  // AudioContext's rate (often 48 kHz) with a resampler that is not Praat's.
  const response = await fetch('speech.wav');
  const bytes = new Uint8Array(await response.arrayBuffer());
  const [sound] = Sound.from_bytes_channels(bytes, 'wav');  // first channel

  // Or from samples you already have: new Sound(Float64Array, sampleRate)
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

### With a bundler (webpack, etc.)

The package is not on npm: use the WASM zip from a GitHub release or your own
`wasm-pack build --target bundler`. With `--target bundler` the bundler
instantiates the module, so there is no `init()` to call.

```javascript
import { Sound } from './pkg/praatfan_gpl.js';

function analyze(fileBytes /* Uint8Array */) {
  const [sound] = Sound.from_bytes_channels(fileBytes, '');

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

// Construction
new Sound(Float64Array samples, sampleRate)
Sound.from_bytes_channels(Uint8Array bytes, ext)  // Array of Sound, one per channel
                                             // (our decoders, file's own rate; ext is a hint, may be "")

// Methods
sound.samples()                              // Float64Array
sound.get_value_at_time(time)               // number | undefined (sinc70, as Praat)
sound.pre_emphasis(fromFreq)                 // Sound (rescaled to 0.99 peak, as Praat)
sound.de_emphasis(fromFreq)                  // Sound (rescaled to 0.99 peak, as Praat)
sound.resample(newRate)                      // Sound (Praat's Resample)
sound.rms()                                  // number
sound.peak()                                 // number

// Analysis methods
sound.to_pitch(dt, floor, ceiling)           // Pitch
sound.to_pitch_cc(dt, floor, ceiling)        // Pitch (cross-correlation)
sound.to_formant_burg(dt, n, maxHz, win, pre) // Formant
sound.to_formant_burg_multi(dt, n, Float64Array ceilings, win, pre) // Array of Formant
sound.to_formant_path_burg(dt, n, middleCeiling, win, pre, ceilingStep, stepsUpDown) // FormantPath
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
pitch.strengths()                            // Float64Array
pitch.times()                                // Float64Array
pitch.get_time_from_frame(frame)             // number (frame is 0-based)
pitch.start_time                             // number
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
formant.get_time_from_frame(frame)                    // number (frame is 0-based)
formant.start_time                                    // number
formant.num_frames                                    // number
formant.time_step                                     // number
formant.max_num_formants                              // number
```

Formant numbers `n` are 1-based, as in Praat.

### FormantPath

```javascript
fp.num_candidates                            // number
fp.ceilings()                                // Float64Array
fp.candidate(i)                              // Formant (throws if out of range)
fp.path_finder(qWeight, freqChangeWeight, stressWeight, ceilingChangeWeight,
               intensityModStep, pathWindowLength, Int32Array params, power)
fp.path()                                    // Uint32Array
fp.set_path(tMin, tMax, candidate)
fp.extract_formant()                         // Formant along the path
fp.get_stress_of_candidate(tMin, tMax, fromF, toF, params, power, candidate)
fp.get_stress_of_candidates(tMin, tMax, fromF, toF, params, power)
fp.num_frames, fp.time_step, fp.start_time   // number
```

### Intensity

```javascript
intensity.get_value_at_time(time, interp)    // number | undefined
intensity.values()                           // Float64Array
intensity.times()                            // Float64Array
intensity.min()                              // number | undefined (Praat "Get minimum", parabolic)
intensity.max()                              // number | undefined (Praat "Get maximum", parabolic)
intensity.mean()                             // number | undefined (Praat "Get mean", energy)
intensity.start_time                         // number
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
spectrogram.get_time_from_frame(frame)       // number (0-based)
spectrogram.get_frequency_from_bin(bin)      // number (0-based)
spectrogram.start_time                       // number
spectrogram.num_frames                       // number
spectrogram.num_freq_bins                    // number
spectrogram.time_step                        // number
spectrogram.freq_step                        // number
spectrogram.freq_min                         // number
spectrogram.freq_max                         // number
```

### Harmonicity

```javascript
harmonicity.get_value_at_time(time, interp)  // number | undefined (interpolates -200 frames in, as Praat)
harmonicity.values()                         // Float64Array (-200 for unvoiced frames)
harmonicity.times()                          // Float64Array
harmonicity.min()                            // number | undefined (-200 frames included, as Praat)
harmonicity.max()                            // number | undefined
harmonicity.mean()                           // number | undefined (-200 frames excluded, as Praat)
harmonicity.start_time                       // number
harmonicity.num_frames                       // number
harmonicity.time_step                        // number
```

## Parameter Options

All option strings are case-insensitive; an unknown string throws a `JsError`.

- **Pitch units**: `"hertz"` (`"hz"`), `"mel"`, `"erb"`, `"semitones"`
  (`"st"`, re 1 Hz), `"semitones_re_100hz"` (`"st100"`),
  `"semitones_re_200hz"` (`"st200"`), `"semitones_re_440hz"` (`"st440"`)
- **Formant units**: `"hertz"` (`"hz"`), `"bark"`, `"mel"`, `"erb"`
- **Interpolation**: `"nearest"` (`"none"`), `"linear"`, `"cubic"`,
  `"sinc70"`, `"sinc700"`. Pitch and Formant follow Praat, which only has
  nearest and linear there: every other choice is linear.
- **Window shapes**: `"rectangular"` (`"rect"`, `"square"`), `"triangular"`
  (`"tri"`, `"bartlett"`), `"parabolic"` (`"para"`, `"welch"`), `"hanning"`
  (`"hann"`), `"hamming"`, `"gaussian"` (`"gauss"`), `"kaiser"`
  (`to_spectrogram` treats it as rectangular; Praat's spectrogram has no
  Kaiser window).

The WASM API is a subset of the Python one: the 0.1.15 summary queries
(ranged mean/quantile/extrema), voice quality and `Sound.x1` are Python and
Rust only.

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
  note         = {Version 0.1.16}
}
```

Please also cite Parselmouth and, if you use `FormantPath`, Weenink (2015):

- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth:
  A Python interface to Praat. *Journal of Phonetics, 71*, 1–15.
  <https://doi.org/10.1016/j.wocn.2018.07.001>
- Weenink, D. (2015). "Improved formant frequency measurements of short
  segments." *Proceedings of ICPhS 2015*, Glasgow.
