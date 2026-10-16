// Load the embedded-V8 bundle as hosts do -- in a bare node:vm context with
// no TextEncoder/TextDecoder/atob/console of its own, shim first -- and run
// one pipe request through each transport.
//
//   node wasm/v8/smoke.mjs <bundle dir> <16-bit PCM mono wav> [native praatfan-gpl-pipe]
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';

const [dir, audio, native] = process.argv.slice(2);
const here = path.dirname(new URL(import.meta.url).pathname);
const ctx = vm.createContext({ WebAssembly });
vm.runInContext(fs.readFileSync(path.join(here, 'test-shim.js'), 'utf8'), ctx, { filename: 'shim.js' });
vm.runInContext(fs.readFileSync(path.join(dir, 'praatfan_gpl.js'), 'utf8'), ctx, { filename: 'praatfan_gpl.js' });
ctx.wasm_b64 = fs.readFileSync(path.join(dir, 'praatfan_gpl_bg.wasm')).toString('base64');
vm.runInContext('wasm_bindgen.initSync({module: praatfan_b64bytes(wasm_b64)}); delete globalThis.wasm_b64;', ctx);

const analyses = [
  { type: 'pitch_ac' }, { type: 'formant_burg' }, { type: 'intensity' },
  { type: 'harmonicity_cc' }, { type: 'spectral_moments', times: [0.5] },
  { type: 'spectrogram' }, { type: 'spectrogram', window_shape: 'hanning', dtype: 'f32le', start_time: 0.5, end_time: 0.7 },
];
const fail = (msg) => { console.error('smoke test failed: ' + msg); process.exit(1); };

ctx.audio_b64 = fs.readFileSync(audio).toString('base64');
ctx.req = JSON.stringify({ wav_path: path.basename(audio), analyses });
const bytesRaw = vm.runInContext('wasm_bindgen.pipe_request_bytes(praatfan_b64bytes(audio_b64), req)', ctx);
const fromBytes = JSON.parse(bytesRaw);
if (!fromBytes.ok) fail('bytes transport: ' + fromBytes.error);
const spec = fromBytes.results[5];
if (spec.n_times === 0 || Buffer.from(spec.power_b64, 'base64').length !== 8 * spec.n_times * spec.n_freqs) {
  fail('spectrogram payload size');
}

// Samples transport, fed the fixture's PCM decoded here, must agree.
const wav = fs.readFileSync(audio);
let off = 12, sr = 0, pcm = null;
while (off + 8 <= wav.length) {
  const id = wav.toString('latin1', off, off + 4), size = wav.readUInt32LE(off + 4);
  if (id === 'fmt ') sr = wav.readUInt32LE(off + 12);
  if (id === 'data') pcm = wav.subarray(off + 8, off + 8 + size);
  off += 8 + size + (size & 1);
}
const x = new Float64Array(pcm.length >> 1);
for (let i = 0; i < x.length; i++) x[i] = pcm.readInt16LE(2 * i) / 32768;
ctx.samples = x; ctx.sr = sr; ctx.req2 = JSON.stringify({ analyses });
const fromSamples = vm.runInContext('wasm_bindgen.pipe_request_samples(samples, sr, req2)', ctx);
if (fromSamples !== JSON.stringify({ ...fromBytes })) {
  const a = JSON.parse(fromSamples);
  if (JSON.stringify(a.results) !== JSON.stringify(fromBytes.results)) fail('samples transport differs from bytes transport');
}

// Against the native binary: same structure, values equal to libm rounding
// (wasm32 uses Rust's libm, not the platform's, so bit identity is not expected).
let worst = 0, worstPower = 0;
if (native) {
  const nat = JSON.parse(execFileSync(native, { input: JSON.stringify({ wav_path: audio, analyses }), maxBuffer: 1 << 28 }));
  const decode = (r) => {
    const b = Buffer.from(r.power_b64, 'base64');
    return Array.from(r.dtype === 'f32le' ? new Float32Array(b.buffer, b.byteOffset, b.length / 4)
                                          : new Float64Array(b.buffer, b.byteOffset, b.length / 8));
  };
  const cmp = (a, b, where) => {
    if (typeof a === 'number' && typeof b === 'number') {
      const rel = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300);
      if (where.endsWith('.power')) worstPower = Math.max(worstPower, rel);
      else worst = Math.max(worst, rel);
    } else if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) {
      a.forEach((v, i) => cmp(v, b[i], where));
    } else if (a && b && typeof a === 'object' && typeof b === 'object') {
      const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
      if (ka.join() !== kb.join()) fail(`keys differ at ${where}`);
      for (const k of ka) {
        if (k === 'power_b64') cmp(decode(a), decode(b), where + (a.dtype === 'f64le' ? '.power' : '.power32'));
        else cmp(a[k], b[k], where + '.' + k);
      }
    } else if (a !== b) fail(`value differs at ${where}: ${a} vs ${b}`);
  };
  cmp(fromBytes, nat, 'response');
  // Spectrogram power: the spec's 1e-10 parity target. Other measures go through
  // iterative refinement (Brent, root polishing), which amplifies libm rounding.
  if (worstPower > 1e-10) fail(`spectrogram power differs from native by ${worstPower.toExponential(2)}`);
  if (worst > 1e-6) fail(`bundle differs from native by ${worst.toExponential(2)} (relative)`);
}

const refused = JSON.parse(vm.runInContext(
  'wasm_bindgen.pipe_request_samples(samples, sr, JSON.stringify({wav_path: "a.wav", analyses: []}))', ctx));
if (refused.ok) fail('samples transport accepted wav_path');

console.log(`ok: praatfan-gpl ${vm.runInContext('wasm_bindgen.version()', ctx)}, ` +
            `${fromBytes.results.length} analyses, spectrogram ${spec.n_times}x${spec.n_freqs}` +
            (native ? `, max relative diff vs native: power ${worstPower.toExponential(2)}, other ${worst.toExponential(2)}` : ''));
