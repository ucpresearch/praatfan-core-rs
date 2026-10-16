// TEST FIXTURE ONLY -- not shipped in praatfan-gpl-wasm-v8.zip.
// Copy of praatfan-core-clean's rust/wasm-v8/shape.mjs at af632a1 (MIT License,
// Copyright (c) 2026 praatfan contributors): the cross-engine black-box shape
// check of the pipe response (fields, types, base64 lengths, errors), never
// values. Run by wasm/v8/build.sh. Refresh from upstream rather than editing.
//
// Black-box shape check of an embedded-V8 pipe bundle -- ours, or the GPL
// engine's praatfan-gpl-wasm-v8.zip -- against docs/pipe-spectrogram.md.
// Loads the bundle with our shim in a bare node:vm context and checks only
// what it returns (fields, types, base64 length, edge cases, errors), never
// its values: the engines' grids and values differ by design.
//
//   node rust/wasm-v8/shape.mjs rust/wasm-v8/shim.js <glue.js> <bg.wasm> <audio.wav>
//
// Clean room: for the GPL bundle, run this; do not open its files.
import fs from 'node:fs'; import vm from 'node:vm';
const [shim, glue, wasm, audio] = process.argv.slice(2);
const ctx = vm.createContext({ WebAssembly });
vm.runInContext(fs.readFileSync(shim, 'utf8'), ctx);
vm.runInContext(fs.readFileSync(glue, 'utf8'), ctx);
ctx.w = fs.readFileSync(wasm).toString('base64');
vm.runInContext('wasm_bindgen.initSync({module: praatfan_b64bytes(w)}); delete globalThis.w;', ctx);
ctx.a = fs.readFileSync(audio).toString('base64');
const run = (analyses, extra = {}) => {
  ctx.req = JSON.stringify(Object.assign({ wav_path: 'x.wav', analyses }, extra));
  return JSON.parse(vm.runInContext('wasm_bindgen.pipe_request_bytes(praatfan_b64bytes(a), req)', ctx));
};
let fails = 0;
const check = (cond, what) => { console.log((cond ? 'ok   ' : 'FAIL ') + what); if (!cond) fails++; };
const b64len = s => { const n = s.replace(/=+$/, '').length; return Math.floor(n * 3 / 4); };

check(typeof vm.runInContext('wasm_bindgen.version()', ctx) === 'string', 'version() is a string: ' + vm.runInContext('wasm_bindgen.version()', ctx));
for (const fn of ['pipe_request_bytes', 'pipe_request_samples'])
  check(typeof vm.runInContext('wasm_bindgen.' + fn, ctx) === 'function', fn + ' exported');

const full = run([{ type: 'spectrogram' }]);
check(full.ok === true, 'default spectrogram ok');
const r = full.results[0];
const keys = ['type','n_times','n_freqs','time_step','frequency_step','times','freqs','layout','dtype','power_b64'];
check(keys.every(k => k in r), 'all result fields present (' + keys.join(',') + ')');
check(Object.keys(r).filter(k => !keys.includes(k)).length === 0, 'no extra fields: ' + Object.keys(r).filter(k => !keys.includes(k)));
check(r.type === 'spectrogram' && r.layout === 'time-major' && r.dtype === 'f64le', 'type/layout/dtype values');
check(Number.isInteger(r.n_times) && Number.isInteger(r.n_freqs) && r.times.length === r.n_times && r.freqs.length === r.n_freqs, `grid ${r.n_times} x ${r.n_freqs} matches arrays`);
check(typeof r.time_step === 'number' && typeof r.frequency_step === 'number', `steps reported (${r.time_step}, ${r.frequency_step})`);
check(/^[A-Za-z0-9+/]*={0,2}$/.test(r.power_b64) && r.power_b64.length % 4 === 0, 'power_b64 is padded standard base64, no line breaks');
check(b64len(r.power_b64) === r.n_times * r.n_freqs * 8, 'base64 length = n_times*n_freqs*8');
const bytes = vm.runInContext('praatfan_b64bytes(s)', Object.assign(ctx, { s: r.power_b64 }));
const vals = new Float64Array(bytes.buffer, bytes.byteOffset, bytes.length / 8);
check(vals.every(v => Number.isFinite(v) && v >= 0), 'power values finite and >= 0');

const f32 = run([{ type: 'spectrogram', dtype: 'f32le', start_time: 0.5, end_time: 0.6 }]).results[0];
check(f32.dtype === 'f32le' && b64len(f32.power_b64) === f32.n_times * f32.n_freqs * 4, 'f32le length');
check(f32.n_times > 0 && f32.times.every(t => t >= 0.5 && t <= 0.6), 'range filters frames');
const i0 = r.times.indexOf(f32.times[0]);
check(i0 >= 0 && f32.times.every((t, k) => t === r.times[i0 + k]), 'range keeps the full grid times');
check(JSON.stringify(run([{ type: 'spectrogram', start_time: null, end_time: null }]).results[0]) === JSON.stringify(r), 'null = absent');
const empty = run([{ type: 'spectrogram', start_time: 99 }]);
check(empty.ok && empty.results[0].n_times === 0 && empty.results[0].times.length === 0 && empty.results[0].power_b64 === '' && empty.results[0].n_freqs > 0, 'empty range -> empty ok result');
for (const mf of [0, -1, 1e6]) {
  const x = run([{ type: 'spectrogram', max_frequency: mf }]);
  check(x.ok, `max_frequency ${mf} clamped, ok`);
}
for (const bad of [{ window_length: 0 }, { time_step: -1 }, { frequency_step: 0 }, { start_time: 0.6, end_time: 0.5 },
                   { window_shape: 'hamming' }, { dtype: 'f16' }, { bogus: 1 }]) {
  const x = run([Object.assign({ type: 'spectrogram' }, bad)]);
  check(x.ok === false && typeof x.error === 'string', 'error for ' + JSON.stringify(bad));
}
const hann = run([{ type: 'spectrogram', window_shape: 'hanning' }]);
check(hann.ok, 'hanning ok');
// samples transport: refuses wav_path/channel, accepts plain samples
ctx.sreq = JSON.stringify({ analyses: [{ type: 'intensity' }] });
const sam = JSON.parse(vm.runInContext('wasm_bindgen.pipe_request_samples(new Float64Array(16000).map((_, i) => Math.sin(i / 10) * 0.1), 16000, sreq)', ctx));
check(sam.ok === true, 'samples transport ok');
ctx.sreq = JSON.stringify({ wav_path: 'x.wav', analyses: [] });
check(JSON.parse(vm.runInContext('wasm_bindgen.pipe_request_samples(new Float64Array(100), 16000, sreq)', ctx)).ok === false, 'samples transport refuses wav_path');
ctx.sreq = JSON.stringify({ analyses: [] });
check(JSON.parse(vm.runInContext('wasm_bindgen.pipe_request_samples(new Float64Array(100), 0, sreq)', ctx)).ok === false, 'samples transport refuses sample_rate 0');
ctx.junk = new Uint8Array([1, 2, 3, 4]); ctx.req = JSON.stringify({ analyses: [{ type: 'intensity' }] });
const junk = JSON.parse(vm.runInContext('wasm_bindgen.pipe_request_bytes(junk, req)', ctx));
check(junk.ok === false && /<bytes>/.test(junk.error), 'undecodable bytes name "<bytes>": ' + junk.error);
// the other analyses keep the existing schema
const all = run([{ type: 'pitch_ac' }, { type: 'formant_burg' }, { type: 'intensity' }, { type: 'harmonicity_cc' },
                 { type: 'spectral_moments', times: [0.5] }, { type: 'band_energy', times: [0.5] }]);
check(all.ok, 'other analyses ok');
console.log('response keys:', Object.keys(all).join(','), '| result keys:', all.results.map(x => Object.keys(x).join('/')).join(' ; '));
console.log(fails ? `${fails} FAILED` : 'all shape checks passed');
process.exit(fails ? 1 : 0);
