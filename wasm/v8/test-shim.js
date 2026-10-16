// TEST FIXTURE ONLY -- not shipped in praatfan-gpl-wasm-v8.zip.
// Copy of praatfan-core-clean's rust/wasm-v8/shim.js (MIT License,
// Copyright (c) 2026 praatfan contributors). Hosts (the R package) load
// their own copy before praatfan_gpl.js; smoke.mjs uses this one to load
// the bundle the same way.
//
// Host shim for the `--target no-modules` bundle in an EMBEDDED V8
// (R's V8 package, Python's mini-racer).
//
// An embedded V8 is the ECMAScript engine alone: no TextEncoder/TextDecoder
// (the wasm-bindgen glue moves every string through them), no atob, and no
// console in mini-racer (the panic hook writes to console.error).  Evaluate
// this file first, then praatfan.js, then
//   wasm_bindgen.initSync({ module: praatfan_b64bytes("<base64 of the .wasm>") })
// Audio goes in as bytes or as a Float64Array built from bytes -- never as
// JSON numbers, which a host encoder may round.

if (typeof globalThis.console === 'undefined') {
  globalThis.__praatfan_log = [];
  var __praatfan_push = function () {
    globalThis.__praatfan_log.push(Array.prototype.join.call(arguments, ' '));
  };
  globalThis.console = { log: __praatfan_push, warn: __praatfan_push, error: __praatfan_push };
}

if (typeof globalThis.TextEncoder === 'undefined') {
  globalThis.TextEncoder = class {
    encode(s) {
      s = s === undefined ? '' : String(s);
      const out = [];
      for (const ch of s) {
        const c = ch.codePointAt(0);
        if (c < 0x80) out.push(c);
        else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
        else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
        else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
      }
      return new Uint8Array(out);
    }
  };
}

if (typeof globalThis.TextDecoder === 'undefined') {
  globalThis.TextDecoder = class {
    constructor() {}
    decode(b) {
      if (!b) return '';
      const parts = [];
      let s = '';
      let i = 0;
      while (i < b.length) {
        let c = b[i++];
        if (c >= 0xf0) c = ((c & 7) << 18) | ((b[i++] & 63) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
        else if (c >= 0xe0) c = ((c & 15) << 12) | ((b[i++] & 63) << 6) | (b[i++] & 63);
        else if (c >= 0xc0) c = ((c & 31) << 6) | (b[i++] & 63);
        s += String.fromCodePoint(c);
        if (s.length >= 8192) { parts.push(s); s = ''; }
      }
      parts.push(s);
      return parts.join('');
    }
  };
}

// base64 -> bytes, written into `out` at `offset` (or a new array).  Returns
// the array when `out` is not given, else the offset after the last byte.
var __praatfan_b64map = (function () {
  const A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const m = new Uint8Array(128);
  for (let i = 0; i < 64; i++) m[A.charCodeAt(i)] = i;
  return m;
})();

function praatfan_b64decode(s, out, offset) {
  const m = __praatfan_b64map;
  let n = s.length;
  while (n > 0 && s.charCodeAt(n - 1) === 61) n--;            // '='
  const fresh = out === undefined;
  if (fresh) { out = new Uint8Array(Math.floor(n * 3 / 4)); offset = 0; }
  let o = offset;
  let i = 0;
  for (; i + 4 <= n; i += 4) {
    const v = (m[s.charCodeAt(i)] << 18) | (m[s.charCodeAt(i + 1)] << 12) |
              (m[s.charCodeAt(i + 2)] << 6) | m[s.charCodeAt(i + 3)];
    out[o++] = v >> 16; out[o++] = (v >> 8) & 255; out[o++] = v & 255;
  }
  const r = n - i;
  if (r >= 2) {
    const w = (m[s.charCodeAt(i)] << 18) | (m[s.charCodeAt(i + 1)] << 12) |
              (r === 3 ? (m[s.charCodeAt(i + 2)] << 6) : 0);
    out[o++] = w >> 16;
    if (r === 3) out[o++] = (w >> 8) & 255;
  }
  return fresh ? out : o;
}

function praatfan_b64bytes(s) { return praatfan_b64decode(s); }
