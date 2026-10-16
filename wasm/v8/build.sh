#!/usr/bin/env bash
# Build the embedded-V8 WASM bundle (praatfan-gpl-wasm-v8.zip) for hosts with
# a bare V8 (R's V8 package, Python's mini-racer).
#
#   wasm/v8/build.sh [native-pipe]   build into wasm/pkg-v8/, check, smoke-test, zip
#
# wasm-bindgen `--target no-modules` (global `wasm_bindgen`, loaded with
# `wasm_bindgen.initSync({module: bytes})`), built without default features so
# nothing touches `console`, with only the `pipe` exports added. Default wasm32
# target features; relaxed SIMD is rejected (older V8s, e.g. Node 18, refuse
# it). Requires V8 >= 9.6.
# The host supplies TextEncoder/TextDecoder via praatfan-core-clean's shim.js.
#
# If a native praatfan-gpl-pipe is given, the smoke test also checks that the
# bundle's output is byte-identical to it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CRATE/.." && pwd)"
OUT="$CRATE/pkg-v8"
NATIVE="${1:-}"

# Panic messages embed source paths; keep the builder's paths out of the .wasm.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
SEP=$'\x1f'
FLAGS="--remap-path-prefix=$ROOT=/src${SEP}--remap-path-prefix=$CARGO_HOME_DIR=/cargo${SEP}--remap-path-prefix=$HOME=/~"
if [[ -n "${RUSTFLAGS:-}" ]]; then
  FLAGS="$(printf '%s' "$RUSTFLAGS" | tr ' ' "$SEP")$SEP$FLAGS"
fi

rm -rf "$OUT"
(cd "$CRATE" && CARGO_ENCODED_RUSTFLAGS="$FLAGS" \
  wasm-pack build --release --target no-modules --out-dir pkg-v8 --out-name praatfan_gpl \
    --no-pack --no-default-features --features pipe)
rm -f "$OUT"/*.d.ts "$OUT/.gitignore"

for f in "$OUT/praatfan_gpl_bg.wasm" "$OUT/praatfan_gpl.js"; do
  if LC_ALL=C grep -a -q -E '/home/|/Users/|/runner/|C:\\Users' "$f"; then
    echo "error: $f contains a builder path:" >&2
    LC_ALL=C grep -a -o -E '(/home/|/Users/|/runner/|C:\\Users)[^"[:cntrl:]]{0,60}' "$f" | sort -u | head >&2
    exit 1
  fi
done

if command -v wasm-objdump >/dev/null; then
  # Dump to a file first: `objdump | grep -q` under pipefail fails silently
  # (grep exits on the first match, objdump dies of SIGPIPE).
  DIS="$(mktemp)"
  wasm-objdump -d "$OUT/praatfan_gpl_bg.wasm" > "$DIS"
  NSIMD="$(grep -c -E '\b(v128|[if](8|16|32|64)x[0-9]+)\.' "$DIS" || true)"
  NRELAXED="$(grep -c 'relaxed' "$DIS" || true)"
  rm -f "$DIS"
  if [[ "$NRELAXED" != 0 ]]; then
    echo "error: the .wasm uses $NRELAXED relaxed-SIMD instructions" >&2
    exit 1
  fi
  # Plain SIMD128 is accepted: faer's pulp/gemm compile SIMD128 paths in but
  # leave them switched off at runtime on this target, and the module already
  # needs V8 >= 9.6 (reference types, multi-value, bulk memory), above
  # SIMD128's 9.1. Native praatfan-gpl-pipe is the route for older hosts.
  echo "SIMD128 instructions present (runtime-disabled): $NSIMD; relaxed SIMD: 0"
else
  echo "warning: wasm-objdump not found; SIMD check skipped" >&2
fi

cp "$ROOT/LICENSE" "$OUT/LICENSE"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$CRATE/Cargo.toml" | head -1)"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
DIRTY=""
git -C "$ROOT" diff --quiet HEAD -- "$ROOT/src" "$CRATE/src" "$ROOT/Cargo.toml" "$CRATE/Cargo.toml" || DIRTY="+dirty"
{
  echo "praatfan-gpl $VERSION, wasm32 --target no-modules (embedded V8)"
  echo "commit: $COMMIT$DIRTY"
  echo "build: wasm/v8/build.sh (wasm-pack build --release --target no-modules --no-default-features --features pipe)"
  echo "remap: <repo>=/src, <cargo home>=/cargo, \$HOME=/~"
  echo "toolchain: $(rustc --version)"
  echo "wasm-pack: $(wasm-pack --version)"
  echo "wasm-bindgen: $(sed -n '/^name = "wasm-bindgen"$/{n;s/^version = "\(.*\)"/\1/p}' "$CRATE/Cargo.lock" | head -1)"
  echo "sha256:"
  (cd "$OUT" && sha256sum praatfan_gpl.js praatfan_gpl_bg.wasm | sed 's/^/  /')
} > "$OUT/PROVENANCE.txt"

node "$HERE/smoke.mjs" "$OUT" "$ROOT/tests/fixtures/one_two_three_four_five.wav" $NATIVE

rm -f "$ROOT/praatfan-gpl-wasm-v8.zip"
(cd "$OUT" && zip -q -X "$ROOT/praatfan-gpl-wasm-v8.zip" praatfan_gpl.js praatfan_gpl_bg.wasm LICENSE PROVENANCE.txt)
echo "wrote praatfan-gpl-wasm-v8.zip"
