#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-web}"
OUT_DIR="${2:-$ROOT/crates/friedflaxia-wasm/pkg}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

cd "$ROOT"
# The default artifact uses wasm SIMD. `FRIEDFLAXIA_WASM_SIMD=0` builds the
# baseline module that browsers without `simd128` load instead. The scalar
# kernels stay available at runtime for bit-exact comparisons.
SIMD="${FRIEDFLAXIA_WASM_SIMD:-1}"
FEATURES=""
if [ "$SIMD" = "1" ]; then
  FEATURES="-C target-feature=+simd128"
fi
# Panic locations embed absolute source paths. Remap them so a local build and
# CI produce the same bytes, which the vendoring check compares by SHA-256.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
REMAP="--remap-path-prefix=${CARGO_HOME_DIR}=/cargo --remap-path-prefix=${ROOT}=/friedflaxia"
RUSTFLAGS="${RUSTFLAGS:-} ${FEATURES} ${REMAP}" \
  cargo build --release --target wasm32-unknown-unknown -p friedflaxia-wasm
rm -rf "$OUT_DIR"
wasm-bindgen \
  --target "$TARGET" \
  --out-dir "$OUT_DIR" \
  "$TARGET_DIR/wasm32-unknown-unknown/release/friedflaxia_wasm.wasm"

echo "wasm package: $OUT_DIR"
ls -lh "$OUT_DIR"
