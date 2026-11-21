#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-web}"
OUT_DIR="${2:-$ROOT/crates/friedflaxia-wasm/pkg}"

cd "$ROOT"
# The scalar fallback still exists for bit-exact comparisons, but the shipped
# artifact uses wasm SIMD. Keep this flag in sync with the vendored manifest.
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
  cargo build --release --target wasm32-unknown-unknown -p friedflaxia-wasm
rm -rf "$OUT_DIR"
wasm-bindgen \
  --target "$TARGET" \
  --out-dir "$OUT_DIR" \
  target/wasm32-unknown-unknown/release/friedflaxia_wasm.wasm

echo "wasm package: $OUT_DIR"
ls -lh "$OUT_DIR"
