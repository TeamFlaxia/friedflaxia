// Node smoke test for the wasm-bindgen `nodejs` target.
// Usage: node scripts/wasm-smoke.cjs <pkg-dir> <embeddinggemma.gguf>

const fs = require('fs');
const path = require('path');

const pkgDir = process.argv[2];
const modelPath = process.argv[3];
if (!pkgDir || !modelPath) {
  console.error('usage: node scripts/wasm-smoke.cjs <pkg-dir> <embeddinggemma.gguf>');
  process.exit(2);
}

const { Model } = require(path.join(pkgDir, 'friedflaxia_wasm.js'));
const bytes = new Uint8Array(fs.readFileSync(modelPath));
const model = new Model(bytes);

if (model.architecture() !== 'gemma-embedding2') {
  throw new Error(`unexpected architecture: ${model.architecture()}`);
}

const ids = [2, 3689, 563, 506, 5279, 529, 7001, 236881, 1];
const embedding = model.embed(new Uint32Array(ids));
let norm = 0;
for (const value of embedding) norm += value * value;
norm = Math.sqrt(norm);

const decoded = model.decode(new Uint32Array(ids));
if (embedding.length !== 768) throw new Error(`expected 768 dims, got ${embedding.length}`);
if (Math.abs(norm - 1) > 1e-4) throw new Error(`embedding norm is ${norm}`);
if (decoded !== 'What is the capital of France?') {
  throw new Error(`unexpected decode: ${decoded}`);
}

console.log(`wasm smoke ok: dims=${embedding.length} norm=${norm.toFixed(6)}`);
