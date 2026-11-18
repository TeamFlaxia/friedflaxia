# FriedFlaxia wasm package

Build the browser/Node bindings with:

```sh
./scripts/build-wasm.sh web          # browser ESM package in crates/friedflaxia-wasm/pkg
./scripts/build-wasm.sh nodejs /tmp/friedflaxia-nodejs
```

The browser package exposes `Model`:

```js
import { Model } from './pkg/friedflaxia_wasm.js';

const model = new Model(new Uint8Array(await (await fetch(ggufUrl)).arrayBuffer()));
const ids = model.encode('Hello world', true);
const embedding = model.embed(ids);
```

`Model` keeps the GGUF bytes resident, so load a model once per worker and reuse
it for every task. The first integration target is `gemma-embedding2`
(`embed` / `embed_text`), followed by d1 (`decision`) and PLaMO (`next_token`).
