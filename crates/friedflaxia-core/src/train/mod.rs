//! Offline training primitives.
//!
//! Everything here is deliberately independent of the inference hot path:
//! the WASM node keeps running forward-only. These helpers are used by the
//! native `friedflaxia-train` CLI during off-time rounds.

pub mod es;
pub mod lora_grad;
pub mod lora_weights;
pub mod optimizer;
pub mod rng;
