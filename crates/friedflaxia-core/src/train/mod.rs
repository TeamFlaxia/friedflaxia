//! Offline training primitives.
//!
//! Everything here is deliberately independent of the inference hot path:
//! the WASM node keeps running forward-only. These helpers are used by the
//! native `friedflaxia-train` CLI during off-time rounds.

pub mod autograd;
pub mod es;
pub mod rng;
