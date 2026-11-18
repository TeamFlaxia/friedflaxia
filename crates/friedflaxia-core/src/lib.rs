//! FriedFlaxia inference core.
//!
//! The first milestone is deliberately narrow: read GGUF headers and tensor
//! metadata, and dequantize the Q4_0/Q8_0 blocks used by the Flaxia baseline
//! models. Model forward passes are built on top of this crate.

pub mod gguf;
pub mod quant;
