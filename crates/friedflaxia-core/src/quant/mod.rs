//! GGML quantization block formats used by GGUF.
//!
//! Only the formats needed by the first Flaxia models are implemented here.
//! Q4_0 and Q8_0 cover the PLaMO, d1-omni and EmbeddingGemma GGUF baselines.

mod bf16;
mod f16;
mod q4_0;
mod q4_1;
mod q6_k;
mod q8_0;

pub use bf16::dequantize_bf16;
pub use f16::f16_to_f32;
pub use q4_0::dequantize_q4_0;
pub use q4_1::dequantize_q4_1;
pub use q6_k::dequantize_q6_k;
pub use q8_0::dequantize_q8_0;

use std::fmt;

/// Number of weights in one Q4_0/Q4_1/Q8_0 block.
pub const QK: usize = 32;
/// Number of weights in one K-quant block.
pub const QK_K: usize = 256;
/// Bytes in one Q4_0 block: one f16 scale plus 16 packed nibble bytes.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes in one Q4_1 block: f16 scale, f16 minimum, 16 packed nibble bytes.
pub const Q4_1_BLOCK_BYTES: usize = 20;
/// Bytes in one Q6_K block: 128 ql + 64 qh + 16 scales + one f16 scale.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Bytes in one Q8_0 block: one f16 scale plus 32 signed bytes.
pub const Q8_0_BLOCK_BYTES: usize = 34;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantError {
    pub kind: QuantErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantErrorKind {
    /// Input length is not a whole number of quant blocks.
    IncompleteBlock,
}

impl QuantError {
    pub(crate) fn incomplete_block() -> Self {
        Self {
            kind: QuantErrorKind::IncompleteBlock,
        }
    }
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            QuantErrorKind::IncompleteBlock => {
                f.write_str("quantized input contains an incomplete block")
            }
        }
    }
}

impl std::error::Error for QuantError {}
