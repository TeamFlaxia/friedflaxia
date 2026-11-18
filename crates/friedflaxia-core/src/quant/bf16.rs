use super::QuantError;

/// Decode a sequence of bfloat16 values (2 bytes each) to f32.
///
/// BF16 is the upper 16 bits of an f32, so widening is a shift. Multimodal
/// projector tensors in the EmbeddingGemma GGUF use this type.
pub fn dequantize_bf16(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if bytes.len() % 2 != 0 {
        return Err(QuantError::incomplete_block());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
        .collect())
}
