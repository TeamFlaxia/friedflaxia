use super::{Q4_0_BLOCK_BYTES, QK, QuantError, f16_to_f32};

/// Dequantize a sequence of complete Q4_0 blocks.
///
/// GGML stores the first 16 values in the low nibbles of the packed bytes and
/// the next 16 values in the high nibbles; each nibble is biased by 8.
pub fn dequantize_q4_0(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if bytes.len() % Q4_0_BLOCK_BYTES != 0 {
        return Err(QuantError::incomplete_block());
    }

    let blocks = bytes.len() / Q4_0_BLOCK_BYTES;
    let mut out = vec![0.0f32; blocks * QK];
    for (block_index, block) in bytes.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = block_index * QK;
        for (j, packed) in block[2..].iter().copied().enumerate() {
            out[base + j] = ((packed & 0x0f) as i8 - 8) as f32 * scale;
            out[base + QK / 2 + j] = ((packed >> 4) as i8 - 8) as f32 * scale;
        }
    }
    Ok(out)
}
