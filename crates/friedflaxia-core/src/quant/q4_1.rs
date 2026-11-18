use super::{Q4_1_BLOCK_BYTES, QK, QuantError, f16_to_f32};

/// Dequantize a sequence of complete Q4_1 blocks.
///
/// Q4_1 stores an f16 scale, an f16 minimum, and 32 unsigned 4-bit values.
/// The first 16 values are the low nibbles, the next 16 are the high nibbles.
pub fn dequantize_q4_1(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if bytes.len() % Q4_1_BLOCK_BYTES != 0 {
        return Err(QuantError::incomplete_block());
    }

    let blocks = bytes.len() / Q4_1_BLOCK_BYTES;
    let mut out = vec![0.0f32; blocks * QK];
    for (block_index, block) in bytes.chunks_exact(Q4_1_BLOCK_BYTES).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let base = block_index * QK;
        for (j, packed) in block[4..].iter().copied().enumerate() {
            out[base + j] = (packed & 0x0f) as f32 * scale + min;
            out[base + QK / 2 + j] = (packed >> 4) as f32 * scale + min;
        }
    }
    Ok(out)
}
