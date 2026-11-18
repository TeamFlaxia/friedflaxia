use super::{Q8_0_BLOCK_BYTES, QK, QuantError, f16_to_f32};

/// Dequantize a sequence of complete Q8_0 blocks.
///
/// One block is an f16 scale followed by 32 signed 8-bit values.
pub fn dequantize_q8_0(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if bytes.len() % Q8_0_BLOCK_BYTES != 0 {
        return Err(QuantError::incomplete_block());
    }

    let blocks = bytes.len() / Q8_0_BLOCK_BYTES;
    let mut out = vec![0.0f32; blocks * QK];
    for (block_index, block) in bytes.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let base = block_index * QK;
        for (i, quant) in block[2..].iter().copied().enumerate() {
            out[base + i] = (quant as i8) as f32 * scale;
        }
    }
    Ok(out)
}
