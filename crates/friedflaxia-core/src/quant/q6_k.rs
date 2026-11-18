use super::{Q6_K_BLOCK_BYTES, QK_K, QuantError, f16_to_f32};

/// Dequantize a sequence of complete Q6_K blocks.
///
/// A Q6_K block holds 256 six-bit values in 128 `ql` bytes + 64 `qh` bytes,
/// followed by 16 signed int8 scales and one f16 block scale. The 6-bit
/// values are biased by -32, and each group of 16 values selects its own
/// scale.
pub fn dequantize_q6_k(bytes: &[u8]) -> Result<Vec<f32>, QuantError> {
    if bytes.len() % Q6_K_BLOCK_BYTES != 0 {
        return Err(QuantError::incomplete_block());
    }

    let blocks = bytes.len() / Q6_K_BLOCK_BYTES;
    let mut out = vec![0.0f32; blocks * QK_K];
    for (block_index, block) in bytes.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let base = block_index * QK_K;

        for n in 0..2usize {
            let ql = &ql[n * 64..n * 64 + 64];
            let qh = &qh[n * 32..n * 32 + 32];
            let scales = &scales[n * 8..n * 8 + 8];
            for l in 0..32usize {
                let is = l / 16;
                let qh_byte = qh[l];
                let q1 = ((ql[l] & 0x0f) as i32 | ((qh_byte & 3) as i32) << 4) - 32;
                let q2 = ((ql[l + 32] & 0x0f) as i32 | (((qh_byte >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[l] >> 4) as i32 | (((qh_byte >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | (((qh_byte >> 6) & 3) as i32) << 4) - 32;

                let offset = base + n * 128 + l;
                out[offset] = d * scales[is] as i8 as f32 * q1 as f32;
                out[offset + 32] = d * scales[is + 2] as i8 as f32 * q2 as f32;
                out[offset + 64] = d * scales[is + 4] as i8 as f32 * q3 as f32;
                out[offset + 96] = d * scales[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}
