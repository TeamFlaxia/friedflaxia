use super::{Q4_0_BLOCK_BYTES, QK, QuantError, QuantErrorKind, f16_to_f32, f32_to_f16};

/// Quantize f32 values into Q4_0 blocks (32 values, 18 bytes each).
///
/// The scale follows the ggml convention used by `convert_dllm.py`: it is the
/// block's extreme element divided by `-8.0`, so a positive extreme yields a
/// negative scale. Quantization is `floor(value / scale + 8.5)` clamped to the
/// biased nibble range `0..=15`; the first 16 nibbles go into the low halves of
/// the packed bytes and the next 16 into the high halves.
///
/// The value count must be a multiple of [`QK`] and every value must be finite;
/// otherwise an error is returned before any bytes are produced.
pub fn quantize_q4_0(values: &[f32]) -> Result<Vec<u8>, QuantError> {
    if values.len() % QK != 0 {
        return Err(QuantError::incomplete_block());
    }

    let mut out = Vec::with_capacity(values.len() / QK * Q4_0_BLOCK_BYTES);
    for block in values.chunks_exact(QK) {
        if block.iter().any(|value| !value.is_finite()) {
            return Err(QuantError {
                kind: QuantErrorKind::NonFiniteValue,
            });
        }

        // np.argmax picks the first element with the largest absolute value.
        let mut extreme = 0.0f32;
        let mut max_abs = -1.0f32;
        for &value in block {
            let magnitude = value.abs();
            if magnitude > max_abs {
                max_abs = magnitude;
                extreme = value;
            }
        }

        let scale = if extreme == 0.0 { 0.0 } else { extreme / -8.0 };
        let divisor = if scale == 0.0 { 1.0 } else { scale };
        out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());

        let mut packed = [0u8; QK / 2];
        for (index, &value) in block.iter().enumerate() {
            let quantized = (value / divisor + 8.5).floor();
            let nibble = if quantized <= 0.0 {
                0u8
            } else if quantized >= 15.0 {
                15u8
            } else {
                quantized as u8
            };
            if index < QK / 2 {
                packed[index] |= nibble;
            } else {
                packed[index - QK / 2] |= nibble << 4;
            }
        }
        out.extend_from_slice(&packed);
    }
    Ok(out)
}

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
