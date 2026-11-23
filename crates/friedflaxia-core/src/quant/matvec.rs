//! Scalar quantized matrix-vector kernels.
//!
//! These are the first compute kernels: deterministic, allocation-free per row,
//! and portable to wasm32. SIMD backends can replace them without changing the
//! public API.

use super::{
    Q4_0_BLOCK_BYTES, Q4_1_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QK, QK_K, f16_to_f32,
};
use crate::simd;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatvecError {
    InvalidShape(&'static str),
    InputLength { expected: usize, actual: usize },
    WeightLength { expected: usize, actual: usize },
    IntegerOverflow,
}

impl fmt::Display for MatvecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(what) => write!(f, "invalid matvec shape: {what}"),
            Self::InputLength { expected, actual } => {
                write!(f, "matvec input length is {actual}, expected {expected}")
            }
            Self::WeightLength { expected, actual } => {
                write!(f, "matvec weight length is {actual}, expected {expected}")
            }
            Self::IntegerOverflow => f.write_str("matvec size overflow"),
        }
    }
}

impl std::error::Error for MatvecError {}

/// Tokens processed per weight-block pass in the batched kernels.
///
/// The block is dequantized once and reused for this many input vectors, which
/// is what turns the per-token memory traffic into a single streaming pass.
pub const MATMUL_TILE_TOKENS: usize = 16;

fn validate_shape(
    weights: &[u8],
    rows: usize,
    cols: usize,
    elements_per_block: usize,
    bytes_per_block: usize,
) -> Result<usize, MatvecError> {
    if rows == 0 || cols == 0 {
        return Err(MatvecError::InvalidShape("rows and cols must be non-zero"));
    }
    if cols % elements_per_block != 0 {
        return Err(MatvecError::InvalidShape(
            "cols must be a whole number of quant blocks",
        ));
    }
    let blocks_per_row = cols / elements_per_block;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(bytes_per_block))
        .ok_or(MatvecError::IntegerOverflow)?;
    if weights.len() != expected {
        return Err(MatvecError::WeightLength {
            expected,
            actual: weights.len(),
        });
    }
    Ok(blocks_per_row)
}

fn validate(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    elements_per_block: usize,
    bytes_per_block: usize,
) -> Result<usize, MatvecError> {
    let blocks_per_row = validate_shape(weights, rows, cols, elements_per_block, bytes_per_block)?;
    if input.len() != cols {
        return Err(MatvecError::InputLength {
            expected: cols,
            actual: input.len(),
        });
    }
    Ok(blocks_per_row)
}

/// Check the shapes shared by every batched kernel.
///
/// `inputs` is `[n_tokens][cols]` and `out` is `[n_tokens][rows]`, so a caller
/// can keep each token's projected vector contiguous.
#[allow(clippy::too_many_arguments)]
fn validate_matmul(
    weights: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    n_tokens: usize,
    out: &[f32],
    elements_per_block: usize,
    bytes_per_block: usize,
) -> Result<usize, MatvecError> {
    let blocks = validate_shape(weights, rows, cols, elements_per_block, bytes_per_block)?;
    if n_tokens == 0 {
        return Err(MatvecError::InvalidShape("n_tokens must be non-zero"));
    }
    let expected_inputs = n_tokens
        .checked_mul(cols)
        .ok_or(MatvecError::IntegerOverflow)?;
    if inputs.len() != expected_inputs {
        return Err(MatvecError::InputLength {
            expected: expected_inputs,
            actual: inputs.len(),
        });
    }
    let expected_out = n_tokens
        .checked_mul(rows)
        .ok_or(MatvecError::IntegerOverflow)?;
    if out.len() != expected_out {
        return Err(MatvecError::WeightLength {
            expected: expected_out,
            actual: out.len(),
        });
    }
    Ok(blocks)
}

/// Batched Q8_0 matmul: `out[token * rows + row] = dot(weights[row], inputs[token])`.
///
/// The loop order is the whole point: one quantized weight block is dequantized
/// once and then reused for every token in the tile, so the weight traffic drops
/// by the tile width instead of streaming the matrix once per token. The matrix
/// is therefore streamed once per 16-token tile, not once per token.
pub fn matmul_q8_0(
    weights: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    n_tokens: usize,
    out: &mut [f32],
) -> Result<(), MatvecError> {
    let blocks = validate_matmul(
        weights,
        rows,
        cols,
        inputs,
        n_tokens,
        out,
        QK,
        Q8_0_BLOCK_BYTES,
    )?;
    if n_tokens == 1 {
        let result = matvec_q8_0(weights, rows, cols, inputs)?;
        out.copy_from_slice(&result);
        return Ok(());
    }
    // Every (token, row) is written exactly once below, so the output does not
    // need a separate zeroing pass.
    for start_token in (0..n_tokens).step_by(MATMUL_TILE_TOKENS) {
        let tile = (n_tokens - start_token).min(MATMUL_TILE_TOKENS);
        for row in 0..rows {
            let row_bytes = &weights[row * blocks * Q8_0_BLOCK_BYTES..];
            let mut acc = [0.0f32; MATMUL_TILE_TOKENS];
            for block in 0..blocks {
                let bytes = &row_bytes[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES];
                let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                let offset = block * QK;
                let mut dequantized = [0.0f32; QK];
                for (index, value) in dequantized.iter_mut().enumerate() {
                    *value = (bytes[2 + index] as i8) as f32;
                }
                for index in 0..tile {
                    let input_block = &inputs[(start_token + index) * cols + offset..][..QK];
                    let dot = if simd::is_enabled() {
                        simd::dot(&dequantized, input_block)
                    } else {
                        let mut dot = 0.0f32;
                        for lane in 0..QK {
                            dot += dequantized[lane] * input_block[lane];
                        }
                        dot
                    };
                    acc[index] += scale * dot;
                }
            }
            for index in 0..tile {
                out[(start_token + index) * rows + row] = acc[index];
            }
        }
    }
    Ok(())
}

/// Batched Q4_0 matmul. See [`matmul_q8_0`] for the loop-order rationale.
pub fn matmul_q4_0(
    weights: &[u8],
    rows: usize,
    cols: usize,
    inputs: &[f32],
    n_tokens: usize,
    out: &mut [f32],
) -> Result<(), MatvecError> {
    let blocks = validate_matmul(
        weights,
        rows,
        cols,
        inputs,
        n_tokens,
        out,
        QK,
        Q4_0_BLOCK_BYTES,
    )?;
    if n_tokens == 1 {
        let result = matvec_q4_0(weights, rows, cols, inputs)?;
        out.copy_from_slice(&result);
        return Ok(());
    }
    // Every (token, row) is written exactly once below, so the output does not
    // need a separate zeroing pass.
    for start_token in (0..n_tokens).step_by(MATMUL_TILE_TOKENS) {
        let tile = (n_tokens - start_token).min(MATMUL_TILE_TOKENS);
        for row in 0..rows {
            let row_bytes = &weights[row * blocks * Q4_0_BLOCK_BYTES..];
            let mut acc = [0.0f32; MATMUL_TILE_TOKENS];
            for block in 0..blocks {
                let bytes = &row_bytes[block * Q4_0_BLOCK_BYTES..(block + 1) * Q4_0_BLOCK_BYTES];
                let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                let offset = block * QK;
                let mut dequantized = [0.0f32; QK];
                for lane in 0..QK / 2 {
                    let packed = bytes[2 + lane];
                    dequantized[lane] = ((packed & 0x0f) as i8 - 8) as f32;
                    dequantized[QK / 2 + lane] = ((packed >> 4) as i8 - 8) as f32;
                }
                for index in 0..tile {
                    let input_block = &inputs[(start_token + index) * cols + offset..][..QK];
                    let dot = if simd::is_enabled() {
                        simd::dot(&dequantized, input_block)
                    } else {
                        let mut dot = 0.0f32;
                        for lane in 0..QK {
                            dot += dequantized[lane] * input_block[lane];
                        }
                        dot
                    };
                    acc[index] += scale * dot;
                }
            }
            for index in 0..tile {
                out[(start_token + index) * rows + row] = acc[index];
            }
        }
    }
    Ok(())
}

pub fn matvec_q8_0(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    let blocks = validate(weights, rows, cols, input, QK, Q8_0_BLOCK_BYTES)?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_bytes = &weights[row * blocks * Q8_0_BLOCK_BYTES..];
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let bytes = &row_bytes[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES];
            let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let input_block = &input[block * QK..(block + 1) * QK];
            let dot = if simd::is_enabled() {
                let mut dequantized = [0.0f32; QK];
                for (index, value) in dequantized.iter_mut().enumerate() {
                    *value = (bytes[2 + index] as i8) as f32;
                }
                simd::dot(&dequantized, input_block)
            } else {
                let mut dot = 0.0f32;
                for index in 0..QK {
                    dot += (bytes[2 + index] as i8) as f32 * input_block[index];
                }
                dot
            };
            sum += scale * dot;
        }
        out.push(sum);
    }
    Ok(out)
}

pub fn matvec_q4_0(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    let blocks = validate(weights, rows, cols, input, QK, Q4_0_BLOCK_BYTES)?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_bytes = &weights[row * blocks * Q4_0_BLOCK_BYTES..];
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let bytes = &row_bytes[block * Q4_0_BLOCK_BYTES..(block + 1) * Q4_0_BLOCK_BYTES];
            let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let input_block = &input[block * QK..(block + 1) * QK];
            let dot = if simd::is_enabled() {
                let mut dequantized = [0.0f32; QK];
                for j in 0..QK / 2 {
                    let packed = bytes[2 + j];
                    dequantized[j] = ((packed & 0x0f) as i8 - 8) as f32;
                    dequantized[QK / 2 + j] = ((packed >> 4) as i8 - 8) as f32;
                }
                simd::dot(&dequantized, input_block)
            } else {
                let mut dot = 0.0f32;
                for j in 0..QK / 2 {
                    let packed = bytes[2 + j];
                    dot += ((packed & 0x0f) as i8 - 8) as f32 * input_block[j];
                    dot += ((packed >> 4) as i8 - 8) as f32 * input_block[QK / 2 + j];
                }
                dot
            };
            sum += scale * dot;
        }
        out.push(sum);
    }
    Ok(out)
}

pub fn matvec_q4_1(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    let blocks = validate(weights, rows, cols, input, QK, Q4_1_BLOCK_BYTES)?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_bytes = &weights[row * blocks * Q4_1_BLOCK_BYTES..];
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let bytes = &row_bytes[block * Q4_1_BLOCK_BYTES..(block + 1) * Q4_1_BLOCK_BYTES];
            let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            let min = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
            let input_block = &input[block * QK..(block + 1) * QK];
            let (dot, input_sum) = if simd::is_enabled() {
                let mut dequantized = [0.0f32; QK];
                for j in 0..QK / 2 {
                    let packed = bytes[4 + j];
                    dequantized[j] = (packed & 0x0f) as f32;
                    dequantized[QK / 2 + j] = (packed >> 4) as f32;
                }
                (simd::dot(&dequantized, input_block), simd::sum(input_block))
            } else {
                let mut dot = 0.0f32;
                let mut input_sum = 0.0f32;
                for j in 0..QK / 2 {
                    let packed = bytes[4 + j];
                    dot += (packed & 0x0f) as f32 * input_block[j];
                    dot += (packed >> 4) as f32 * input_block[QK / 2 + j];
                    input_sum += input_block[j] + input_block[QK / 2 + j];
                }
                (dot, input_sum)
            };
            sum += scale * dot + min * input_sum;
        }
        out.push(sum);
    }
    Ok(out)
}

pub fn matvec_q6_k(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    let blocks = validate(weights, rows, cols, input, QK_K, Q6_K_BLOCK_BYTES)?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_bytes = &weights[row * blocks * Q6_K_BLOCK_BYTES..];
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let bytes = &row_bytes[block * Q6_K_BLOCK_BYTES..(block + 1) * Q6_K_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([bytes[208], bytes[209]]));
            let ql = &bytes[..128];
            let qh = &bytes[128..192];
            let scales = &bytes[192..208];
            let input_base = block * QK_K;

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

                    let offset = input_base + n * 128 + l;
                    sum += d * scales[is] as i8 as f32 * q1 as f32 * input[offset];
                    sum += d * scales[is + 2] as i8 as f32 * q2 as f32 * input[offset + 32];
                    sum += d * scales[is + 4] as i8 as f32 * q3 as f32 * input[offset + 64];
                    sum += d * scales[is + 6] as i8 as f32 * q4 as f32 * input[offset + 96];
                }
            }
        }
        out.push(sum);
    }
    Ok(out)
}

pub fn matvec_f32(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    if rows == 0 || cols == 0 {
        return Err(MatvecError::InvalidShape("rows and cols must be non-zero"));
    }
    if input.len() != cols {
        return Err(MatvecError::InputLength {
            expected: cols,
            actual: input.len(),
        });
    }
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or(MatvecError::IntegerOverflow)?;
    if weights.len() != expected {
        return Err(MatvecError::WeightLength {
            expected,
            actual: weights.len(),
        });
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_bytes = &weights[row * cols * 4..(row + 1) * cols * 4];
        let sum = if simd::is_enabled() {
            simd::dot_f32_le_bytes(row_bytes, input)
        } else {
            let mut sum = 0.0f32;
            for (col, input_value) in input.iter().copied().enumerate() {
                let offset = col * 4;
                let value = f32::from_le_bytes([
                    row_bytes[offset],
                    row_bytes[offset + 1],
                    row_bytes[offset + 2],
                    row_bytes[offset + 3],
                ]);
                sum += value * input_value;
            }
            sum
        };
        out.push(sum);
    }
    Ok(out)
}

pub fn matvec_bf16(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, MatvecError> {
    if rows == 0 || cols == 0 {
        return Err(MatvecError::InvalidShape("rows and cols must be non-zero"));
    }
    if input.len() != cols {
        return Err(MatvecError::InputLength {
            expected: cols,
            actual: input.len(),
        });
    }
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or(MatvecError::IntegerOverflow)?;
    if weights.len() != expected {
        return Err(MatvecError::WeightLength {
            expected,
            actual: weights.len(),
        });
    }
    let mut out = Vec::with_capacity(rows);
    let mut scratch = if simd::is_enabled() {
        vec![0.0f32; cols]
    } else {
        Vec::new()
    };
    for row in 0..rows {
        let sum = if simd::is_enabled() {
            for (col, value) in scratch.iter_mut().enumerate() {
                let offset = (row * cols + col) * 2;
                let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                *value = f32::from_bits((bits as u32) << 16);
            }
            simd::dot(&scratch, input)
        } else {
            let mut sum = 0.0f32;
            for (col, input_value) in input.iter().copied().enumerate() {
                let offset = (row * cols + col) * 2;
                let bits = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                sum += f32::from_bits((bits as u32) << 16) * input_value;
            }
            sum
        };
        out.push(sum);
    }
    Ok(out)
}
