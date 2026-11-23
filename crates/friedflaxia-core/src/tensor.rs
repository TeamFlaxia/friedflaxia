//! Access to individual GGUF tensors with shape-checked operations.

use crate::gguf::{Gguf, GgufError, TensorInfo, TensorType};
use crate::quant::{
    MatvecError, QuantError, dequantize_bf16, dequantize_q4_0, dequantize_q6_k, dequantize_q8_0,
    matmul_q4_0, matmul_q8_0, matvec_bf16, matvec_f32, matvec_q4_0, matvec_q4_1, matvec_q6_k,
    matvec_q8_0,
};
use std::fmt;

pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn rows(&self) -> usize {
        if self.info.dims.len() >= 2 {
            usize::try_from(self.info.dims[1]).unwrap_or(0)
        } else {
            1
        }
    }

    pub fn cols(&self) -> usize {
        self.info
            .dims
            .first()
            .and_then(|dim| usize::try_from(*dim).ok())
            .unwrap_or(0)
    }

    /// Matrix-vector product for a row-major GGUF matrix `[cols, rows]`.
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, TensorError> {
        let rows = self.rows();
        let cols = self.cols();
        if input.len() != cols {
            return Err(TensorError::Shape {
                tensor: self.info.name.clone(),
                expected: cols,
                actual: input.len(),
            });
        }
        let result = match self.info.tensor_type {
            TensorType::F32 => matvec_f32(self.bytes, rows, cols, input),
            TensorType::BF16 => matvec_bf16(self.bytes, rows, cols, input),
            TensorType::Q4_0 => matvec_q4_0(self.bytes, rows, cols, input),
            TensorType::Q4_1 => matvec_q4_1(self.bytes, rows, cols, input),
            TensorType::Q6K => matvec_q6_k(self.bytes, rows, cols, input),
            TensorType::Q8_0 => matvec_q8_0(self.bytes, rows, cols, input),
            other => return Err(TensorError::UnsupportedType(other)),
        };
        result.map_err(TensorError::Matvec)
    }

    /// Batched matrix-matrix product: `inputs` is `[n_tokens][cols]` and the
    /// result is `[n_tokens][rows]`.
    ///
    /// Types without a dedicated batched kernel fall back to one matvec per
    /// token, so callers never have to special-case the quant format.
    pub fn matmul(&self, inputs: &[f32], n_tokens: usize) -> Result<Vec<f32>, TensorError> {
        let rows = self.rows();
        let cols = self.cols();
        if n_tokens == 0 {
            return Err(TensorError::Shape {
                tensor: self.info.name.clone(),
                expected: 1,
                actual: 0,
            });
        }
        if inputs.len() != n_tokens.saturating_mul(cols) {
            return Err(TensorError::Shape {
                tensor: self.info.name.clone(),
                expected: n_tokens.saturating_mul(cols),
                actual: inputs.len(),
            });
        }
        if n_tokens == 1 {
            return self.matvec(inputs);
        }
        let mut out = vec![0.0f32; n_tokens * rows];
        let result = match self.info.tensor_type {
            TensorType::Q8_0 => matmul_q8_0(self.bytes, rows, cols, inputs, n_tokens, &mut out),
            TensorType::Q4_0 => matmul_q4_0(self.bytes, rows, cols, inputs, n_tokens, &mut out),
            _ => {
                for token in 0..n_tokens {
                    let start = token * cols;
                    let projected = self.matvec(&inputs[start..start + cols])?;
                    out[token * rows..token * rows + projected.len()].copy_from_slice(&projected);
                }
                Ok(())
            }
        };
        result.map_err(TensorError::Matvec)?;
        Ok(out)
    }

    /// Read one row of a matrix (or the whole 1-D tensor) as f32.
    pub fn row_vec(&self, row: usize) -> Result<Vec<f32>, TensorError> {
        let rows = self.rows();
        let cols = self.cols();
        if row >= rows {
            return Err(TensorError::RowOutOfRange {
                tensor: self.info.name.clone(),
                row,
                rows,
            });
        }
        let elements = self
            .info
            .element_count()
            .checked_div(rows.max(1) as u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(cols);
        let offset = row.checked_mul(elements).ok_or(TensorError::Shape {
            tensor: self.info.name.clone(),
            expected: elements,
            actual: 0,
        })?;
        let values = match self.info.tensor_type {
            TensorType::F32 => self.bytes[offset * 4..]
                .chunks_exact(4)
                .take(elements)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect(),
            TensorType::BF16 => dequantize_bf16(&self.bytes[offset * 2..offset * 2 + elements * 2])
                .map_err(TensorError::Quant)?,
            TensorType::Q4_0 => {
                let block_bytes = elements / 32 * 18;
                dequantize_q4_0(&self.bytes[offset / 32 * 18..offset / 32 * 18 + block_bytes])
                    .map_err(TensorError::Quant)?
            }
            TensorType::Q8_0 => {
                let block_bytes = elements / 32 * 34;
                dequantize_q8_0(&self.bytes[offset / 32 * 34..offset / 32 * 34 + block_bytes])
                    .map_err(TensorError::Quant)?
            }
            TensorType::Q6K => {
                let block_bytes = elements / 256 * 210;
                dequantize_q6_k(&self.bytes[offset / 256 * 210..offset / 256 * 210 + block_bytes])
                    .map_err(TensorError::Quant)?
            }
            other => return Err(TensorError::UnsupportedType(other)),
        };
        Ok(values)
    }

    /// Read a 1-D tensor (or a single-element tensor) as f32.
    pub fn vec(&self) -> Result<Vec<f32>, TensorError> {
        let elements =
            usize::try_from(self.info.element_count()).map_err(|_| TensorError::Shape {
                tensor: self.info.name.clone(),
                expected: 0,
                actual: 0,
            })?;
        match self.info.tensor_type {
            TensorType::F32 => Ok(self
                .bytes
                .chunks_exact(4)
                .take(elements)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect()),
            TensorType::BF16 => {
                dequantize_bf16(&self.bytes[..elements * 2]).map_err(TensorError::Quant)
            }
            other => Err(TensorError::UnsupportedType(other)),
        }
    }

    pub fn scalar(&self) -> Result<f32, TensorError> {
        let values = self.vec()?;
        values.first().copied().ok_or_else(|| TensorError::Shape {
            tensor: self.info.name.clone(),
            expected: 1,
            actual: 0,
        })
    }
}

pub fn tensor<'a>(
    gguf: &'a Gguf,
    file: &'a [u8],
    name: &str,
) -> Result<TensorView<'a>, TensorError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| TensorError::Missing(name.to_owned()))?;
    let bytes = gguf.tensor_data(file, name).map_err(TensorError::Gguf)?;
    Ok(TensorView { info, bytes })
}

#[derive(Debug)]
pub enum TensorError {
    Missing(String),
    Gguf(GgufError),
    Quant(QuantError),
    Matvec(MatvecError),
    UnsupportedType(TensorType),
    Shape {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    RowOutOfRange {
        tensor: String,
        row: usize,
        rows: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "missing tensor {name}"),
            Self::Gguf(error) => write!(f, "GGUF error: {error}"),
            Self::Quant(error) => write!(f, "quantization error: {error}"),
            Self::Matvec(error) => write!(f, "matvec error: {error}"),
            Self::UnsupportedType(tensor_type) => {
                write!(f, "unsupported tensor type {tensor_type:?}")
            }
            Self::Shape {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor {tensor} shape mismatch: expected {expected}, got {actual}"
            ),
            Self::RowOutOfRange { tensor, row, rows } => {
                write!(f, "tensor {tensor} row {row} is outside 0..{rows}")
            }
        }
    }
}

impl std::error::Error for TensorError {}
