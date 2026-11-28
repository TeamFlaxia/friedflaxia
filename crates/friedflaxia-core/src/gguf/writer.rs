//! GGUF v3 writer for `qwen3-dllm-lora` adapters and dLLM weights.
//!
//! The output is intentionally narrow: four LoRA metadata entries followed by
//! F32 `lora_a` / `lora_b` tensors, plus optional Q4_0 weight tensors. The byte
//! layout mirrors what the `Gguf` parser expects, including 32-byte alignment
//! before the tensor data, so a freshly written file can be loaded without an
//! extra conversion step.

use super::{TensorType, Value};
use crate::quant::{QuantErrorKind, quantize_q4_0};
use std::fmt;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
const ALIGNMENT: usize = 32;
const ARCHITECTURE: &str = "qwen3-dllm-lora";

#[derive(Debug, Clone, PartialEq)]
pub enum WriterError {
    ZeroRank,
    InvalidAlpha(f32),
    ValueCountMismatch,
    /// Q4_0 needs complete 32-value blocks; carry the offending count.
    UnalignedQ4_0(usize),
    /// NaN/inf cannot be quantized into a finite GGML block.
    NonFiniteValue,
    Overflow,
}

impl fmt::Display for WriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRank => f.write_str("lora rank must be non-zero"),
            Self::InvalidAlpha(alpha) => {
                write!(f, "lora alpha must be finite and positive, got {alpha}")
            }
            Self::ValueCountMismatch => f.write_str("tensor value count does not match dims"),
            Self::UnalignedQ4_0(count) => {
                write!(f, "Q4_0 value count must be a multiple of 32, got {count}")
            }
            Self::NonFiniteValue => f.write_str("tensor contains a non-finite value"),
            Self::Overflow => f.write_str("GGUF size overflow"),
        }
    }
}

impl std::error::Error for WriterError {}

/// On-disk encoding requested for a tensor added to the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    F32,
    Q4_0,
}

impl QuantKind {
    fn tensor_type(self) -> TensorType {
        match self {
            Self::F32 => TensorType::F32,
            Self::Q4_0 => TensorType::Q4_0,
        }
    }
}

struct PendingTensor {
    name: String,
    dims: Vec<u64>,
    kind: QuantKind,
    /// Encoded tensor payload, ready to be appended to the data section.
    data: Vec<u8>,
}

/// Builds one LoRA adapter GGUF entirely in memory.
pub struct Writer {
    metadata: Vec<(String, Value)>,
    tensors: Vec<PendingTensor>,
}

impl Writer {
    /// Start an adapter. `rank` and `alpha` follow the loader's constraints in
    /// `models::lora`: a zero rank or a non-finite / non-positive alpha is
    /// rejected instead of emitting an adapter that could never be loaded.
    pub fn new(rank: u32, alpha: f32, expert: &str) -> Result<Self, WriterError> {
        if rank == 0 {
            return Err(WriterError::ZeroRank);
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(WriterError::InvalidAlpha(alpha));
        }
        Ok(Self {
            metadata: vec![
                (
                    "general.architecture".to_owned(),
                    Value::String(ARCHITECTURE.to_owned()),
                ),
                (format!("{ARCHITECTURE}.rank"), Value::U32(rank)),
                (format!("{ARCHITECTURE}.alpha"), Value::F32(alpha)),
                (
                    format!("{ARCHITECTURE}.expert"),
                    Value::String(expert.to_owned()),
                ),
            ],
            tensors: Vec::new(),
        })
    }

    /// Append one F32 tensor with GGML dimensions `[cols, rows, ...]`.
    ///
    /// `values` must hold exactly `dims.iter().product()` elements; the byte
    /// length is derived from that count so a truncated adapter is rejected
    /// before anything is serialized.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        values: &[f32],
    ) -> Result<(), WriterError> {
        self.add_quantized_tensor(name, dims, values, QuantKind::F32)
    }

    /// Append one tensor encoded as F32 or Q4_0 with GGML dimensions
    /// `[cols, rows, ...]`.
    ///
    /// `values` must hold exactly `dims.iter().product()` elements. Q4_0
    /// additionally requires that count to be a multiple of 32 and every value
    /// to be finite; its payload is 18 bytes per 32 weights.
    pub fn add_quantized_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        values: &[f32],
        quant: QuantKind,
    ) -> Result<(), WriterError> {
        let expected = dims.iter().try_fold(1usize, |acc, dim| {
            let dim = usize::try_from(*dim).map_err(|_| WriterError::Overflow)?;
            acc.checked_mul(dim).ok_or(WriterError::Overflow)
        })?;
        if expected != values.len() {
            return Err(WriterError::ValueCountMismatch);
        }
        let data = match quant {
            QuantKind::F32 => {
                let mut data = Vec::with_capacity(values.len().saturating_mul(4));
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                data
            }
            QuantKind::Q4_0 => quantize_q4_0(values).map_err(|error| match error.kind {
                QuantErrorKind::IncompleteBlock => WriterError::UnalignedQ4_0(values.len()),
                QuantErrorKind::NonFiniteValue => WriterError::NonFiniteValue,
            })?,
        };
        self.tensors.push(PendingTensor {
            name: name.to_owned(),
            dims: dims.to_vec(),
            kind: quant,
            data,
        });
        Ok(())
    }

    /// Serialize the complete GGUF file, including aligned tensor payloads.
    pub fn finish(self) -> Vec<u8> {
        let data_len: usize = self.tensors.iter().map(|tensor| tensor.data.len()).sum();
        let mut out = Vec::with_capacity(256 + data_len);
        self.write_header(&mut out);
        self.write_metadata(&mut out);
        self.write_tensor_infos(&mut out);
        Self::write_padding(&mut out);
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
            Self::write_padding(&mut out);
        }
        out
    }

    fn write_header(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
    }

    fn write_metadata(&self, out: &mut Vec<u8>) {
        for (key, value) in &self.metadata {
            put_string(out, key);
            put_value(out, value);
        }
    }

    /// GGUF tensor offsets are the cumulative sum of *padded* tensor sizes.
    /// llama.cpp rejects files that use raw byte lengths instead.
    fn padded_len(len: usize) -> usize {
        len.div_ceil(ALIGNMENT) * ALIGNMENT
    }

    fn write_tensor_infos(&self, out: &mut Vec<u8>) {
        let mut offset = 0u64;
        for tensor in &self.tensors {
            put_string(out, &tensor.name);
            put_u32(out, tensor.dims.len() as u32);
            for dim in &tensor.dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            put_u32(out, tensor.kind.tensor_type().as_u32());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += Self::padded_len(tensor.data.len()) as u64;
        }
    }

    fn write_padding(out: &mut Vec<u8>) {
        while out.len() % ALIGNMENT != 0 {
            out.push(0);
        }
    }
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// GGUF metadata type ids, matching `Gguf::parse` and gguf-py.
fn value_type_id(value: &Value) -> u32 {
    match value {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(values) => values.first().map(value_type_id).unwrap_or(0),
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(out: &mut Vec<u8>, value: &Value) {
    put_u32(out, value_type_id(value));
    put_payload(out, value);
}

fn put_payload(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U8(value) => out.push(*value),
        Value::I8(value) => out.push(*value as u8),
        Value::U16(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::I16(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::U32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::I32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::F32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::Bool(value) => out.push(u8::from(*value)),
        Value::String(value) => put_string(out, value),
        Value::Array(values) => {
            put_u32(out, values.first().map(value_type_id).unwrap_or(0));
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for item in values {
                put_payload(out, item);
            }
        }
        Value::U64(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::I64(value) => out.extend_from_slice(&value.to_le_bytes()),
        Value::F64(value) => out.extend_from_slice(&value.to_le_bytes()),
    }
}
