//! GGUF v2/v3 header and tensor-index parser.
//!
//! This is intentionally a small, dependency-free parser. It reads metadata
//! and tensor descriptors, and exposes tensor payloads by name. The caller can
//! hold the header in memory while streaming tensor byte ranges on demand.

use std::collections::HashMap;
use std::fmt;

mod reader;
mod value;

pub mod config;
pub mod tokenizer;
pub mod writer;
pub use config::{ConfigError, ModelConfig};
pub use value::{TensorType, Value};

use reader::Reader;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 10_000_000;
const MAX_ARRAY_DEPTH: u32 = 4;
const MAX_TENSOR_DIMS: u32 = 8;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_ALIGNMENT: u64 = 1 << 20;

/// One tensor as described by the GGUF tensor index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// GGML dimensions, `ne[0]` first (the contiguous dimension).
    pub dims: Vec<u64>,
    pub tensor_type: TensorType,
    /// Byte offset relative to the start of the tensor-data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of scalar elements, saturating on malformed dimensions.
    pub fn element_count(&self) -> u64 {
        self.dims
            .iter()
            .fold(1u64, |acc, dim| acc.saturating_mul(*dim))
    }

    /// Encoded byte length, or `None` for a type this parser does not know.
    pub fn byte_len(&self) -> Option<usize> {
        self.tensor_type.byte_len(self.element_count())
    }
}

/// Options for header parsing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParseOptions {
    /// Skip every `tokenizer.*` metadata value. Forward passes only need the
    /// tensor index and model configuration, and tokenizer vocabularies can be
    /// tens of megabytes.
    pub skip_tokenizer: bool,
}

/// Parsed GGUF header plus tensor metadata.
#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    metadata: HashMap<String, Value>,
    tensors: Vec<TensorInfo>,
    data_offset: usize,
    alignment: u64,
}

impl Gguf {
    /// Parse a complete header (and the alignment padding before tensor data).
    ///
    /// Tensor payload bytes are not required for this call; they are validated
    /// lazily by [`Gguf::tensor_data`].
    pub fn parse(bytes: &[u8]) -> Result<Self, GgufError> {
        Self::parse_with_options(bytes, ParseOptions::default())
    }

    /// Parse a header, optionally skipping large tokenizer metadata.
    pub fn parse_with_options(bytes: &[u8], options: ParseOptions) -> Result<Self, GgufError> {
        let mut reader = Reader::new(bytes);

        if reader.take(4)? != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = reader.u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::LimitExceeded("tensor count"));
        }
        if metadata_count > MAX_METADATA_ENTRIES {
            return Err(GgufError::LimitExceeded("metadata count"));
        }

        let mut metadata = HashMap::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = reader.string(MAX_STRING_BYTES)?;
            if options.skip_tokenizer && key.starts_with("tokenizer.") {
                skip_value(&mut reader, 0)?;
                continue;
            }
            let value = read_value(&mut reader, 0)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(value) => value
                .as_u64()
                .ok_or(GgufError::InvalidMetadata("general.alignment"))?,
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > MAX_ALIGNMENT {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.string(MAX_STRING_BYTES)?;
            let n_dims = reader.u32()?;
            if n_dims > MAX_TENSOR_DIMS {
                return Err(GgufError::LimitExceeded("tensor dimensions"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.u64()?);
            }
            let tensor_type = TensorType::from_u32(reader.u32()?);
            let offset = reader.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                tensor_type,
                offset,
            });
        }

        let data_offset = align_up(reader.position(), alignment)?;
        if data_offset > bytes.len() {
            return Err(GgufError::UnexpectedEof);
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Iterate over every metadata key/value pair.
    pub fn metadata_entries(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Absolute byte offset of the tensor-data section in the source buffer.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Alignment declared by the file (defaults to 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Return the encoded bytes of one tensor.
    pub fn tensor_data<'a>(&self, bytes: &'a [u8], name: &str) -> Result<&'a [u8], GgufError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| GgufError::UnknownTensor(name.to_owned()))?;
        let len = tensor
            .byte_len()
            .ok_or(GgufError::UnsupportedTensorType(tensor.tensor_type))?;
        let offset = usize::try_from(tensor.offset).map_err(|_| GgufError::IntegerOverflow)?;
        let start = self
            .data_offset
            .checked_add(offset)
            .ok_or(GgufError::IntegerOverflow)?;
        let end = start.checked_add(len).ok_or(GgufError::IntegerOverflow)?;
        bytes.get(start..end).ok_or(GgufError::TensorDataOutOfRange)
    }
}

fn align_up(value: usize, alignment: u64) -> Result<usize, GgufError> {
    let alignment = usize::try_from(alignment).map_err(|_| GgufError::IntegerOverflow)?;
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|padded| padded & !mask)
        .ok_or(GgufError::IntegerOverflow)
}

fn read_value(reader: &mut Reader<'_>, depth: u32) -> Result<Value, GgufError> {
    let value_type = reader.u32()?;
    read_value_of_type(reader, value_type, depth)
}

fn read_value_of_type(
    reader: &mut Reader<'_>,
    value_type: u32,
    depth: u32,
) -> Result<Value, GgufError> {
    let value = match value_type {
        0 => Value::U8(reader.u8()?),
        1 => Value::I8(reader.u8()? as i8),
        2 => Value::U16(reader.u16()?),
        3 => Value::I16(reader.u16()? as i16),
        4 => Value::U32(reader.u32()?),
        5 => Value::I32(reader.u32()? as i32),
        6 => Value::F32(f32::from_bits(reader.u32()?)),
        7 => match reader.u8()? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            _ => return Err(GgufError::InvalidMetadata("boolean value")),
        },
        8 => Value::String(reader.string(MAX_STRING_BYTES)?),
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(GgufError::LimitExceeded("array nesting"));
            }
            let element_type = reader.u32()?;
            if element_type == 9 {
                return Err(GgufError::InvalidMetadata(
                    "nested arrays are not supported",
                ));
            }
            let count = reader.u64()?;
            if count > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::LimitExceeded("array length"));
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_value_of_type(reader, element_type, depth + 1)?);
            }
            Value::Array(values)
        }
        10 => Value::U64(reader.u64()?),
        11 => Value::I64(reader.u64()? as i64),
        12 => Value::F64(f64::from_bits(reader.u64()?)),
        other => return Err(GgufError::UnknownMetadataType(other)),
    };
    Ok(value)
}

/// Advance past one metadata value without allocating its contents.
fn skip_value(reader: &mut Reader<'_>, depth: u32) -> Result<(), GgufError> {
    let value_type = reader.u32()?;
    skip_value_of_type(reader, value_type, depth)
}

fn skip_value_of_type(
    reader: &mut Reader<'_>,
    value_type: u32,
    depth: u32,
) -> Result<(), GgufError> {
    match value_type {
        0 | 1 | 7 => {
            reader.take(1)?;
        }
        2 | 3 => {
            reader.take(2)?;
        }
        4..=6 => {
            reader.take(4)?;
        }
        10..=12 => {
            reader.take(8)?;
        }
        8 => {
            let len = reader.u64()?;
            if len > MAX_STRING_BYTES {
                return Err(GgufError::LimitExceeded("string length"));
            }
            reader.take(usize::try_from(len).map_err(|_| GgufError::IntegerOverflow)?)?;
        }
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(GgufError::LimitExceeded("array nesting"));
            }
            let element_type = reader.u32()?;
            if element_type == 9 {
                return Err(GgufError::InvalidMetadata(
                    "nested arrays are not supported",
                ));
            }
            let count = reader.u64()?;
            if count > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::LimitExceeded("array length"));
            }
            for _ in 0..count {
                skip_value_of_type(reader, element_type, depth + 1)?;
            }
        }
        other => return Err(GgufError::UnknownMetadataType(other)),
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    BadMagic,
    UnsupportedVersion(u32),
    UnexpectedEof,
    LimitExceeded(&'static str),
    InvalidMetadata(&'static str),
    InvalidAlignment(u64),
    UnknownMetadataType(u32),
    UnknownTensor(String),
    UnsupportedTensorType(TensorType),
    TensorDataOutOfRange,
    IntegerOverflow,
    InvalidUtf8,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::BadMagic => f.write_str("not a GGUF file (bad magic)"),
            GgufError::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version {version}")
            }
            GgufError::UnexpectedEof => f.write_str("unexpected end of GGUF data"),
            GgufError::LimitExceeded(what) => write!(f, "GGUF {what} exceeds the parser limit"),
            GgufError::InvalidMetadata(what) => write!(f, "invalid GGUF metadata: {what}"),
            GgufError::InvalidAlignment(alignment) => {
                write!(f, "invalid GGUF alignment {alignment}")
            }
            GgufError::UnknownMetadataType(value_type) => {
                write!(f, "unknown GGUF metadata type {value_type}")
            }
            GgufError::UnknownTensor(name) => write!(f, "unknown GGUF tensor {name}"),
            GgufError::UnsupportedTensorType(tensor_type) => {
                write!(f, "unsupported GGUF tensor type {tensor_type:?}")
            }
            GgufError::TensorDataOutOfRange => {
                f.write_str("GGUF tensor data range is outside the buffer")
            }
            GgufError::IntegerOverflow => f.write_str("GGUF integer overflow"),
            GgufError::InvalidUtf8 => f.write_str("GGUF string is not valid UTF-8"),
        }
    }
}

impl std::error::Error for GgufError {}
