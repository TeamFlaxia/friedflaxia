use std::fmt;

/// GGML tensor data type identifiers stored in a GGUF tensor index.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
    /// A type this engine does not implement yet, kept for diagnostics.
    Other(u32),
}

impl TensorType {
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::BF16 => 30,
            Self::Other(value) => value,
        }
    }

    /// `(elements per block, bytes per block)` for supported GGML types.
    pub fn block_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q4_0 => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::BF16 => Some((1, 2)),
            Self::Other(_) => None,
        }
    }

    /// Encoded byte length for `elements` scalar values.
    pub fn byte_len(self, elements: u64) -> Option<usize> {
        if elements == 0 {
            return Some(0);
        }
        let (elements_per_block, bytes_per_block) = self.block_layout()?;
        let elements_per_block = elements_per_block as u64;
        let blocks = elements.div_ceil(elements_per_block);
        let bytes = blocks.checked_mul(bytes_per_block as u64)?;
        usize::try_from(bytes).ok()
    }
}

impl fmt::Debug for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("F32"),
            Self::F16 => f.write_str("F16"),
            Self::Q4_0 => f.write_str("Q4_0"),
            Self::Q4_1 => f.write_str("Q4_1"),
            Self::Q5_0 => f.write_str("Q5_0"),
            Self::Q5_1 => f.write_str("Q5_1"),
            Self::Q8_0 => f.write_str("Q8_0"),
            Self::Q8_1 => f.write_str("Q8_1"),
            Self::Q2K => f.write_str("Q2_K"),
            Self::Q3K => f.write_str("Q3_K"),
            Self::Q4K => f.write_str("Q4_K"),
            Self::Q5K => f.write_str("Q5_K"),
            Self::Q6K => f.write_str("Q6_K"),
            Self::Q8K => f.write_str("Q8_K"),
            Self::BF16 => f.write_str("BF16"),
            Self::Other(value) => write!(f, "Other({value})"),
        }
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Numeric metadata as u64. Negative signed values are rejected.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(*value as u64),
            Self::U16(value) => Some(*value as u64),
            Self::U32(value) => Some(*value as u64),
            Self::U64(value) => Some(*value),
            Self::I8(value) => u64::try_from(*value).ok(),
            Self::I16(value) => u64::try_from(*value).ok(),
            Self::I32(value) => u64::try_from(*value).ok(),
            Self::I64(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|value| u32::try_from(value).ok())
    }
}
