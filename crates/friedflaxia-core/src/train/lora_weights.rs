//! Flat-parameter view of a `qwen3-dllm-lora` adapter.
//!
//! Training only cares about the parameter vector, not about GGUF plumbing.
//! This module converts between the adapter format produced by
//! [`crate::gguf::writer::Writer`] and a flat `f32` vector, and writes the
//! updated values back into a loadable adapter.

use crate::gguf::writer::{Writer, WriterError};
use crate::gguf::{ConfigError, Gguf, GgufError, ModelConfig, TensorType};
use crate::models::lora::{LoraModule, LORA_ARCHITECTURE};
use crate::tensor::{tensor, TensorError};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

// `LoraModule` lives in this crate but does not implement `Ord`; supplying the
// ordering here lets it serve as a stable BTreeMap key without changing the
// inference-side module. `index()` is a bijection over `ALL`, so this agrees
// with the derived `PartialEq`.
impl PartialOrd for LoraModule {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LoraModule {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index().cmp(&other.index())
    }
}

/// Which half of an A/B low-rank pair a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    A,
    B,
}

impl Side {
    pub fn segment(self) -> &'static str {
        match self {
            Self::A => "lora_a",
            Self::B => "lora_b",
        }
    }
}

/// LoRA tensors keyed by `(layer, module, side)` plus adapter metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraWeights {
    tensors: BTreeMap<(usize, LoraModule, Side), Vec<f32>>,
    rank: usize,
    alpha: f32,
    expert: String,
}

impl LoraWeights {
    pub fn from_gguf(gguf: &Gguf, bytes: &[u8]) -> Result<Self, WeightError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture() != LORA_ARCHITECTURE {
            return Err(WeightError::Architecture(config.architecture().to_owned()));
        }
        let rank = config.u32("rank")?.ok_or(WeightError::MissingRank)? as usize;
        if rank == 0 {
            return Err(WeightError::InvalidRank);
        }
        let alpha = config.f32("alpha")?.unwrap_or(rank as f32);
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(WeightError::InvalidAlpha(alpha));
        }
        let expert = config.string("expert")?.unwrap_or("").to_owned();

        let mut tensors = BTreeMap::new();
        for info in gguf.tensors() {
            let Some(key) = parse_tensor_name(&info.name) else {
                if info.name.contains("lora_") {
                    return Err(WeightError::MalformedName(info.name.clone()));
                }
                continue;
            };
            if info.tensor_type != TensorType::F32 {
                return Err(WeightError::UnsupportedType(info.tensor_type));
            }
            let values = tensor(gguf, bytes, &info.name)?.vec()?;
            if values.is_empty() || values.len() % rank != 0 {
                return Err(WeightError::InvalidShape(info.name.clone()));
            }
            if tensors.insert(key, values).is_some() {
                return Err(WeightError::Duplicate(info.name.clone()));
            }
        }
        Ok(Self {
            tensors,
            rank,
            alpha,
            expert,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<(usize, LoraModule, Side), Vec<f32>> {
        &self.tensors
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn expert(&self) -> &str {
        &self.expert
    }

    fn total_len(&self) -> usize {
        self.tensors.values().map(Vec::len).sum()
    }

    /// Concatenate all values in BTreeMap key order.
    pub fn flatten(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.total_len());
        for values in self.tensors.values() {
            flat.extend_from_slice(values);
        }
        flat
    }

    /// Rebuild weights from a vector in [`Self::flatten`] order.
    ///
    /// # Panics
    /// Panics when `flat` has the wrong length; use [`Self::try_from_flat`] to
    /// handle that case without panicking.
    pub fn from_flat(&self, flat: &[f32]) -> Self {
        self.try_from_flat(flat)
            .expect("flat weight length mismatch")
    }

    pub fn try_from_flat(&self, flat: &[f32]) -> Result<Self, WeightError> {
        let expected = self.total_len();
        if flat.len() != expected {
            return Err(WeightError::LengthMismatch {
                expected,
                actual: flat.len(),
            });
        }
        let mut tensors = BTreeMap::new();
        let mut cursor = 0;
        for (key, values) in &self.tensors {
            let end = cursor + values.len();
            tensors.insert(*key, flat[cursor..end].to_vec());
            cursor = end;
        }
        Ok(Self {
            tensors,
            rank: self.rank,
            alpha: self.alpha,
            expert: self.expert.clone(),
        })
    }

    /// Serialize into a GGUF that `models::lora::LoraModel` can load.
    pub fn write_gguf(&self) -> Result<Vec<u8>, WeightError> {
        let rank = u32::try_from(self.rank).map_err(|_| WeightError::InvalidRank)?;
        let mut writer = Writer::new(rank, self.alpha, &self.expert)?;
        for ((layer, module, side), values) in &self.tensors {
            let name = format!("blk.{layer}.{}.{}.weight", module.suffix(), side.segment());
            let dims = tensor_dims(*side, values.len(), self.rank, &name)?;
            writer.add_tensor(&name, &dims, values)?;
        }
        Ok(writer.finish())
    }
}

/// GGML `[cols, rows]` dimensions for one low-rank half.
fn tensor_dims(side: Side, len: usize, rank: usize, name: &str) -> Result<[u64; 2], WeightError> {
    let dim = len.checked_div(rank).ok_or(WeightError::InvalidRank)?;
    if dim == 0 || dim.checked_mul(rank) != Some(len) {
        return Err(WeightError::InvalidShape(name.to_owned()));
    }
    let dim = u64::try_from(dim).map_err(|_| WeightError::InvalidShape(name.to_owned()))?;
    let rank = u64::try_from(rank).map_err(|_| WeightError::InvalidRank)?;
    Ok(match side {
        Side::A => [dim, rank],
        Side::B => [rank, dim],
    })
}

fn parse_tensor_name(name: &str) -> Option<(usize, LoraModule, Side)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() != 5 || parts[0] != "blk" || parts[4] != "weight" {
        return None;
    }
    let layer = parts[1].parse().ok()?;
    let module = LoraModule::ALL
        .into_iter()
        .find(|module| module.suffix() == parts[2])?;
    let side = match parts[3] {
        "lora_a" => Side::A,
        "lora_b" => Side::B,
        _ => return None,
    };
    Some((layer, module, side))
}

#[derive(Debug)]
pub enum WeightError {
    Gguf(GgufError),
    Config(ConfigError),
    Tensor(TensorError),
    Writer(WriterError),
    Architecture(String),
    MissingRank,
    InvalidRank,
    InvalidAlpha(f32),
    UnsupportedType(TensorType),
    MalformedName(String),
    Duplicate(String),
    InvalidShape(String),
    LengthMismatch { expected: usize, actual: usize },
}

impl From<GgufError> for WeightError {
    fn from(error: GgufError) -> Self {
        Self::Gguf(error)
    }
}

impl From<ConfigError> for WeightError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<TensorError> for WeightError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<WriterError> for WeightError {
    fn from(error: WriterError) -> Self {
        Self::Writer(error)
    }
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf(error) => write!(f, "GGUF error: {error}"),
            Self::Config(error) => write!(f, "adapter config error: {error}"),
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
            Self::Writer(error) => write!(f, "adapter writer error: {error}"),
            Self::Architecture(arch) => write!(f, "expected {LORA_ARCHITECTURE}, got {arch}"),
            Self::MissingRank => f.write_str("missing lora rank"),
            Self::InvalidRank => f.write_str("lora rank must be non-zero"),
            Self::InvalidAlpha(alpha) => {
                write!(f, "lora alpha must be finite and positive, got {alpha}")
            }
            Self::UnsupportedType(tensor_type) => {
                write!(f, "adapter tensor must be F32, got {tensor_type:?}")
            }
            Self::MalformedName(name) => write!(f, "malformed lora tensor name: {name}"),
            Self::Duplicate(name) => write!(f, "duplicate lora tensor: {name}"),
            Self::InvalidShape(name) => write!(f, "invalid shape for lora tensor {name}"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "flat weight length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::writer::Writer;
    use crate::gguf::{Gguf, TensorType};
    use crate::models::lora::LoraModule;
    use std::collections::BTreeMap;

    const RANK: usize = 2;
    const ALPHA: f32 = 4.0;
    const EXPERT: &str = "code";

    /// Four tensors over two modules, with deliberately distinct values so a
    /// reordered or truncated vector cannot pass the round trip unnoticed.
    fn sample_weights() -> LoraWeights {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            (0, LoraModule::AttnQ, Side::A),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        tensors.insert(
            (0, LoraModule::AttnQ, Side::B),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        );
        tensors.insert(
            (0, LoraModule::FfnDown, Side::A),
            vec![15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
        );
        tensors.insert(
            (0, LoraModule::FfnDown, Side::B),
            vec![23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
        );
        LoraWeights {
            tensors,
            rank: RANK,
            alpha: ALPHA,
            expert: EXPERT.to_owned(),
        }
    }

    /// Build the same adapter with the production GGUF writer.
    fn adapter_bytes() -> Vec<u8> {
        let mut writer = Writer::new(RANK as u32, ALPHA, EXPERT).unwrap();
        writer
            .add_tensor(
                "blk.0.attn_q.lora_a.weight",
                &[3, RANK as u64],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap();
        writer
            .add_tensor(
                "blk.0.attn_q.lora_b.weight",
                &[RANK as u64, 4],
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            )
            .unwrap();
        writer
            .add_tensor(
                "blk.0.ffn_down.lora_a.weight",
                &[4, RANK as u64],
                &[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            )
            .unwrap();
        writer
            .add_tensor(
                "blk.0.ffn_down.lora_b.weight",
                &[RANK as u64, 3],
                &[23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
            )
            .unwrap();
        writer.finish()
    }

    /// Patch one tensor-type field so an otherwise valid adapter contains a
    /// non-F32 tensor (the writer itself only emits F32).
    fn patch_tensor_type(bytes: &mut [u8], name: &str, tensor_type: TensorType) {
        let pos = bytes
            .windows(name.len())
            .position(|window| window == name.as_bytes())
            .expect("tensor name is present");
        let mut cursor = pos + name.len();
        let n_dims = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + n_dims * 8;
        bytes[cursor..cursor + 4].copy_from_slice(&tensor_type.as_u32().to_le_bytes());
    }

    #[test]
    fn gguf_round_trip_preserves_values_and_metadata() {
        let bytes = adapter_bytes();
        let gguf = Gguf::parse(&bytes).unwrap();
        let first = LoraWeights::from_gguf(&gguf, &bytes).unwrap();
        assert_eq!(first, sample_weights());
        assert_eq!(first.rank(), RANK);
        assert_eq!(first.alpha(), ALPHA);
        assert_eq!(first.expert(), EXPERT);

        let written = first.write_gguf().unwrap();
        let rewritten_gguf = Gguf::parse(&written).unwrap();
        assert_eq!(rewritten_gguf.tensor_count(), 4);
        let second = LoraWeights::from_gguf(&rewritten_gguf, &written).unwrap();
        assert_eq!(second, first);
        assert_eq!(written, bytes);
    }

    #[test]
    fn flatten_and_from_flat_are_inverses_in_key_order() {
        let weights = sample_weights();
        let flat = weights.flatten();
        let expected: Vec<f32> = weights
            .tensors()
            .values()
            .flat_map(|values| values.iter().copied())
            .collect();
        assert_eq!(flat, expected);

        let rebuilt = weights.from_flat(&flat);
        assert_eq!(rebuilt, weights);

        let mut changed = flat.clone();
        changed[0] = 99.0;
        let changed = weights.from_flat(&changed);
        assert_eq!(changed.tensors()[&(0, LoraModule::AttnQ, Side::A)][0], 99.0);
        assert_eq!(weights.tensors()[&(0, LoraModule::AttnQ, Side::A)][0], 1.0);
    }

    #[test]
    #[should_panic(expected = "flat weight length")]
    fn from_flat_panics_on_wrong_length() {
        let weights = sample_weights();
        let flat = weights.flatten();
        let _ = weights.from_flat(&flat[..flat.len() - 1]);
    }

    #[test]
    fn try_from_flat_reports_length_mismatch() {
        let weights = sample_weights();
        let flat = weights.flatten();
        let error = weights.try_from_flat(&[0.0]).unwrap_err();
        assert!(
            matches!(
                error,
                WeightError::LengthMismatch { expected, actual }
                    if expected == flat.len() && actual == 1
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn from_gguf_rejects_non_f32_tensor() {
        let mut bytes = adapter_bytes();
        patch_tensor_type(&mut bytes, "blk.0.attn_q.lora_a.weight", TensorType::F16);
        let gguf = Gguf::parse(&bytes).unwrap();
        let error = LoraWeights::from_gguf(&gguf, &bytes).unwrap_err();
        assert!(
            matches!(error, WeightError::UnsupportedType(TensorType::F16)),
            "unexpected error: {error}"
        );
    }
}
