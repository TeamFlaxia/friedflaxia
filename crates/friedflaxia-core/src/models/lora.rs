//! LoRA adapters for the qwen3-dllm denoiser.
//!
//! Theia's v1 experts are LoRA adapters over one base dLLM. An adapter is a
//! separate GGUF with `blk.{i}.{module}.lora_a.weight` / `lora_b.weight`
//! tensors and a small amount of metadata:
//!
//! ```text
//! general.architecture      = "qwen3-dllm-lora"
//! qwen3-dllm-lora.rank      = 16
//! qwen3-dllm-lora.alpha     = 32.0        (optional, defaults to rank)
//! qwen3-dllm-lora.expert    = "code"      (optional, informational)
//! ```
//!
//! `A` has shape `[in, rank]` and `B` has shape `[rank, out]`, matching the
//! `[cols, rows]` layout of the rest of the engine. A projection becomes
//! `W x + (alpha / rank) * B (A x)`.

// `dllm` imports this module to apply adapters; this module imports
// `DllmConfig` for shape validation. The cycle is type-only; keep it that way.
use super::ModelError;
use super::dllm::DllmConfig;
use crate::gguf::{Gguf, ModelConfig, TensorInfo};
use crate::tensor::TensorView;
use std::collections::HashMap;

pub const LORA_ARCHITECTURE: &str = "qwen3-dllm-lora";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraModule {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl LoraModule {
    pub const ALL: [Self; 7] = [
        Self::AttnQ,
        Self::AttnK,
        Self::AttnV,
        Self::AttnOutput,
        Self::FfnGate,
        Self::FfnUp,
        Self::FfnDown,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::AttnQ => 0,
            Self::AttnK => 1,
            Self::AttnV => 2,
            Self::AttnOutput => 3,
            Self::FfnGate => 4,
            Self::FfnUp => 5,
            Self::FfnDown => 6,
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnOutput => "attn_output",
            Self::FfnGate => "ffn_gate",
            Self::FfnUp => "ffn_up",
            Self::FfnDown => "ffn_down",
        }
    }

    fn from_suffix(suffix: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|module| module.suffix() == suffix)
    }

    /// `(input dim, output dim)` of the base projection this module adapts.
    fn dims(self, base: &DllmConfig) -> (usize, usize) {
        match self {
            Self::AttnQ => (base.n_embd, base.q_dim()),
            Self::AttnK | Self::AttnV => (base.n_embd, base.kv_dim()),
            Self::AttnOutput => (base.q_dim(), base.n_embd),
            Self::FfnGate | Self::FfnUp => (base.n_embd, base.n_ff),
            Self::FfnDown => (base.n_ff, base.n_embd),
        }
    }
}

struct Resolved {
    info: TensorInfo,
    start: usize,
    len: usize,
}

impl Resolved {
    fn view<'a>(&'a self, bytes: &'a [u8]) -> TensorView<'a> {
        TensorView {
            info: &self.info,
            bytes: &bytes[self.start..self.start + self.len],
        }
    }
}

struct Pair {
    a: Resolved,
    b: Resolved,
}

struct Layer {
    pairs: [Option<Pair>; 7],
}

/// A loaded adapter. `Gguf` and its bytes are owned, so this can live next to
/// the base engine without a self-referential struct.
pub struct LoraModel {
    bytes: Vec<u8>,
    rank: usize,
    scale: f32,
    expert: Option<String>,
    layers: Vec<Layer>,
    pair_count: usize,
}

impl LoraModel {
    pub fn from_gguf(gguf: Gguf, bytes: Vec<u8>, base: &DllmConfig) -> Result<Self, ModelError> {
        let config = ModelConfig::from_gguf(&gguf)?;
        if config.architecture() != LORA_ARCHITECTURE {
            return Err(ModelError::Invalid("expected qwen3-dllm-lora architecture"));
        }
        let rank = config
            .u32("rank")?
            .ok_or_else(|| ModelError::Shape("missing lora rank".into()))?
            as usize;
        if rank == 0 {
            return Err(ModelError::Shape("lora rank must be non-zero".into()));
        }
        let alpha = config.f32("alpha")?.unwrap_or(rank as f32);
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(ModelError::Shape("lora alpha must be positive".into()));
        }
        let expert = config.string("expert")?.map(str::to_owned);
        if base.n_layer == 0 {
            return Err(ModelError::Invalid("base model has no layers"));
        }

        // Collect exactly the tensors we will apply. Any tensor containing
        // `lora_` must match the canonical `blk.{i}.{module}.{side}.weight`
        // name, so a typo cannot silently skip an expert.
        let mut found: HashMap<(usize, usize, usize), String> = HashMap::new();
        for info in gguf.tensors() {
            if !info.name.contains("lora_") {
                continue;
            }
            let Some((layer, module_suffix, side)) = parse_lora_name(&info.name) else {
                return Err(ModelError::Shape(format!(
                    "malformed lora tensor name: {}",
                    info.name
                )));
            };
            let module = LoraModule::from_suffix(module_suffix).ok_or_else(|| {
                ModelError::Shape(format!(
                    "lora adapter has an unsupported module: {module_suffix}"
                ))
            })?;
            if layer >= base.n_layer {
                return Err(ModelError::Shape(format!(
                    "lora adapter has layer {layer} but the base has {}",
                    base.n_layer
                )));
            }
            let canonical = format!("blk.{layer}.{}.{side}.weight", module.suffix());
            if info.name != canonical {
                return Err(ModelError::Shape(format!(
                    "non-canonical lora tensor name: {} (expected {canonical})",
                    info.name
                )));
            }
            let side_index = if side == "lora_a" { 0 } else { 1 };
            if found
                .insert((layer, module.index(), side_index), info.name.clone())
                .is_some()
            {
                return Err(ModelError::Shape(format!(
                    "duplicate lora tensor: {}",
                    info.name
                )));
            }
        }

        let data_offset = gguf.data_offset();
        let mut layers = Vec::with_capacity(base.n_layer);
        let mut pair_count = 0usize;
        for layer in 0..base.n_layer {
            let mut pairs: [Option<Pair>; 7] = std::array::from_fn(|_| None);
            for module in LoraModule::ALL {
                let a_name = found.get(&(layer, module.index(), 0));
                let b_name = found.get(&(layer, module.index(), 1));
                match (a_name, b_name) {
                    (None, None) => {}
                    (Some(a_name), Some(b_name)) => {
                        let a = resolve(&gguf, &bytes, data_offset, a_name)?;
                        let b = resolve(&gguf, &bytes, data_offset, b_name)?;
                        check_pair(&a, &b, module, rank, base)?;
                        pairs[module.index()] = Some(Pair { a, b });
                        pair_count += 1;
                    }
                    _ => {
                        return Err(ModelError::Shape(format!(
                            "lora layer {layer} {} has only one of lora_a/lora_b",
                            module.suffix()
                        )));
                    }
                }
            }
            layers.push(Layer { pairs });
        }

        if pair_count == 0 {
            return Err(ModelError::Shape("lora adapter matched no modules".into()));
        }

        Ok(Self {
            bytes,
            rank,
            scale: alpha / rank as f32,
            expert,
            layers,
            pair_count,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn expert(&self) -> Option<&str> {
        self.expert.as_deref()
    }

    /// Number of A/B pairs actually applied by this adapter.
    pub fn pair_count(&self) -> usize {
        self.pair_count
    }

    /// Add `scale * B (A input)` to an already-computed base projection.
    ///
    /// `out` is row-major `[n][out_dim]` and `input` is `[n][in_dim]`.
    pub fn apply(
        &self,
        layer: usize,
        module: LoraModule,
        input: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let Some(pair) = self
            .layers
            .get(layer)
            .and_then(|layer| layer.pairs[module.index()].as_ref())
        else {
            return Ok(());
        };
        if n == 0 || input.len() % n != 0 || out.len() % n != 0 {
            return Err(ModelError::Shape("lora projection rows".into()));
        }
        let projected = pair.a.view(&self.bytes).matmul(input, n)?;
        let delta = pair.b.view(&self.bytes).matmul(&projected, n)?;
        if delta.len() != out.len() {
            return Err(ModelError::Shape("lora delta length".into()));
        }
        for (value, add) in out.iter_mut().zip(&delta) {
            *value += self.scale * add;
        }
        Ok(())
    }
}

fn parse_lora_name(name: &str) -> Option<(usize, &str, &str)> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() != 5 || parts[0] != "blk" || parts[4] != "weight" {
        return None;
    }
    let layer = parts[1].parse::<usize>().ok()?;
    let side = match parts[3] {
        "lora_a" | "lora_b" => parts[3],
        _ => return None,
    };
    Some((layer, parts[2], side))
}

fn resolve(
    gguf: &Gguf,
    bytes: &[u8],
    data_offset: usize,
    name: &str,
) -> Result<Resolved, ModelError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| ModelError::Shape(format!("missing lora tensor {name}")))?
        .clone();
    let len = info
        .byte_len()
        .ok_or_else(|| ModelError::Shape(format!("unsupported lora tensor type {name}")))?;
    let offset = usize::try_from(info.offset)
        .map_err(|_| ModelError::Shape(format!("lora tensor offset {name}")))?;
    let start = data_offset
        .checked_add(offset)
        .ok_or_else(|| ModelError::Shape(format!("lora tensor offset overflow: {name}")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| ModelError::Shape(format!("lora tensor length overflow: {name}")))?;
    if end > bytes.len() {
        return Err(ModelError::Shape(format!(
            "lora tensor out of bounds: {name}"
        )));
    }
    Ok(Resolved { info, start, len })
}

fn info_dim(info: &TensorInfo, index: usize) -> Result<usize, ModelError> {
    let value = info.dims.get(index).copied().ok_or_else(|| {
        ModelError::Shape(format!(
            "lora tensor {} has no dimension {index}",
            info.name
        ))
    })?;
    usize::try_from(value).map_err(|_| {
        ModelError::Shape(format!(
            "lora tensor {} dimension {index} is too large",
            info.name
        ))
    })
}

fn info_cols(info: &TensorInfo) -> Result<usize, ModelError> {
    info_dim(info, 0)
}

fn info_rows(info: &TensorInfo) -> Result<usize, ModelError> {
    if info.dims.len() >= 2 {
        info_dim(info, 1)
    } else {
        Ok(1)
    }
}

fn check_pair(
    a: &Resolved,
    b: &Resolved,
    module: LoraModule,
    rank: usize,
    base: &DllmConfig,
) -> Result<(), ModelError> {
    let (in_dim, out_dim) = module.dims(base);
    let a_cols = info_cols(&a.info)?;
    let a_rows = info_rows(&a.info)?;
    if a_cols != in_dim || a_rows != rank {
        return Err(ModelError::Shape(format!(
            "lora {} A shape: expected [{in_dim}, {rank}], got [{a_cols}, {a_rows}]",
            module.suffix()
        )));
    }
    let b_cols = info_cols(&b.info)?;
    let b_rows = info_rows(&b.info)?;
    if b_cols != rank || b_rows != out_dim {
        return Err(ModelError::Shape(format!(
            "lora {} B shape: expected [{rank}, {out_dim}], got [{b_cols}, {b_rows}]",
            module.suffix()
        )));
    }
    Ok(())
}
