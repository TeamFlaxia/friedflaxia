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

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::tensor::tensor;

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
}

struct Pair {
    a: String,
    b: String,
}

struct Layer {
    pairs: [Option<Pair>; 7],
}

/// A loaded adapter. `Gguf` is owned, so this can live next to the base engine.
pub struct LoraModel {
    gguf: Gguf,
    bytes: Vec<u8>,
    rank: usize,
    scale: f32,
    expert: Option<String>,
    layers: Vec<Layer>,
}

impl LoraModel {
    pub fn from_gguf(gguf: Gguf, bytes: Vec<u8>, base_layers: usize) -> Result<Self, ModelError> {
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
        if base_layers == 0 {
            return Err(ModelError::Invalid("base model has no layers"));
        }

        let mut layers = Vec::with_capacity(base_layers);
        for layer in 0..base_layers {
            let mut pairs: [Option<Pair>; 7] = std::array::from_fn(|_| None);
            for module in LoraModule::ALL {
                let a = format!("blk.{layer}.{}.lora_a.weight", module.suffix());
                let b = format!("blk.{layer}.{}.lora_b.weight", module.suffix());
                match (gguf.tensor(&a).is_some(), gguf.tensor(&b).is_some()) {
                    (true, true) => pairs[module.index()] = Some(Pair { a, b }),
                    (false, false) => {}
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

        Ok(Self {
            gguf,
            bytes,
            rank,
            scale: alpha / rank as f32,
            expert,
            layers,
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
        let in_dim = input.len() / n;
        let out_dim = out.len() / n;
        let a = tensor(&self.gguf, &self.bytes, &pair.a)?;
        let b = tensor(&self.gguf, &self.bytes, &pair.b)?;
        if a.cols() != in_dim || a.rows() != self.rank {
            return Err(ModelError::Shape(format!(
                "lora {} A shape: expected [{in_dim}, {}], got [{}, {}]",
                module.suffix(),
                self.rank,
                a.cols(),
                a.rows()
            )));
        }
        if b.cols() != self.rank || b.rows() != out_dim {
            return Err(ModelError::Shape(format!(
                "lora {} B shape: expected [{}, {out_dim}], got [{}, {}]",
                module.suffix(),
                self.rank,
                b.cols(),
                b.rows()
            )));
        }
        let projected = a.matmul(input, n)?;
        let delta = b.matmul(&projected, n)?;
        if delta.len() != out.len() {
            return Err(ModelError::Shape("lora delta length".into()));
        }
        for (value, add) in out.iter_mut().zip(&delta) {
            *value += self.scale * add;
        }
        Ok(())
    }
}
