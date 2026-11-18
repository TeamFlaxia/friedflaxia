//! PLaMo 3 Translate generation forward.
//!
//! PLaMo 3 is a decoder-only model with a fused QKV projection, GQA (16 query
//! heads, one KV head), per-head Q/K RMSNorm, YaRN RoPE, alternating
//! sliding-window/full attention and post-attention/post-FFN RMSNorm.

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::ops::{rms_norm, rope_neox_yarn_inplace, silu, softmax_in_place};
use crate::tensor::tensor;

#[derive(Debug, Clone)]
pub struct PlamoConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub n_swa: usize,
    pub swa_pattern: Vec<bool>,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub n_ctx_orig: usize,
}

impl PlamoConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture() != "plamo3" {
            return Err(ModelError::Invalid("expected plamo3 architecture"));
        }
        let n_layer = required_u32(&config, "block_count")? as usize;
        let n_embd = required_u32(&config, "embedding_length")? as usize;
        let n_ff = required_u32(&config, "feed_forward_length")? as usize;
        let n_head = required_u32(&config, "attention.head_count")? as usize;
        let n_head_kv = required_u32(&config, "attention.head_count_kv")? as usize;
        let head_dim = config
            .u32("attention.key_length")?
            .unwrap_or((n_embd / n_head) as u32) as usize;
        let eps = config
            .f32("attention.layer_norm_rms_epsilon")?
            .unwrap_or(1e-6);
        let n_swa = config.u32("attention.sliding_window")?.unwrap_or(0) as usize;
        let pattern = config.u32("attention.sliding_window_pattern")?.unwrap_or(8) as usize;
        let swa_pattern = if pattern == 0 {
            vec![false; n_layer]
        } else {
            (0..n_layer)
                .map(|layer| layer % pattern < pattern - 1)
                .collect()
        };
        let rope_freq_base = config.f32("rope.freq_base")?.unwrap_or(1_000_000.0);
        let rope_freq_base_swa = config.f32("rope.freq_base_swa")?.unwrap_or(10_000.0);
        let scaling = config.string("rope.scaling.type")?.unwrap_or("none");
        let yarn = scaling == "yarn";
        let factor = config.f32("rope.scaling.factor")?.unwrap_or(1.0);
        let rope_freq_scale = if factor > 0.0 { 1.0 / factor } else { 1.0 };
        let n_ctx_orig = config
            .u32("rope.scaling.original_context_length")?
            .unwrap_or(4096) as usize;
        let yarn_beta_fast = config.f32("rope.scaling.yarn_beta_fast")?.unwrap_or(32.0);
        let yarn_beta_slow = config.f32("rope.scaling.yarn_beta_slow")?.unwrap_or(1.0);
        let rope_attn_factor = config.f32("rope.scaling.attn_factor")?.unwrap_or(1.0);
        let yarn_attn_factor = if yarn && factor > 1.0 {
            let mscale = 1.0 + 0.1 * factor.ln();
            (mscale / (1.0 + 0.1 * factor.ln())) * rope_attn_factor
        } else {
            1.0
        };

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            eps,
            n_swa,
            swa_pattern,
            rope_freq_base,
            rope_freq_base_swa,
            rope_freq_scale,
            yarn_ext_factor: if yarn { 1.0 } else { 0.0 },
            yarn_attn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            n_ctx_orig,
        })
    }
}

fn required_u32(config: &ModelConfig<'_>, key: &str) -> Result<u32, ModelError> {
    config
        .u32(key)?
        .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// Single-sequence PLaMO decoder with a persistent KV cache.
pub struct PlamoSession<'a> {
    gguf: &'a Gguf,
    file: &'a [u8],
    config: PlamoConfig,
    layers: Vec<LayerCache>,
    pos: usize,
}

impl<'a> PlamoSession<'a> {
    pub fn new(gguf: &'a Gguf, file: &'a [u8]) -> Result<Self, ModelError> {
        let config = PlamoConfig::from_gguf(gguf)?;
        let layers = (0..config.n_layer).map(|_| LayerCache::default()).collect();
        Ok(Self {
            gguf,
            file,
            config,
            layers,
            pos: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Feed every prompt token and return the final token's logits.
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, ModelError> {
        if token_ids.is_empty() {
            return Err(ModelError::Invalid("token_ids must not be empty"));
        }
        let mut logits = Vec::new();
        for token in token_ids {
            logits = self.step(*token)?;
        }
        Ok(logits)
    }

    /// Decode one token and return its logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, ModelError> {
        self.step(token)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, ModelError> {
        let token_embd = tensor(self.gguf, self.file, "token_embd.weight")?;
        let mut hidden = token_embd.row_vec(token as usize)?;
        let position = self.pos;

        for layer in 0..self.config.n_layer {
            let prefix = format!("blk.{layer}");
            let attn_norm =
                tensor(self.gguf, self.file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
            let wqkv = tensor(self.gguf, self.file, &format!("{prefix}.attn_qkv.weight"))?;
            let q_norm = tensor(
                self.gguf,
                self.file,
                &format!("{prefix}.attn_q_norm.weight"),
            )?
            .vec()?;
            let k_norm = tensor(
                self.gguf,
                self.file,
                &format!("{prefix}.attn_k_norm.weight"),
            )?
            .vec()?;
            let wo = tensor(
                self.gguf,
                self.file,
                &format!("{prefix}.attn_output.weight"),
            )?;
            let attn_post_norm = tensor(
                self.gguf,
                self.file,
                &format!("{prefix}.post_attention_norm"),
            )?
            .vec()?;
            let ffn_norm =
                tensor(self.gguf, self.file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
            let ffn_up = tensor(self.gguf, self.file, &format!("{prefix}.ffn_up.weight"))?;
            let ffn_down = tensor(self.gguf, self.file, &format!("{prefix}.ffn_down.weight"))?;
            let ffn_post_norm =
                tensor(self.gguf, self.file, &format!("{prefix}.post_ffw_norm"))?.vec()?;

            let (freq_base, freq_scale, ext_factor, attn_factor) = if self.config.swa_pattern[layer]
            {
                (self.config.rope_freq_base_swa, 1.0, 0.0, 1.0)
            } else {
                (
                    self.config.rope_freq_base,
                    self.config.rope_freq_scale,
                    self.config.yarn_ext_factor,
                    self.config.yarn_attn_factor,
                )
            };

            let normed = rms_norm(&hidden, Some(&attn_norm), self.config.eps);
            let qkv = wqkv.matvec(&normed)?;
            let q_dim = self.config.n_head * self.config.head_dim;
            let k_dim = self.config.n_head_kv * self.config.head_dim;
            if qkv.len() != q_dim + k_dim + k_dim {
                return Err(ModelError::Shape(format!("{prefix} qkv dimension")));
            }
            let mut q = qkv[..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..].to_vec();

            for head in 0..self.config.n_head {
                let start = head * self.config.head_dim;
                let value = rms_norm(
                    &q[start..start + self.config.head_dim],
                    Some(&q_norm),
                    self.config.eps,
                );
                q[start..start + self.config.head_dim].copy_from_slice(&value);
            }
            for head in 0..self.config.n_head_kv {
                let start = head * self.config.head_dim;
                let value = rms_norm(
                    &k[start..start + self.config.head_dim],
                    Some(&k_norm),
                    self.config.eps,
                );
                k[start..start + self.config.head_dim].copy_from_slice(&value);
            }

            rope_neox_yarn_inplace(
                &mut q,
                self.config.n_head,
                self.config.head_dim,
                position,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                self.config.yarn_beta_fast,
                self.config.yarn_beta_slow,
                self.config.n_ctx_orig,
            );
            rope_neox_yarn_inplace(
                &mut k,
                self.config.n_head_kv,
                self.config.head_dim,
                position,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                self.config.yarn_beta_fast,
                self.config.yarn_beta_slow,
                self.config.n_ctx_orig,
            );

            self.layers[layer].keys.push(k);
            self.layers[layer].values.push(v);

            let start = if self.config.swa_pattern[layer] && self.config.n_swa > 0 {
                position.saturating_sub(self.config.n_swa - 1)
            } else {
                0
            };
            let keys = &self.layers[layer].keys;
            let values = &self.layers[layer].values;
            let mut concat = vec![0.0f32; q_dim];
            for head in 0..self.config.n_head {
                let kv_head = head / (self.config.n_head / self.config.n_head_kv);
                let mut scores: Vec<f32> = (start..=position)
                    .map(|other| {
                        q[head * self.config.head_dim..(head + 1) * self.config.head_dim]
                            .iter()
                            .zip(
                                &keys[other][kv_head * self.config.head_dim
                                    ..(kv_head + 1) * self.config.head_dim],
                            )
                            .map(|(q, k)| q * k)
                            .sum::<f32>()
                            / (self.config.head_dim as f32).sqrt()
                    })
                    .collect();
                softmax_in_place(&mut scores);
                let dest =
                    &mut concat[head * self.config.head_dim..(head + 1) * self.config.head_dim];
                for (weight, other) in scores.iter().zip(start..=position) {
                    for (index, value) in values[other]
                        [kv_head * self.config.head_dim..(kv_head + 1) * self.config.head_dim]
                        .iter()
                        .enumerate()
                    {
                        dest[index] += weight * value;
                    }
                }
            }
            let projected = wo.matvec(&concat)?;
            let post = rms_norm(&projected, Some(&attn_post_norm), self.config.eps);
            let residual = add(&hidden, &post);

            let ffn_in = rms_norm(&residual, Some(&ffn_norm), self.config.eps);
            let up = ffn_up.matvec(&ffn_in)?;
            if up.len() != 2 * self.config.n_ff {
                return Err(ModelError::Shape(format!("{prefix} ffn_up dimension")));
            }
            let activated: Vec<f32> = up[..self.config.n_ff]
                .iter()
                .zip(&up[self.config.n_ff..])
                .map(|(gate, up)| silu(*gate) * up)
                .collect();
            let down = ffn_down.matvec(&activated)?;
            let post = rms_norm(&down, Some(&ffn_post_norm), self.config.eps);
            hidden = add(&residual, &post);
        }

        self.pos += 1;
        let output_norm = tensor(self.gguf, self.file, "output_norm.weight")?.vec()?;
        let normed = rms_norm(&hidden, Some(&output_norm), self.config.eps);
        tensor(self.gguf, self.file, "output.weight")?
            .matvec(&normed)
            .map_err(Into::into)
    }
}

/// Greedy next-token id for the final prompt position.
pub fn next_token(gguf: &Gguf, file: &[u8], token_ids: &[u32]) -> Result<u32, ModelError> {
    let logits = last_token_logits(gguf, file, token_ids)?;
    let (index, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| ModelError::Shape("empty logits".into()))?;
    Ok(index as u32)
}

/// Compute logits for the final prompt token.
pub fn last_token_logits(
    gguf: &Gguf,
    file: &[u8],
    token_ids: &[u32],
) -> Result<Vec<f32>, ModelError> {
    if token_ids.is_empty() {
        return Err(ModelError::Invalid("token_ids must not be empty"));
    }
    let config = PlamoConfig::from_gguf(gguf)?;
    let hidden = forward_hidden(gguf, file, &config, token_ids)?;
    let last = hidden
        .last()
        .ok_or_else(|| ModelError::Shape("empty hidden state".into()))?;
    let output_norm = tensor(gguf, file, "output_norm.weight")?.vec()?;
    let normed = rms_norm(last, Some(&output_norm), config.eps);
    let output = tensor(gguf, file, "output.weight")?;
    output.matvec(&normed).map_err(Into::into)
}

fn forward_hidden(
    gguf: &Gguf,
    file: &[u8],
    config: &PlamoConfig,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, ModelError> {
    forward_hidden_until(gguf, file, config, token_ids, config.n_layer)
}

fn forward_hidden_until(
    gguf: &Gguf,
    file: &[u8],
    config: &PlamoConfig,
    token_ids: &[u32],
    layers: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let token_embd = tensor(gguf, file, "token_embd.weight")?;
    let mut hidden: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|id| token_embd.row_vec(*id as usize))
        .collect::<Result<_, _>>()?;

    for layer in 0..layers.min(config.n_layer) {
        let prefix = format!("blk.{layer}");
        let attn_norm = tensor(gguf, file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
        let wqkv = tensor(gguf, file, &format!("{prefix}.attn_qkv.weight"))?;
        let q_norm = tensor(gguf, file, &format!("{prefix}.attn_q_norm.weight"))?.vec()?;
        let k_norm = tensor(gguf, file, &format!("{prefix}.attn_k_norm.weight"))?.vec()?;
        let wo = tensor(gguf, file, &format!("{prefix}.attn_output.weight"))?;
        let attn_post_norm = tensor(gguf, file, &format!("{prefix}.post_attention_norm"))?.vec()?;
        let ffn_norm = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
        let ffn_up = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;
        let ffn_post_norm = tensor(gguf, file, &format!("{prefix}.post_ffw_norm"))?.vec()?;

        let q_dim = config.n_head * config.head_dim;
        let k_dim = config.n_head_kv * config.head_dim;
        let v_dim = config.n_head_kv * config.head_dim;
        let (freq_base, freq_scale, ext_factor, attn_factor) = if config.swa_pattern[layer] {
            // PLaMO uses plain RoPE (theta 10000) on sliding-window layers;
            // YaRN applies only to the full-attention layers.
            (config.rope_freq_base_swa, 1.0, 0.0, 1.0)
        } else {
            (
                config.rope_freq_base,
                config.rope_freq_scale,
                config.yarn_ext_factor,
                config.yarn_attn_factor,
            )
        };
        let window = config.swa_pattern[layer].then_some(config.n_swa);

        let mut q_heads = Vec::with_capacity(hidden.len());
        let mut k_heads = Vec::with_capacity(hidden.len());
        let mut v_heads = Vec::with_capacity(hidden.len());
        for (position, row) in hidden.iter().enumerate() {
            let normed = rms_norm(row, Some(&attn_norm), config.eps);
            let qkv = wqkv.matvec(&normed)?;
            if qkv.len() != q_dim + k_dim + v_dim {
                return Err(ModelError::Shape(format!("{prefix} qkv dimension")));
            }
            let mut q = qkv[..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
            let v = qkv[q_dim + k_dim..].to_vec();

            for head in 0..config.n_head {
                let start = head * config.head_dim;
                let value = rms_norm(
                    &q[start..start + config.head_dim],
                    Some(&q_norm),
                    config.eps,
                );
                q[start..start + config.head_dim].copy_from_slice(&value);
            }
            for head in 0..config.n_head_kv {
                let start = head * config.head_dim;
                let value = rms_norm(
                    &k[start..start + config.head_dim],
                    Some(&k_norm),
                    config.eps,
                );
                k[start..start + config.head_dim].copy_from_slice(&value);
            }

            rope_neox_yarn_inplace(
                &mut q,
                config.n_head,
                config.head_dim,
                position,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                config.yarn_beta_fast,
                config.yarn_beta_slow,
                config.n_ctx_orig,
            );
            rope_neox_yarn_inplace(
                &mut k,
                config.n_head_kv,
                config.head_dim,
                position,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                config.yarn_beta_fast,
                config.yarn_beta_slow,
                config.n_ctx_orig,
            );

            q_heads.push(
                (0..config.n_head)
                    .map(|head| q[head * config.head_dim..(head + 1) * config.head_dim].to_vec())
                    .collect::<Vec<_>>(),
            );
            k_heads.push(
                (0..config.n_head_kv)
                    .map(|head| k[head * config.head_dim..(head + 1) * config.head_dim].to_vec())
                    .collect::<Vec<_>>(),
            );
            v_heads.push(
                (0..config.n_head_kv)
                    .map(|head| v[head * config.head_dim..(head + 1) * config.head_dim].to_vec())
                    .collect::<Vec<_>>(),
            );
        }

        let mut next = Vec::with_capacity(hidden.len());
        for token in 0..hidden.len() {
            let start = match window {
                Some(window) => token.saturating_sub(window.saturating_sub(1)),
                None => 0,
            };
            let mut concat = vec![0.0f32; q_dim];
            for head in 0..config.n_head {
                let kv_head = head / (config.n_head / config.n_head_kv);
                let mut scores: Vec<f32> = (start..=token)
                    .map(|other| {
                        q_heads[token][head]
                            .iter()
                            .zip(&k_heads[other][kv_head])
                            .map(|(q, k)| q * k)
                            .sum::<f32>()
                            / (config.head_dim as f32).sqrt()
                    })
                    .collect();
                softmax_in_place(&mut scores);
                let dest = &mut concat[head * config.head_dim..(head + 1) * config.head_dim];
                for (weight, other) in scores.iter().zip(start..=token) {
                    for (index, value) in v_heads[other][kv_head].iter().enumerate() {
                        dest[index] += weight * value;
                    }
                }
            }
            let projected = wo.matvec(&concat)?;
            let post = rms_norm(&projected, Some(&attn_post_norm), config.eps);
            let residual = add(&hidden[token], &post);

            let ffn_in = rms_norm(&residual, Some(&ffn_norm), config.eps);
            let up = ffn_up.matvec(&ffn_in)?;
            if up.len() != 2 * config.n_ff {
                return Err(ModelError::Shape(format!("{prefix} ffn_up dimension")));
            }
            let activated: Vec<f32> = up[..config.n_ff]
                .iter()
                .zip(&up[config.n_ff..])
                .map(|(gate, up)| silu(*gate) * up)
                .collect();
            let down = ffn_down.matvec(&activated)?;
            let post = rms_norm(&down, Some(&ffn_post_norm), config.eps);
            next.push(add(&residual, &post));
        }
        hidden = next;
    }

    Ok(hidden)
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}
