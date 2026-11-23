//! PLaMo 3 Translate generation forward.
//!
//! PLaMo 3 is a decoder-only model with a fused QKV projection, GQA (16 query
//! heads, one KV head), per-head Q/K RMSNorm, YaRN RoPE, alternating
//! sliding-window/full attention and post-attention/post-FFN RMSNorm.

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::ops::{rms_norm, rms_norm_batch, rope_neox_yarn_inplace, silu, softmax_in_place};
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
    ///
    /// The whole prompt goes through each projection as one batch, so the
    /// weight matrices are streamed once per 16-token tile instead of once per
    /// token.
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>, ModelError> {
        if token_ids.is_empty() {
            return Err(ModelError::Invalid("token_ids must not be empty"));
        }
        let n_tokens = token_ids.len();
        let width = self.config.n_embd;
        let token_embd = tensor(self.gguf, self.file, "token_embd.weight")?;
        let mut hidden = Vec::with_capacity(n_tokens * width);
        for token in token_ids {
            hidden.extend_from_slice(&token_embd.row_vec(*token as usize)?);
        }

        let base_pos = self.pos;
        for layer in 0..self.config.n_layer {
            let cache = &mut self.layers[layer];
            hidden = run_layer(
                self.gguf,
                self.file,
                &self.config,
                layer,
                &hidden,
                n_tokens,
                &mut cache.keys,
                &mut cache.values,
                base_pos,
            )?;
        }
        self.pos += n_tokens;

        let last = &hidden[(n_tokens - 1) * width..];
        let output_norm = tensor(self.gguf, self.file, "output_norm.weight")?.vec()?;
        let normed = rms_norm(last, Some(&output_norm), self.config.eps);
        tensor(self.gguf, self.file, "output.weight")?
            .matvec(&normed)
            .map_err(Into::into)
    }

    /// Decode one token and return its logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, ModelError> {
        self.step(token)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, ModelError> {
        let width = self.config.n_embd;
        let token_embd = tensor(self.gguf, self.file, "token_embd.weight")?;
        let mut hidden = token_embd.row_vec(token as usize)?;
        let base_pos = self.pos;

        for layer in 0..self.config.n_layer {
            let cache = &mut self.layers[layer];
            hidden = run_layer(
                self.gguf,
                self.file,
                &self.config,
                layer,
                &hidden,
                1,
                &mut cache.keys,
                &mut cache.values,
                base_pos,
            )?;
        }

        self.pos += 1;
        let output_norm = tensor(self.gguf, self.file, "output_norm.weight")?.vec()?;
        debug_assert_eq!(hidden.len(), width);
        let normed = rms_norm(&hidden[..width], Some(&output_norm), self.config.eps);
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
    let width = config.n_embd;
    let last_start = (token_ids.len() - 1) * width;
    let last = &hidden[last_start..last_start + width];
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
) -> Result<Vec<f32>, ModelError> {
    forward_hidden_until(gguf, file, config, token_ids, config.n_layer)
}

/// Run the decoder stack over a whole prompt, returning flat
/// `[n_tokens][n_embd]` hidden states.
fn forward_hidden_until(
    gguf: &Gguf,
    file: &[u8],
    config: &PlamoConfig,
    token_ids: &[u32],
    layers: usize,
) -> Result<Vec<f32>, ModelError> {
    let width = config.n_embd;
    let n_tokens = token_ids.len();
    let token_embd = tensor(gguf, file, "token_embd.weight")?;
    let mut hidden = Vec::with_capacity(n_tokens * width);
    for id in token_ids {
        hidden.extend_from_slice(&token_embd.row_vec(*id as usize)?);
    }

    for layer in 0..layers.min(config.n_layer) {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        hidden = run_layer(
            gguf,
            file,
            config,
            layer,
            &hidden,
            n_tokens,
            &mut keys,
            &mut values,
            0,
        )?;
    }

    Ok(hidden)
}

/// One decoder layer for `n_tokens` positions, appending the KV cache entries.
///
/// With `n_tokens == 1` every projection falls back to the single-token kernel,
/// so decode stays numerically identical to the pre-batching implementation.
#[allow(clippy::too_many_arguments)]
fn run_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &PlamoConfig,
    layer: usize,
    hidden: &[f32],
    n_tokens: usize,
    keys: &mut Vec<Vec<f32>>,
    values: &mut Vec<Vec<f32>>,
    base_pos: usize,
) -> Result<Vec<f32>, ModelError> {
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

    let width = config.n_embd;
    if hidden.len() != n_tokens * width {
        return Err(ModelError::Shape(format!("{prefix} hidden shape")));
    }
    // A failed prefill can leave entries behind without advancing the session;
    // reject that state instead of appending duplicates.
    if keys.len() != base_pos || values.len() != base_pos {
        return Err(ModelError::Shape(format!(
            "{prefix} attention cache length"
        )));
    }
    let q_dim = config.n_head * config.head_dim;
    let k_dim = config.n_head_kv * config.head_dim;
    let v_dim = config.n_head_kv * config.head_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
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
    if wo.rows() != width || wo.cols() != q_dim {
        return Err(ModelError::Shape(format!(
            "{prefix} attention output shape"
        )));
    }
    if ffn_down.rows() != width || ffn_down.cols() != config.n_ff {
        return Err(ModelError::Shape(format!("{prefix} ffn_down shape")));
    }

    let normed = rms_norm_batch(hidden, n_tokens, width, Some(&attn_norm), config.eps);
    let mut qkv = wqkv.matmul(&normed, n_tokens)?;
    drop(normed);
    if qkv.len() != n_tokens * qkv_dim {
        return Err(ModelError::Shape(format!("{prefix} qkv dimension")));
    }

    for token in 0..n_tokens {
        let position = base_pos + token;
        let row = &mut qkv[token * qkv_dim..(token + 1) * qkv_dim];
        let (q, rest) = row.split_at_mut(q_dim);
        let (k, v) = rest.split_at_mut(k_dim);

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
            q,
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
            k,
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

        keys.push(k.to_vec());
        values.push(v.to_vec());
    }

    let softmax_scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut concat = vec![0.0f32; n_tokens * q_dim];
    let mut scores: Vec<f32> = Vec::new();
    for token in 0..n_tokens {
        let position = base_pos + token;
        let start = match window {
            Some(window) if window > 0 => position.saturating_sub(window - 1),
            _ => 0,
        };
        if keys.len() < position + 1 || values.len() != keys.len() {
            return Err(ModelError::Shape(format!("{prefix} attention cache")));
        }
        let q_row = &qkv[token * qkv_dim..token * qkv_dim + q_dim];
        for head in 0..config.n_head {
            let kv_head = head / (config.n_head / config.n_head_kv);
            let q_head = &q_row[head * config.head_dim..(head + 1) * config.head_dim];
            scores.clear();
            scores.extend((start..=position).map(|other| {
                let key = &keys[other][kv_head * config.head_dim..(kv_head + 1) * config.head_dim];
                q_head.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * softmax_scale
            }));
            softmax_in_place(&mut scores);
            let dest = &mut concat[token * q_dim + head * config.head_dim
                ..token * q_dim + (head + 1) * config.head_dim];
            for (weight, other) in scores.iter().zip(start..=position) {
                let value =
                    &values[other][kv_head * config.head_dim..(kv_head + 1) * config.head_dim];
                for (index, value) in value.iter().enumerate() {
                    dest[index] += weight * value;
                }
            }
        }
    }

    let projected = wo.matmul(&concat, n_tokens)?;
    let post = rms_norm_batch(
        &projected,
        n_tokens,
        width,
        Some(&attn_post_norm),
        config.eps,
    );
    let residual: Vec<f32> = hidden
        .iter()
        .zip(&post)
        .map(|(residual, value)| residual + value)
        .collect();

    let ffn_in = rms_norm_batch(&residual, n_tokens, width, Some(&ffn_norm), config.eps);
    let up = ffn_up.matmul(&ffn_in, n_tokens)?;
    if up.len() != n_tokens * 2 * config.n_ff {
        return Err(ModelError::Shape(format!("{prefix} ffn_up dimension")));
    }
    let mut activated = vec![0.0f32; n_tokens * config.n_ff];
    let row_stride = 2 * config.n_ff;
    for token in 0..n_tokens {
        let row = &up[token * row_stride..(token + 1) * row_stride];
        let dest = &mut activated[token * config.n_ff..(token + 1) * config.n_ff];
        for (index, value) in dest.iter_mut().enumerate() {
            *value = silu(row[index]) * row[config.n_ff + index];
        }
    }
    let down = ffn_down.matmul(&activated, n_tokens)?;
    let post = rms_norm_batch(&down, n_tokens, width, Some(&ffn_post_norm), config.eps);

    Ok(residual
        .iter()
        .zip(&post)
        .map(|(residual, value)| residual + value)
        .collect())
}
