//! EmbeddingGemma 2 text forward pass.
//!
//! This mirrors llama.cpp's `gemma-embedding2` graph: scaled token embeddings,
//! a per-layer projection tower, non-causal attention with alternating
//! sliding-window/full layers, per-layer input gates, and mean pooling after
//! the output projection.

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::ops::{gelu_tanh, l2_normalize, rms_norm, rope_neox_inplace, softmax_in_place};
use crate::tensor::tensor;

const DEFAULT_SLIDING_WINDOW: u32 = 1024;

#[derive(Debug, Clone)]
pub struct EmbeddingGemmaConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_embd_out: usize,
    pub n_embd_per_layer: usize,
    pub n_swa: usize,
    pub eps: f32,
    pub head_count_kv: Vec<usize>,
    pub swa_pattern: Vec<bool>,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
}

impl EmbeddingGemmaConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture() != "gemma-embedding2" {
            return Err(ModelError::Invalid(
                "expected gemma-embedding2 architecture",
            ));
        }
        let n_layer = required_u32(&config, "block_count")? as usize;
        let n_embd = required_u32(&config, "embedding_length")? as usize;
        let n_ff = required_u32(&config, "feed_forward_length")? as usize;
        let n_head = required_u32(&config, "attention.head_count")? as usize;
        let n_embd_out = required_u32(&config, "embedding_length_out")? as usize;
        let n_embd_per_layer = required_u32(&config, "embedding_length_per_layer_input")? as usize;
        let n_swa = config
            .u32("attention.sliding_window")?
            .unwrap_or(DEFAULT_SLIDING_WINDOW) as usize;
        let eps = config
            .f32("attention.layer_norm_rms_epsilon")?
            .unwrap_or(1e-6);
        let head_count_kv = match config.i32_array("attention.head_count_kv")? {
            Some(values) => values
                .into_iter()
                .map(|value| usize::try_from(value).unwrap_or(0))
                .collect(),
            None => vec![n_head; n_layer],
        };
        if head_count_kv.len() != n_layer {
            return Err(ModelError::Shape("head_count_kv length".into()));
        }
        let swa_pattern = match config.bool_array("attention.sliding_window_pattern")? {
            Some(values) => values,
            None => (0..n_layer).map(|layer| (layer + 1) % 6 != 0).collect(),
        };
        if swa_pattern.len() != n_layer {
            return Err(ModelError::Shape("sliding_window_pattern length".into()));
        }
        let rope_freq_base = config.f32("rope.freq_base")?.unwrap_or(10_000.0);
        let rope_freq_base_swa = config.f32("rope.freq_base_swa")?.unwrap_or(rope_freq_base);

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_embd_out,
            n_embd_per_layer,
            n_swa,
            eps,
            head_count_kv,
            swa_pattern,
            rope_freq_base,
            rope_freq_base_swa,
        })
    }
}

fn required_u32(config: &ModelConfig<'_>, key: &str) -> Result<u32, ModelError> {
    config
        .u32(key)?
        .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
}

/// Run the EmbeddingGemma 2 text encoder over one token sequence.
///
/// The returned vector is mean-pooled and L2-normalized, matching the
/// `sentence_embedding` output of the ONNX conversion.
pub fn embed_text(gguf: &Gguf, file: &[u8], token_ids: &[u32]) -> Result<Vec<f32>, ModelError> {
    embed_text_with_features(gguf, file, token_ids, &[], &[], &[])
}

/// Run the encoder over a sequence that contains image, video and audio soft
/// tokens. Feature rows replace the corresponding placeholder token
/// embeddings, in placeholder order.
pub fn embed_text_with_features(
    gguf: &Gguf,
    file: &[u8],
    token_ids: &[u32],
    image_features: &[f32],
    video_features: &[f32],
    audio_features: &[f32],
) -> Result<Vec<f32>, ModelError> {
    if token_ids.is_empty() {
        return Err(ModelError::Invalid("token_ids must not be empty"));
    }
    const IMAGE_TOKEN: u32 = 258_880;
    const AUDIO_TOKEN: u32 = 258_881;
    const VIDEO_TOKEN: u32 = 258_884;
    let config = EmbeddingGemmaConfig::from_gguf(gguf)?;
    let token_embd = tensor(gguf, file, "token_embd.weight")?;
    if token_embd.cols() != config.n_embd {
        return Err(ModelError::Shape("token_embd columns".into()));
    }
    let width = config.n_embd;
    if image_features.len() % width != 0
        || video_features.len() % width != 0
        || audio_features.len() % width != 0
    {
        return Err(ModelError::Shape("multimodal feature width".into()));
    }
    let image_rows = image_features.len() / width;
    let video_rows = video_features.len() / width;
    let audio_rows = audio_features.len() / width;
    let expected = |token: u32| token_ids.iter().filter(|id| **id == token).count();
    if expected(IMAGE_TOKEN) != image_rows
        || expected(VIDEO_TOKEN) != video_rows
        || expected(AUDIO_TOKEN) != audio_rows
    {
        return Err(ModelError::Shape("multimodal feature count".into()));
    }

    let input_scale = (config.n_embd as f32).sqrt();
    let mut hidden = Vec::with_capacity(token_ids.len());
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut audio_index = 0usize;
    for token_id in token_ids {
        let mut row = token_embd.row_vec(*token_id as usize)?;
        for value in &mut row {
            *value *= input_scale;
        }
        let feature = match *token_id {
            IMAGE_TOKEN => {
                let start = image_index * width;
                image_index += 1;
                Some(&image_features[start..start + width])
            }
            VIDEO_TOKEN => {
                let start = video_index * width;
                video_index += 1;
                Some(&video_features[start..start + width])
            }
            AUDIO_TOKEN => {
                let start = audio_index * width;
                audio_index += 1;
                Some(&audio_features[start..start + width])
            }
            _ => None,
        };
        if let Some(feature) = feature {
            hidden.push(feature.to_vec());
        } else {
            hidden.push(row);
        }
    }

    let per_layer = build_per_layer_inputs(gguf, file, &config, &hidden)?;

    for (layer, per_layer_input) in per_layer.iter().enumerate() {
        hidden = run_layer(gguf, file, &config, layer, hidden, per_layer_input)?;
    }

    let output_norm = tensor(gguf, file, "output_norm.weight")?.vec()?;
    let mut pooled = vec![0.0f32; config.n_embd];
    for row in &hidden {
        let normed = rms_norm(row, Some(&output_norm), config.eps);
        for (index, value) in normed.iter().enumerate() {
            pooled[index] += value;
        }
    }
    let scale = 1.0 / hidden.len() as f32;
    for value in &mut pooled {
        *value *= scale;
    }

    let output = tensor(gguf, file, "output.weight")?;
    let embedding = output.matvec(&pooled)?;
    if embedding.len() != config.n_embd_out {
        return Err(ModelError::Shape("output projection dimension".into()));
    }
    Ok(l2_normalize(&embedding))
}

fn build_per_layer_inputs(
    gguf: &Gguf,
    file: &[u8],
    config: &EmbeddingGemmaConfig,
    hidden: &[Vec<f32>],
) -> Result<Vec<Vec<Vec<f32>>>, ModelError> {
    let projection = tensor(gguf, file, "per_layer_model_proj.weight")?;
    let norm = tensor(gguf, file, "per_layer_proj_norm.weight")?.vec()?;
    let scale = 1.0 / (config.n_embd as f32).sqrt();

    let mut out = vec![vec![vec![0.0f32; config.n_embd_per_layer]; hidden.len()]; config.n_layer];
    for (token, row) in hidden.iter().enumerate() {
        let projected = projection.matvec(row)?;
        let expected = config.n_layer * config.n_embd_per_layer;
        if projected.len() != expected {
            return Err(ModelError::Shape("per_layer_model_proj output".into()));
        }
        for (layer, out_layer) in out.iter_mut().enumerate() {
            let start = layer * config.n_embd_per_layer;
            let slice = &projected[start..start + config.n_embd_per_layer];
            let scaled: Vec<f32> = slice.iter().map(|value| value * scale).collect();
            out_layer[token] = rms_norm(&scaled, Some(&norm), config.eps);
        }
    }
    Ok(out)
}

fn run_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &EmbeddingGemmaConfig,
    layer: usize,
    hidden: Vec<Vec<f32>>,
    per_layer_input: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let prefix = format!("blk.{layer}");
    let attn_norm = tensor(gguf, file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
    let wq = tensor(gguf, file, &format!("{prefix}.attn_q.weight"))?;
    let wk = tensor(gguf, file, &format!("{prefix}.attn_k.weight"))?;
    let wv = tensor(gguf, file, &format!("{prefix}.attn_v.weight"))?;
    let wo = tensor(gguf, file, &format!("{prefix}.attn_output.weight"))?;
    let q_norm = tensor(gguf, file, &format!("{prefix}.attn_q_norm.weight"))?.vec()?;
    let k_norm = tensor(gguf, file, &format!("{prefix}.attn_k_norm.weight"))?.vec()?;
    let attn_post_norm =
        tensor(gguf, file, &format!("{prefix}.post_attention_norm.weight"))?.vec()?;

    let ffn_norm = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
    let ffn_gate = tensor(gguf, file, &format!("{prefix}.ffn_gate.weight"))?;
    let ffn_up = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
    let ffn_down = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;
    let ffn_post_norm = tensor(gguf, file, &format!("{prefix}.post_ffw_norm.weight"))?.vec()?;

    let inp_gate = tensor(gguf, file, &format!("{prefix}.inp_gate.weight"))?;
    let per_layer_proj = tensor(gguf, file, &format!("{prefix}.proj.weight"))?;
    let per_layer_post_norm = tensor(gguf, file, &format!("{prefix}.post_norm.weight"))?.vec()?;
    let out_scale = tensor(gguf, file, &format!("{prefix}.layer_output_scale.weight"))?.scalar()?;

    let n_tokens = hidden.len();
    let q_dim = wq.rows();
    if q_dim % config.n_head != 0 {
        return Err(ModelError::Shape(format!("{prefix} q dimension")));
    }
    let head_dim = q_dim / config.n_head;
    let kv_dim = wk.rows();
    if head_dim == 0 || kv_dim % head_dim != 0 {
        return Err(ModelError::Shape(format!("{prefix} kv dimension")));
    }
    let n_head_kv = kv_dim / head_dim;
    if n_head_kv == 0 || config.n_head % n_head_kv != 0 {
        return Err(ModelError::Shape(format!("{prefix} kv head count")));
    }
    if q_norm.len() != head_dim || k_norm.len() != head_dim {
        return Err(ModelError::Shape(format!("{prefix} q/k norm length")));
    }
    let freq_base = if config.swa_pattern[layer] {
        config.rope_freq_base_swa
    } else {
        config.rope_freq_base
    };
    let window = if config.swa_pattern[layer] {
        Some(config.n_swa)
    } else {
        None
    };

    let mut q_heads = Vec::with_capacity(n_tokens);
    let mut k_heads = Vec::with_capacity(n_tokens);
    let mut v_heads = Vec::with_capacity(n_tokens);
    for (position, row) in hidden.iter().enumerate() {
        let normed = rms_norm(row, Some(&attn_norm), config.eps);
        let q = wq.matvec(&normed)?;
        let k = wk.matvec(&normed)?;
        let v = wv.matvec(&normed)?;

        let mut qn = vec![0.0f32; q_dim];
        for head in 0..config.n_head {
            let start = head * head_dim;
            let normalized = rms_norm(&q[start..start + head_dim], Some(&q_norm), config.eps);
            qn[start..start + head_dim].copy_from_slice(&normalized);
        }
        rope_neox_inplace(&mut qn, config.n_head, head_dim, position, freq_base);

        let mut kn = vec![0.0f32; kv_dim];
        for head in 0..n_head_kv {
            let start = head * head_dim;
            let normalized = rms_norm(&k[start..start + head_dim], Some(&k_norm), config.eps);
            kn[start..start + head_dim].copy_from_slice(&normalized);
        }
        rope_neox_inplace(&mut kn, n_head_kv, head_dim, position, freq_base);

        let mut vn = vec![0.0f32; kv_dim];
        for head in 0..n_head_kv {
            let start = head * head_dim;
            let normalized = rms_norm(&v[start..start + head_dim], None, config.eps);
            vn[start..start + head_dim].copy_from_slice(&normalized);
        }

        q_heads.push(
            (0..config.n_head)
                .map(|h| qn[h * head_dim..(h + 1) * head_dim].to_vec())
                .collect::<Vec<_>>(),
        );
        k_heads.push(
            (0..n_head_kv)
                .map(|h| kn[h * head_dim..(h + 1) * head_dim].to_vec())
                .collect::<Vec<_>>(),
        );
        v_heads.push(
            (0..n_head_kv)
                .map(|h| vn[h * head_dim..(h + 1) * head_dim].to_vec())
                .collect::<Vec<_>>(),
        );
    }

    let mut next = Vec::with_capacity(n_tokens);
    for token in 0..n_tokens {
        let (start, end) = match window {
            Some(window) => {
                let half = window / 2;
                (token.saturating_sub(half), (token + half).min(n_tokens - 1))
            }
            None => (0, n_tokens - 1),
        };
        let mut concat = vec![0.0f32; config.n_head * head_dim];
        for head in 0..config.n_head {
            let kv_head = head / (config.n_head / n_head_kv);
            let keys = &k_heads[start..=end];
            let values = &v_heads[start..=end];
            let mut scores: Vec<f32> = keys
                .iter()
                .map(|key| {
                    q_heads[token][head]
                        .iter()
                        .zip(&key[kv_head])
                        .map(|(q, k)| q * k)
                        .sum()
                })
                .collect();
            softmax_in_place(&mut scores);
            let out = &mut concat[head * head_dim..(head + 1) * head_dim];
            for (weight, value) in scores.iter().zip(values) {
                for (index, value) in value[kv_head].iter().enumerate() {
                    out[index] += weight * value;
                }
            }
        }

        let projected = wo.matvec(&concat)?;
        let post = rms_norm(&projected, Some(&attn_post_norm), config.eps);
        let attn_out: Vec<f32> = hidden[token]
            .iter()
            .zip(post)
            .map(|(residual, value)| residual + value)
            .collect();

        let ffn_in = rms_norm(&attn_out, Some(&ffn_norm), config.eps);
        let gate = ffn_gate.matvec(&ffn_in)?;
        let up = ffn_up.matvec(&ffn_in)?;
        let activated: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| gelu_tanh(*gate) * up)
            .collect();
        let down = ffn_down.matvec(&activated)?;
        let ffn_post = rms_norm(&down, Some(&ffn_post_norm), config.eps);
        let mut cur: Vec<f32> = attn_out
            .iter()
            .zip(ffn_post)
            .map(|(residual, value)| residual + value)
            .collect();

        let gate = inp_gate.matvec(&cur)?;
        let gated: Vec<f32> = gate
            .iter()
            .zip(&per_layer_input[token])
            .map(|(gate, input)| gelu_tanh(*gate) * input)
            .collect();
        let projected = per_layer_proj.matvec(&gated)?;
        let normed = rms_norm(&projected, Some(&per_layer_post_norm), config.eps);
        for (value, add) in cur.iter_mut().zip(normed) {
            *value += add;
        }
        for value in &mut cur {
            *value *= out_scale;
        }
        next.push(cur);
    }

    Ok(next)
}
