//! Gemma 4 vision encoder used by EmbeddingGemma 2.
//!
//! The encoder consumes raw 16x16 patch vectors (RGB, `[0, 1]`) plus their
//! `(column, row)` patch coordinates, runs 16 bidirectional ViT layers with
//! axial 2D RoPE, average-pools 3x3 patch blocks, and projects the result to
//! the text model width.

use super::ModelError;
use crate::gguf::Gguf;
use crate::ops::{gelu_tanh, rms_norm, softmax_in_place};
use crate::tensor::tensor;

use crate::manifest::PATCH_SIZE;
const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;
const POOL: usize = 3;
const ROPE_THETA: f32 = 100.0;

pub struct GemmaVisionConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub output_dim: usize,
}

impl GemmaVisionConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let get_u32 = |key: &str| -> Result<usize, ModelError> {
            gguf.metadata(key)
                .and_then(|value| value.as_u32())
                .map(|value| value as usize)
                .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
        };
        let n_layer = get_u32("clip.vision.block_count")?;
        let n_embd = get_u32("clip.vision.embedding_length")?;
        let n_ff = get_u32("clip.vision.feed_forward_length")?;
        let n_head = get_u32("clip.vision.attention.head_count")?;
        let output_dim = get_u32("clip.vision.projection_dim")?;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(ModelError::Shape("vision head count".into()));
        }
        let eps = gguf
            .metadata("clip.vision.attention.layer_norm_epsilon")
            .and_then(|value| value.as_f32())
            .unwrap_or(1e-6);
        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            head_dim: n_embd / n_head,
            eps,
            output_dim,
        })
    }
}

/// Encode a patch grid into `[soft_tokens, output_dim]` rows.
///
/// `patches` is row-major over `(row, column)` and every row holds
/// `PATCH_SIZE * PATCH_SIZE * 3` RGB values in `(y, x, channel)` order.
pub fn encode_image(
    gguf: &Gguf,
    file: &[u8],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaVisionConfig::from_gguf(gguf)?;
    if grid_h == 0 || grid_w == 0 {
        return Err(ModelError::Invalid("image grid must not be empty"));
    }
    if grid_h % POOL != 0 || grid_w % POOL != 0 {
        return Err(ModelError::Invalid(
            "image grid dimensions must be divisible by 3",
        ));
    }
    let n_patches = grid_h * grid_w;
    if patches.len() != n_patches * PATCH_DIM {
        return Err(ModelError::Shape("image patches".into()));
    }

    let mut hidden = image_embedding(gguf, file, &config, patches, grid_h, grid_w)?;

    for layer in 0..config.n_layer {
        hidden = run_layer(gguf, file, &config, layer, hidden, grid_w)?;
    }

    let mut pooled = Vec::with_capacity((grid_h / POOL) * (grid_w / POOL));
    let scale = (config.n_embd as f32).sqrt();
    for pool_row in 0..grid_h / POOL {
        for pool_column in 0..grid_w / POOL {
            let mut sum = vec![0.0f32; config.n_embd];
            for dy in 0..POOL {
                for dx in 0..POOL {
                    let row = pool_row * POOL + dy;
                    let column = pool_column * POOL + dx;
                    for (index, value) in hidden[row * grid_w + column].iter().enumerate() {
                        sum[index] += value;
                    }
                }
            }
            let inv = 1.0 / (POOL * POOL) as f32;
            for value in &mut sum {
                *value *= inv * scale;
            }
            let normed = rms_norm(&sum, None, config.eps);
            pooled.push(normed);
        }
    }

    let projection = tensor(gguf, file, "mm.input_projection.weight")?;
    if projection.cols() != config.n_embd || projection.rows() != config.output_dim {
        return Err(ModelError::Shape("vision projection".into()));
    }
    let mut out = Vec::with_capacity(pooled.len() * config.output_dim);
    for row in &pooled {
        out.extend(projection.matvec(row)?);
    }
    Ok(out)
}

fn image_embedding(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaVisionConfig,
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let patch_weight = tensor(gguf, file, "v.patch_embd.weight")?.vec()?;
    if patch_weight.len() != config.n_embd * PATCH_DIM {
        return Err(ModelError::Shape("patch embedding".into()));
    }
    let position = tensor(gguf, file, "v.position_embd.weight")?.vec()?;
    if position.len() != config.n_embd * 10_240 * 2 {
        return Err(ModelError::Shape("vision position embedding".into()));
    }

    let mut hidden = Vec::with_capacity(grid_h * grid_w);
    for row in 0..grid_h {
        for column in 0..grid_w {
            let patch = &patches[(row * grid_w + column) * PATCH_DIM..][..PATCH_DIM];
            let mut values = vec![0.0f32; config.n_embd];
            for (channel, value) in values.iter_mut().enumerate() {
                let base = channel * PATCH_DIM;
                let mut sum = 0.0f32;
                for ky in 0..PATCH_SIZE {
                    for kx in 0..PATCH_SIZE {
                        for c in 0..3 {
                            let pixel = patch[(ky * PATCH_SIZE + kx) * 3 + c];
                            let weight =
                                patch_weight[base + (c * PATCH_SIZE + ky) * PATCH_SIZE + kx];
                            sum += (pixel - 0.5) * 2.0 * weight;
                        }
                    }
                }
                *value = sum;
            }
            let x_offset = column * config.n_embd;
            let y_offset = 10_240 * config.n_embd + row * config.n_embd;
            for (index, value) in values.iter_mut().enumerate() {
                *value += position[x_offset + index] + position[y_offset + index];
            }
            hidden.push(values);
        }
    }
    Ok(hidden)
}

fn run_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaVisionConfig,
    layer: usize,
    hidden: Vec<Vec<f32>>,
    grid_w: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let prefix = format!("v.blk.{layer}");
    let ln1 = tensor(gguf, file, &format!("{prefix}.ln1.weight"))?.vec()?;
    let wq = tensor(gguf, file, &format!("{prefix}.attn_q.weight"))?;
    let wk = tensor(gguf, file, &format!("{prefix}.attn_k.weight"))?;
    let wv = tensor(gguf, file, &format!("{prefix}.attn_v.weight"))?;
    let q_norm = tensor(gguf, file, &format!("{prefix}.attn_q_norm.weight"))?.vec()?;
    let k_norm = tensor(gguf, file, &format!("{prefix}.attn_k_norm.weight"))?.vec()?;
    let wo = tensor(gguf, file, &format!("{prefix}.attn_out.weight"))?;
    let attn_post_norm = tensor(gguf, file, &format!("{prefix}.attn_post_norm.weight"))?.vec()?;
    let ln2 = tensor(gguf, file, &format!("{prefix}.ln2.weight"))?.vec()?;
    let gate = tensor(gguf, file, &format!("{prefix}.ffn_gate.weight"))?;
    let up = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
    let down = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;
    let ffn_post_norm = tensor(gguf, file, &format!("{prefix}.ffn_post_norm.weight"))?.vec()?;

    let n_patches = hidden.len();
    let mut q = Vec::with_capacity(n_patches);
    let mut k = Vec::with_capacity(n_patches);
    let mut v = Vec::with_capacity(n_patches);
    for (index, row) in hidden.iter().enumerate() {
        let normed = rms_norm(row, Some(&ln1), config.eps);
        let mut q_row = wq.matvec(&normed)?;
        let mut k_row = wk.matvec(&normed)?;
        let v_row = wv.matvec(&normed)?;
        for head in 0..config.n_head {
            let start = head * config.head_dim;
            let normalized = rms_norm(
                &q_row[start..start + config.head_dim],
                Some(&q_norm),
                config.eps,
            );
            q_row[start..start + config.head_dim].copy_from_slice(&normalized);
            let normalized = rms_norm(
                &k_row[start..start + config.head_dim],
                Some(&k_norm),
                config.eps,
            );
            k_row[start..start + config.head_dim].copy_from_slice(&normalized);
        }
        let column = index % grid_w;
        let row_index = index / grid_w;
        apply_axial_rope(
            &mut q_row,
            config.n_head,
            config.head_dim,
            column,
            row_index,
        );
        apply_axial_rope(
            &mut k_row,
            config.n_head,
            config.head_dim,
            column,
            row_index,
        );
        let mut v_row = v_row;
        for head in 0..config.n_head {
            let start = head * config.head_dim;
            let normalized = rms_norm(&v_row[start..start + config.head_dim], None, config.eps);
            v_row[start..start + config.head_dim].copy_from_slice(&normalized);
        }
        q.push(q_row);
        k.push(k_row);
        v.push(v_row);
    }

    let mut attention = vec![vec![0.0f32; config.n_embd]; n_patches];
    let head_scale = 1.0f32;
    for head in 0..config.n_head {
        let start = head * config.head_dim;
        for (index, row) in attention.iter_mut().enumerate() {
            let mut scores = vec![0.0f32; n_patches];
            for (other, k_row) in k.iter().enumerate() {
                let mut sum = 0.0f32;
                for dim in 0..config.head_dim {
                    sum += q[index][start + dim] * k_row[start + dim];
                }
                scores[other] = sum * head_scale;
            }
            softmax_in_place(&mut scores);
            for (other, score) in scores.iter().enumerate() {
                for dim in 0..config.head_dim {
                    row[start + dim] += score * v[other][start + dim];
                }
            }
        }
    }

    let mut out = Vec::with_capacity(n_patches);
    for (index, row) in attention.iter().enumerate() {
        let projected = wo.matvec(row)?;
        let normed = rms_norm(&projected, Some(&attn_post_norm), config.eps);
        let mut residual: Vec<f32> = hidden[index]
            .iter()
            .zip(&normed)
            .map(|(left, right)| left + right)
            .collect();
        let normed = rms_norm(&residual, Some(&ln2), config.eps);
        let gate_out = gate.matvec(&normed)?;
        let up_out = up.matvec(&normed)?;
        let activated: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(gate, up)| gelu_tanh(*gate) * up)
            .collect();
        let down_out = down.matvec(&activated)?;
        let normed = rms_norm(&down_out, Some(&ffn_post_norm), config.eps);
        for (value, add) in residual.iter_mut().zip(&normed) {
            *value += add;
        }
        out.push(residual);
    }
    Ok(out)
}

fn apply_axial_rope(values: &mut [f32], n_head: usize, head_dim: usize, column: usize, row: usize) {
    let half = head_dim / 2;
    for head in 0..n_head {
        let base = head * head_dim;
        rope_chunk(&mut values[base..base + half], column as f32, ROPE_THETA);
        rope_chunk(
            &mut values[base + half..base + head_dim],
            row as f32,
            ROPE_THETA,
        );
    }
}

/// NeoX RoPE over one half-head chunk (16 frequency pairs for Gemma 4 vision).
fn rope_chunk(values: &mut [f32], position: f32, theta: f32) {
    let half = values.len() / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for (index, (cos, sin)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        let angle = position * theta.powf(-2.0 * index as f32 / values.len() as f32);
        *cos = angle.cos();
        *sin = angle.sin();
    }
    for index in 0..half {
        let x0 = values[index];
        let x1 = values[index + half];
        values[index] = x0 * cos[index] - x1 * sin[index];
        values[index + half] = x1 * cos[index] + x0 * sin[index];
    }
}
