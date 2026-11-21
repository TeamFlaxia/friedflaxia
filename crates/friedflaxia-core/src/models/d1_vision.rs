//! d1-omni SigLIP2 vision tower and 2x2 pixel-shuffle projector.

use super::ModelError;
use crate::gguf::Gguf;
use crate::ops::{gelu_tanh, layer_norm, softmax_in_place};
use crate::tensor::tensor;

use crate::manifest::PATCH_SIZE;
const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;
const BASE_GRID: usize = 16;

pub struct D1VisionConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub output_dim: usize,
}

impl D1VisionConfig {
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
            return Err(ModelError::Shape("d1 vision head count".into()));
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

fn linear(
    gguf: &Gguf,
    file: &[u8],
    weight: &str,
    bias: &str,
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let weight = tensor(gguf, file, weight)?;
    let bias = tensor(gguf, file, bias)?.vec()?;
    let mut out = weight.matvec(input)?;
    if out.len() != bias.len() {
        return Err(ModelError::Shape(format!(
            "{weight_name} bias",
            weight_name = weight.info.name
        )));
    }
    for (value, bias) in out.iter_mut().zip(&bias) {
        *value += bias;
    }
    Ok(out)
}

fn layer_norm_weighted(
    gguf: &Gguf,
    file: &[u8],
    weight: &str,
    bias: &str,
    input: &[f32],
    eps: f32,
) -> Result<Vec<f32>, ModelError> {
    let weight = tensor(gguf, file, weight)?.vec()?;
    let bias = tensor(gguf, file, bias)?.vec()?;
    Ok(layer_norm(input, &weight, &bias, eps))
}

/// Encode a 512x512 image tile into `[256, output_dim]` soft tokens.
#[allow(clippy::needless_range_loop)]
pub fn encode_image(
    gguf: &Gguf,
    file: &[u8],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = D1VisionConfig::from_gguf(gguf)?;
    if grid_h == 0 || grid_w == 0 || grid_h % 2 != 0 || grid_w % 2 != 0 {
        return Err(ModelError::Invalid(
            "d1 vision grid must be non-empty and even",
        ));
    }
    let n_patches = grid_h * grid_w;
    if patches.len() != n_patches * PATCH_DIM {
        return Err(ModelError::Shape("d1 image patches".into()));
    }

    let patch_weight = tensor(gguf, file, "v.patch_embd.weight")?.vec()?;
    let patch_bias = tensor(gguf, file, "v.patch_embd.bias")?.vec()?;
    let position = tensor(gguf, file, "v.position_embd.weight")?.vec()?;
    if patch_weight.len() != config.n_embd * PATCH_DIM
        || patch_bias.len() != config.n_embd
        || position.len() != config.n_embd * BASE_GRID * BASE_GRID
    {
        return Err(ModelError::Shape("d1 vision embedding".into()));
    }

    let mut hidden = Vec::with_capacity(n_patches);
    for row in 0..grid_h {
        for column in 0..grid_w {
            let patch = &patches[(row * grid_w + column) * PATCH_DIM..][..PATCH_DIM];
            let mut values = vec![0.0f32; config.n_embd];
            for (channel, value) in values.iter_mut().enumerate() {
                let base = channel * PATCH_DIM;
                let mut sum = patch_bias[channel];
                for ky in 0..PATCH_SIZE {
                    for kx in 0..PATCH_SIZE {
                        for c in 0..3 {
                            let pixel = patch[(ky * PATCH_SIZE + kx) * 3 + c];
                            let weight =
                                patch_weight[base + (c * PATCH_SIZE + ky) * PATCH_SIZE + kx];
                            sum += pixel * weight;
                        }
                    }
                }
                *value = sum;
            }
            add_interpolated_position(
                &mut values,
                &position,
                config.n_embd,
                row,
                column,
                grid_h,
                grid_w,
            );
            hidden.push(values);
        }
    }

    for layer in 0..config.n_layer {
        hidden = run_layer(gguf, file, &config, layer, hidden, n_patches)?;
    }

    let normed: Vec<Vec<f32>> = hidden
        .iter()
        .map(|row| {
            layer_norm_weighted(
                gguf,
                file,
                "v.post_ln.weight",
                "v.post_ln.bias",
                row,
                config.eps,
            )
        })
        .collect::<Result<_, _>>()?;

    let tokens_h = grid_h / 2;
    let tokens_w = grid_w / 2;
    let mut tokens = Vec::with_capacity(tokens_h * tokens_w);
    for token_y in 0..tokens_h {
        for token_x in 0..tokens_w {
            let mut token = Vec::with_capacity(config.n_embd * 4);
            for by in 0..2 {
                for bx in 0..2 {
                    for c in 0..config.n_embd {
                        token.push(normed[(token_y * 2 + by) * grid_w + token_x * 2 + bx][c]);
                    }
                }
            }
            let mut projected = linear(gguf, file, "mm.1.weight", "mm.1.bias", &token)?;
            for value in &mut projected {
                *value = gelu_tanh(*value);
            }
            let out = linear(gguf, file, "mm.2.weight", "mm.2.bias", &projected)?;
            if out.len() != config.output_dim {
                return Err(ModelError::Shape("d1 vision projector".into()));
            }
            tokens.push(out);
        }
    }
    Ok(tokens.into_iter().flatten().collect())
}

fn add_interpolated_position(
    values: &mut [f32],
    table: &[f32],
    width: usize,
    row: usize,
    column: usize,
    grid_h: usize,
    grid_w: usize,
) {
    let radius_y = (BASE_GRID as f32 / grid_h as f32).max(1.0);
    let radius_x = (BASE_GRID as f32 / grid_w as f32).max(1.0);
    let center_y = (row as f32 + 0.5) * (BASE_GRID as f32 / grid_h as f32) - 0.5;
    let center_x = (column as f32 + 0.5) * (BASE_GRID as f32 / grid_w as f32) - 0.5;
    let kernel_y = (radius_y * 2.0 + 2.0).ceil() as isize;
    let kernel_x = (radius_x * 2.0 + 2.0).ceil() as isize;
    let start_y = (center_y - radius_y).floor() as isize;
    let start_x = (center_x - radius_x).floor() as isize;

    let mut total = 0.0f32;
    let mut samples = Vec::with_capacity((kernel_y * kernel_x) as usize);
    for ky in 0..kernel_y {
        let y = start_y + ky;
        if y < 0 || y >= BASE_GRID as isize {
            continue;
        }
        let wy = (1.0 - (y as f32 - center_y).abs() / radius_y).max(0.0);
        if wy == 0.0 {
            continue;
        }
        for kx in 0..kernel_x {
            let x = start_x + kx;
            if x < 0 || x >= BASE_GRID as isize {
                continue;
            }
            let wx = (1.0 - (x as f32 - center_x).abs() / radius_x).max(0.0);
            if wx == 0.0 {
                continue;
            }
            let weight = wy * wx;
            total += weight;
            samples.push((y as usize, x as usize, weight));
        }
    }
    if total <= 0.0 {
        return;
    }
    let inv = 1.0 / total;
    for (y, x, weight) in samples {
        let offset = (y * BASE_GRID + x) * width;
        let scale = weight * inv;
        for (index, value) in values.iter_mut().enumerate() {
            *value += table[offset + index] * scale;
        }
    }
}

fn run_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &D1VisionConfig,
    layer: usize,
    hidden: Vec<Vec<f32>>,
    n_patches: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let prefix = format!("v.blk.{layer}");
    let pressed: Vec<Vec<f32>> = hidden
        .iter()
        .map(|row| {
            layer_norm_weighted(
                gguf,
                file,
                &format!("{prefix}.ln1.weight"),
                &format!("{prefix}.ln1.bias"),
                row,
                config.eps,
            )
        })
        .collect::<Result<_, _>>()?;

    let mut queries = Vec::with_capacity(n_patches);
    let mut keys = Vec::with_capacity(n_patches);
    let mut values = Vec::with_capacity(n_patches);
    for row in &pressed {
        queries.push(linear(
            gguf,
            file,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_q.bias"),
            row,
        )?);
        keys.push(linear(
            gguf,
            file,
            &format!("{prefix}.attn_k.weight"),
            &format!("{prefix}.attn_k.bias"),
            row,
        )?);
        values.push(linear(
            gguf,
            file,
            &format!("{prefix}.attn_v.weight"),
            &format!("{prefix}.attn_v.bias"),
            row,
        )?);
    }

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut attention = vec![vec![0.0f32; config.n_embd]; n_patches];
    let mut scores = vec![0.0f32; n_patches];
    for (frame, row) in attention.iter_mut().enumerate() {
        for head in 0..config.n_head {
            let start = head * config.head_dim;
            for (other, key) in keys.iter().enumerate() {
                let mut sum = 0.0f32;
                for dim in 0..config.head_dim {
                    sum += queries[frame][start + dim] * key[start + dim];
                }
                scores[other] = sum * scale;
            }
            softmax_in_place(&mut scores);
            for (other, weight) in scores.iter().enumerate() {
                for dim in 0..config.head_dim {
                    row[start + dim] += weight * values[other][start + dim];
                }
            }
        }
    }

    let mut out = Vec::with_capacity(n_patches);
    for (index, row) in attention.iter().enumerate() {
        let projected = linear(
            gguf,
            file,
            &format!("{prefix}.attn_out.weight"),
            &format!("{prefix}.attn_out.bias"),
            row,
        )?;
        let mut residual: Vec<f32> = hidden[index]
            .iter()
            .zip(&projected)
            .map(|(left, right)| left + right)
            .collect();
        let normed = layer_norm_weighted(
            gguf,
            file,
            &format!("{prefix}.ln2.weight"),
            &format!("{prefix}.ln2.bias"),
            &residual,
            config.eps,
        )?;
        let up = linear(
            gguf,
            file,
            &format!("{prefix}.ffn_up.weight"),
            &format!("{prefix}.ffn_up.bias"),
            &normed,
        )?;
        let activated: Vec<f32> = up.iter().map(|value| gelu_tanh(*value)).collect();
        let down = linear(
            gguf,
            file,
            &format!("{prefix}.ffn_down.weight"),
            &format!("{prefix}.ffn_down.bias"),
            &activated,
        )?;
        for (value, add) in residual.iter_mut().zip(&down) {
            *value += add;
        }
        out.push(residual);
    }
    Ok(out)
}
