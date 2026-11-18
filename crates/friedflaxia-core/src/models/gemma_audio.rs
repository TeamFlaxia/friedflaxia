//! Gemma 4 audio encoder used by EmbeddingGemma 2.
//!
//! The encoder consumes log-mel features `[frames, 128]`, runs two stride-2
//! subsampling convolutions, 12 chunked-attention conformer layers with
//! relative position bias and a causal depthwise convolution, then projects
//! the result to the text model width.

use super::ModelError;
use crate::gguf::Gguf;
use crate::ops::{rms_norm, silu, softmax_in_place};
use crate::tensor::{TensorView, tensor};

const MEL_BINS: usize = 128;
const SUBSAMPLE_CHANNELS: [usize; 2] = [128, 32];
const CHUNK: usize = 12;
const PAST: usize = 12;
const CONTEXT: usize = CHUNK + PAST;
const REL_POSITIONS: usize = PAST + 1;
const SOFTCAP: f32 = 50.0;
const GRADIENT_CLIP: f32 = 1.0e10;

pub struct GemmaAudioConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub output_dim: usize,
}

impl GemmaAudioConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let get_u32 = |key: &str| -> Result<usize, ModelError> {
            gguf.metadata(key)
                .and_then(|value| value.as_u32())
                .map(|value| value as usize)
                .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
        };
        let n_layer = get_u32("clip.audio.block_count")?;
        let n_embd = get_u32("clip.audio.embedding_length")?;
        let n_ff = get_u32("clip.audio.feed_forward_length")?;
        let n_head = get_u32("clip.audio.attention.head_count")?;
        let output_dim = get_u32("clip.audio.projection_dim")?;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(ModelError::Shape("audio head count".into()));
        }
        let eps = gguf
            .metadata("clip.audio.attention.layer_norm_epsilon")
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

/// Debug helper: return the subsampled rows before the conformer stack.
#[doc(hidden)]
pub fn debug_audio_subsample(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    Ok(subsample(gguf, file, mel, frames)?
        .into_iter()
        .flatten()
        .collect())
}

/// Debug helper: return the first layer's feed-forward output.
#[doc(hidden)]
pub fn debug_audio_ffn1(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let prefix = "a.blk.0";
    let out = feed_forward(
        gguf,
        file,
        &config,
        &hidden,
        &format!("{prefix}.ffn_norm.weight"),
        &format!("{prefix}.ffn_up.weight"),
        &format!("{prefix}.ffn_up"),
        &format!("{prefix}.ffn_down.weight"),
        &format!("{prefix}.ffn_down"),
        &format!("{prefix}.ffn_post_norm.weight"),
    )?;
    Ok(out.into_iter().flatten().collect())
}

/// Debug helper: return the first layer's attention residual.
#[doc(hidden)]
pub fn debug_audio_attention(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let positions = relative_positions(&config);
    let relative_keys = build_relative_keys(gguf, file, &config, &positions)?;
    let out = run_layer(
        gguf,
        file,
        &config,
        0,
        hidden,
        &relative_keys,
        frames / 4,
        1,
    )?;
    Ok(out.into_iter().flatten().collect())
}

/// Debug helper: return layer 0 query/key/value tensors after their clamps.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn debug_audio_qkv(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let prefix = "a.blk.0";
    let ffn1 = feed_forward(
        gguf,
        file,
        &config,
        &hidden,
        &format!("{prefix}.ffn_norm.weight"),
        &format!("{prefix}.ffn_up.weight"),
        &format!("{prefix}.ffn_up"),
        &format!("{prefix}.ffn_down.weight"),
        &format!("{prefix}.ffn_down"),
        &format!("{prefix}.ffn_post_norm.weight"),
    )?;
    let norm = tensor(gguf, file, &format!("{prefix}.attn_pre_norm.weight"))?.vec()?;
    let mut q_all = Vec::new();
    let mut k_all = Vec::new();
    let mut v_all = Vec::new();
    for row in &ffn1 {
        let normed = rms_norm(row, Some(&norm), config.eps);
        q_all.extend(clamped_linear(
            gguf,
            file,
            &format!("{prefix}.attn_q"),
            &normed,
        )?);
        k_all.extend(clamped_linear(
            gguf,
            file,
            &format!("{prefix}.attn_k"),
            &normed,
        )?);
        v_all.extend(clamped_linear(
            gguf,
            file,
            &format!("{prefix}.attn_v"),
            &normed,
        )?);
    }
    Ok((q_all, k_all, v_all))
}

/// Debug helper: return the relative key projection rows.
#[doc(hidden)]
pub fn debug_audio_relative_keys(gguf: &Gguf, file: &[u8]) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let positions = relative_positions(&config);
    Ok(build_relative_keys(gguf, file, &config, &positions)?
        .into_iter()
        .flatten()
        .collect())
}

/// Debug helper: return the first layer's attention projection before norm/residual.
#[doc(hidden)]
pub fn debug_audio_attn_out(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let positions = relative_positions(&config);
    let relative_keys = build_relative_keys(gguf, file, &config, &positions)?;
    let out = run_layer(
        gguf,
        file,
        &config,
        0,
        hidden,
        &relative_keys,
        frames / 4,
        2,
    )?;
    Ok(out.into_iter().flatten().collect())
}

/// Debug helper: return the first layer's raw attention output before projection.
#[doc(hidden)]
pub fn debug_audio_raw_attention(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let positions = relative_positions(&config);
    let relative_keys = build_relative_keys(gguf, file, &config, &positions)?;
    let out = run_layer(
        gguf,
        file,
        &config,
        0,
        hidden,
        &relative_keys,
        frames / 4,
        3,
    )?;
    Ok(out.into_iter().flatten().collect())
}

/// Debug helper: run layer 0 up to a stage (4 = light conv, 5 = feed-forward 2).
#[doc(hidden)]
pub fn debug_audio_layer0(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
    stage: u8,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    let hidden = subsample(gguf, file, mel, frames)?;
    let positions = relative_positions(&config);
    let relative_keys = build_relative_keys(gguf, file, &config, &positions)?;
    let out = run_layer(
        gguf,
        file,
        &config,
        0,
        hidden,
        &relative_keys,
        frames / 4,
        stage,
    )?;
    Ok(out.into_iter().flatten().collect())
}

/// Encode log-mel features into `[soft_tokens, output_dim]` rows.
pub fn encode_audio(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<f32>, ModelError> {
    let config = GemmaAudioConfig::from_gguf(gguf)?;
    if frames == 0 || mel.len() != frames * MEL_BINS {
        return Err(ModelError::Shape("audio mel features".into()));
    }

    let mut hidden = subsample(gguf, file, mel, frames)?;
    let positions = relative_positions(&config);
    let relative_keys = build_relative_keys(gguf, file, &config, &positions)?;
    let frame_count = hidden.len();

    for layer in 0..config.n_layer {
        hidden = run_layer(
            gguf,
            file,
            &config,
            layer,
            hidden,
            &relative_keys,
            frame_count,
            0,
        )?;
    }

    let out_weight = tensor(gguf, file, "a.pre_encode.out.weight")?;
    let out_bias = tensor(gguf, file, "a.pre_encode.out.bias")?.vec()?;
    let mut out = Vec::with_capacity(frame_count);
    for row in &hidden {
        let mut projected = out_weight.matvec(row)?;
        for (value, bias) in projected.iter_mut().zip(&out_bias) {
            *value += bias;
        }
        out.push(projected);
    }

    let normed: Vec<Vec<f32>> = out
        .iter()
        .map(|row| rms_norm(row, None, config.eps))
        .collect();
    let projection = tensor(gguf, file, "mm.a.input_projection.weight")?;
    if projection.cols() != normed[0].len() || projection.rows() != config.output_dim {
        return Err(ModelError::Shape("audio projection".into()));
    }
    let mut result = Vec::with_capacity(frame_count * config.output_dim);
    for row in &normed {
        result.extend(projection.matvec(row)?);
    }
    Ok(result)
}

fn subsample(
    gguf: &Gguf,
    file: &[u8],
    mel: &[f32],
    frames: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let mut width = MEL_BINS;
    let mut height = frames;
    let mut channels = 1usize;
    let mut data = mel.to_vec();

    for (index, out_channels) in SUBSAMPLE_CHANNELS.iter().enumerate() {
        let weight = tensor(gguf, file, &format!("a.conv1d.{index}.weight"))?.vec()?;
        let norm = tensor(gguf, file, &format!("a.conv1d.{index}.norm.weight"))?.vec()?;
        if weight.len() != out_channels * channels * 9 || norm.len() != *out_channels {
            return Err(ModelError::Shape("audio subsample convolution".into()));
        }
        let out_height = (height + 2 - 3) / 2 + 1;
        let out_width = (width + 2 - 3) / 2 + 1;
        let mut out = vec![0.0f32; out_channels * out_height * out_width];
        for o in 0..*out_channels {
            let weight_base = o * channels * 9;
            for y in 0..out_height {
                for x in 0..out_width {
                    let mut sum = 0.0f32;
                    for c in 0..channels {
                        for ky in 0..3usize {
                            let iy = y * 2 + ky;
                            if iy == 0 || iy > height {
                                continue;
                            }
                            for kx in 0..3usize {
                                let ix = x * 2 + kx;
                                if ix == 0 || ix > width {
                                    continue;
                                }
                                let input = data[(c * height + (iy - 1)) * width + (ix - 1)];
                                let weight = weight[weight_base + (c * 3 + ky) * 3 + kx];
                                sum += input * weight;
                            }
                        }
                    }
                    out[(o * out_height + y) * out_width + x] = sum;
                }
            }
        }
        // LayerNorm over the channel axis, then ReLU.
        let plane = out_height * out_width;
        for position in 0..plane {
            let mut mean = 0.0f32;
            for c in 0..*out_channels {
                mean += out[c * plane + position];
            }
            mean /= *out_channels as f32;
            let mut variance = 0.0f32;
            for c in 0..*out_channels {
                let centered = out[c * plane + position] - mean;
                variance += centered * centered;
            }
            variance /= *out_channels as f32;
            let inv = 1.0 / (variance + 1e-6).sqrt();
            for c in 0..*out_channels {
                let value = (out[c * plane + position] - mean) * inv * norm[c];
                out[c * plane + position] = value.max(0.0);
            }
        }
        data = out;
        channels = *out_channels;
        height = out_height;
        width = out_width;
    }

    // [channels][height][width] -> [height][width * channels] with channel fastest.
    let row_width = width * channels;
    let mut rows = Vec::with_capacity(height);
    for y in 0..height {
        let mut row = Vec::with_capacity(row_width);
        for x in 0..width {
            for c in 0..channels {
                row.push(data[(c * height + y) * width + x]);
            }
        }
        rows.push(row);
    }

    let projection = tensor(gguf, file, "a.input_projection.weight")?;
    if projection.cols() != row_width {
        return Err(ModelError::Shape("audio input projection".into()));
    }
    let mut out = Vec::with_capacity(height);
    for row in &rows {
        out.push(projection.matvec(row)?);
    }
    Ok(out)
}

fn relative_positions(config: &GemmaAudioConfig) -> Vec<f32> {
    let half = config.n_embd / 2;
    let increment = (10_000.0f32).ln() / (half.max(2) - 1) as f32;
    let mut out = vec![0.0f32; REL_POSITIONS * config.n_embd];
    for (row, position) in (0..=PAST).rev().enumerate() {
        for index in 0..half {
            let inv = (-(index as f32) * increment).exp();
            let angle = position as f32 * inv;
            out[row * config.n_embd + index] = angle.sin();
            out[row * config.n_embd + half + index] = angle.cos();
        }
    }
    out
}

fn build_relative_keys(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaAudioConfig,
    positions: &[f32],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let projection = tensor(gguf, file, "a.blk.0.attn_k_rel.weight")?;
    if projection.cols() != config.n_embd {
        return Err(ModelError::Shape("audio relative key projection".into()));
    }
    let mut keys = Vec::with_capacity(REL_POSITIONS);
    for row in 0..REL_POSITIONS {
        keys.push(projection.matvec(&positions[row * config.n_embd..(row + 1) * config.n_embd])?);
    }
    Ok(keys)
}

fn clamped_linear(
    gguf: &Gguf,
    file: &[u8],
    prefix: &str,
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let weight = tensor(gguf, file, &format!("{prefix}.weight"))?;
    let input_min = tensor(gguf, file, &format!("{prefix}.input_min"));
    let input_max = tensor(gguf, file, &format!("{prefix}.input_max"));
    let output_min = tensor(gguf, file, &format!("{prefix}.output_min"));
    let output_max = tensor(gguf, file, &format!("{prefix}.output_max"));
    clamped_matvec(
        &weight,
        input,
        &input_min,
        &input_max,
        &output_min,
        &output_max,
    )
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn run_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaAudioConfig,
    layer: usize,
    hidden: Vec<Vec<f32>>,
    relative_keys: &[Vec<f32>],
    frame_count: usize,
    stop_stage: u8,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let prefix = format!("a.blk.{layer}");
    let ffn1 = feed_forward(
        gguf,
        file,
        config,
        &hidden,
        &format!("{prefix}.ffn_norm.weight"),
        &format!("{prefix}.ffn_up.weight"),
        &format!("{prefix}.ffn_up"),
        &format!("{prefix}.ffn_down.weight"),
        &format!("{prefix}.ffn_down"),
        &format!("{prefix}.ffn_post_norm.weight"),
    )?;

    let attn_pre = tensor(gguf, file, &format!("{prefix}.attn_pre_norm.weight"))?.vec()?;
    let normed: Vec<Vec<f32>> = ffn1
        .iter()
        .map(|row| rms_norm(row, Some(&attn_pre), config.eps))
        .collect();

    let per_dim = tensor(gguf, file, &format!("{prefix}.per_dim_scale.weight"))?.vec()?;

    let q_scale = (1.0 / (config.head_dim as f32).sqrt()) / std::f32::consts::LN_2;
    let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
    let mut queries = Vec::with_capacity(frame_count);
    let mut keys = Vec::with_capacity(frame_count);
    let mut values = Vec::with_capacity(frame_count);
    for row in &normed {
        let mut q = clamped_linear(gguf, file, &format!("{prefix}.attn_q"), row)?;
        let mut k = clamped_linear(gguf, file, &format!("{prefix}.attn_k"), row)?;
        let v = clamped_linear(gguf, file, &format!("{prefix}.attn_v"), row)?;
        for head in 0..config.n_head {
            let start = head * config.head_dim;
            for dim in 0..config.head_dim {
                q[start + dim] *= q_scale * per_dim[dim];
                k[start + dim] *= k_scale;
            }
        }
        queries.push(q);
        keys.push(k);
        values.push(v);
    }

    let attn_post = tensor(gguf, file, &format!("{prefix}.attn_post_norm.weight"))?.vec()?;
    let raw_attention = vec![vec![0.0f32; config.n_embd]; frame_count];
    let mut attention = raw_attention.clone();
    let mut scores = [0.0f32; CONTEXT];
    for frame in 0..frame_count {
        let first = frame.saturating_sub(PAST - 1);
        let last = frame + 1;
        for head in 0..config.n_head {
            let start = head * config.head_dim;
            let mut count = 0;
            for key_index in first..last {
                let mut score = 0.0f32;
                for dim in 0..config.head_dim {
                    score += queries[frame][start + dim] * keys[key_index][start + dim];
                }
                let relative = key_index as isize - frame as isize + PAST as isize;
                if (0..REL_POSITIONS as isize).contains(&relative) {
                    let relative = relative as usize;
                    for dim in 0..config.head_dim {
                        score += queries[frame][start + dim] * relative_keys[relative][start + dim];
                    }
                }
                scores[count] = score;
                count += 1;
            }
            for score in scores.iter_mut().take(count) {
                *score = (*score / SOFTCAP).tanh() * SOFTCAP;
            }
            softmax_in_place(&mut scores[..count]);
            for (offset, score) in scores[..count].iter().enumerate() {
                let key_index = first + offset;
                for dim in 0..config.head_dim {
                    attention[frame][start + dim] += score * values[key_index][start + dim];
                }
            }
        }
        if stop_stage == 3 {
            continue;
        }
        attention[frame] =
            clamped_linear(gguf, file, &format!("{prefix}.attn_out"), &attention[frame])?;
        if stop_stage == 2 {
            continue;
        }
        for value in &mut attention[frame] {
            *value = value.clamp(-GRADIENT_CLIP, GRADIENT_CLIP);
        }
        attention[frame] = rms_norm(&attention[frame], Some(&attn_post), config.eps);
        for (value, residual) in attention[frame].iter_mut().zip(&ffn1[frame]) {
            *value += residual;
        }
    }

    if stop_stage == 1 {
        return Ok(attention);
    }
    let conv = light_conv(gguf, file, config, &attention, &prefix)?;
    if stop_stage == 4 {
        return Ok(conv);
    }
    let ffn2 = feed_forward(
        gguf,
        file,
        config,
        &conv,
        &format!("{prefix}.ffn_norm_1.weight"),
        &format!("{prefix}.ffn_up_1.weight"),
        &format!("{prefix}.ffn_up_1"),
        &format!("{prefix}.ffn_down_1.weight"),
        &format!("{prefix}.ffn_down_1"),
        &format!("{prefix}.ffn_post_norm_1.weight"),
    )?;

    if stop_stage == 5 {
        return Ok(ffn2);
    }
    let out_norm = tensor(gguf, file, &format!("{prefix}.ln2.weight"))?.vec()?;
    Ok(ffn2
        .iter()
        .map(|row| rms_norm(row, Some(&out_norm), config.eps))
        .collect())
}

fn light_conv(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaAudioConfig,
    hidden: &[Vec<f32>],
    prefix: &str,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let norm = tensor(gguf, file, &format!("{prefix}.conv_norm.weight"))?.vec()?;
    let pw1_prefix = format!("{prefix}.conv_pw1");
    let pw2_prefix = format!("{prefix}.conv_pw2");
    let dw = tensor(gguf, file, &format!("{prefix}.conv_dw.weight"))?.vec()?;
    let conv_norm = tensor(gguf, file, &format!("{prefix}.norm_conv.weight"))?.vec()?;
    if dw.len() != 5 * config.n_embd {
        return Err(ModelError::Shape("audio depthwise conv".into()));
    }

    let mut gated_rows = Vec::with_capacity(hidden.len());
    for row in hidden {
        let normed = rms_norm(row, Some(&norm), config.eps);
        let start = clamped_linear(gguf, file, &pw1_prefix, &normed)?;
        let half = start.len() / 2;
        let gated: Vec<f32> = (0..half)
            .map(|index| start[index] * (1.0 / (1.0 + (-start[half + index]).exp())))
            .collect();
        gated_rows.push(gated);
    }

    let mut out = Vec::with_capacity(hidden.len());
    for (frame, row) in hidden.iter().enumerate() {
        let mut conv = vec![0.0f32; config.n_embd];
        for c in 0..config.n_embd {
            let mut sum = 0.0f32;
            for k in 0..5usize {
                if frame + k < 4 {
                    continue;
                }
                let input_frame = frame + k - 4;
                sum += gated_rows[input_frame][c] * dw[c * 5 + k];
            }
            conv[c] = sum.clamp(-GRADIENT_CLIP, GRADIENT_CLIP);
        }
        let normed = rms_norm(&conv, Some(&conv_norm), config.eps);
        let activated: Vec<f32> = normed.iter().map(|value| silu(*value)).collect();
        let projected = clamped_linear(gguf, file, &pw2_prefix, &activated)?;
        out.push(
            projected
                .iter()
                .zip(row)
                .map(|(value, residual)| value + residual)
                .collect(),
        );
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn feed_forward(
    gguf: &Gguf,
    file: &[u8],
    config: &GemmaAudioConfig,
    hidden: &[Vec<f32>],
    norm_name: &str,
    up_name: &str,
    up_prefix: &str,
    down_name: &str,
    down_prefix: &str,
    post_norm_name: &str,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let norm = tensor(gguf, file, norm_name)?.vec()?;
    let up = tensor(gguf, file, up_name)?;
    let up_min = tensor(gguf, file, &format!("{up_prefix}.input_min"));
    let up_max = tensor(gguf, file, &format!("{up_prefix}.input_max"));
    let up_out_min = tensor(gguf, file, &format!("{up_prefix}.output_min"));
    let up_out_max = tensor(gguf, file, &format!("{up_prefix}.output_max"));
    let down = tensor(gguf, file, down_name)?;
    let down_min = tensor(gguf, file, &format!("{down_prefix}.input_min"));
    let down_max = tensor(gguf, file, &format!("{down_prefix}.input_max"));
    let down_out_min = tensor(gguf, file, &format!("{down_prefix}.output_min"));
    let down_out_max = tensor(gguf, file, &format!("{down_prefix}.output_max"));
    let post_norm = tensor(gguf, file, post_norm_name)?.vec()?;

    let mut out = Vec::with_capacity(hidden.len());
    for row in hidden {
        let clamped: Vec<f32> = row
            .iter()
            .map(|value| value.clamp(-GRADIENT_CLIP, GRADIENT_CLIP))
            .collect();
        let normed = rms_norm(&clamped, Some(&norm), config.eps);
        let mut up_out = clamped_matvec(&up, &normed, &up_min, &up_max, &up_out_min, &up_out_max)?;
        for value in &mut up_out {
            *value = silu(*value);
        }
        let down_out = clamped_matvec(
            &down,
            &up_out,
            &down_min,
            &down_max,
            &down_out_min,
            &down_out_max,
        )?;
        let post = rms_norm(&down_out, Some(&post_norm), config.eps);
        out.push(
            post.iter()
                .zip(row)
                .map(|(value, residual)| value * 0.5 + residual)
                .collect(),
        );
    }
    Ok(out)
}

fn clamped_matvec(
    weight: &TensorView<'_>,
    input: &[f32],
    input_min: &Result<TensorView<'_>, crate::tensor::TensorError>,
    input_max: &Result<TensorView<'_>, crate::tensor::TensorError>,
    output_min: &Result<TensorView<'_>, crate::tensor::TensorError>,
    output_max: &Result<TensorView<'_>, crate::tensor::TensorError>,
) -> Result<Vec<f32>, ModelError> {
    let min = input_min.as_ref().ok().and_then(|view| view.scalar().ok());
    let max = input_max.as_ref().ok().and_then(|view| view.scalar().ok());
    let clamped: Vec<f32> = input
        .iter()
        .map(|value| {
            let mut value = *value;
            if let Some(min) = min {
                value = value.max(min);
            }
            if let Some(max) = max {
                value = value.min(max);
            }
            value
        })
        .collect();
    let mut out = weight.matvec(&clamped)?;
    let out_min = output_min.as_ref().ok().and_then(|view| view.scalar().ok());
    let out_max = output_max.as_ref().ok().and_then(|view| view.scalar().ok());
    for value in &mut out {
        if let Some(min) = out_min {
            *value = value.max(min);
        }
        if let Some(max) = out_max {
            *value = value.min(max);
        }
    }
    Ok(out)
}
