//! d1-omni audio frontend, FastConformer encoder, adapter and residual.

use super::ModelError;
use crate::gguf::Gguf;
use crate::ops::{gelu_erf, layer_norm, silu, softmax_in_place};
use crate::tensor::tensor;
use std::f32::consts::PI;

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 512;
const WINDOW: usize = 400;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const MIN_SAMPLES: usize = 8_000;
const MAX_SAMPLES: usize = 30 * SAMPLE_RATE;
const CHANNELS: usize = 256;
const D_MODEL: usize = 512;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEADS;
const CONV_KERNEL: usize = 9;

pub struct D1AudioConfig {
    pub n_layer: usize,
    pub eps: f32,
}

impl D1AudioConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let n_layer = gguf
            .metadata("clip.audio.block_count")
            .and_then(|value| value.as_u32())
            .map(|value| value as usize)
            .ok_or_else(|| ModelError::Shape("missing clip.audio.block_count".into()))?;
        let eps = gguf
            .metadata("clip.audio.attention.layer_norm_epsilon")
            .and_then(|value| value.as_f32())
            .unwrap_or(1e-5);
        Ok(Self { n_layer, eps })
    }
}

/// Encode 16 kHz mono samples into `[tokens, 1024]` prefix embeddings.
pub fn encode_audio(gguf: &Gguf, file: &[u8], samples: &[f32]) -> Result<Vec<f32>, ModelError> {
    if samples.is_empty() {
        return Err(ModelError::Invalid("audio samples must not be empty"));
    }
    let config = D1AudioConfig::from_gguf(gguf)?;
    let samples = &samples[..samples.len().min(MAX_SAMPLES)];
    let mut padded = samples.to_vec();
    if padded.len() < MIN_SAMPLES {
        padded.resize(MIN_SAMPLES, 0.0);
    }

    let (mel, _) = mel_frontend(&padded);
    let (mut hidden, token_count) = pre_encode(gguf, file, &mel, mel.len())?;
    let positions = relative_positions(token_count);

    for layer in 0..config.n_layer {
        hidden = conformer_layer(
            gguf,
            file,
            &config,
            layer,
            hidden,
            &positions,
            token_count,
            0,
        )?;
    }

    let adapter = adapter(gguf, file, &hidden)?;
    let residual = residual(gguf, file, &adapter)?;
    Ok(residual.into_iter().flatten().collect())
}

fn hz_to_mel(frequency: f32) -> f32 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = (6.4f32).ln() / 27.0;
    if frequency >= min_log_hz {
        min_log_mel + (frequency / min_log_hz).ln() / logstep
    } else {
        frequency / (200.0 / 3.0)
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / (200.0 / 3.0);
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        (200.0 / 3.0) * mel
    }
}

fn slaney_filterbank() -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let mut points = vec![0.0f32; MEL_BINS + 2];
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let step = (mel_max - mel_min) / (MEL_BINS + 1) as f32;
    for (index, point) in points.iter_mut().enumerate() {
        *point = mel_to_hz(mel_min + step * index as f32);
    }
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|index| index as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect();
    let mut filters = vec![0.0f32; MEL_BINS * bins];
    for filter in 0..MEL_BINS {
        let down = points[filter + 1] - points[filter];
        let up = points[filter + 2] - points[filter + 1];
        let enorm = 2.0 / (points[filter + 2] - points[filter]);
        for (bin, frequency) in fft_freqs.iter().enumerate() {
            let lower = (frequency - points[filter]) / down;
            let upper = (points[filter + 2] - frequency) / up;
            filters[filter * bins + bin] = lower.min(upper).max(0.0) * enorm;
        }
    }
    filters
}

#[allow(clippy::needless_range_loop)]
fn mel_frontend(samples: &[f32]) -> (Vec<Vec<f32>>, usize) {
    let window = hann_window();
    let frames = samples.len() / HOP + 1;
    let valid_frames = samples.len() / HOP;
    let filters = slaney_filterbank();
    let bins = N_FFT / 2 + 1;
    let mut mel = vec![vec![0.0f32; MEL_BINS]; frames];
    let mut real = vec![0.0f32; N_FFT];
    let mut imag = vec![0.0f32; N_FFT];
    let mut preemphasized = vec![0.0f32; samples.len()];
    if let Some(first) = samples.first() {
        preemphasized[0] = *first;
    }
    for index in 1..samples.len() {
        preemphasized[index] = samples[index] - 0.97 * samples[index - 1];
    }
    for frame in 0..frames {
        real.fill(0.0);
        imag.fill(0.0);
        let start = frame as isize * HOP as isize - (N_FFT / 2) as isize;
        for j in 0..N_FFT {
            let source = start + j as isize;
            if source < 0 || source >= preemphasized.len() as isize {
                continue;
            }
            let window_index = j as isize - ((N_FFT - WINDOW) / 2) as isize;
            let window_value = if (0..WINDOW as isize).contains(&window_index) {
                window[window_index as usize]
            } else {
                0.0
            };
            real[j] = preemphasized[source as usize] * window_value;
        }
        fft(&mut real, &mut imag);
        let mut power = vec![0.0f32; bins];
        for (bin, value) in power.iter_mut().enumerate() {
            *value = real[bin] * real[bin] + imag[bin] * imag[bin];
        }
        for filter in 0..MEL_BINS {
            let base = filter * bins;
            let mut sum = 0.0f32;
            for bin in 0..bins {
                sum += filters[base + bin] * power[bin];
            }
            mel[frame][filter] = (sum + 2f32.powi(-24)).ln();
        }
    }
    // Per-feature normalization over the valid frames only.
    for filter in 0..MEL_BINS {
        let count = valid_frames.max(1) as f32;
        let mut mean = 0.0f32;
        for frame in 0..valid_frames {
            mean += mel[frame][filter];
        }
        mean /= count;
        let mut variance = 0.0f32;
        for frame in 0..valid_frames {
            let centered = mel[frame][filter] - mean;
            variance += centered * centered;
        }
        let denominator = if valid_frames > 1 {
            valid_frames as f32 - 1.0
        } else {
            1.0
        };
        let std = (variance / denominator).sqrt();
        let std = if std.is_nan() { 0.0 } else { std };
        for frame in 0..frames {
            if frame < valid_frames {
                mel[frame][filter] = (mel[frame][filter] - mean) / (std + 1e-5);
            } else {
                mel[frame][filter] = 0.0;
            }
        }
    }
    (mel, valid_frames)
}

fn hann_window() -> Vec<f32> {
    (0..WINDOW)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / (WINDOW - 1) as f32).cos())
        .collect()
}

fn fft(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut length = 2usize;
    while length <= n {
        let angle = -2.0 * PI / length as f32;
        let w_real = angle.cos();
        let w_imag = angle.sin();
        let mut start = 0usize;
        while start < n {
            let mut cur_real = 1.0f32;
            let mut cur_imag = 0.0f32;
            for offset in 0..length / 2 {
                let even = start + offset;
                let odd = even + length / 2;
                let odd_real = real[odd] * cur_real - imag[odd] * cur_imag;
                let odd_imag = real[odd] * cur_imag + imag[odd] * cur_real;
                real[odd] = real[even] - odd_real;
                imag[odd] = imag[even] - odd_imag;
                real[even] += odd_real;
                imag[even] += odd_imag;
                let next_real = cur_real * w_real - cur_imag * w_imag;
                cur_imag = cur_real * w_imag + cur_imag * w_real;
                cur_real = next_real;
            }
            start += length;
        }
        length <<= 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn conv2d(
    input: &[f32],
    height: usize,
    width: usize,
    in_channels: usize,
    out_channels: usize,
    weight: &[f32],
    bias: &[f32],
    depthwise: bool,
    groups: usize,
    kernel: usize,
    stride: usize,
) -> (Vec<f32>, usize, usize) {
    let pad = if kernel == 3 { 1 } else { 0 };
    let out_height = (height + 2 * pad - kernel) / stride + 1;
    let out_width = (width + 2 * pad - kernel) / stride + 1;
    let kn = kernel * kernel;
    let mut out = vec![0.0f32; out_channels * out_height * out_width];
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    for group in 0..groups {
        for oc in 0..out_per_group {
            let o = group * out_per_group + oc;
            let bias_value = bias.get(o).copied().unwrap_or(0.0);
            for y in 0..out_height {
                for x in 0..out_width {
                    let mut sum = bias_value;
                    for ic in 0..in_per_group {
                        let c = group * in_per_group + ic;
                        for ky in 0..kernel {
                            let iy = y * stride + ky;
                            if iy < pad || iy - pad >= height {
                                continue;
                            }
                            for kx in 0..kernel {
                                let ix = x * stride + kx;
                                if ix < pad || ix - pad >= width {
                                    continue;
                                }
                                let input_value =
                                    input[(c * height + (iy - pad)) * width + (ix - pad)];
                                let weight_index = if depthwise {
                                    o * kn + ky * kernel + kx
                                } else {
                                    (o * in_channels + c) * kn + ky * kernel + kx
                                };
                                sum += input_value * weight[weight_index];
                            }
                        }
                    }
                    out[(o * out_height + y) * out_width + x] = sum;
                }
            }
        }
    }
    (out, out_height, out_width)
}

fn mask_time(data: &mut [f32], channels: usize, height: usize, width: usize, valid: usize) {
    if valid >= height {
        return;
    }
    for c in 0..channels {
        for frame in valid..height {
            let start = (c * height + frame) * width;
            data[start..start + width].fill(0.0);
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn pre_encode_rows(
    gguf: &Gguf,
    file: &[u8],
    mel: &[Vec<f32>],
    frames: usize,
) -> Result<(Vec<Vec<f32>>, usize), ModelError> {
    let mut data = vec![0.0f32; frames * MEL_BINS];
    for (frame, row) in mel.iter().enumerate().take(frames) {
        data[frame * MEL_BINS..(frame + 1) * MEL_BINS].copy_from_slice(row);
    }
    let mut height = frames;
    let mut width = MEL_BINS;
    let mut channels = 1usize;
    let mut length = frames;
    let mut valid = frames.saturating_sub(1);
    for index in 0..7 {
        let (kernel, stride, depthwise, out_channels) = match index {
            0 => (3, 2, false, CHANNELS),
            2 => (3, 2, true, CHANNELS),
            3 => (1, 1, false, CHANNELS),
            5 => (3, 2, true, CHANNELS),
            6 => (1, 1, false, CHANNELS),
            _ => continue,
        };
        mask_time(&mut data, channels, height, width, valid);
        let weight = tensor(gguf, file, &format!("a.conv1d.{index}.weight"))?.vec()?;
        let bias = tensor(gguf, file, &format!("a.conv1d.{index}.bias"))?.vec()?;
        let groups = if depthwise { out_channels } else { 1 };
        let (mut out, out_h, out_w) = conv2d(
            &data,
            height,
            width,
            channels,
            out_channels,
            &weight,
            &bias,
            depthwise,
            groups,
            kernel,
            stride,
        );
        if index == 0 || index == 3 || index == 6 {
            for value in &mut out {
                *value = value.max(0.0);
            }
        }
        data = out;
        channels = out_channels;
        height = out_h;
        width = out_w;
        if stride == 2 {
            length = (length + 2 - 3) / 2 + 1;
            valid = (valid + 2 - 3) / 2 + 1;
        }
    }
    mask_time(&mut data, channels, height, width, valid);
    let mut rows = Vec::with_capacity(height);
    for frame in 0..height {
        let mut row = Vec::with_capacity(channels * width);
        for c in 0..channels {
            for f in 0..width {
                row.push(data[(c * height + frame) * width + f]);
            }
        }
        rows.push(row);
    }
    Ok((rows, height))
}

fn pre_encode(
    gguf: &Gguf,
    file: &[u8],
    mel: &[Vec<f32>],
    frames: usize,
) -> Result<(Vec<Vec<f32>>, usize), ModelError> {
    let (rows, height) = pre_encode_rows(gguf, file, mel, frames)?;
    let projection = tensor(gguf, file, "a.pre_encode.out.weight")?;
    let bias = tensor(gguf, file, "a.pre_encode.out.bias")?.vec()?;
    let mut hidden = Vec::with_capacity(height);
    for row in rows {
        let mut out = projection.matvec(&row)?;
        for (value, bias) in out.iter_mut().zip(&bias) {
            *value += bias;
        }
        hidden.push(out);
    }
    Ok((hidden, height))
}

fn relative_positions(length: usize) -> Vec<Vec<f32>> {
    let count = length * 2 - 1;
    let mut positions = vec![vec![0.0f32; D_MODEL]; count];
    for (row, position) in (0..count)
        .map(|index| length as isize - 1 - index as isize)
        .enumerate()
    {
        for index in 0..D_MODEL / 2 {
            let div = (-((2 * index) as f32) * (10_000f32.ln()) / D_MODEL as f32).exp();
            let angle = position as f32 * div;
            positions[row][index * 2] = angle.sin();
            positions[row][index * 2 + 1] = angle.cos();
        }
    }
    positions
}

fn linear_with_bias(
    gguf: &Gguf,
    file: &[u8],
    weight: &str,
    bias: &str,
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let weight = tensor(gguf, file, weight)?;
    let bias = tensor(gguf, file, bias)?.vec()?;
    let mut out = weight.matvec(input)?;
    for (value, bias) in out.iter_mut().zip(&bias) {
        *value += bias;
    }
    Ok(out)
}

fn feed_forward(
    gguf: &Gguf,
    file: &[u8],
    prefix: &str,
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let up = linear_with_bias(
        gguf,
        file,
        &format!("{prefix}.ffn_up.weight"),
        &format!("{prefix}.ffn_up.bias"),
        input,
    )?;
    let activated: Vec<f32> = up.iter().map(|value| silu(*value)).collect();
    linear_with_bias(
        gguf,
        file,
        &format!("{prefix}.ffn_down.weight"),
        &format!("{prefix}.ffn_down.bias"),
        &activated,
    )
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn conformer_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &D1AudioConfig,
    layer: usize,
    hidden: Vec<Vec<f32>>,
    positions: &[Vec<f32>],
    token_count: usize,
    stop: u8,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let prefix = format!("a.blk.{layer}");
    let norm1 = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
    let norm1_bias = tensor(gguf, file, &format!("{prefix}.ffn_norm.bias"))?.vec()?;
    let mut after_ffn: Vec<Vec<f32>> = Vec::with_capacity(token_count);
    for row in &hidden {
        let normed = layer_norm(row, &norm1, &norm1_bias, config.eps);
        let ffn = feed_forward(gguf, file, &prefix, &normed)?;
        after_ffn.push(
            row.iter()
                .zip(&ffn)
                .map(|(value, ffn)| value + ffn * 0.5)
                .collect(),
        );
    }

    if stop == 1 {
        return Ok(after_ffn);
    }
    let attn_norm = tensor(gguf, file, &format!("{prefix}.ln1.weight"))?.vec()?;
    let attn_norm_bias = tensor(gguf, file, &format!("{prefix}.ln1.bias"))?.vec()?;
    let normed: Vec<Vec<f32>> = after_ffn
        .iter()
        .map(|row| layer_norm(row, &attn_norm, &attn_norm_bias, config.eps))
        .collect();
    let mut q = Vec::with_capacity(token_count);
    let mut k = Vec::with_capacity(token_count);
    let mut v = Vec::with_capacity(token_count);
    for row in &normed {
        q.push(linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.attn_q.weight"),
            &format!("{prefix}.attn_q.bias"),
            row,
        )?);
        k.push(linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.attn_k.weight"),
            &format!("{prefix}.attn_k.bias"),
            row,
        )?);
        v.push(linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.attn_v.weight"),
            &format!("{prefix}.attn_v.bias"),
            row,
        )?);
    }
    let pos_weight = tensor(gguf, file, &format!("{prefix}.linear_pos.weight"))?;
    let mut pos = Vec::with_capacity(positions.len());
    for row in positions {
        pos.push(pos_weight.matvec(row)?);
    }
    let bias_u = tensor(gguf, file, &format!("{prefix}.pos_bias_u"))?.vec()?;
    let bias_v = tensor(gguf, file, &format!("{prefix}.pos_bias_v"))?.vec()?;

    let pos_len = positions.len();
    let mut attended = vec![vec![0.0f32; D_MODEL]; token_count];
    for head in 0..N_HEADS {
        let start = head * HEAD_DIM;
        let mut content = vec![vec![0.0f32; token_count]; token_count];
        let mut relative = vec![vec![0.0f32; pos_len]; token_count];
        for frame in 0..token_count {
            for other in 0..token_count {
                let mut sum = 0.0f32;
                for dim in 0..HEAD_DIM {
                    let q_value = q[frame][start + dim] + bias_u[head * HEAD_DIM + dim];
                    sum += q_value * k[other][start + dim];
                }
                content[frame][other] = sum;
            }
            for (other, pos_row) in pos.iter().enumerate() {
                let mut sum = 0.0f32;
                for dim in 0..HEAD_DIM {
                    let q_value = q[frame][start + dim] + bias_v[head * HEAD_DIM + dim];
                    sum += q_value * pos_row[start + dim];
                }
                relative[frame][other] = sum;
            }
        }
        // Transformer-XL relative shift over the whole query block.
        let mut shifted = vec![vec![0.0f32; token_count]; token_count];
        for frame in 0..token_count {
            for pos in 0..token_count {
                let index = pos + token_count - 1 - frame;
                if index < pos_len {
                    shifted[frame][pos] = relative[frame][index];
                }
            }
        }
        for frame in 0..token_count {
            let mut scores: Vec<f32> = (0..token_count)
                .map(|other| {
                    (content[frame][other] + shifted[frame][other]) / (HEAD_DIM as f32).sqrt()
                })
                .collect();
            softmax_in_place(&mut scores);
            for (other, weight) in scores.iter().enumerate() {
                for dim in 0..HEAD_DIM {
                    attended[frame][start + dim] += weight * v[other][start + dim];
                }
            }
        }
    }

    for frame in 0..token_count {
        attended[frame] = linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.attn_out.weight"),
            &format!("{prefix}.attn_out.bias"),
            &attended[frame],
        )?;
        for (value, residual) in attended[frame].iter_mut().zip(&after_ffn[frame]) {
            *value += residual;
        }
    }

    if stop == 2 {
        return Ok(attended);
    }
    let conv_norm = tensor(gguf, file, &format!("{prefix}.norm_conv.weight"))?.vec()?;
    let conv_norm_bias = tensor(gguf, file, &format!("{prefix}.norm_conv.bias"))?.vec()?;
    let normed: Vec<Vec<f32>> = attended
        .iter()
        .map(|row| layer_norm(row, &conv_norm, &conv_norm_bias, config.eps))
        .collect();
    let conv = conv_module(gguf, file, &prefix, &normed)?;
    let after_conv: Vec<Vec<f32>> = attended
        .iter()
        .zip(&conv)
        .map(|(row, conv)| {
            row.iter()
                .zip(conv)
                .map(|(value, conv)| value + conv)
                .collect()
        })
        .collect();

    if stop == 3 {
        return Ok(after_conv);
    }
    let norm2 = tensor(gguf, file, &format!("{prefix}.ffn_norm_1.weight"))?.vec()?;
    let norm2_bias = tensor(gguf, file, &format!("{prefix}.ffn_norm_1.bias"))?.vec()?;
    let mut after_ffn2: Vec<Vec<f32>> = Vec::with_capacity(token_count);
    for row in &after_conv {
        let normed = layer_norm(row, &norm2, &norm2_bias, config.eps);
        let ffn = feed_forward_1(gguf, file, &prefix, &normed)?;
        after_ffn2.push(
            row.iter()
                .zip(&ffn)
                .map(|(value, ffn)| value + ffn * 0.5)
                .collect(),
        );
    }

    if stop == 4 {
        return Ok(after_ffn2);
    }
    let out_norm = tensor(gguf, file, &format!("{prefix}.ln2.weight"))?.vec()?;
    let out_norm_bias = tensor(gguf, file, &format!("{prefix}.ln2.bias"))?.vec()?;
    Ok(after_ffn2
        .iter()
        .map(|row| layer_norm(row, &out_norm, &out_norm_bias, config.eps))
        .collect())
}

fn feed_forward_1(
    gguf: &Gguf,
    file: &[u8],
    prefix: &str,
    input: &[f32],
) -> Result<Vec<f32>, ModelError> {
    let up = linear_with_bias(
        gguf,
        file,
        &format!("{prefix}.ffn_up_1.weight"),
        &format!("{prefix}.ffn_up_1.bias"),
        input,
    )?;
    let activated: Vec<f32> = up.iter().map(|value| silu(*value)).collect();
    linear_with_bias(
        gguf,
        file,
        &format!("{prefix}.ffn_down_1.weight"),
        &format!("{prefix}.ffn_down_1.bias"),
        &activated,
    )
}

fn conv_module(
    gguf: &Gguf,
    file: &[u8],
    prefix: &str,
    input: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let mut gated = Vec::with_capacity(input.len());
    for row in input {
        let pw1 = linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.conv_pw1.weight"),
            &format!("{prefix}.conv_pw1.bias"),
            row,
        )?;
        let half = pw1.len() / 2;
        gated.push(
            (0..half)
                .map(|index| pw1[index] * (1.0 / (1.0 + (-pw1[half + index]).exp())))
                .collect::<Vec<f32>>(),
        );
    }
    let dw = tensor(gguf, file, &format!("{prefix}.conv_dw.weight"))?.vec()?;
    let dw_bias = tensor(gguf, file, &format!("{prefix}.conv_dw.bias"))?.vec()?;
    let conv_norm = tensor(gguf, file, &format!("{prefix}.conv_norm.weight"))?.vec()?;
    let conv_norm_bias = tensor(gguf, file, &format!("{prefix}.conv_norm.bias"))?.vec()?;
    let mut out = Vec::with_capacity(input.len());
    for frame in 0..input.len() {
        let mut conv = vec![0.0f32; D_MODEL];
        for c in 0..D_MODEL {
            let mut sum = dw_bias[c];
            for k in 0..CONV_KERNEL {
                let source = frame as isize + k as isize - (CONV_KERNEL as isize / 2);
                if source < 0 || source >= input.len() as isize {
                    continue;
                }
                sum += gated[source as usize][c] * dw[c * CONV_KERNEL + k];
            }
            let normalized = sum * conv_norm[c] + conv_norm_bias[c];
            conv[c] = silu(normalized);
        }
        out.push(linear_with_bias(
            gguf,
            file,
            &format!("{prefix}.conv_pw2.weight"),
            &format!("{prefix}.conv_pw2.bias"),
            &conv,
        )?);
    }
    Ok(out)
}

fn adapter(gguf: &Gguf, file: &[u8], hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, ModelError> {
    let norm = tensor(gguf, file, "mm.a.mlp.0.weight")?.vec()?;
    let norm_bias = tensor(gguf, file, "mm.a.mlp.0.bias")?.vec()?;
    let mut out = Vec::with_capacity(hidden.len());
    for row in hidden {
        let normed = layer_norm(row, &norm, &norm_bias, 1e-5);
        let mut up = linear_with_bias(gguf, file, "mm.a.mlp.1.weight", "mm.a.mlp.1.bias", &normed)?;
        for value in &mut up {
            *value = gelu_erf(*value);
        }
        out.push(linear_with_bias(
            gguf,
            file,
            "mm.a.mlp.3.weight",
            "mm.a.mlp.3.bias",
            &up,
        )?);
    }
    Ok(out)
}

fn residual(gguf: &Gguf, file: &[u8], hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, ModelError> {
    let norm = tensor(gguf, file, "mm.a.mlp.4.weight")?.vec()?;
    let norm_bias = tensor(gguf, file, "mm.a.mlp.4.bias")?.vec()?;
    let mut out = Vec::with_capacity(hidden.len());
    for row in hidden {
        let normed = layer_norm(row, &norm, &norm_bias, 1e-5);
        let mut down =
            linear_with_bias(gguf, file, "mm.a.mlp.5.weight", "mm.a.mlp.5.bias", &normed)?;
        for value in &mut down {
            *value = gelu_erf(*value);
        }
        let up = linear_with_bias(gguf, file, "mm.a.mlp.6.weight", "mm.a.mlp.6.bias", &down)?;
        out.push(row.iter().zip(&up).map(|(value, up)| value + up).collect());
    }
    Ok(out)
}
