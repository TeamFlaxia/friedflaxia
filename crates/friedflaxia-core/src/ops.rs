//! Scalar building blocks shared by the model forward passes.

/// RMSNorm with an optional elementwise weight.
pub fn rms_norm(input: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0 / (mean_square + eps).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let scale = weight.map_or(1.0, |weight| weight[index]);
            value * inv_rms * scale
        })
        .collect()
}

/// LayerNorm with weight and bias.
pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean = input.iter().sum::<f32>() / input.len() as f32;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / input.len() as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let scale = weight.get(index).copied().unwrap_or(1.0);
            let shift = bias.get(index).copied().unwrap_or(0.0);
            (value - mean) * inv_std * scale + shift
        })
        .collect()
}

/// Sigmoid-linear unit.
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// GELU with the exact erf formulation used by the d1 decision scorer.
pub fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value / std::f32::consts::SQRT_2))
}

/// GELU with the tanh approximation used by Gemma-family models.
pub fn gelu_tanh(value: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * value * (1.0 + (SQRT_2_OVER_PI * (value + 0.044_715 * value * value * value)).tanh())
}

/// In-place numerically stable softmax.
pub fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        let exp = (*value - max).exp();
        *value = exp;
        sum += exp;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

/// Apply NeoX-style RoPE to every head, in place.
///
/// `x` is laid out as `[head][head_dim]`. `pos` is the token position.
pub fn rope_neox_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    freq_base: f32,
) {
    if n_heads == 0 || head_dim < 2 {
        return;
    }
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for (index, (cos, sin)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        let theta = pos as f32 * freq_base.powf(-2.0 * index as f32 / head_dim as f32);
        *cos = theta.cos();
        *sin = theta.sin();
    }

    for head in 0..n_heads {
        let base = head * head_dim;
        for index in 0..half {
            let x0 = x[base + index];
            let x1 = x[base + half + index];
            x[base + index] = x0 * cos[index] - x1 * sin[index];
            x[base + half + index] = x1 * cos[index] + x0 * sin[index];
        }
    }
}

/// L2-normalize a vector, leaving an all-zero vector unchanged.
pub fn l2_normalize(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return values.to_vec();
    }
    values.iter().map(|value| value / norm).collect()
}

/// Mean-pool rows into one vector.
pub fn mean_pool(rows: &[Vec<f32>]) -> Vec<f32> {
    if rows.is_empty() {
        return Vec::new();
    }
    let width = rows[0].len();
    let mut out = vec![0.0f32; width];
    for row in rows {
        for (index, value) in row.iter().enumerate() {
            out[index] += value;
        }
    }
    let scale = 1.0 / rows.len() as f32;
    for value in &mut out {
        *value *= scale;
    }
    out
}
