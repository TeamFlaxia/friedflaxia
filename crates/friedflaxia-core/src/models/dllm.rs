//! Masked-diffusion denoising step.
//!
//! A dLLM denoises a whole canvas at once. The forward pass here is a
//! Qwen3-style decoder evaluated with a block-diffusion attention mask:
//!
//! - queries inside the active block `[start, end)` attend to the clean prefix
//!   and bidirectionally to every token in the block,
//! - queries outside the block fall back to causal attention. Their outputs are
//!   not scored; the fallback only keeps the batched forward well-defined.
//!
//! The host owns sampling, Self-Repulsion and distribution merging. This module
//! returns per-position logits (or top-k candidates) and the max-softmax
//! confidence the host uses for its commit rule.

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::ops::{rms_norm, rms_norm_batch, rope_neox_inplace, silu};
use crate::tensor::{TensorView, tensor};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

/// Architecture string used by Theia's pinned dLLM GGUFs.
pub const DLLM_ARCHITECTURE: &str = "qwen3-dllm";

#[derive(Debug, Clone)]
pub struct DllmConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_freq_base: f32,
    pub vocab_size: usize,
}

impl DllmConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture() != DLLM_ARCHITECTURE {
            return Err(ModelError::Invalid("expected qwen3-dllm architecture"));
        }
        let n_layer = required_u32(&config, "block_count")? as usize;
        let n_embd = required_u32(&config, "embedding_length")? as usize;
        let n_ff = required_u32(&config, "feed_forward_length")? as usize;
        let n_head = required_u32(&config, "attention.head_count")? as usize;
        let n_head_kv = config
            .u32("attention.head_count_kv")?
            .unwrap_or(n_head as u32) as usize;
        let head_dim = match config.u32("attention.key_length")? {
            Some(value) => value as usize,
            None => {
                // Qwen3 has head_dim != n_embd / n_head, so a missing key length
                // must not silently truncate the head.
                if n_head == 0 || n_embd % n_head != 0 {
                    return Err(ModelError::Shape(
                        "qwen3-dllm needs attention.key_length when n_embd is not divisible by head_count"
                            .into(),
                    ));
                }
                n_embd / n_head
            }
        };
        let eps = config
            .f32("attention.layer_norm_rms_epsilon")?
            .unwrap_or(1e-6);
        // Qwen3 uses a 1e6 base by default; a converted dLLM GGUF overrides it.
        let rope_freq_base = config.f32("rope.freq_base")?.unwrap_or(1_000_000.0);

        if n_layer == 0 || n_embd == 0 || n_ff == 0 {
            return Err(ModelError::Shape("dllm layer/embedding sizes".into()));
        }
        if n_head == 0 || n_head_kv == 0 || head_dim == 0 || n_head % n_head_kv != 0 {
            return Err(ModelError::Shape("dllm head configuration".into()));
        }
        if head_dim % 2 != 0 {
            // RoPE rotates pairs; an odd head_dim leaves a lane un-rotated.
            return Err(ModelError::Shape("dllm head_dim must be even".into()));
        }
        let token_embd = gguf
            .tensor("token_embd.weight")
            .ok_or_else(|| ModelError::Shape("missing token_embd.weight".into()))?;
        if token_embd.dims.len() != 2 {
            return Err(ModelError::Shape("token_embd.weight rank".into()));
        }
        let vocab_size = usize::try_from(token_embd.dims[1])
            .map_err(|_| ModelError::Shape("dllm vocab size".into()))?;
        if usize::try_from(token_embd.dims[0]).ok() != Some(n_embd) {
            return Err(ModelError::Shape("token_embd.weight columns".into()));
        }
        if vocab_size == 0 {
            return Err(ModelError::Shape("dllm vocab size".into()));
        }

        Ok(Self {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            eps,
            rope_freq_base,
            vocab_size,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
}

fn required_u32(config: &ModelConfig<'_>, key: &str) -> Result<u32, ModelError> {
    config
        .u32(key)?
        .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
}

/// Full-vocabulary logits for the requested positions.
#[derive(Debug, Clone)]
pub struct DenoiseStepOutput {
    /// Row-major `[positions.len()][vocab_size]`.
    pub logits: Vec<f32>,
    pub vocab_size: usize,
    pub positions: Vec<usize>,
    /// Max softmax probability per position.
    pub confidence: Vec<f32>,
}

/// Sparse candidates for the requested positions, suitable for delta merging.
#[derive(Debug, Clone)]
pub struct DenoiseTopKOutput {
    pub positions: Vec<usize>,
    /// Row-major `[positions.len()][top_k]`.
    pub token_ids: Vec<u32>,
    /// Softmax probabilities aligned with `token_ids`.
    pub probabilities: Vec<f32>,
    /// Max softmax probability per position (equal to the first probability).
    pub confidence: Vec<f32>,
    pub vocab_size: usize,
    /// Effective k after clamping to the vocabulary size.
    pub top_k: usize,
}

fn validate_input(
    config: &DllmConfig,
    tokens: &[u32],
    positions: &[usize],
    block: (usize, usize),
) -> Result<(), ModelError> {
    if tokens.is_empty() {
        return Err(ModelError::Invalid("tokens must not be empty"));
    }
    if positions.is_empty() {
        return Err(ModelError::Invalid("positions must not be empty"));
    }
    if block.0 >= block.1 || block.1 > tokens.len() {
        return Err(ModelError::Invalid(
            "block must be a non-empty range inside tokens",
        ));
    }
    let mut seen = HashSet::with_capacity(positions.len());
    for position in positions {
        if *position >= block.1 {
            return Err(ModelError::Invalid(
                "positions must be before the block end (the tail is not computed)",
            ));
        }
        if !seen.insert(*position) {
            return Err(ModelError::Invalid("positions must be unique"));
        }
    }
    if tokens.iter().any(|id| *id as usize >= config.vocab_size) {
        return Err(ModelError::Invalid("token id is outside the vocabulary"));
    }
    Ok(())
}

/// One denoising step over `tokens`, returning full logits for `positions`.
pub fn denoise_step(
    gguf: &Gguf,
    file: &[u8],
    tokens: &[u32],
    positions: &[usize],
    block: (usize, usize),
) -> Result<DenoiseStepOutput, ModelError> {
    let config = DllmConfig::from_gguf(gguf)?;
    validate_input(&config, tokens, positions, block)?;
    // Tokens at or after block.1 cannot influence a query before block.1 under
    // the block mask, so the backbone only needs the prefix plus the block.
    let visible = &tokens[..block.1];
    let hidden = forward_hidden(gguf, file, &config, visible, block)?;
    let logits = projected_logits(gguf, file, &config, &hidden, positions)?;
    let confidence = logits
        .chunks_exact(config.vocab_size)
        .map(|row| {
            let (_, sum) = softmax_stats(row);
            if sum > 0.0 { 1.0 / sum } else { 0.0 }
        })
        .collect();

    Ok(DenoiseStepOutput {
        logits,
        vocab_size: config.vocab_size,
        positions: positions.to_vec(),
        confidence,
    })
}

/// One denoising step returning the top-`k` candidates per position.
///
/// This is the shape the Theia workload needs: a 4B model's vocabulary is large
/// enough that shipping full logits for every position wastes memory and
/// bandwidth, while the host only merges a handful of candidates.
pub fn denoise_topk(
    gguf: &Gguf,
    file: &[u8],
    tokens: &[u32],
    positions: &[usize],
    block: (usize, usize),
    top_k: usize,
) -> Result<DenoiseTopKOutput, ModelError> {
    if top_k == 0 {
        return Err(ModelError::Invalid("top_k must be at least 1"));
    }
    let config = DllmConfig::from_gguf(gguf)?;
    validate_input(&config, tokens, positions, block)?;
    let visible = &tokens[..block.1];
    let hidden = forward_hidden(gguf, file, &config, visible, block)?;
    // Keep all requested positions in one output projection. The output matrix
    // is the largest tensor in a 4B model, so chunking by position would stream
    // it p/chunk times; p is bounded by the caller's block size.
    let logits = projected_logits(gguf, file, &config, &hidden, positions)?;
    let vocab_size = config.vocab_size;
    let k = top_k.min(vocab_size);
    let mut token_ids = Vec::with_capacity(positions.len() * k);
    let mut probabilities = Vec::with_capacity(positions.len() * k);
    let mut confidence = Vec::with_capacity(positions.len());

    for row in logits.chunks_exact(vocab_size) {
        let (max, sum) = softmax_stats(row);
        confidence.push(if sum > 0.0 { 1.0 / sum } else { 0.0 });
        for index in top_k_indices(row, k) {
            token_ids.push(index as u32);
            probabilities.push(if sum > 0.0 {
                (score(row, index) - max).exp() / sum
            } else {
                0.0
            });
        }
    }

    Ok(DenoiseTopKOutput {
        positions: positions.to_vec(),
        token_ids,
        probabilities,
        confidence,
        vocab_size,
        top_k: k,
    })
}

fn projected_logits(
    gguf: &Gguf,
    file: &[u8],
    config: &DllmConfig,
    hidden: &[f32],
    positions: &[usize],
) -> Result<Vec<f32>, ModelError> {
    let width = config.n_embd;
    let output_norm = tensor(gguf, file, "output_norm.weight")?.vec()?;
    if output_norm.len() != width {
        return Err(ModelError::Shape("output_norm length".into()));
    }
    let mut selected = Vec::with_capacity(positions.len() * width);
    for position in positions {
        let start = position * width;
        selected.extend_from_slice(&hidden[start..start + width]);
    }
    let normed = rms_norm_batch(
        &selected,
        positions.len(),
        width,
        Some(&output_norm),
        config.eps,
    );
    let output = output_tensor(gguf, file, config)?;
    let logits = output.matmul(&normed, positions.len())?;
    if logits.len() != positions.len() * config.vocab_size {
        return Err(ModelError::Shape("dllm output projection".into()));
    }
    Ok(logits)
}

fn output_tensor<'a>(
    gguf: &'a Gguf,
    file: &'a [u8],
    config: &DllmConfig,
) -> Result<TensorView<'a>, ModelError> {
    let name = if gguf.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        // Tied embeddings are common for small checkpoints.
        "token_embd.weight"
    };
    let output = tensor(gguf, file, name)?;
    if output.cols() != config.n_embd || output.rows() != config.vocab_size {
        return Err(ModelError::Shape("output projection shape".into()));
    }
    Ok(output)
}

fn forward_hidden(
    gguf: &Gguf,
    file: &[u8],
    config: &DllmConfig,
    tokens: &[u32],
    block: (usize, usize),
) -> Result<Vec<f32>, ModelError> {
    let width = config.n_embd;
    let n = tokens.len();
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let token_embd = tensor(gguf, file, "token_embd.weight")?;
    if token_embd.cols() != width {
        return Err(ModelError::Shape("token_embd columns".into()));
    }
    let mut hidden = Vec::with_capacity(n * width);
    for token in tokens {
        hidden.extend_from_slice(&token_embd.row_vec(*token as usize)?);
    }
    let mut concat = vec![0.0f32; n * q_dim];
    let mut activated = vec![0.0f32; n * config.n_ff];

    for layer in 0..config.n_layer {
        let prefix = format!("blk.{layer}");
        let attn_norm = tensor(gguf, file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
        let wq = tensor(gguf, file, &format!("{prefix}.attn_q.weight"))?;
        let wk = tensor(gguf, file, &format!("{prefix}.attn_k.weight"))?;
        let wv = tensor(gguf, file, &format!("{prefix}.attn_v.weight"))?;
        let wo = tensor(gguf, file, &format!("{prefix}.attn_output.weight"))?;
        let q_norm = tensor(gguf, file, &format!("{prefix}.attn_q_norm.weight"))?.vec()?;
        let k_norm = tensor(gguf, file, &format!("{prefix}.attn_k_norm.weight"))?.vec()?;
        let ffn_norm = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
        let ffn_gate = tensor(gguf, file, &format!("{prefix}.ffn_gate.weight"))?;
        let ffn_up = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
        let ffn_down = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;

        if attn_norm.len() != width || ffn_norm.len() != width {
            return Err(ModelError::Shape(format!("{prefix} norm length")));
        }
        if q_norm.len() != config.head_dim || k_norm.len() != config.head_dim {
            return Err(ModelError::Shape(format!("{prefix} q/k norm length")));
        }
        if wq.rows() != q_dim
            || wq.cols() != width
            || wk.rows() != kv_dim
            || wk.cols() != width
            || wv.rows() != kv_dim
            || wv.cols() != width
            || wo.rows() != width
            || wo.cols() != q_dim
        {
            return Err(ModelError::Shape(format!("{prefix} attention shape")));
        }
        if ffn_gate.rows() != config.n_ff
            || ffn_gate.cols() != width
            || ffn_up.rows() != config.n_ff
            || ffn_up.cols() != width
            || ffn_down.rows() != width
            || ffn_down.cols() != config.n_ff
        {
            return Err(ModelError::Shape(format!("{prefix} ffn shape")));
        }

        let normed = rms_norm_batch(&hidden, n, width, Some(&attn_norm), config.eps);
        let mut q = wq.matmul(&normed, n)?;
        let mut k = wk.matmul(&normed, n)?;
        let v = wv.matmul(&normed, n)?;
        if q.len() != n * q_dim || k.len() != n * kv_dim || v.len() != n * kv_dim {
            return Err(ModelError::Shape(format!("{prefix} projection length")));
        }

        for position in 0..n {
            let q_row = &mut q[position * q_dim..(position + 1) * q_dim];
            for head in 0..config.n_head {
                let start = head * config.head_dim;
                let value = rms_norm(
                    &q_row[start..start + config.head_dim],
                    Some(&q_norm),
                    config.eps,
                );
                q_row[start..start + config.head_dim].copy_from_slice(&value);
            }
            rope_neox_inplace(
                q_row,
                config.n_head,
                config.head_dim,
                position,
                config.rope_freq_base,
            );

            let k_row = &mut k[position * kv_dim..(position + 1) * kv_dim];
            for head in 0..config.n_head_kv {
                let start = head * config.head_dim;
                let value = rms_norm(
                    &k_row[start..start + config.head_dim],
                    Some(&k_norm),
                    config.eps,
                );
                k_row[start..start + config.head_dim].copy_from_slice(&value);
            }
            rope_neox_inplace(
                k_row,
                config.n_head_kv,
                config.head_dim,
                position,
                config.rope_freq_base,
            );
        }

        let scale = 1.0 / (config.head_dim as f32).sqrt();
        concat.fill(0.0);
        let mut scores: Vec<f32> = Vec::new();
        for query in 0..n {
            // The active block is bidirectional; everything else is causal.
            let key_end = if query >= block.0 && query < block.1 {
                block.1
            } else {
                query + 1
            };
            for head in 0..config.n_head {
                let kv_head = head / (config.n_head / config.n_head_kv);
                let q_head = &q[query * q_dim + head * config.head_dim
                    ..query * q_dim + (head + 1) * config.head_dim];
                scores.clear();
                scores.extend((0..key_end).map(|other| {
                    let key = &k[other * kv_dim + kv_head * config.head_dim
                        ..other * kv_dim + (kv_head + 1) * config.head_dim];
                    crate::simd::dot(q_head, key) * scale
                }));
                crate::ops::softmax_in_place(&mut scores);
                let out = &mut concat[query * q_dim + head * config.head_dim
                    ..query * q_dim + (head + 1) * config.head_dim];
                for (weight, other) in scores.iter().zip(0..key_end) {
                    let value = &v[other * kv_dim + kv_head * config.head_dim
                        ..other * kv_dim + (kv_head + 1) * config.head_dim];
                    for (index, value) in value.iter().enumerate() {
                        out[index] += weight * value;
                    }
                }
            }
        }

        let projected = wo.matmul(&concat, n)?;
        if projected.len() != hidden.len() {
            return Err(ModelError::Shape(format!("{prefix} attention output")));
        }
        for (value, add) in hidden.iter_mut().zip(&projected) {
            *value += add;
        }

        let ffn_in = rms_norm_batch(&hidden, n, width, Some(&ffn_norm), config.eps);
        let gate = ffn_gate.matmul(&ffn_in, n)?;
        let up = ffn_up.matmul(&ffn_in, n)?;
        for (index, (gate, up)) in gate.iter().zip(&up).enumerate() {
            activated[index] = silu(*gate) * up;
        }
        let down = ffn_down.matmul(&activated, n)?;
        if down.len() != hidden.len() {
            return Err(ModelError::Shape(format!("{prefix} ffn output")));
        }
        for (value, add) in hidden.iter_mut().zip(&down) {
            *value += add;
        }
    }

    Ok(hidden)
}

/// Max logit and the softmax denominator. NaN logits are ignored; a row with
/// no finite value reports a zero denominator (and therefore zero confidence).
fn softmax_stats(row: &[f32]) -> (f32, f32) {
    let mut max = f32::NEG_INFINITY;
    for value in row {
        if value.is_finite() && *value > max {
            max = *value;
        }
    }
    if !max.is_finite() {
        return (f32::NEG_INFINITY, 0.0);
    }
    let mut sum = 0.0f32;
    for value in row {
        if value.is_finite() {
            sum += (value - max).exp();
        }
    }
    (max, sum)
}

/// A NaN logit must never outrank a real candidate.
fn score(row: &[f32], index: usize) -> f32 {
    let value = row[index];
    if value.is_nan() {
        f32::NEG_INFINITY
    } else {
        value
    }
}

/// Indices of the `k` largest values, descending. Ties break by index.
fn top_k_indices(row: &[f32], k: usize) -> Vec<usize> {
    if k >= row.len() {
        let mut indices: Vec<usize> = (0..row.len()).collect();
        indices.sort_by(|a, b| compare_scores(row, *a, *b));
        return indices;
    }
    // Reverse turns BinaryHeap into a min-heap over quality: the worst of the
    // k kept candidates sits at the top and is the one we evict.
    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k);
    for index in 0..row.len() {
        let entry = HeapEntry {
            score: score(row, index),
            index,
        };
        if heap.len() < k {
            heap.push(Reverse(entry));
        } else if let Some(Reverse(worst)) = heap.peek() {
            if entry > *worst {
                heap.pop();
                heap.push(Reverse(entry));
            }
        }
    }
    let mut indices: Vec<usize> = heap.into_iter().map(|Reverse(entry)| entry.index).collect();
    indices.sort_by(|a, b| compare_scores(row, *a, *b));
    indices
}

fn compare_scores(row: &[f32], a: usize, b: usize) -> Ordering {
    score(row, b)
        .partial_cmp(&score(row, a))
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.cmp(&b))
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    score: f32,
    index: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            // Higher quality is "greater"; ties prefer the earlier index.
            .then_with(|| other.index.cmp(&self.index))
    }
}

#[cfg(test)]
mod tests {
    use super::{softmax_stats, top_k_indices};

    #[test]
    fn top_k_orders_by_score_then_index() {
        let row = [0.1, 0.9, 0.5, 0.9, 0.2];
        assert_eq!(top_k_indices(&row, 2), vec![1, 3]);
        assert_eq!(top_k_indices(&row, 5), vec![1, 3, 2, 4, 0]);
        assert_eq!(top_k_indices(&row, 9), vec![1, 3, 2, 4, 0]);
    }

    #[test]
    fn nan_scores_never_outrank_real_candidates() {
        let row = [f32::NAN, 0.1, f32::NAN, 0.2];
        assert_eq!(top_k_indices(&row, 1), vec![3]);
        assert_eq!(top_k_indices(&row, 4)[..2], [3, 1]);
    }

    #[test]
    fn confidence_is_the_max_softmax_probability() {
        let (_, sum) = softmax_stats(&[0.0, 0.0, 0.0, 0.0]);
        assert!((1.0 / sum - 0.25).abs() < 1e-6);
        let (_, sum) = softmax_stats(&[0.0, 10.0]);
        assert!(1.0 / sum > 0.999);
        assert_eq!(softmax_stats(&[]).1, 0.0);
        assert_eq!(softmax_stats(&[f32::NEG_INFINITY]).1, 0.0);
        assert_eq!(softmax_stats(&[f32::NAN]).1, 0.0);
    }
}
