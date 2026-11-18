//! LiquidAI d1-omni decision forward.
//!
//! The GGUF stores a 16-layer bidirectional LFM2 encoder (alternating
//! short-convolution and attention layers) plus a 2-layer decision head. The
//! head scores marker positions for one of three question types.

use super::ModelError;
use crate::gguf::{Gguf, ModelConfig};
use crate::ops::{gelu_erf, layer_norm, rms_norm, rope_neox_inplace, silu, softmax_in_place};
use crate::tensor::{TensorView, tensor};

const DECISION_TYPES: usize = 3;

#[derive(Debug, Clone)]
pub struct D1Config {
    pub n_layer: usize,
    pub n_layer_decision: usize,
    pub n_embd: usize,
    pub n_ff: Vec<usize>,
    pub n_head: usize,
    pub head_count_kv: Vec<usize>,
    pub eps_rms: f32,
    pub eps_layer: f32,
    pub rope_freq_base: f32,
    pub shortconv_cache: usize,
}

impl D1Config {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, ModelError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture() != "lfm2" {
            return Err(ModelError::Invalid("expected lfm2 architecture"));
        }
        let n_layer = required_u32(&config, "block_count")? as usize;
        let n_layer_decision = config.u32("decision.block_count")?.unwrap_or(0) as usize;
        let n_embd = required_u32(&config, "embedding_length")? as usize;
        let n_ff = config
            .i32_array("feed_forward_length")?
            .map(|values| values.into_iter().map(|value| value as usize).collect())
            .unwrap_or_else(|| {
                vec![required_u32(&config, "feed_forward_length").unwrap_or(0) as usize; n_layer]
            });
        let n_head = required_u32(&config, "attention.head_count")? as usize;
        let head_count_kv = config
            .i32_array("attention.head_count_kv")?
            .map(|values| values.into_iter().map(|value| value as usize).collect())
            .unwrap_or_else(|| vec![n_head; n_layer]);
        let eps_rms = config
            .f32("attention.layer_norm_rms_epsilon")?
            .unwrap_or(1e-5);
        let eps_layer = config.f32("attention.layer_norm_epsilon")?.unwrap_or(1e-5);
        let rope_freq_base = config.f32("rope.freq_base")?.unwrap_or(10_000.0);
        let shortconv_cache = config.u32("shortconv.l_cache")?.unwrap_or(3) as usize;
        if n_ff.len() != n_layer || head_count_kv.len() != n_layer {
            return Err(ModelError::Shape("lfm2 per-layer arrays".into()));
        }
        Ok(Self {
            n_layer,
            n_layer_decision,
            n_embd,
            n_ff,
            n_head,
            head_count_kv,
            eps_rms,
            eps_layer,
            rope_freq_base,
            shortconv_cache,
        })
    }
}

fn required_u32(config: &ModelConfig<'_>, key: &str) -> Result<u32, ModelError> {
    config
        .u32(key)?
        .ok_or_else(|| ModelError::Shape(format!("missing {key}")))
}

/// Run the d1-omni decision model over one token sequence.
///
/// Returns one raw logit per marker position. Temperature calibration and
/// softmax stay in the host (they are payload-specific).
pub fn decision_logits(
    gguf: &Gguf,
    file: &[u8],
    token_ids: &[u32],
    marker_positions: &[usize],
    qtype: usize,
) -> Result<Vec<f32>, ModelError> {
    if token_ids.is_empty() {
        return Err(ModelError::Invalid("token_ids must not be empty"));
    }
    if qtype >= DECISION_TYPES {
        return Err(ModelError::Invalid("qtype must be 0..3"));
    }
    let config = D1Config::from_gguf(gguf)?;
    let hidden = trunk_forward(gguf, file, &config, token_ids)?;
    let scored = decision_head(gguf, file, &config, hidden, marker_positions, qtype)?;
    Ok(scored)
}

fn trunk_forward(
    gguf: &Gguf,
    file: &[u8],
    config: &D1Config,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let token_embd = tensor(gguf, file, "token_embd.weight")?;
    let mut hidden: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|id| token_embd.row_vec(*id as usize))
        .collect::<Result<_, _>>()?;

    let n_layer_enc = config.n_layer - config.n_layer_decision;
    for layer in 0..n_layer_enc {
        let prefix = format!("blk.{layer}");
        let attn_norm = tensor(gguf, file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
        let residual = hidden.clone();
        let normalized: Vec<Vec<f32>> = hidden
            .iter()
            .map(|row| rms_norm(row, Some(&attn_norm), config.eps_rms))
            .collect();

        let attended = if config.head_count_kv[layer] == 0 {
            shortconv_layer(gguf, file, config, &prefix, &normalized)?
        } else {
            attention_layer(gguf, file, config, layer, &prefix, &normalized)?
        };

        hidden = residual
            .into_iter()
            .zip(attended)
            .map(|(residual, value)| add(&residual, &value))
            .collect();

        let ffn_norm = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
        let gate = tensor(gguf, file, &format!("{prefix}.ffn_gate.weight"))?;
        let up = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
        let down = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;
        for row in &mut hidden {
            let normed = rms_norm(row, Some(&ffn_norm), config.eps_rms);
            let gate = gate.matvec(&normed)?;
            let up = up.matvec(&normed)?;
            let activated: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(gate, up)| silu(*gate) * up)
                .collect();
            let ffn = down.matvec(&activated)?;
            *row = add(row, &ffn);
        }
    }

    let output_norm = tensor(gguf, file, "token_embd_norm.weight")?.vec()?;
    Ok(hidden
        .iter()
        .map(|row| rms_norm(row, Some(&output_norm), config.eps_rms))
        .collect())
}

fn shortconv_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &D1Config,
    prefix: &str,
    normalized: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let in_proj = tensor(gguf, file, &format!("{prefix}.shortconv.in_proj.weight"))?;
    let out_proj = tensor(gguf, file, &format!("{prefix}.shortconv.out_proj.weight"))?;
    let conv = tensor(gguf, file, &format!("{prefix}.shortconv.conv.weight"))?.vec()?;
    if config.shortconv_cache != 3 {
        return Err(ModelError::Invalid("only a 3-tap shortconv is supported"));
    }
    let n_embd = config.n_embd;
    if conv.len() != 3 * n_embd {
        return Err(ModelError::Shape("shortconv taps".into()));
    }

    let mut b = Vec::with_capacity(normalized.len());
    let mut c = Vec::with_capacity(normalized.len());
    let mut x = Vec::with_capacity(normalized.len());
    for row in normalized {
        let projected = in_proj.matvec(row)?;
        if projected.len() != 3 * n_embd {
            return Err(ModelError::Shape("shortconv in_proj output".into()));
        }
        b.push(projected[..n_embd].to_vec());
        c.push(projected[n_embd..2 * n_embd].to_vec());
        x.push(projected[2 * n_embd..].to_vec());
    }
    let bx: Vec<Vec<f32>> = b
        .iter()
        .zip(&x)
        .map(|(b, x)| b.iter().zip(x).map(|(b, x)| b * x).collect())
        .collect();

    let mut out = Vec::with_capacity(normalized.len());
    for token in 0..normalized.len() {
        let prev = if token > 0 {
            Some(&bx[token - 1])
        } else {
            None
        };
        let next = if token + 1 < bx.len() {
            Some(&bx[token + 1])
        } else {
            None
        };
        let mut y = vec![0.0f32; n_embd];
        for channel in 0..n_embd {
            let tap0 = conv[channel * 3];
            let tap1 = conv[channel * 3 + 1];
            let tap2 = conv[channel * 3 + 2];
            let mut value = bx[token][channel] * tap1;
            if let Some(prev) = prev {
                value += prev[channel] * tap0;
            }
            if let Some(next) = next {
                value += next[channel] * tap2;
            }
            y[channel] = value;
        }
        let gated: Vec<f32> = c[token].iter().zip(y).map(|(c, y)| c * y).collect();
        out.push(out_proj.matvec(&gated)?);
    }
    Ok(out)
}

fn attention_layer(
    gguf: &Gguf,
    file: &[u8],
    config: &D1Config,
    layer: usize,
    prefix: &str,
    normalized: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, ModelError> {
    let q_proj = tensor(gguf, file, &format!("{prefix}.attn_q.weight"))?;
    let k_proj = tensor(gguf, file, &format!("{prefix}.attn_k.weight"))?;
    let v_proj = tensor(gguf, file, &format!("{prefix}.attn_v.weight"))?;
    let out_proj = tensor(gguf, file, &format!("{prefix}.attn_output.weight"))?;
    let q_norm = tensor(gguf, file, &format!("{prefix}.attn_q_norm.weight"))?.vec()?;
    let k_norm = tensor(gguf, file, &format!("{prefix}.attn_k_norm.weight"))?.vec()?;

    let n_head = config.n_head;
    let head_dim = q_proj.rows() / n_head;
    let n_head_kv = config.head_count_kv[layer];
    if head_dim == 0 || n_head % n_head_kv != 0 {
        return Err(ModelError::Shape("d1 attention heads".into()));
    }

    let mut q_heads = Vec::with_capacity(normalized.len());
    let mut k_heads = Vec::with_capacity(normalized.len());
    let mut v_heads = Vec::with_capacity(normalized.len());
    for (position, row) in normalized.iter().enumerate() {
        let q = q_proj.matvec(row)?;
        let k = k_proj.matvec(row)?;
        let v = v_proj.matvec(row)?;
        let mut qn = vec![0.0f32; q.len()];
        for head in 0..n_head {
            let start = head * head_dim;
            let value = rms_norm(&q[start..start + head_dim], Some(&q_norm), config.eps_rms);
            qn[start..start + head_dim].copy_from_slice(&value);
        }
        rope_neox_inplace(&mut qn, n_head, head_dim, position, config.rope_freq_base);
        let mut kn = vec![0.0f32; k.len()];
        for head in 0..n_head_kv {
            let start = head * head_dim;
            let value = rms_norm(&k[start..start + head_dim], Some(&k_norm), config.eps_rms);
            kn[start..start + head_dim].copy_from_slice(&value);
        }
        rope_neox_inplace(
            &mut kn,
            n_head_kv,
            head_dim,
            position,
            config.rope_freq_base,
        );
        q_heads.push(
            (0..n_head)
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
                .map(|h| v[h * head_dim..(h + 1) * head_dim].to_vec())
                .collect::<Vec<_>>(),
        );
    }

    let mut out = Vec::with_capacity(normalized.len());
    for q_rows in &q_heads {
        let mut concat = vec![0.0f32; n_head * head_dim];
        for head in 0..n_head {
            let kv_head = head / (n_head / n_head_kv);
            let mut scores: Vec<f32> = k_heads
                .iter()
                .map(|keys| {
                    q_rows[head]
                        .iter()
                        .zip(&keys[kv_head])
                        .map(|(q, k)| q * k)
                        .sum::<f32>()
                        / (head_dim as f32).sqrt()
                })
                .collect();
            softmax_in_place(&mut scores);
            let dest = &mut concat[head * head_dim..(head + 1) * head_dim];
            for (weight, values) in scores.iter().zip(&v_heads) {
                for (index, value) in values[kv_head].iter().enumerate() {
                    dest[index] += weight * value;
                }
            }
        }
        out.push(out_proj.matvec(&concat)?);
    }
    Ok(out)
}

fn decision_head(
    gguf: &Gguf,
    file: &[u8],
    config: &D1Config,
    mut hidden: Vec<Vec<f32>>,
    marker_positions: &[usize],
    qtype: usize,
) -> Result<Vec<f32>, ModelError> {
    let type_embd = tensor(gguf, file, "token_types.weight")?.row_vec(qtype)?;
    for row in &mut hidden {
        if row.len() != type_embd.len() {
            return Err(ModelError::Shape("token_types".into()));
        }
        for (value, type_value) in row.iter_mut().zip(&type_embd) {
            *value += type_value;
        }
    }

    for layer in config.n_layer - config.n_layer_decision..config.n_layer {
        let prefix = format!("blk.{layer}");
        let attn_norm_w = tensor(gguf, file, &format!("{prefix}.attn_norm.weight"))?.vec()?;
        let attn_norm_b = tensor(gguf, file, &format!("{prefix}.attn_norm.bias"))?.vec()?;
        let qkv_w = tensor(gguf, file, &format!("{prefix}.attn_qkv.weight"))?;
        let qkv_b = tensor(gguf, file, &format!("{prefix}.attn_qkv.bias"))?.vec()?;
        let out_w = tensor(gguf, file, &format!("{prefix}.attn_output.weight"))?;
        let out_b = tensor(gguf, file, &format!("{prefix}.attn_output.bias"))?.vec()?;
        let ffn_norm_w = tensor(gguf, file, &format!("{prefix}.ffn_norm.weight"))?.vec()?;
        let ffn_norm_b = tensor(gguf, file, &format!("{prefix}.ffn_norm.bias"))?.vec()?;
        let up_w = tensor(gguf, file, &format!("{prefix}.ffn_up.weight"))?;
        let up_b = tensor(gguf, file, &format!("{prefix}.ffn_up.bias"))?.vec()?;
        let down_w = tensor(gguf, file, &format!("{prefix}.ffn_down.weight"))?;
        let down_b = tensor(gguf, file, &format!("{prefix}.ffn_down.bias"))?.vec()?;

        let n_head = config.n_head;
        let head_dim = config.n_embd / n_head;
        let n_head_kv = config.head_count_kv[layer];
        if n_head % n_head_kv != 0 {
            return Err(ModelError::Shape("d1 head attention heads".into()));
        }

        let mut residual = Vec::with_capacity(hidden.len());
        let mut attended = Vec::with_capacity(hidden.len());
        for row in &hidden {
            let normed = layer_norm(row, &attn_norm_w, &attn_norm_b, config.eps_layer);
            let mut qkv = qkv_w.matvec(&normed)?;
            for (value, bias) in qkv.iter_mut().zip(&qkv_b) {
                *value += bias;
            }
            if qkv.len() != 3 * config.n_embd {
                return Err(ModelError::Shape("d1 head qkv".into()));
            }
            let q = &qkv[..config.n_embd];
            let k = &qkv[config.n_embd..2 * config.n_embd];
            let v = &qkv[2 * config.n_embd..];
            residual.push(row.clone());
            attended.push((q.to_vec(), k.to_vec(), v.to_vec()));
        }

        let mut next = Vec::with_capacity(hidden.len());
        for token in 0..hidden.len() {
            let mut concat = vec![0.0f32; config.n_embd];
            for head in 0..n_head {
                let kv_head = head / (n_head / n_head_kv);
                let q = &attended[token].0[head * head_dim..(head + 1) * head_dim];
                let mut scores: Vec<f32> = attended
                    .iter()
                    .map(|(_, k, _)| {
                        let k = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
                        q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() / (head_dim as f32).sqrt()
                    })
                    .collect();
                softmax_in_place(&mut scores);
                let dest = &mut concat[head * head_dim..(head + 1) * head_dim];
                for (weight, (_, _, v)) in scores.iter().zip(&attended) {
                    let v = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
                    for (index, value) in v.iter().enumerate() {
                        dest[index] += weight * value;
                    }
                }
            }
            let projected = out_w.matvec(&concat)?;
            let mut cur: Vec<f32> = projected
                .iter()
                .zip(&out_b)
                .map(|(value, bias)| value + bias)
                .collect();
            let ffn_inp = add(&residual[token], &cur);
            let normed = layer_norm(&ffn_inp, &ffn_norm_w, &ffn_norm_b, config.eps_layer);
            let mut up = up_w.matvec(&normed)?;
            for (value, bias) in up.iter_mut().zip(&up_b) {
                *value += bias;
            }
            for value in &mut up {
                *value = value.max(0.0);
            }
            let mut ffn = down_w.matvec(&up)?;
            for (value, bias) in ffn.iter_mut().zip(&down_b) {
                *value += bias;
            }
            cur = add(&ffn_inp, &ffn);
            next.push(cur);
        }
        hidden = next;
    }

    let cls_norm_w = tensor(gguf, file, "cls.norm.weight")?.vec()?;
    let cls_norm_b = tensor(gguf, file, "cls.norm.bias")?.vec()?;
    let cls_w = tensor(gguf, file, "cls.weight")?;
    let cls_b = tensor(gguf, file, "cls.bias")?.vec()?;
    let cls_out_w = tensor(gguf, file, "cls.output.weight")?;
    let cls_out_b = tensor(gguf, file, "cls.output.bias")?.vec()?;

    let mut logits = Vec::with_capacity(marker_positions.len());
    for position in marker_positions {
        let row = hidden.get(*position).ok_or_else(|| {
            ModelError::Shape(format!("marker position {position} is out of range"))
        })?;
        let normed = layer_norm(row, &cls_norm_w, &cls_norm_b, config.eps_layer);
        let mut projected = cls_w.matvec(&normed)?;
        for (value, bias) in projected.iter_mut().zip(&cls_b) {
            *value += bias;
        }
        for value in &mut projected {
            *value = gelu_erf(*value);
        }
        let mut score = cls_out_w.matvec(&projected)?;
        for (value, bias) in score.iter_mut().zip(&cls_out_b) {
            *value += bias;
        }
        logits.push(score[0]);
    }
    Ok(logits)
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}

#[allow(dead_code)]
fn _tensor_type_name(view: &TensorView<'_>) -> String {
    format!("{:?}", view.info.tensor_type)
}
