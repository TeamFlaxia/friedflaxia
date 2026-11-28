//! Produce deterministic `qwen3-dllm-lora` GGUF adapters for Theia's internal test.
//!
//! WARNING: the weights are deterministic SplitMix64 seeds, NOT trained
//! experts. They exist to prove the artifact format and the load path. Replace
//! them with a trained adapter before making any quality claim.
//!
//! ```sh
//! # Preferred: derive all shapes from the pinned base GGUF.
//! cargo run -p friedflaxia-core --example make_lora -- \
//!   --base /path/to/qwen3-bd3lm-q4_0.gguf --expert code \
//!   --out artifacts/theia-code-f32.gguf
//!
//! # Fallback: verified pinned-base config (no base GGUF required).
//! cargo run -p friedflaxia-core --example make_lora -- \
//!   --pinned-config --expert math --out artifacts/theia-math-f32.gguf
//! ```
//!
//! `--round-out` additionally writes a deterministic SFT round JSON for
//! `blk.0.attn_q`: its `x` and `logits` are computed from the generated
//! adapter itself, and `target` is the argmax of those synthetic logits. It is
//! a mechanical training-input example, not an evaluation result.

use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::dllm::DllmConfig;
use friedflaxia_core::tensor::tensor;
use friedflaxia_core::train::lora_factory::{
    DEFAULT_ALPHA, DEFAULT_RANK, DEFAULT_SEED, build_seed_adapter, expert_seed,
};
use friedflaxia_core::train::rng::SplitMix64;
use std::env;
use std::fs;

/// Verified from the pinned base GGUF (sha256
/// `0f13278bde4d700118c8cf13d293af897a8974e74293de887aa383c567352a8e`,
/// 342059424 bytes) with `cargo run -p friedflaxia-core --example inspect`.
fn pinned_config() -> DllmConfig {
    DllmConfig {
        n_layer: 28,
        n_embd: 1024,
        n_ff: 3072,
        n_head: 16,
        n_head_kv: 8,
        head_dim: 128,
        eps: 1e-6,
        rope_freq_base: 1_000_000.0,
        vocab_size: 151_936,
    }
}

struct Args {
    base: Option<String>,
    pinned_config: bool,
    expert: String,
    out: String,
    rank: u32,
    alpha: f32,
    seed: u64,
    round_out: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut base = None;
    let mut pinned_config = false;
    let mut expert = None;
    let mut out = None;
    let mut rank = DEFAULT_RANK;
    let mut alpha = DEFAULT_ALPHA;
    let mut seed = DEFAULT_SEED;
    let mut round_out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--base" => base = Some(args.next().ok_or("--base needs a path")?),
            "--pinned-config" => pinned_config = true,
            "--expert" => expert = Some(args.next().ok_or("--expert needs an id")?),
            "--out" => out = Some(args.next().ok_or("--out needs a path")?),
            "--round-out" => round_out = Some(args.next().ok_or("--round-out needs a path")?),
            "--rank" => {
                rank = args
                    .next()
                    .ok_or("--rank needs a number")?
                    .parse()
                    .map_err(|_| "bad rank")?
            }
            "--alpha" => {
                alpha = args
                    .next()
                    .ok_or("--alpha needs a number")?
                    .parse()
                    .map_err(|_| "bad alpha")?
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or("--seed needs a number")?
                    .parse()
                    .map_err(|_| "bad seed")?
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    let expert = expert.ok_or("missing --expert")?;
    if expert.is_empty()
        || expert.len() > 51
        || !expert
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("expert id must be 1-51 lowercase letters, digits or '-'".into());
    }
    if base.is_some() == pinned_config {
        return Err("pass exactly one of --base <path> or --pinned-config".into());
    }
    Ok(Args {
        base,
        pinned_config,
        expert,
        out: out.ok_or("missing --out")?,
        rank,
        alpha,
        seed,
        round_out,
    })
}

fn base_config(args: &Args) -> Result<DllmConfig, String> {
    if args.pinned_config {
        return Ok(pinned_config());
    }
    let path = args.base.as_deref().expect("base path");
    if path.starts_with("http://") || path.starts_with("https://") {
        return Err("URLs are not fetched by this offline CLI; download the pinned GGUF with curl and pass its local path".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read base GGUF: {error}"))?;
    let gguf = Gguf::parse_with_options(
        &bytes,
        ParseOptions {
            skip_tokenizer: true,
        },
    )
    .map_err(|error| format!("cannot parse base GGUF: {error}"))?;
    DllmConfig::from_gguf(&gguf).map_err(|error| format!("not a qwen3-dllm base: {error}"))
}

fn join_floats(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// A tiny SFT round derived from the adapter's own `blk.0.attn_q` pair.
/// No base forward and no evaluation is involved: `target` is simply the
/// largest synthetic logit, so this only demonstrates the offline train path.
fn make_round(
    bytes: &[u8],
    rank: usize,
    alpha: f32,
    expert: &str,
    seed: u64,
) -> Result<String, String> {
    let gguf = Gguf::parse(bytes).map_err(|error| error.to_string())?;
    let a =
        tensor(&gguf, bytes, "blk.0.attn_q.lora_a.weight").map_err(|error| error.to_string())?;
    let b =
        tensor(&gguf, bytes, "blk.0.attn_q.lora_b.weight").map_err(|error| error.to_string())?;
    let mut x = vec![0.0f32; a.cols()];
    SplitMix64::new(expert_seed(seed, expert) ^ 0x726f_756e_6431).fill_normal(&mut x);
    for value in &mut x {
        *value *= 0.01;
    }
    let hidden = a.matmul(&x, 1).map_err(|error| error.to_string())?;
    let mut logits = b.matmul(&hidden, 1).map_err(|error| error.to_string())?;
    let scale = alpha / rank as f32;
    for value in &mut logits {
        *value *= scale;
    }
    let target = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .ok_or("empty logits")?;
    Ok(format!(
        "{{\"lr\":0.001,\"examples\":[{{\"inputs\":{{\"blk.0.attn_q\":{{\"x\":[{}]}}}},\"logits\":[{}],\"target\":{}}}]}}",
        join_floats(&x),
        join_floats(&logits),
        target
    ))
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let config = base_config(&args)?;
    let rank_usize = usize::try_from(args.rank).map_err(|_| "rank is too large")?;
    let bytes = build_seed_adapter(&config, args.rank, args.alpha, &args.expert, args.seed)
        .map_err(|error| error.to_string())?;
    eprintln!(
        "INTERNAL TEST SEED (not trained): expert={} rank={} alpha={} layers={} n_embd={} q_dim={} kv_dim={} n_ff={} tensors={} bytes={}",
        args.expert,
        args.rank,
        args.alpha,
        config.n_layer,
        config.n_embd,
        config.q_dim(),
        config.kv_dim(),
        config.n_ff,
        Gguf::parse(&bytes)
            .map_err(|error| error.to_string())?
            .tensor_count(),
        bytes.len()
    );
    fs::write(&args.out, &bytes).map_err(|error| format!("cannot write adapter: {error}"))?;
    if let Some(path) = &args.round_out {
        let round = make_round(&bytes, rank_usize, args.alpha, &args.expert, args.seed)?;
        fs::write(path, round).map_err(|error| format!("cannot write round JSON: {error}"))?;
    }
    Ok(())
}
