//! Deterministic LoRA *seed* adapter factory for the Theia internal test.
//!
//! This module writes real, loadable `qwen3-dllm-lora` GGUFs through
//! [`crate::gguf::writer::Writer`], using the projection shapes of a parsed
//! `qwen3-dllm` base config. It is the production conversion path for adapter
//! artifacts; **the weights it emits are deterministic test seeds, not trained
//! experts**. They exercise the format, the loader and the attribution path,
//! and carry no quality claim. A trained adapter must replace them before any
//! quality evaluation.

use super::rng::SplitMix64;
use crate::gguf::writer::{Writer, WriterError};
use crate::models::dllm::DllmConfig;
use crate::models::lora::LoraModule;

/// Default rank/alpha of the internal-test adapters (`alpha / rank = 2`).
pub const DEFAULT_RANK: u32 = 32;
pub const DEFAULT_ALPHA: f32 = 64.0;
/// Domain-separation seed for the three internal-test experts.
pub const DEFAULT_SEED: u64 = 0x0074_6865_6961_7631; // "theiav1"

const A_SCALE: f32 = 0.02;
const B_SCALE: f32 = 0.01;

/// FNV-1a over the expert id, mixed into the caller's seed. Different experts
/// therefore get disjoint deterministic weight streams.
pub fn expert_seed(seed: u64, expert: &str) -> u64 {
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for byte in expert.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn nonzero(value: f32, fallback: f32) -> f32 {
    if value == 0.0 { fallback } else { value }
}

/// Build one complete F32 adapter: every LoRA module of every base layer.
///
/// `A` has GGML dims `[in_dim, rank]`, `B` has `[rank, out_dim]`, with
/// canonical names `blk.{layer}.{module}.lora_a/lora_b.weight`. Values are
/// small N(0, 1) draws from [`SplitMix64`] scaled by `A_SCALE` / `B_SCALE`;
/// the seed and expert id fully determine the bytes.
pub fn build_seed_adapter(
    base: &DllmConfig,
    rank: u32,
    alpha: f32,
    expert: &str,
    seed: u64,
) -> Result<Vec<u8>, WriterError> {
    let mut writer = Writer::new(rank, alpha, expert)?;
    let mut rng = SplitMix64::new(expert_seed(seed, expert));
    let rank_usize = usize::try_from(rank).map_err(|_| WriterError::Overflow)?;

    for layer in 0..base.n_layer {
        for module in LoraModule::ALL {
            let (in_dim, out_dim) = module.dims(base);
            let a_len = in_dim
                .checked_mul(rank_usize)
                .ok_or(WriterError::Overflow)?;
            let b_len = rank_usize
                .checked_mul(out_dim)
                .ok_or(WriterError::Overflow)?;
            let mut a = vec![0.0f32; a_len];
            let mut b = vec![0.0f32; b_len];
            rng.fill_normal(&mut a);
            rng.fill_normal(&mut b);
            for value in &mut a {
                *value = nonzero(*value * A_SCALE, A_SCALE);
            }
            for value in &mut b {
                *value = nonzero(*value * B_SCALE, B_SCALE);
            }

            let prefix = format!("blk.{layer}.{}.", module.suffix());
            writer.add_tensor(
                &format!("{prefix}lora_a.weight"),
                &[in_dim as u64, u64::from(rank)],
                &a,
            )?;
            writer.add_tensor(
                &format!("{prefix}lora_b.weight"),
                &[u64::from(rank), out_dim as u64],
                &b,
            )?;
        }
    }
    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_ALPHA, DEFAULT_RANK, DEFAULT_SEED, build_seed_adapter};

    fn tiny_config() -> crate::models::dllm::DllmConfig {
        crate::models::dllm::DllmConfig {
            n_layer: 1,
            n_embd: 8,
            n_ff: 16,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            eps: 1e-6,
            rope_freq_base: 10_000.0,
            vocab_size: 5,
        }
    }

    #[test]
    fn seed_adapter_defaults_are_rank_32_alpha_64() {
        assert_eq!(DEFAULT_RANK, 32);
        assert_eq!(DEFAULT_ALPHA / DEFAULT_RANK as f32, 2.0);
        assert_ne!(DEFAULT_SEED, 0);
    }

    #[test]
    fn seed_adapter_covers_every_module_of_every_layer() {
        let bytes = build_seed_adapter(&tiny_config(), 4, 8.0, "code", 1).unwrap();
        let gguf = crate::gguf::Gguf::parse(&bytes).unwrap();
        assert_eq!(gguf.tensor_count(), 14);
        assert!(gguf.tensor("blk.0.ffn_down.lora_b.weight").is_some());
    }
}
