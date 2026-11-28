//! Offline LoRA training CLI. Never touches the inference hot path.

use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::train::es::{EsConfig, EsSample, es_step};
use friedflaxia_core::train::lora_weights::LoraWeights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliError {
    InvalidSamples,
    InvalidAlpha,
    InvalidAdapter,
}

fn es_update(
    adapter_bytes: &[u8],
    samples: &[EsSample],
    alpha: f32,
) -> Result<Vec<u8>, CliError> {
    if samples.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(CliError::InvalidAlpha);
    }
    let gguf = Gguf::parse(adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let weights = LoraWeights::from_gguf(&gguf, adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let mut flat = weights.flatten();
    es_step(&mut flat, samples, EsConfig { alpha }).map_err(|_| CliError::InvalidSamples)?;
    let updated = weights.try_from_flat(&flat).map_err(|_| CliError::InvalidAdapter)?;
    updated.write_gguf().map_err(|_| CliError::InvalidAdapter)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || args[1] != "es-lora" {
        eprintln!("usage: friedflaxia-train es-lora <adapter.gguf> <samples.json> <out.gguf>");
        std::process::exit(2);
    }
    let adapter = match std::fs::read(&args[2]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read adapter: {error}");
            std::process::exit(1);
        }
    };
    let samples_json = match std::fs::read_to_string(&args[3]) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read samples: {error}");
            std::process::exit(1);
        }
    };
    // Minimal JSON: [{"seed": 1, "reward": 0.5}, ...]
    let samples: Vec<EsSample> = samples_json
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|entry| !entry.trim().is_empty())
        .filter_map(|entry| {
            let seed = entry.split("\"seed\"").nth(1)?.split(':').nth(1)?.trim().trim_end_matches(',').parse().ok()?;
            let reward = entry.split("\"reward\"").nth(1)?.split(':').nth(1)?.trim().trim_end_matches('}').parse().ok()?;
            Some(EsSample { seed, reward })
        })
        .collect();
    if samples.is_empty() {
        eprintln!("no valid samples");
        std::process::exit(1);
    }
    // alpha is fixed by the offline recipe; it can be made a flag later.
    match es_update(&adapter, &samples, 0.001) {
        Ok(bytes) => {
            if let Err(error) = std::fs::write(&args[4], bytes) {
                eprintln!("cannot write adapter: {error}");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("es-lora failed: {error:?}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use friedflaxia_core::gguf::Value;
    use friedflaxia_core::gguf::writer::Writer;

    fn adapter_bytes() -> Vec<u8> {
        let mut writer = Writer::new(2, 64.0, "code").unwrap();
        writer
            .add_tensor("blk.0.attn_q.lora_a.weight", &[2, 2], &[0.1, 0.2, 0.3, 0.4])
            .unwrap();
        writer
            .add_tensor("blk.0.attn_q.lora_b.weight", &[2, 2], &[0.5, 0.6, 0.7, 0.8])
            .unwrap();
        writer.finish()
    }

    #[test]
    fn es_update_produces_a_loadable_adapter() {
        let bytes = adapter_bytes();
        let samples = [
            EsSample { seed: 1, reward: 1.0 },
            EsSample { seed: 2, reward: -1.0 },
        ];
        let out = es_update(&bytes, &samples, 0.01).unwrap();
        let gguf = Gguf::parse(&out).unwrap();
        assert_eq!(
            gguf.metadata("general.architecture"),
            Some(&Value::String("qwen3-dllm-lora".to_string()))
        );
        assert_eq!(gguf.metadata("qwen3-dllm-lora.expert"), Some(&Value::String("code".to_string())));
    }

    #[test]
    fn es_update_changes_weights_for_non_constant_rewards() {
        let bytes = adapter_bytes();
        let parsed = Gguf::parse(&bytes).unwrap();
        let before = LoraWeights::from_gguf(&parsed, &bytes).unwrap().flatten();
        let out = es_update(&bytes, &[EsSample { seed: 3, reward: 1.0 }], 0.02).unwrap();
        let parsed = Gguf::parse(&out).unwrap();
        let after = LoraWeights::from_gguf(&parsed, &out).unwrap().flatten();
        assert_ne!(before, after);
    }

    #[test]
    fn es_update_rejects_empty_samples() {
        assert_eq!(
            es_update(&adapter_bytes(), &[], 0.01),
            Err(CliError::InvalidSamples)
        );
    }
}
