//! Offline LoRA training CLI. Never touches the inference hot path.

use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::train::es::{EsConfig, EsSample, es_step};
use friedflaxia_core::train::lora_weights::LoraWeights;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliError {
    InvalidSamples,
    InvalidAlpha,
    InvalidAdapter,
}

#[derive(Debug, Deserialize)]
struct SampleJson {
    seed: u64,
    reward: f32,
}

#[derive(Debug, Deserialize)]
struct RoundJson {
    alpha: f32,
    samples: Vec<SampleJson>,
}

fn parse_round(text: &str) -> Result<(f32, Vec<EsSample>), CliError> {
    let round: RoundJson = serde_json::from_str(text).map_err(|_| CliError::InvalidSamples)?;
    if round.samples.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    let samples = round
        .samples
        .into_iter()
        .map(|sample| EsSample {
            seed: sample.seed,
            reward: sample.reward,
        })
        .collect();
    Ok((round.alpha, samples))
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
    let (alpha, samples) = match parse_round(&samples_json) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("invalid round file: {error:?}");
            std::process::exit(1);
        }
    };
    match es_update(&adapter, &samples, alpha) {
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
    fn parse_round_reads_alpha_and_samples() {
        let (alpha, samples) = parse_round(
            r#"{"alpha":0.002,"samples":[{"seed":1,"reward":0.5},{"seed":9,"reward":-0.25}]}"#,
        )
        .unwrap();
        assert_eq!(alpha, 0.002);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].seed, 1);
        assert_eq!(samples[1].reward, -0.25);
    }

    #[test]
    fn parse_round_rejects_malformed_or_empty_input() {
        assert_eq!(parse_round("not json"), Err(CliError::InvalidSamples));
        assert_eq!(
            parse_round(r#"{"alpha":0.1,"samples":[]}"#),
            Err(CliError::InvalidSamples)
        );
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
