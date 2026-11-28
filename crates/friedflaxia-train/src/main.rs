//! Offline LoRA training CLI. Never touches the inference hot path.

use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::train::es::{EsConfig, EsSample, es_step};
use friedflaxia_core::train::lora_grad::{LoraPair, cross_entropy_grad, grpo_grad};
use friedflaxia_core::train::lora_weights::{LoraWeights, Side};
use friedflaxia_core::train::optimizer::AdamW;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliError {
    InvalidSamples,
    InvalidAlpha,
    InvalidAdapter,
    InvalidLearningRate,
    InvalidTarget,
    ShapeMismatch,
    InvalidCommand,
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

/// Which supervised signal a round file carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainMode {
    Sft,
    Grpo,
}

/// One cached-base training example: `x` is the frozen hidden activation.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainExample {
    pub x: Vec<f32>,
    pub logits: Vec<f32>,
    pub target: usize,
    pub sampled: usize,
    pub advantage: f32,
}

#[derive(Debug, Deserialize)]
struct TrainExampleJson {
    x: Vec<f32>,
    logits: Vec<f32>,
    #[serde(default)]
    target: Option<usize>,
    #[serde(default)]
    sampled: Option<usize>,
    #[serde(default)]
    advantage: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct TrainRoundJson {
    lr: f32,
    examples: Vec<TrainExampleJson>,
}

fn parse_train_round(text: &str, mode: TrainMode) -> Result<(f32, Vec<TrainExample>), CliError> {
    let round: TrainRoundJson = serde_json::from_str(text).map_err(|_| CliError::InvalidSamples)?;
    if !round.lr.is_finite() || round.lr <= 0.0 {
        return Err(CliError::InvalidLearningRate);
    }
    if round.examples.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    let examples = round
        .examples
        .into_iter()
        .map(|example| {
            if !example.x.iter().all(|v| v.is_finite())
                || !example.logits.iter().all(|v| v.is_finite())
                || example.logits.is_empty()
            {
                return Err(CliError::InvalidSamples);
            }
            let target = match example.target {
                Some(target) => target,
                None if mode == TrainMode::Sft => return Err(CliError::InvalidSamples),
                None => 0,
            };
            let (sampled, advantage) = match mode {
                TrainMode::Sft => (0, 0.0),
                TrainMode::Grpo => (
                    example.sampled.ok_or(CliError::InvalidSamples)?,
                    example.advantage.ok_or(CliError::InvalidSamples)?,
                ),
            };
            let parsed = TrainExample {
                x: example.x,
                logits: example.logits,
                target,
                sampled,
                advantage,
            };
            validate_example(&parsed, mode)?;
            Ok(parsed)
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    Ok((round.lr, examples))
}

/// High-level parsers used by the `sft` / `grpo` subcommands.
fn parse_sft(text: &str) -> Result<(f32, Vec<TrainExample>), CliError> {
    parse_train_round(text, TrainMode::Sft)
}

fn parse_grpo(text: &str) -> Result<(f32, Vec<TrainExample>), CliError> {
    parse_train_round(text, TrainMode::Grpo)
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

fn es_update(adapter_bytes: &[u8], samples: &[EsSample], alpha: f32) -> Result<Vec<u8>, CliError> {
    if samples.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(CliError::InvalidAlpha);
    }
    let gguf = Gguf::parse(adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let weights =
        LoraWeights::from_gguf(&gguf, adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let mut flat = weights.flatten();
    es_step(&mut flat, samples, EsConfig { alpha }).map_err(|_| CliError::InvalidSamples)?;
    let updated = weights
        .try_from_flat(&flat)
        .map_err(|_| CliError::InvalidAdapter)?;
    updated.write_gguf().map_err(|_| CliError::InvalidAdapter)
}

fn validate_example(example: &TrainExample, mode: TrainMode) -> Result<(), CliError> {
    if example.logits.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    if !example.x.iter().all(|v| v.is_finite()) || !example.logits.iter().all(|v| v.is_finite()) {
        return Err(CliError::InvalidSamples);
    }
    match mode {
        TrainMode::Sft => {
            if example.target >= example.logits.len() {
                return Err(CliError::InvalidTarget);
            }
        }
        TrainMode::Grpo => {
            if example.sampled >= example.logits.len() {
                return Err(CliError::InvalidTarget);
            }
            if !example.advantage.is_finite() {
                return Err(CliError::InvalidSamples);
            }
        }
    }
    Ok(())
}

/// An A/B pair positioned in the flat parameter vector (`flatten` order).
struct PairSlot {
    pair: LoraPair,
    a_start: usize,
    b_start: usize,
}

/// Rebuild trainable `LoraPair`s from an adapter. No base forward is involved:
/// `x` comes from the round file and only A/B are updated.
fn pair_slots(weights: &LoraWeights) -> Result<Vec<PairSlot>, CliError> {
    let tensors = weights.tensors();
    let a_keys: Vec<_> = tensors
        .keys()
        .filter(|(_, _, side)| *side == Side::A)
        .map(|(layer, module, _)| (*layer, *module))
        .collect();
    let b_keys: Vec<_> = tensors
        .keys()
        .filter(|(_, _, side)| *side == Side::B)
        .map(|(layer, module, _)| (*layer, *module))
        .collect();
    if a_keys.is_empty() || a_keys != b_keys {
        return Err(CliError::InvalidAdapter);
    }

    let rank = weights.rank();
    let scale = weights.alpha() / rank as f32;
    let mut slots = Vec::with_capacity(a_keys.len());
    let mut cursor = 0usize;
    for (layer, module) in a_keys {
        let a = &tensors[&(layer, module, Side::A)];
        let b = &tensors[&(layer, module, Side::B)];
        let in_dim = a.len() / rank;
        let out_dim = b.len() / rank;
        if in_dim == 0 || out_dim == 0 || a.len() % rank != 0 || b.len() % rank != 0 {
            return Err(CliError::InvalidAdapter);
        }
        let pair = LoraPair {
            a: a.clone(),
            b: b.clone(),
            in_dim,
            out_dim,
            rank,
            scale,
        };
        pair.validate().map_err(|_| CliError::InvalidAdapter)?;
        slots.push(PairSlot {
            pair,
            a_start: cursor,
            b_start: cursor + a.len(),
        });
        cursor += a.len() + b.len();
    }
    Ok(slots)
}

/// Run one batched AdamW step over all examples and return a loadable adapter.
fn train_update(
    adapter_bytes: &[u8],
    examples: &[TrainExample],
    lr: f32,
    mode: TrainMode,
) -> Result<Vec<u8>, CliError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(CliError::InvalidLearningRate);
    }
    if examples.is_empty() {
        return Err(CliError::InvalidSamples);
    }
    for example in examples {
        validate_example(example, mode)?;
    }

    let gguf = Gguf::parse(adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let weights =
        LoraWeights::from_gguf(&gguf, adapter_bytes).map_err(|_| CliError::InvalidAdapter)?;
    let slots = pair_slots(&weights)?;
    let mut flat = weights.flatten();
    let mut grads = vec![0.0_f32; flat.len()];

    for example in examples {
        let dout = match mode {
            TrainMode::Sft => cross_entropy_grad(&example.logits, example.target),
            TrainMode::Grpo => grpo_grad(&example.logits, example.sampled, example.advantage),
        };
        for slot in &slots {
            if example.x.len() != slot.pair.in_dim || dout.len() != slot.pair.out_dim {
                return Err(CliError::ShapeMismatch);
            }
            let local = slot
                .pair
                .backward(&example.x, &dout)
                .map_err(|_| CliError::InvalidAdapter)?;
            for (dst, value) in grads[slot.a_start..slot.a_start + local.da.len()]
                .iter_mut()
                .zip(local.da)
            {
                *dst += value;
            }
            for (dst, value) in grads[slot.b_start..slot.b_start + local.db.len()]
                .iter_mut()
                .zip(local.db)
            {
                *dst += value;
            }
        }
    }

    // No decoupled decay: a zero advantage must leave every .weight untouched.
    let mut optimizer = AdamW::new(flat.len(), lr, 0.0);
    optimizer
        .step(&mut flat, &grads)
        .map_err(|_| CliError::InvalidSamples)?;
    let updated = weights
        .try_from_flat(&flat)
        .map_err(|_| CliError::InvalidAdapter)?;
    updated.write_gguf().map_err(|_| CliError::InvalidAdapter)
}

/// Shared dispatch for the CLI and its tests.
fn run_command(command: &str, adapter_bytes: &[u8], text: &str) -> Result<Vec<u8>, CliError> {
    match command {
        "es-lora" => {
            let (alpha, samples) = parse_round(text)?;
            es_update(adapter_bytes, &samples, alpha)
        }
        "sft" => {
            let (lr, examples) = parse_sft(text)?;
            train_update(adapter_bytes, &examples, lr, TrainMode::Sft)
        }
        "grpo" => {
            let (lr, examples) = parse_grpo(text)?;
            train_update(adapter_bytes, &examples, lr, TrainMode::Grpo)
        }
        _ => Err(CliError::InvalidCommand),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || !matches!(args[1].as_str(), "es-lora" | "sft" | "grpo") {
        eprintln!(
            "usage: friedflaxia-train <es-lora|sft|grpo> <adapter.gguf> <round.json> <out.gguf>"
        );
        std::process::exit(2);
    }
    let adapter = match std::fs::read(&args[2]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read adapter: {error}");
            std::process::exit(1);
        }
    };
    let round_json = match std::fs::read_to_string(&args[3]) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cannot read round file: {error}");
            std::process::exit(1);
        }
    };
    match run_command(&args[1], &adapter, &round_json) {
        Ok(bytes) => {
            if let Err(error) = std::fs::write(&args[4], bytes) {
                eprintln!("cannot write adapter: {error}");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{} failed: {error:?}", args[1]);
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
            EsSample {
                seed: 1,
                reward: 1.0,
            },
            EsSample {
                seed: 2,
                reward: -1.0,
            },
        ];
        let out = es_update(&bytes, &samples, 0.01).unwrap();
        let gguf = Gguf::parse(&out).unwrap();
        assert_eq!(
            gguf.metadata("general.architecture"),
            Some(&Value::String("qwen3-dllm-lora".to_string()))
        );
        assert_eq!(
            gguf.metadata("qwen3-dllm-lora.expert"),
            Some(&Value::String("code".to_string()))
        );
    }

    #[test]
    fn es_update_changes_weights_for_non_constant_rewards() {
        let bytes = adapter_bytes();
        let parsed = Gguf::parse(&bytes).unwrap();
        let before = LoraWeights::from_gguf(&parsed, &bytes).unwrap().flatten();
        let out = es_update(
            &bytes,
            &[EsSample {
                seed: 3,
                reward: 1.0,
            }],
            0.02,
        )
        .unwrap();
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

    const SFT_JSON: &str =
        r#"{"lr":0.001,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":0}]}"#;
    const GRPO_ZERO_JSON: &str = r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":0,"advantage":0.0}]}"#;

    fn train_weights(bytes: &[u8]) -> LoraWeights {
        let gguf = Gguf::parse(bytes).unwrap();
        LoraWeights::from_gguf(&gguf, bytes).unwrap()
    }

    #[test]
    fn sft_updates_adapter_weights_and_round_trips() {
        let bytes = adapter_bytes();
        let before = train_weights(&bytes);
        let (lr, examples) = parse_sft(SFT_JSON).unwrap();
        assert_eq!(lr, 0.001);
        assert_eq!(examples.len(), 1);

        let out = train_update(&bytes, &examples, lr, TrainMode::Sft).unwrap();
        let gguf = Gguf::parse(&out).unwrap();
        assert_eq!(
            gguf.metadata("general.architecture"),
            Some(&Value::String("qwen3-dllm-lora".to_string()))
        );
        let after = train_weights(&out);
        assert_ne!(after, before, "cross-entropy gradients must update .weight");
        assert_eq!(after.rank(), before.rank());
        assert_eq!(after.alpha(), before.alpha());
        assert_eq!(after.expert(), before.expert());
    }

    #[test]
    fn grpo_zero_advantage_leaves_weights_unchanged() {
        let bytes = adapter_bytes();
        let before = train_weights(&bytes);
        let (lr, examples) = parse_grpo(GRPO_ZERO_JSON).unwrap();
        let out = train_update(&bytes, &examples, lr, TrainMode::Grpo).unwrap();
        assert_eq!(train_weights(&out), before);
        assert_eq!(
            out, bytes,
            "unchanged weights must re-serialize identically"
        );
    }

    #[test]
    fn grpo_nonzero_advantage_updates_weights() {
        let bytes = adapter_bytes();
        let (lr, examples) = parse_grpo(
            r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":0,"advantage":0.7}]}"#,
        )
        .unwrap();
        let out = train_update(&bytes, &examples, lr, TrainMode::Grpo).unwrap();
        assert_ne!(train_weights(&out), train_weights(&bytes));
    }

    #[test]
    fn sft_rejects_invalid_target_and_shape_mismatch() {
        assert_eq!(
            parse_sft(
                r#"{"lr":0.001,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":2}]}"#
            ),
            Err(CliError::InvalidTarget)
        );
        assert_eq!(
            parse_sft(r#"{"lr":0.001,"examples":[]}"#),
            Err(CliError::InvalidSamples)
        );
        assert_eq!(parse_sft("not json"), Err(CliError::InvalidSamples));
        assert_eq!(
            parse_sft(r#"{"lr":0.0,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":0}]}"#),
            Err(CliError::InvalidLearningRate)
        );

        // x はキャッシュ済み隠れ活性なので in_dim と一致しなければならない。
        let bad_x = vec![TrainExample {
            x: vec![0.1, 0.2, 0.3],
            logits: vec![0.5, -0.2],
            target: 0,
            sampled: 0,
            advantage: 0.0,
        }];
        assert_eq!(
            train_update(&adapter_bytes(), &bad_x, 0.001, TrainMode::Sft),
            Err(CliError::ShapeMismatch)
        );
        // logits の長さは B の out_dim と一致しなければならない。
        let bad_logits = vec![TrainExample {
            x: vec![0.1, 0.2],
            logits: vec![0.5, -0.2, 0.1],
            target: 0,
            sampled: 0,
            advantage: 0.0,
        }];
        assert_eq!(
            train_update(&adapter_bytes(), &bad_logits, 0.001, TrainMode::Sft),
            Err(CliError::ShapeMismatch)
        );
    }

    #[test]
    fn grpo_rejects_invalid_sampled_and_advantage() {
        assert_eq!(
            parse_grpo(
                r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":2,"advantage":0.7}]}"#
            ),
            Err(CliError::InvalidTarget)
        );
        assert_eq!(
            parse_grpo(
                r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":0}]}"#
            ),
            Err(CliError::InvalidSamples)
        );
        let bad_advantage = vec![TrainExample {
            x: vec![0.1, 0.2],
            logits: vec![0.5, -0.2],
            target: 0,
            sampled: 0,
            advantage: f32::NAN,
        }];
        assert_eq!(
            train_update(&adapter_bytes(), &bad_advantage, 0.01, TrainMode::Grpo),
            Err(CliError::InvalidSamples)
        );
    }

    #[test]
    fn training_is_reproducible() {
        let bytes = adapter_bytes();
        let (lr, examples) = parse_sft(SFT_JSON).unwrap();
        let first = train_update(&bytes, &examples, lr, TrainMode::Sft).unwrap();
        let second = train_update(&bytes, &examples, lr, TrainMode::Sft).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, bytes);
    }

    #[test]
    fn run_command_dispatches_all_subcommands() {
        let bytes = adapter_bytes();
        let sft = parse_sft(SFT_JSON).unwrap();
        assert_eq!(
            run_command("sft", &bytes, SFT_JSON).unwrap(),
            train_update(&bytes, &sft.1, sft.0, TrainMode::Sft).unwrap()
        );
        let grpo = parse_grpo(GRPO_ZERO_JSON).unwrap();
        assert_eq!(
            run_command("grpo", &bytes, GRPO_ZERO_JSON).unwrap(),
            train_update(&bytes, &grpo.1, grpo.0, TrainMode::Grpo).unwrap()
        );
        assert!(
            run_command(
                "es-lora",
                &bytes,
                r#"{"alpha":0.01,"samples":[{"seed":1,"reward":1.0}]}"#
            )
            .is_ok()
        );
        assert_eq!(
            run_command("unknown", &bytes, SFT_JSON),
            Err(CliError::InvalidCommand)
        );
    }
}
