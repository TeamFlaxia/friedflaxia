//! Offline LoRA training CLI. Never touches the inference hot path.

use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::train::es::{EsConfig, EsSample, es_step};
use friedflaxia_core::train::lora_grad::{cross_entropy_grad, grpo_grad};
use friedflaxia_core::train::lora_weights::{LoraPairSlot, LoraWeights};
use friedflaxia_core::train::optimizer::AdamW;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliError {
    InvalidSamples,
    InvalidAlpha,
    InvalidAdapter,
    InvalidLearningRate,
    InvalidTarget,
    ShapeMismatch,
    /// An `inputs` key did not name an A/B pair present in the adapter.
    UnknownPair,
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

/// Inputs for one named A/B pair. `dout` is optional: when absent it is
/// derived from the example's `logits` (and `target` / `sampled`).
#[derive(Debug, Clone, PartialEq)]
pub struct PairInput {
    /// Frozen base activation `[in_dim]`; always required.
    pub x: Vec<f32>,
    /// Upstream gradient `[out_dim]`, if the caller already has it.
    pub dout: Option<Vec<f32>>,
}

/// One cached-base training example.
///
/// Two JSON shapes are accepted (but not both at once):
///
/// - legacy `"x": [...]`: the same activation is applied to every A/B pair;
/// - `"inputs": {"blk.<layer>.<module>": {"x": [...], "dout": [...]}}`:
///   each named pair gets its own activation and optional upstream gradient.
///   Pairs absent from `inputs` are not updated.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainExample {
    /// Legacy shared activation; empty when `inputs` is used.
    pub x: Vec<f32>,
    /// Per-pair inputs keyed by `blk.<layer>.<module>`; empty in legacy mode.
    pub inputs: BTreeMap<String, PairInput>,
    pub logits: Vec<f32>,
    pub target: usize,
    pub sampled: usize,
    pub advantage: f32,
}

#[derive(Debug, Deserialize)]
struct PairInputJson {
    x: Option<Vec<f32>>,
    dout: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct TrainExampleJson {
    #[serde(default)]
    x: Option<Vec<f32>>,
    #[serde(default)]
    inputs: Option<BTreeMap<String, PairInputJson>>,
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
            if example.x.is_some() == example.inputs.is_some() {
                // Exactly one of legacy `x` and named `inputs` is allowed.
                return Err(CliError::InvalidSamples);
            }
            if example.logits.is_empty() || !example.logits.iter().all(|v| v.is_finite()) {
                return Err(CliError::InvalidSamples);
            }
            let (x, inputs) = match (example.x, example.inputs) {
                (Some(x), None) => {
                    if x.is_empty() || !x.iter().all(|v| v.is_finite()) {
                        return Err(CliError::InvalidSamples);
                    }
                    (x, BTreeMap::new())
                }
                (None, Some(entries)) => {
                    if entries.is_empty() {
                        return Err(CliError::InvalidSamples);
                    }
                    let mut inputs = BTreeMap::new();
                    for (name, input) in entries {
                        let Some(x) = input.x else {
                            return Err(CliError::InvalidSamples);
                        };
                        if x.is_empty() || !x.iter().all(|v| v.is_finite()) {
                            return Err(CliError::InvalidSamples);
                        }
                        if let Some(dout) = &input.dout {
                            if dout.is_empty() || !dout.iter().all(|v| v.is_finite()) {
                                return Err(CliError::InvalidSamples);
                            }
                        }
                        inputs.insert(
                            name,
                            PairInput {
                                x,
                                dout: input.dout,
                            },
                        );
                    }
                    (Vec::new(), inputs)
                }
                _ => return Err(CliError::InvalidSamples),
            };
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
                x,
                inputs,
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
    if example.logits.is_empty() || !example.logits.iter().all(|v| v.is_finite()) {
        return Err(CliError::InvalidSamples);
    }
    if example.inputs.is_empty() {
        if example.x.is_empty() || !example.x.iter().all(|v| v.is_finite()) {
            return Err(CliError::InvalidSamples);
        }
    } else {
        if !example.x.is_empty() {
            return Err(CliError::InvalidSamples);
        }
        for input in example.inputs.values() {
            if input.x.is_empty()
                || !input.x.iter().all(|v| v.is_finite())
                || input
                    .dout
                    .as_ref()
                    .is_some_and(|dout| dout.is_empty() || !dout.iter().all(|v| v.is_finite()))
            {
                return Err(CliError::InvalidSamples);
            }
        }
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
    // Named pairs come straight from the adapter iterator: no base forward,
    // no assumption that every pair shares the same in/out dimensions.
    let slots: Vec<LoraPairSlot> = weights
        .iter_pairs()
        .collect::<Result<_, _>>()
        .map_err(|_| CliError::InvalidAdapter)?;
    if slots.is_empty() {
        return Err(CliError::InvalidAdapter);
    }
    let names: BTreeSet<_> = slots.iter().map(LoraPairSlot::name).collect();
    for example in examples {
        if example.inputs.keys().any(|name| !names.contains(name)) {
            return Err(CliError::UnknownPair);
        }
    }

    let mut flat = weights.flatten();
    let mut grads = vec![0.0_f32; flat.len()];

    for example in examples {
        let derived_dout = match mode {
            TrainMode::Sft => cross_entropy_grad(&example.logits, example.target),
            TrainMode::Grpo => grpo_grad(&example.logits, example.sampled, example.advantage),
        };
        for slot in &slots {
            // Named inputs select exactly the pairs to update; legacy shared
            // `x` still applies to all pairs. Unnamed pairs keep zero grads.
            let (x, dout): (&[f32], &[f32]) = if example.inputs.is_empty() {
                (&example.x, &derived_dout)
            } else {
                let Some(input) = example.inputs.get(&slot.name()) else {
                    continue;
                };
                (&input.x, input.dout.as_deref().unwrap_or(&derived_dout))
            };
            if x.len() != slot.pair.in_dim || dout.len() != slot.pair.out_dim {
                return Err(CliError::ShapeMismatch);
            }
            let local = slot
                .pair
                .backward(x, dout)
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
    use friedflaxia_core::models::lora::LoraModule;
    use friedflaxia_core::train::lora_weights::Side;

    /// Same rank, but attn_q is [in=2, out=2] and ffn_down is [in=4, out=3].
    /// A single shared `x` / `dout` cannot drive both of them.
    fn heterogeneous_adapter_bytes() -> Vec<u8> {
        let mut writer = Writer::new(2, 4.0, "code").unwrap();
        writer
            .add_tensor("blk.0.attn_q.lora_a.weight", &[2, 2], &[0.1, 0.2, 0.3, 0.4])
            .unwrap();
        writer
            .add_tensor("blk.0.attn_q.lora_b.weight", &[2, 2], &[0.5, 0.6, 0.7, 0.8])
            .unwrap();
        writer
            .add_tensor(
                "blk.0.ffn_down.lora_a.weight",
                &[4, 2],
                &[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            )
            .unwrap();
        writer
            .add_tensor(
                "blk.0.ffn_down.lora_b.weight",
                &[2, 3],
                &[0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
            )
            .unwrap();
        writer.finish()
    }

    /// Per-pair `x` and `dout`; logits only carry the SFT target here.
    const SFT_INPUTS_JSON: &str = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.attn_q":{"x":[0.1,0.2],"dout":[0.5,-0.2]},"blk.0.ffn_down":{"x":[0.1,0.2,0.3,0.4],"dout":[0.5,-0.2,0.1]}},"logits":[0.5,-0.2],"target":0}]}"#;

    fn tensor(weights: &LoraWeights, module: LoraModule, side: Side) -> &[f32] {
        &weights.tensors()[&(0, module, side)]
    }

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
            inputs: BTreeMap::new(),
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
            inputs: BTreeMap::new(),
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
            inputs: BTreeMap::new(),
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

    #[test]
    fn sft_updates_both_pairs_with_distinct_named_shapes() {
        let bytes = heterogeneous_adapter_bytes();
        let before = train_weights(&bytes);
        let (lr, examples) = parse_sft(SFT_INPUTS_JSON).unwrap();
        assert_eq!(examples.len(), 1);

        let out = run_command("sft", &bytes, SFT_INPUTS_JSON).unwrap();
        let after = train_weights(&out);
        for module in [LoraModule::AttnQ, LoraModule::FfnDown] {
            for side in [Side::A, Side::B] {
                assert_ne!(
                    tensor(&after, module, side),
                    tensor(&before, module, side),
                    "{module:?}/{side:?} must be updated with its own x/dout"
                );
            }
        }
        // No ShapeMismatch from a single shared vector, and lr is still parsed.
        assert_eq!(lr, 0.001);
    }

    #[test]
    fn unspecified_pairs_are_left_unchanged() {
        let bytes = heterogeneous_adapter_bytes();
        let before = train_weights(&bytes);
        let json = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.attn_q":{"x":[0.1,0.2],"dout":[0.5,-0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        let out = run_command("sft", &bytes, json).unwrap();
        let after = train_weights(&out);

        assert_ne!(
            tensor(&after, LoraModule::AttnQ, Side::A),
            tensor(&before, LoraModule::AttnQ, Side::A)
        );
        assert_ne!(
            tensor(&after, LoraModule::AttnQ, Side::B),
            tensor(&before, LoraModule::AttnQ, Side::B)
        );
        for side in [Side::A, Side::B] {
            assert_eq!(
                tensor(&after, LoraModule::FfnDown, side),
                tensor(&before, LoraModule::FfnDown, side),
                "omitted pairs must not be updated"
            );
        }
    }

    #[test]
    fn legacy_x_json_still_works_and_matches_named_form() {
        let bytes = adapter_bytes();
        let legacy = run_command("sft", &bytes, SFT_JSON).unwrap();
        let (_, examples) = parse_sft(SFT_JSON).unwrap();
        assert_eq!(examples[0].x, vec![0.1, 0.2]);

        // Without an explicit per-pair dout the named form derives it from
        // logits/target, exactly as the legacy shared-x form does.
        let named = run_command(
            "sft",
            &bytes,
            r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.attn_q":{"x":[0.1,0.2]}},"logits":[0.5,-0.2],"target":0}]}"#,
        )
        .unwrap();
        assert_eq!(named, legacy, "legacy JSON must remain backward compatible");
    }

    #[test]
    fn grpo_supports_named_inputs() {
        let bytes = heterogeneous_adapter_bytes();
        let json = r#"{"lr":0.01,"examples":[{"inputs":{"blk.0.ffn_down":{"x":[0.1,0.2,0.3,0.4],"dout":[0.5,-0.2,0.1]}},"logits":[0.5,-0.2],"sampled":0,"advantage":0.7}]}"#;
        let out = run_command("grpo", &bytes, json).unwrap();
        let before = train_weights(&bytes);
        let after = train_weights(&out);
        assert_ne!(
            tensor(&after, LoraModule::FfnDown, Side::A),
            tensor(&before, LoraModule::FfnDown, Side::A)
        );
        assert_eq!(
            tensor(&after, LoraModule::AttnQ, Side::A),
            tensor(&before, LoraModule::AttnQ, Side::A)
        );
    }

    #[test]
    fn unknown_pair_names_are_rejected() {
        let bytes = heterogeneous_adapter_bytes();
        let json = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.nope":{"x":[0.1,0.2],"dout":[0.5,-0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(run_command("sft", &bytes, json), Err(CliError::UnknownPair));
        let malformed = r#"{"lr":0.001,"examples":[{"inputs":{"attn_q":{"x":[0.1,0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(
            run_command("sft", &bytes, malformed),
            Err(CliError::UnknownPair)
        );
    }

    #[test]
    fn invalid_named_input_lengths_are_rejected() {
        let bytes = heterogeneous_adapter_bytes();
        let bad_x = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.attn_q":{"x":[0.1,0.2,0.3],"dout":[0.5,-0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(
            run_command("sft", &bytes, bad_x),
            Err(CliError::ShapeMismatch)
        );
        let bad_dout = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.ffn_down":{"x":[0.1,0.2,0.3,0.4],"dout":[0.5]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(
            run_command("sft", &bytes, bad_dout),
            Err(CliError::ShapeMismatch)
        );
        // A pair without explicit dout must match the logits-derived one.
        let bad_derived = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.ffn_down":{"x":[0.1,0.2,0.3,0.4]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(
            run_command("sft", &bytes, bad_derived),
            Err(CliError::ShapeMismatch)
        );
    }

    #[test]
    fn ambiguous_or_empty_inputs_are_rejected() {
        let both = r#"{"lr":0.001,"examples":[{"x":[0.1,0.2],"inputs":{"blk.0.attn_q":{"x":[0.1,0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(parse_sft(both), Err(CliError::InvalidSamples));
        let neither = r#"{"lr":0.001,"examples":[{"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(parse_sft(neither), Err(CliError::InvalidSamples));
        let missing_x = r#"{"lr":0.001,"examples":[{"inputs":{"blk.0.attn_q":{"dout":[0.5,-0.2]}},"logits":[0.5,-0.2],"target":0}]}"#;
        assert_eq!(parse_sft(missing_x), Err(CliError::InvalidSamples));
    }
}
