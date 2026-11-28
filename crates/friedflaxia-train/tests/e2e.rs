//! End-to-end tests for the `friedflaxia-train` binary.
//!
//! Unlike the CLI's unit tests, these spawn the real executable via
//! `std::process::Command` and exchange GGUF / JSON files through the
//! filesystem, so argument parsing and exit codes are covered as well.

use friedflaxia_core::gguf::writer::Writer;
use friedflaxia_core::gguf::{Gguf, Value};
use friedflaxia_core::train::lora_weights::LoraWeights;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const TRAIN_BIN: &str = env!("CARGO_BIN_EXE_friedflaxia-train");
const ARCHITECTURE: &str = "qwen3-dllm-lora";
const RANK: u32 = 2;
const ALPHA: f32 = 64.0;
const EXPERT: &str = "code";

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

/// An isolated directory holding one adapter plus per-test round files.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "friedflaxia-train-e2e-{}-{label}-{id}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("remove stale fixture directory");
        }
        fs::create_dir_all(&dir).expect("create fixture directory");
        Self { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, bytes).expect("write fixture file");
        path
    }

    /// rank=2, alpha=64, expert="code" with one `[2, 2]` A/B pair.
    fn adapter(&self) -> PathBuf {
        let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("create writer");
        writer
            .add_tensor(
                "blk.0.attn_q.lora_a.weight",
                &[2, u64::from(RANK)],
                &[0.1, 0.2, 0.3, 0.4],
            )
            .expect("add lora_a");
        writer
            .add_tensor(
                "blk.0.attn_q.lora_b.weight",
                &[u64::from(RANK), 2],
                &[0.5, 0.6, 0.7, 0.8],
            )
            .expect("add lora_b");
        self.write("adapter.gguf", &writer.finish())
    }

    fn round(&self, name: &str, json: &str) -> PathBuf {
        self.write(name, json.as_bytes())
    }

    fn run(&self, subcommand: &str, round: &Path, out: &str) -> Output {
        Command::new(TRAIN_BIN)
            .arg(subcommand)
            .arg(self.path("adapter.gguf"))
            .arg(round)
            .arg(self.path(out))
            .output()
            .expect("spawn friedflaxia-train")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn read_output(output: &Output) -> String {
    format!(
        "status: {:?}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_success(output: &Output) {
    assert!(output.status.success(), "{}", read_output(output));
}

fn load(path: &Path) -> LoraWeights {
    let bytes = fs::read(path).expect("read output adapter");
    let gguf = Gguf::parse(&bytes).expect("parse output GGUF");
    LoraWeights::from_gguf(&gguf, &bytes).expect("load output LoRA weights")
}

fn weights(path: &Path) -> Vec<f32> {
    load(path).flatten()
}

/// The output must still be the same adapter family as the input.
fn assert_identity_preserved(path: &Path) {
    let bytes = fs::read(path).expect("read output adapter");
    let gguf = Gguf::parse(&bytes).expect("parse output GGUF");
    assert_eq!(
        gguf.metadata("general.architecture"),
        Some(&Value::String(ARCHITECTURE.to_owned()))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.expert"),
        Some(&Value::String(EXPERT.to_owned()))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.rank"),
        Some(&Value::U32(RANK))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.alpha"),
        Some(&Value::F32(ALPHA))
    );

    let adapter = LoraWeights::from_gguf(&gguf, &bytes).expect("load output LoRA weights");
    assert_eq!(adapter.rank(), RANK as usize);
    assert_eq!(adapter.alpha(), ALPHA);
    assert_eq!(adapter.expert(), EXPERT);
}

#[test]
fn es_changes_weights_and_preserves_adapter_identity() {
    let fixture = Fixture::new("es");
    let before = weights(&fixture.adapter());
    let round = fixture.round(
        "es.json",
        r#"{"alpha":0.01,"samples":[{"seed":1,"reward":1.0}]}"#,
    );
    let output = fixture.run("es-lora", &round, "es-out.gguf");
    assert_success(&output);

    let out = fixture.path("es-out.gguf");
    assert!(out.exists(), "ES must write the output adapter");
    assert_identity_preserved(&out);
    assert_ne!(before, weights(&out), "ES must change the LoRA weights");
}

#[test]
fn sft_changes_weights_and_preserves_adapter_identity() {
    let fixture = Fixture::new("sft");
    let before = weights(&fixture.adapter());
    let round = fixture.round(
        "sft.json",
        r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":0}]}"#,
    );
    let output = fixture.run("sft", &round, "sft-out.gguf");
    assert_success(&output);

    let out = fixture.path("sft-out.gguf");
    assert!(out.exists(), "SFT must write the output adapter");
    assert_identity_preserved(&out);
    assert_ne!(before, weights(&out), "SFT must change the LoRA weights");
}

#[test]
fn grpo_zero_advantage_leaves_weights_unchanged() {
    let fixture = Fixture::new("grpo-zero");
    let adapter = fixture.adapter();
    let before = weights(&adapter);
    let round = fixture.round(
        "grpo-zero.json",
        r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":0,"advantage":0.0}]}"#,
    );
    let output = fixture.run("grpo", &round, "grpo-zero-out.gguf");
    assert_success(&output);

    let out = fixture.path("grpo-zero-out.gguf");
    assert!(out.exists(), "GRPO must write the output adapter");
    assert_identity_preserved(&out);
    assert_eq!(
        before,
        weights(&out),
        "zero advantage must leave every weight untouched"
    );
}

#[test]
fn grpo_positive_advantage_changes_weights() {
    let fixture = Fixture::new("grpo-positive");
    let before = weights(&fixture.adapter());
    let round = fixture.round(
        "grpo-positive.json",
        r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":0,"advantage":0.7}]}"#,
    );
    let output = fixture.run("grpo", &round, "grpo-positive-out.gguf");
    assert_success(&output);

    let out = fixture.path("grpo-positive-out.gguf");
    assert!(out.exists(), "GRPO must write the output adapter");
    assert_identity_preserved(&out);
    assert_ne!(
        before,
        weights(&out),
        "positive advantage must update weights"
    );
}

#[test]
fn malformed_json_fails_without_writing_an_adapter() {
    for (index, subcommand) in ["es-lora", "sft", "grpo"].into_iter().enumerate() {
        let fixture = Fixture::new(&format!("bad-json-{subcommand}"));
        fixture.adapter();
        let round = fixture.round("bad.json", "not json");
        let out_name = format!("bad-out-{index}.gguf");
        let output = fixture.run(subcommand, &round, &out_name);
        assert!(
            !output.status.success(),
            "{subcommand} accepted malformed JSON: {}",
            read_output(&output)
        );
        assert!(
            !fixture.path(&out_name).exists(),
            "{subcommand} wrote an adapter despite malformed JSON"
        );
    }
}

#[test]
fn out_of_range_target_and_sampled_fail() {
    let fixture = Fixture::new("bad-target");
    fixture.adapter();

    let sft_round = fixture.round(
        "bad-target.json",
        r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"target":2}]}"#,
    );
    let output = fixture.run("sft", &sft_round, "bad-target-out.gguf");
    assert!(
        !output.status.success(),
        "SFT accepted an out-of-range target: {}",
        read_output(&output)
    );
    assert!(!fixture.path("bad-target-out.gguf").exists());

    let grpo_round = fixture.round(
        "bad-sampled.json",
        r#"{"lr":0.01,"examples":[{"x":[0.1,0.2],"logits":[0.5,-0.2],"sampled":2,"advantage":0.7}]}"#,
    );
    let output = fixture.run("grpo", &grpo_round, "bad-sampled-out.gguf");
    assert!(
        !output.status.success(),
        "GRPO accepted an out-of-range sampled index: {}",
        read_output(&output)
    );
    assert!(!fixture.path("bad-sampled-out.gguf").exists());
}
