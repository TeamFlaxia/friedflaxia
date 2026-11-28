//! Uncached vs. cached dLLM denoise step timings.
//!
//! `FRIEDFLAXIA_DLLM_GGUF=/path/to/qwen3-dllm.gguf cargo run --release --example dllm_bench`
//!
//! The two paths are checked for identical top-8 candidates on every step; the
//! numbers therefore compare the same work, not a shortcut.

use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::models::dllm::{DllmSession, denoise_topk};
use std::env;
use std::time::Instant;

const BLOCK: usize = 8;
const STEPS: usize = 8;

fn main() {
    let Ok(path) = env::var("FRIEDFLAXIA_DLLM_GGUF") else {
        eprintln!("set FRIEDFLAXIA_DLLM_GGUF to a qwen3-dllm GGUF to run this benchmark");
        return;
    };
    let bytes = std::fs::read(path).expect("read dLLM GGUF");
    let gguf = Gguf::parse(&bytes).expect("parse dLLM GGUF");
    let mut tokens = vec![1u32; BLOCK * STEPS];
    tokens[0] = 151_644;
    tokens[1] = 872;
    tokens[2] = 151_645;
    tokens[3] = 151_669;

    let mut uncached_ms = 0u128;
    let mut cached_ms = 0u128;
    let mut session = DllmSession::new(&gguf, &bytes).expect("session");

    for step in 0..STEPS {
        if step > 0 {
            for offset in 0..BLOCK {
                tokens[(step - 1) * BLOCK + offset] = (1000 + step * BLOCK + offset) as u32;
            }
        }
        let block = (step * BLOCK, (step + 1) * BLOCK);
        let positions: Vec<usize> = (block.0..block.1).collect();

        let started = Instant::now();
        let expected =
            denoise_topk(&gguf, &bytes, &tokens, &positions, block, 8).expect("uncached");
        let plain = started.elapsed().as_millis();

        let started = Instant::now();
        let actual = session
            .denoise_topk(&tokens, &positions, block, 8)
            .expect("cached");
        let warm = started.elapsed().as_millis();

        assert_eq!(actual.token_ids, expected.token_ids, "step {step} diverged");
        uncached_ms += plain;
        cached_ms += warm;
        println!(
            "step {step} block {block:?}: uncached {plain:>5} ms, cached {warm:>5} ms, prefix {}",
            session.prefix_len()
        );
    }

    println!(
        "total: uncached {uncached_ms} ms, cached {cached_ms} ms ({:.2}x)",
        uncached_ms as f64 / cached_ms.max(1) as f64
    );
}
