use friedflaxia_wasm::Engine;

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn minimal_gguf(architecture: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());
    put_string(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    put_string(&mut out, architecture);
    put_string(&mut out, "general.alignment");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

#[test]
fn parses_a_supported_model_and_rejects_bad_input() {
    let engine = Engine::new(minimal_gguf("gemma-embedding2")).expect("supported architecture");
    assert_eq!(engine.architecture(), "gemma-embedding2");
    assert_eq!(engine.tensor_count(), 0);
    assert!(engine.encode("hello", true).is_err());
    assert!(engine.embed(&[]).is_err());

    let mut bad = minimal_gguf("gemma-embedding2");
    bad[..4].copy_from_slice(b"GGUX");
    assert!(Engine::new(bad).is_err());

    assert!(Engine::new(minimal_gguf("unknown-arch")).is_err());
}

#[test]
fn rejects_cross_model_calls() {
    let engine = Engine::new(minimal_gguf("plamo3")).expect("supported architecture");
    assert!(engine.embed(&[]).is_err());
    assert!(engine.decision(&[], &[], 0).is_err());
}
