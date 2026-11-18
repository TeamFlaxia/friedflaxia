use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::gguf::config::ModelConfig;

const GGUF_VERSION: u32 = 3;

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_header(out: &mut Vec<u8>, key: &str, value_type: u32) {
    put_string(out, key);
    out.extend_from_slice(&value_type.to_le_bytes());
}

fn put_string_kv(out: &mut Vec<u8>, key: &str, value: &str) {
    put_header(out, key, 8);
    put_string(out, value);
}

fn put_u32_kv(out: &mut Vec<u8>, key: &str, value: u32) {
    put_header(out, key, 4);
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32_kv(out: &mut Vec<u8>, key: &str, value: f32) {
    put_header(out, key, 6);
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn put_bool_kv(out: &mut Vec<u8>, key: &str, value: bool) {
    put_header(out, key, 7);
    out.push(u8::from(value));
}

fn build_config_fixture() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&7u64.to_le_bytes());

    put_string_kv(&mut out, "general.architecture", "gemma-embedding2");
    put_u32_kv(&mut out, "gemma-embedding2.block_count", 24);
    put_u32_kv(&mut out, "gemma-embedding2.embedding_length", 512);
    put_u32_kv(&mut out, "gemma-embedding2.embedding_length_out", 768);
    put_f32_kv(
        &mut out,
        "gemma-embedding2.attention.layer_norm_rms_epsilon",
        0.000001,
    );
    put_bool_kv(&mut out, "gemma-embedding2.attention.causal", false);
    put_u32_kv(&mut out, "gemma-embedding2.pooling_type", 1);
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

#[test]
fn reads_architecture_scoped_values() {
    let bytes = build_config_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    let config = ModelConfig::from_gguf(&gguf).expect("architecture metadata");

    assert_eq!(config.architecture(), "gemma-embedding2");
    assert_eq!(config.u32("block_count").unwrap(), Some(24));
    assert_eq!(config.u32("embedding_length").unwrap(), Some(512));
    assert_eq!(config.u32("embedding_length_out").unwrap(), Some(768));
    assert_eq!(
        config.f32("attention.layer_norm_rms_epsilon").unwrap(),
        Some(0.000001)
    );
    assert_eq!(config.bool("attention.causal").unwrap(), Some(false));
    assert_eq!(config.u32("missing").unwrap(), None);
}

#[test]
fn reports_type_mismatches() {
    let bytes = build_config_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    let config = ModelConfig::from_gguf(&gguf).expect("architecture metadata");

    assert!(config.u32("attention.causal").is_err());
    assert!(config.f32("block_count").is_err());
    assert!(config.bool("embedding_length").is_err());
}

#[test]
fn requires_general_architecture() {
    let mut empty = Vec::new();
    empty.extend_from_slice(b"GGUF");
    empty.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    empty.extend_from_slice(&0u64.to_le_bytes());
    empty.extend_from_slice(&0u64.to_le_bytes());
    while empty.len() % 32 != 0 {
        empty.push(0);
    }
    let gguf = Gguf::parse(&empty).expect("empty GGUF");
    assert!(ModelConfig::from_gguf(&gguf).is_err());
}
