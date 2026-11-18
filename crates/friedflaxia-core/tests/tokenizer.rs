use friedflaxia_core::gguf::Gguf;
use friedflaxia_core::gguf::tokenizer::{TokenType, TokenizerSpec};

const GGUF_VERSION: u32 = 3;

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_kv_header(out: &mut Vec<u8>, key: &str, value_type: u32) {
    put_string(out, key);
    out.extend_from_slice(&value_type.to_le_bytes());
}

fn put_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    put_kv_header(out, key, 8);
    put_string(out, value);
}

fn put_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    put_kv_header(out, key, 4);
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_kv_bool(out: &mut Vec<u8>, key: &str, value: bool) {
    put_kv_header(out, key, 7);
    out.push(u8::from(value));
}

fn put_array_header(out: &mut Vec<u8>, key: &str, element_type: u32, count: usize) {
    put_kv_header(out, key, 9);
    out.extend_from_slice(&element_type.to_le_bytes());
    out.extend_from_slice(&(count as u64).to_le_bytes());
}

fn put_kv_string_array(out: &mut Vec<u8>, key: &str, values: &[&str]) {
    put_array_header(out, key, 8, values.len());
    for value in values {
        put_string(out, value);
    }
}

fn put_kv_f32_array(out: &mut Vec<u8>, key: &str, values: &[f32]) {
    put_array_header(out, key, 6, values.len());
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn put_kv_i32_array(out: &mut Vec<u8>, key: &str, values: &[i32]) {
    put_array_header(out, key, 5, values.len());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn build_tokenizer_fixture() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // no tensors
    out.extend_from_slice(&12u64.to_le_bytes()); // metadata entries

    put_kv_string(&mut out, "tokenizer.ggml.model", "gpt2");
    put_kv_string_array(
        &mut out,
        "tokenizer.ggml.tokens",
        &["a", "b", "<s>", "<unk>"],
    );
    put_kv_f32_array(&mut out, "tokenizer.ggml.scores", &[0.0, -1.5, 0.0, 0.0]);
    put_kv_i32_array(&mut out, "tokenizer.ggml.token_type", &[1, 1, 3, 2]);
    put_kv_string_array(&mut out, "tokenizer.ggml.merges", &["a b"]);
    put_kv_u32(&mut out, "tokenizer.ggml.bos_token_id", 2);
    put_kv_u32(&mut out, "tokenizer.ggml.eos_token_id", 0);
    put_kv_bool(&mut out, "tokenizer.ggml.add_bos_token", false);
    put_kv_bool(&mut out, "tokenizer.ggml.add_eos_token", true);
    put_kv_string(&mut out, "tokenizer.chat_template", "{{ messages }}");
    put_kv_string(&mut out, "general.architecture", "test");
    put_kv_u32(&mut out, "general.alignment", 32);

    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

#[test]
fn reads_gpt2_style_tokenizer_metadata() {
    let bytes = build_tokenizer_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    let tokenizer = TokenizerSpec::from_gguf(&gguf).expect("tokenizer metadata");

    assert_eq!(tokenizer.model, "gpt2");
    assert_eq!(tokenizer.tokens, vec!["a", "b", "<s>", "<unk>"]);
    assert_eq!(tokenizer.scores, vec![0.0, -1.5, 0.0, 0.0]);
    assert_eq!(
        tokenizer.token_types,
        vec![
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Control,
            TokenType::Unknown
        ]
    );
    assert_eq!(tokenizer.merges, vec!["a b"]);
    assert_eq!(tokenizer.bos_token_id, 2);
    assert_eq!(tokenizer.eos_token_id, 0);
    assert!(!tokenizer.add_bos_token);
    assert!(tokenizer.add_eos_token);
    assert_eq!(tokenizer.chat_template.as_deref(), Some("{{ messages }}"));
    assert_eq!(tokenizer.vocab_size(), 4);
}

#[test]
fn rejects_missing_or_malformed_arrays() {
    let mut bytes = build_tokenizer_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    assert!(TokenizerSpec::from_gguf(&gguf).is_ok());

    // Truncate the file before the token array is complete.
    bytes.truncate(120);
    assert!(Gguf::parse(&bytes).is_err());
}

#[test]
fn token_types_map_to_known_values() {
    assert_eq!(TokenType::from_i32(1), Some(TokenType::Normal));
    assert_eq!(TokenType::from_i32(2), Some(TokenType::Unknown));
    assert_eq!(TokenType::from_i32(3), Some(TokenType::Control));
    assert_eq!(TokenType::from_i32(4), Some(TokenType::UserDefined));
    assert_eq!(TokenType::from_i32(5), Some(TokenType::Unused));
    assert_eq!(TokenType::from_i32(6), Some(TokenType::Byte));
    assert_eq!(TokenType::from_i32(99), None);
}
