use friedflaxia_core::gguf::{Gguf, GgufError, TensorType, Value};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u32 = 32;

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    put_string(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    put_string(out, value);
}

fn put_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    put_string(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_kv_u64(out: &mut Vec<u8>, key: &str, value: u64) {
    put_string(out, key);
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_kv_string_array(out: &mut Vec<u8>, key: &str, values: &[&str]) {
    put_string(out, key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        put_string(out, value);
    }
}

/// A complete, aligned GGUF v3 file with one Q8_0 tensor of shape [32, 2].
fn build_fixture() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    out.extend_from_slice(&4u64.to_le_bytes()); // metadata count

    put_kv_u32(&mut out, "general.alignment", ALIGNMENT);
    put_kv_string(&mut out, "general.architecture", "test-arch");
    put_kv_string_array(&mut out, "tokenizer.ggml.tokens", &["a", "b"]);
    put_kv_u64(&mut out, "test.value", 42);

    // Tensor info: name, n_dims, dims (ne[0] first), type, offset.
    put_string(&mut out, "token_embd.weight");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&32u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
    out.extend_from_slice(&0u64.to_le_bytes());

    while out.len() % ALIGNMENT as usize != 0 {
        out.push(0);
    }

    // Two Q8_0 blocks: 2 * 34 bytes.
    for block in 0..2u8 {
        out.extend_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
        for i in 0..32u8 {
            out.push(i.wrapping_add(block));
        }
    }
    out
}

#[test]
fn parses_metadata_and_one_tensor() {
    let bytes = build_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");

    assert_eq!(gguf.version, GGUF_VERSION);
    assert_eq!(gguf.tensor_count(), 1);
    assert_eq!(
        gguf.metadata("general.architecture")
            .and_then(Value::as_str),
        Some("test-arch")
    );
    assert_eq!(
        gguf.metadata("general.alignment").and_then(Value::as_u64),
        Some(32)
    );
    assert_eq!(
        gguf.metadata("test.value").and_then(Value::as_u64),
        Some(42)
    );

    let tokens = gguf
        .metadata("tokenizer.ggml.tokens")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].as_str(), Some("a"));
    assert_eq!(tokens[1].as_str(), Some("b"));

    let tensor = gguf.tensor("token_embd.weight").expect("tensor info");
    assert_eq!(tensor.dims, vec![32, 2]);
    assert_eq!(tensor.tensor_type, TensorType::Q8_0);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.element_count(), 64);
    assert_eq!(tensor.byte_len(), Some(2 * 34));

    let data = gguf
        .tensor_data(&bytes, "token_embd.weight")
        .expect("tensor bytes");
    assert_eq!(data.len(), 68);
    assert_eq!(&data[..2], &0x3c00u16.to_le_bytes());
}

#[test]
fn alignment_pads_the_data_section() {
    let bytes = build_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    assert_eq!(gguf.data_offset() % ALIGNMENT as usize, 0);
    assert_eq!(
        bytes[gguf.data_offset()..gguf.data_offset() + 2],
        0x3c00u16.to_le_bytes()
    );
}

#[test]
fn rejects_bad_magic_and_unsupported_versions() {
    let mut bytes = build_fixture();
    bytes[..4].copy_from_slice(b"GGUX");
    assert!(matches!(Gguf::parse(&bytes), Err(GgufError::BadMagic)));

    let mut bytes = build_fixture();
    bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        Gguf::parse(&bytes),
        Err(GgufError::UnsupportedVersion(1))
    ));
}

#[test]
fn rejects_truncated_header_and_tensor_data() {
    let bytes = build_fixture();
    assert!(Gguf::parse(&bytes[..20]).is_err());

    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    assert!(
        gguf.tensor_data(&bytes[..bytes.len() - 1], "token_embd.weight")
            .is_err()
    );
}

#[test]
fn unknown_tensor_is_reported() {
    let bytes = build_fixture();
    let gguf = Gguf::parse(&bytes).expect("valid GGUF");
    assert!(gguf.tensor("missing").is_none());
    assert!(matches!(
        gguf.tensor_data(&bytes, "missing"),
        Err(GgufError::UnknownTensor(_))
    ));
}
