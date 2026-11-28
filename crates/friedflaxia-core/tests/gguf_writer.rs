//! Tests for the Rust GGUF writer used to publish qwen3-dllm LoRA adapters.
//!
//! Every file produced here is read back with the production `Gguf` parser, so
//! the tests cover the wire format, not just the writer's internal state.

use friedflaxia_core::gguf::writer::Writer;
use friedflaxia_core::gguf::{Gguf, TensorType, Value};

const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const EXPERT: &str = "code";
const A_NAME: &str = "blk.0.attn_q.lora_a.weight";
const B_NAME: &str = "blk.0.attn_q.lora_b.weight";
const ALIGNMENT: usize = 32;

fn a_values() -> Vec<f32> {
    (0..12).map(|index| index as f32 * 0.5).collect()
}

fn b_values() -> Vec<f32> {
    (0..8).map(|index| -1.0 - index as f32).collect()
}

fn build_writer() -> Writer {
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_tensor(A_NAME, &[3, 4], &a_values())
        .expect("lora_a tensor");
    writer
        .add_tensor(B_NAME, &[4, 2], &b_values())
        .expect("lora_b tensor");
    writer
}

fn assert_f32_data(bytes: &[u8], gguf: &Gguf, name: &str, expected: &[f32]) {
    let data = gguf.tensor_data(bytes, name).expect("tensor payload");
    assert_eq!(data.len(), expected.len() * 4);
    for (index, value) in expected.iter().enumerate() {
        let chunk: [u8; 4] = data[index * 4..index * 4 + 4].try_into().expect("one f32");
        assert_eq!(f32::from_le_bytes(chunk), *value, "{name}[{index}]");
    }
}

#[test]
fn gguf_writer_writes_header_in_spec_order() {
    let bytes = build_writer().finish();

    assert_eq!(&bytes[..4], b"GGUF");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 4);
}

#[test]
fn gguf_writer_round_trips_metadata_through_parser() {
    let bytes = build_writer().finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");

    assert_eq!(gguf.version, 3);
    assert_eq!(
        gguf.metadata("general.architecture"),
        Some(&Value::String("qwen3-dllm-lora".to_owned()))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.rank"),
        Some(&Value::U32(RANK))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.alpha"),
        Some(&Value::F32(ALPHA))
    );
    assert_eq!(
        gguf.metadata("qwen3-dllm-lora.expert"),
        Some(&Value::String(EXPERT.to_owned()))
    );
}

#[test]
fn gguf_writer_round_trips_tensor_names_dims_offsets_and_values() {
    let bytes = build_writer().finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");

    assert_eq!(gguf.tensor_count(), 2);
    let a = gguf.tensor(A_NAME).expect("lora_a info");
    assert_eq!(a.dims, vec![3, 4]);
    assert_eq!(a.tensor_type, TensorType::F32);
    assert_eq!(a.offset, 0);

    let b = gguf.tensor(B_NAME).expect("lora_b info");
    assert_eq!(b.dims, vec![4, 2]);
    assert_eq!(b.tensor_type, TensorType::F32);
    assert_eq!(b.offset, 48); // 12 f32 values * 4 bytes

    assert_eq!(gguf.alignment(), ALIGNMENT as u64);
    assert_eq!(gguf.data_offset() % ALIGNMENT, 0);
    assert_eq!(bytes.len(), gguf.data_offset() + 48 + 32);

    assert_f32_data(&bytes, &gguf, A_NAME, &a_values());
    assert_f32_data(&bytes, &gguf, B_NAME, &b_values());
}

#[test]
fn gguf_writer_rejects_zero_rank() {
    let error = Writer::new(0, ALPHA, EXPERT)
        .err()
        .expect("rank 0 must be rejected");
    assert!(error.to_string().contains("rank"), "{error}");
}

#[test]
fn gguf_writer_rejects_non_positive_alpha() {
    for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(Writer::new(RANK, alpha, EXPERT).is_err(), "alpha={alpha}");
    }
}

#[test]
fn gguf_writer_rejects_tensor_values_that_do_not_fill_dims() {
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    assert!(writer.add_tensor(A_NAME, &[3, 4], &[0.0; 11]).is_err());
}
