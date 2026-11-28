//! Tests for the Rust GGUF writer used to publish qwen3-dllm LoRA adapters.
//!
//! Every file produced here is read back with the production `Gguf` parser, so
//! the tests cover the wire format, not just the writer's internal state.

use friedflaxia_core::gguf::writer::{QuantKind, Writer};
use friedflaxia_core::gguf::{Gguf, TensorType, Value};
use friedflaxia_core::quant::{Q4_0_BLOCK_BYTES, QK, dequantize_q4_0, f16_to_f32};
use friedflaxia_core::tensor::{TensorError, tensor};

const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const EXPERT: &str = "code";
const A_NAME: &str = "blk.0.attn_q.lora_a.weight";
const B_NAME: &str = "blk.0.attn_q.lora_b.weight";
const ALIGNMENT: usize = 32;

const Q4_NAME: &str = "blk.0.attn_q.weight";
/// Q4_0 source values round to a 4-bit grid whose step is `|extreme| / 8`,
/// and the block scale itself is rounded to f16 (~2^-11 relative error).
/// For the test data below (`|extreme| <= 0.65`) that is at most ~0.041 for
/// half a step plus ~0.003 for the scale, so 0.05 is an explicit bound.
const Q4_TOLERANCE: f32 = 0.05;

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

fn q4_ladder() -> Vec<f32> {
    // Exact multiples of 0.125: with extreme -1.0 the ggml scale is +0.125,
    // so every value lands exactly on a nibble and the round trip is lossless.
    (0..2 * QK)
        .map(|index| (index % (QK / 2)) as f32 * 0.125 - 1.0)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: element count");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= Q4_TOLERANCE,
            "{context}[{index}]: got {got}, expected {want} (tol {Q4_TOLERANCE})"
        );
    }
}

#[test]
fn gguf_writer_writes_q4_0_tensor_type_and_exact_nibbles() {
    let source = q4_ladder();
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_quantized_tensor(Q4_NAME, &[QK as u64, 2], &source, QuantKind::Q4_0)
        .expect("Q4_0 tensor");

    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    let info = gguf.tensor(Q4_NAME).expect("Q4_0 info");
    assert_eq!(info.tensor_type, TensorType::Q4_0);
    assert_eq!(info.dims, vec![QK as u64, 2]);
    assert_eq!(info.offset, 0);

    let data = gguf.tensor_data(&bytes, Q4_NAME).expect("Q4_0 payload");
    assert_eq!(data.len(), 2 * Q4_0_BLOCK_BYTES);
    // -1.0 / -8 = +0.125, encoded as IEEE binary16 0x3000.
    assert_eq!(u16::from_le_bytes(data[..2].try_into().unwrap()), 0x3000);
    for j in 0..QK / 2 {
        // q[j] = j, q[16 + j] = 16 + j, both truncated to four bits.
        assert_eq!(data[2 + j], (j as u8) | ((j as u8) << 4), "nibble {j}");
    }
    assert_eq!(dequantize_q4_0(data).expect("decode Q4_0"), source);
}

#[test]
fn gguf_writer_q4_0_matches_convert_dllm_py_bytes() {
    // Expected bytes were produced by `quantize_q4_0` in tools/convert_dllm.py
    // for the same 32 f32 values (extreme -1.5500001, scale +0.19375).
    let source: Vec<f32> = (0..QK).map(|index| (index as f32 - 15.5) * 0.1).collect();
    let expected = [
        0x33, 0x32, 0x80, 0x91, 0x91, 0xa2, 0xa2, 0xb3, 0xb3, 0xc4, 0xc4, 0xd5, 0xd5, 0xe6, 0xe6,
        0xf7, 0xf7, 0xf8,
    ];

    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_quantized_tensor(Q4_NAME, &[QK as u64], &source, QuantKind::Q4_0)
        .expect("Q4_0 tensor");
    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    assert_eq!(
        gguf.tensor_data(&bytes, Q4_NAME).expect("payload"),
        expected
    );
}

#[test]
fn gguf_writer_q4_0_round_trips_through_tensor_view() {
    // Non-grid values exercise the real quantization error of Q4_0.
    let source: Vec<f32> = (0..2 * QK)
        .map(|index| index as f32 * 0.0137 - 0.23)
        .collect();
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_quantized_tensor(Q4_NAME, &[2 * QK as u64], &source, QuantKind::Q4_0)
        .expect("Q4_0 tensor");

    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    assert_eq!(gguf.tensor(Q4_NAME).unwrap().tensor_type, TensorType::Q4_0);

    let view = tensor(&gguf, &bytes, Q4_NAME).expect("Q4_0 tensor view");
    match view.vec() {
        Ok(values) => assert_close(&values, &source, "TensorView::vec Q4_0"),
        Err(TensorError::UnsupportedType(TensorType::Q4_0)) => {
            // `tensor.rs` is outside this task's editable files. `row_vec(0)`
            // uses the same dequantize_q4_0 path and spans this 1-D tensor.
            assert_close(
                &view.row_vec(0).expect("Q4_0 row dequantization"),
                &source,
                "TensorView::row_vec Q4_0",
            );
        }
        Err(error) => panic!("unexpected Q4_0 vec error: {error}"),
    }
}

#[test]
fn gguf_writer_q4_0_uses_signed_extreme_scale() {
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_quantized_tensor("positive", &[QK as u64], &[1.0; QK], QuantKind::Q4_0)
        .expect("positive block");
    writer
        .add_quantized_tensor("negative", &[QK as u64], &[-1.0; QK], QuantKind::Q4_0)
        .expect("negative block");

    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    let scale_of = |name: &str| {
        let data = gguf.tensor_data(&bytes, name).expect("Q4_0 payload");
        f16_to_f32(u16::from_le_bytes(data[..2].try_into().unwrap()))
    };
    assert_eq!(scale_of("positive"), -0.125); // extreme +1.0 -> d = -1.0 / 8
    assert_eq!(scale_of("negative"), 0.125); // extreme -1.0 -> d = +1.0 / 8
}

#[test]
fn gguf_writer_q4_0_scale_is_always_finite() {
    let cases: [(&str, f32); 4] = [
        ("huge_positive", f32::MAX),
        ("huge_negative", -f32::MAX),
        ("tiny_positive", f32::MIN_POSITIVE),
        ("zero", 0.0),
    ];
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    for (name, value) in cases {
        writer
            .add_quantized_tensor(name, &[QK as u64], &[value; QK], QuantKind::Q4_0)
            .expect("finite scale");
    }

    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    for (name, _) in cases {
        let data = gguf.tensor_data(&bytes, name).expect("Q4_0 payload");
        let bits = u16::from_le_bytes(data[..2].try_into().unwrap());
        assert_ne!(bits & 0x7c00, 0x7c00, "{name}: f16 scale is inf/NaN");
        assert!(
            f16_to_f32(bits).is_finite(),
            "{name}: decoded scale must be finite"
        );
        assert!(
            dequantize_q4_0(data)
                .expect("Q4_0 payload")
                .iter()
                .all(|v| v.is_finite()),
            "{name}: dequantized values must be finite"
        );
    }
}

#[test]
fn gguf_writer_rejects_q4_0_input_that_is_not_32_values() {
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    for count in [1, 12, 31, 33, 64 + 7] {
        let values = vec![0.25f32; count];
        assert!(
            writer
                .add_quantized_tensor("bad", &[count as u64], &values, QuantKind::Q4_0)
                .is_err(),
            "value count {count} must be rejected"
        );
    }
    // A rejected tensor must not corrupt the writer: valid data still works.
    writer
        .add_quantized_tensor(Q4_NAME, &[QK as u64], &[0.5; QK], QuantKind::Q4_0)
        .expect("valid Q4_0 tensor after errors");
    assert_eq!(Gguf::parse(&writer.finish()).unwrap().tensor_count(), 1);
}

#[test]
fn gguf_writer_rejects_non_finite_q4_0_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
        assert!(
            writer
                .add_quantized_tensor("bad", &[QK as u64], &[value; QK], QuantKind::Q4_0)
                .is_err(),
            "non-finite value {value} must be rejected"
        );
    }
}

#[test]
fn gguf_writer_mixed_f32_and_q4_0_offsets_are_correct() {
    let mut writer = Writer::new(RANK, ALPHA, EXPERT).expect("valid adapter metadata");
    writer
        .add_tensor(A_NAME, &[3, 4], &a_values())
        .expect("F32 tensor");
    writer
        .add_quantized_tensor(Q4_NAME, &[QK as u64], &[0.125; QK], QuantKind::Q4_0)
        .expect("Q4_0 tensor");
    writer
        .add_tensor(B_NAME, &[4, 2], &b_values())
        .expect("F32 tensor");

    let bytes = writer.finish();
    let gguf = Gguf::parse(&bytes).expect("writer output parses");
    assert_eq!(gguf.tensor(A_NAME).unwrap().tensor_type, TensorType::F32);
    assert_eq!(gguf.tensor(A_NAME).unwrap().offset, 0);
    assert_eq!(gguf.tensor(Q4_NAME).unwrap().tensor_type, TensorType::Q4_0);
    assert_eq!(gguf.tensor(Q4_NAME).unwrap().offset, 48);
    assert_eq!(gguf.tensor(B_NAME).unwrap().tensor_type, TensorType::F32);
    assert_eq!(
        gguf.tensor(B_NAME).unwrap().offset,
        48 + Q4_0_BLOCK_BYTES as u64
    );
    assert_eq!(bytes.len(), gguf.data_offset() + 48 + Q4_0_BLOCK_BYTES + 32);
    assert_f32_data(&bytes, &gguf, A_NAME, &a_values());
    assert_f32_data(&bytes, &gguf, B_NAME, &b_values());
}
