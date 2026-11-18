use friedflaxia_core::gguf::TensorType;

#[test]
fn bf16_is_a_supported_two_byte_type() {
    assert_eq!(TensorType::from_u32(30), TensorType::BF16);
    assert_eq!(TensorType::BF16.as_u32(), 30);
    assert_eq!(TensorType::BF16.block_layout(), Some((1, 2)));
    assert_eq!(TensorType::BF16.byte_len(6_291_456), Some(12_582_912));
}

#[test]
fn q8_0_and_q4_0_byte_lengths_match_ggml_blocks() {
    assert_eq!(TensorType::Q8_0.byte_len(64), Some(68));
    assert_eq!(TensorType::Q4_0.byte_len(64), Some(36));
    assert_eq!(TensorType::Q4_0.byte_len(33), Some(36));
    assert_eq!(TensorType::Q8_0.byte_len(0), Some(0));
}

#[test]
fn q6_k_byte_length_matches_the_ggml_block() {
    assert_eq!(TensorType::Q6K.block_layout(), Some((256, 210)));
    assert_eq!(TensorType::Q6K.byte_len(1536 * 107_520), Some(135_475_200));
}
