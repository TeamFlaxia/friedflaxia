use friedflaxia_core::quant::{
    Q4_0_BLOCK_BYTES, Q4_1_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, dequantize_bf16,
    dequantize_q4_0, dequantize_q4_1, dequantize_q6_k, dequantize_q8_0,
};

/// IEEE-754 binary16 bit patterns used by the GGML block headers.
const F16_0_5: u16 = 0x3800;
const F16_1_0: u16 = 0x3c00;
const F16_2_0: u16 = 0x4000;

#[test]
fn q8_0_block_uses_block_scale_and_signed_quants() {
    let mut block = [0u8; Q8_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_0_5.to_le_bytes());
    for i in 0..32 {
        block[2 + i] = (i as i8 - 16) as u8;
    }

    let values = dequantize_q8_0(&block).expect("one complete Q8_0 block");
    assert_eq!(values.len(), 32);
    for (i, value) in values.iter().enumerate() {
        assert_eq!(*value, (i as f32 - 16.0) * 0.5);
    }
}

#[test]
fn q4_0_block_dequantizes_low_nibbles_then_high_nibbles() {
    let mut block = [0u8; Q4_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_2_0.to_le_bytes());
    for j in 0..16usize {
        let low = j as u8 & 0x0f;
        let high = (15 - j) as u8 & 0x0f;
        block[2 + j] = low | (high << 4);
    }

    let values = dequantize_q4_0(&block).expect("one complete Q4_0 block");
    assert_eq!(values.len(), 32);
    for j in 0..16usize {
        assert_eq!(values[j], (j as f32 - 8.0) * 2.0);
        assert_eq!(values[16 + j], ((15 - j) as f32 - 8.0) * 2.0);
    }
}

#[test]
fn zero_scale_produces_zeroes() {
    let mut q8 = [0u8; Q8_0_BLOCK_BYTES];
    q8[2..].fill(0x7f);
    assert!(
        dequantize_q8_0(&q8)
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );

    let mut q4 = [0u8; Q4_0_BLOCK_BYTES];
    q4[2..].fill(0xf0);
    assert!(
        dequantize_q4_0(&q4)
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

#[test]
fn q4_1_block_applies_scale_and_min() {
    let mut block = [0u8; Q4_1_BLOCK_BYTES];
    block[..2].copy_from_slice(&F16_1_0.to_le_bytes());
    block[2..4].copy_from_slice(&F16_0_5.to_le_bytes());
    for j in 0..16usize {
        let low = j as u8 & 0x0f;
        let high = (15 - j) as u8 & 0x0f;
        block[4 + j] = low | (high << 4);
    }

    let values = dequantize_q4_1(&block).expect("one complete Q4_1 block");
    assert_eq!(values.len(), 32);
    for j in 0..16usize {
        assert_eq!(values[j], j as f32 + 0.5);
        assert_eq!(values[16 + j], (15 - j) as f32 + 0.5);
    }
}

#[test]
fn q6_k_block_uses_six_bit_values_and_signed_scales() {
    let mut block = [0u8; Q6_K_BLOCK_BYTES];
    // ql=0 and qh=0 encode q=-32 everywhere; d=1 and every scale=1.
    block[208..210].copy_from_slice(&F16_1_0.to_le_bytes());
    for scale in &mut block[192..208] {
        *scale = 1;
    }
    let values = dequantize_q6_k(&block).expect("one complete Q6_K block");
    assert_eq!(values.len(), 256);
    assert!(values.iter().all(|value| *value == -32.0));

    // Encode 31 at output[0]: ql[0] low nibble = 15, qh[0] bits 0..1 = 3.
    block[0] = 0x0f;
    block[128] = 0b0000_0011;
    let values = dequantize_q6_k(&block).expect("one complete Q6_K block");
    assert_eq!(values[0], 31.0);
    assert_eq!(values[32], -32.0);
    assert_eq!(values[64], -32.0);
    assert_eq!(values[96], -32.0);

    // Scales are signed int8: -2 * -32 = 64.
    block[0] = 0;
    block[128] = 0;
    block[192..208].fill(-2i8 as u8);
    let values = dequantize_q6_k(&block).expect("one complete Q6_K block");
    assert!(values.iter().all(|value| *value == 64.0));
}

#[test]
fn bf16_values_widen_to_f32() {
    let values = dequantize_bf16(
        &[
            0x0000, // 0.0
            0x3f80, // 1.0
            0x4000, // 2.0
            0xbf80, // -1.0
        ]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>(),
    )
    .expect("whole BF16 values");

    assert_eq!(values, vec![0.0, 1.0, 2.0, -1.0]);
    assert!(dequantize_bf16(&[0x00]).is_err());
}

#[test]
fn incomplete_blocks_are_rejected() {
    assert!(dequantize_q8_0(&[0u8; Q8_0_BLOCK_BYTES - 1]).is_err());
    assert!(dequantize_q4_0(&[0u8; Q4_0_BLOCK_BYTES - 1]).is_err());
}

#[test]
fn f16_scale_decoding_handles_normal_and_subnormal_values() {
    assert_eq!(friedflaxia_core::quant::f16_to_f32(F16_0_5), 0.5);
    assert_eq!(friedflaxia_core::quant::f16_to_f32(F16_1_0), 1.0);
    assert_eq!(friedflaxia_core::quant::f16_to_f32(F16_2_0), 2.0);
    // Smallest positive subnormal: 2^-24.
    assert_eq!(friedflaxia_core::quant::f16_to_f32(0x0001), 2f32.powi(-24));
    // Positive infinity and a quiet NaN.
    assert!(friedflaxia_core::quant::f16_to_f32(0x7c00).is_infinite());
    assert!(friedflaxia_core::quant::f16_to_f32(0x7e00).is_nan());
}
