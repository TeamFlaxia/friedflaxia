use friedflaxia_core::quant::{matvec_bf16, matvec_f32, matvec_q4_0, matvec_q6_k, matvec_q8_0};

const F16_0_5: u16 = 0x3800;
const F16_1_0: u16 = 0x3c00;
const F16_2_0: u16 = 0x4000;

fn push_f16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[test]
fn q8_0_matvec_projects_each_row() {
    let mut weights = Vec::new();
    push_f16(&mut weights, F16_1_0);
    for value in 1..=32u8 {
        weights.push(value);
    }
    push_f16(&mut weights, F16_0_5);
    for value in 0..32u8 {
        weights.push(value);
    }

    let input = vec![1.0f32; 32];
    let out = matvec_q8_0(&weights, 2, 32, &input).expect("valid Q8_0 matvec");
    assert_eq!(out, vec![528.0, 248.0]);
}

#[test]
fn q4_0_matvec_dequantizes_low_and_high_nibbles() {
    let mut weights = Vec::new();
    push_f16(&mut weights, F16_2_0);
    for j in 0..16u8 {
        weights.push((j & 0x0f) | (((15 - j) & 0x0f) << 4));
    }

    let out = matvec_q4_0(&weights, 1, 32, &[1.0f32; 32]).expect("valid Q4_0 matvec");
    assert_eq!(out, vec![-32.0]);
}

#[test]
fn q6_k_matvec_projects_all_256_values() {
    let mut weights = vec![0u8; 210];
    weights[208..210].copy_from_slice(&F16_1_0.to_le_bytes());
    for scale in &mut weights[192..208] {
        *scale = 1;
    }

    let out = matvec_q6_k(&weights, 1, 256, &[1.0f32; 256]).expect("valid Q6_K matvec");
    assert_eq!(out, vec![-8192.0]);
}

#[test]
fn f32_and_bf16_matvec_are_supported() {
    let f32_weights: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(
        matvec_f32(&f32_weights, 2, 3, &[1.0, 1.0, 1.0]).unwrap(),
        vec![6.0, 15.0]
    );

    let bf16_weights: Vec<u8> = [0x3f80u16, 0x4000, 0x4040, 0x4080]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(
        matvec_bf16(&bf16_weights, 2, 2, &[2.0, 1.0]).unwrap(),
        vec![4.0, 10.0]
    );
}

#[test]
fn matvec_rejects_bad_shapes() {
    assert!(matvec_q8_0(&[0u8; 34], 1, 31, &[0.0; 31]).is_err());
    assert!(matvec_q8_0(&[0u8; 34], 1, 32, &[0.0; 31]).is_err());
    assert!(matvec_q8_0(&[0u8; 33], 1, 32, &[0.0; 32]).is_err());
    assert!(matvec_f32(&[0u8; 12], 1, 4, &[0.0; 4]).is_err());
}
