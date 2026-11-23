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

/// Deterministic small generator so the batched-vs-single comparison is
/// reproducible without a dependency.
fn pseudo(seed: u64) -> f32 {
    let mixed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((mixed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

#[test]
fn q8_0_matmul_matches_per_token_matvec() {
    use friedflaxia_core::quant::matmul_q8_0;

    let rows = 3usize;
    let cols = 64usize;
    let mut weights = Vec::new();
    for row in 0..rows {
        for block in 0..cols / 32 {
            push_f16(&mut weights, F16_0_5 + row as u16 + block as u16);
            for lane in 0..32u8 {
                weights.push(lane.wrapping_add((row * 7 + block) as u8));
            }
        }
    }
    // 5 stays inside one tile; 17 and 33 cross the 16-token tile boundary.
    for n_tokens in [5usize, 17, 33] {
        let inputs: Vec<f32> = (0..n_tokens * cols).map(|i| pseudo(i as u64)).collect();
        let mut batched = vec![0.0f32; n_tokens * rows];
        matmul_q8_0(&weights, rows, cols, &inputs, n_tokens, &mut batched).expect("batched");
        for token in 0..n_tokens {
            let expected = matvec_q8_0(
                &weights,
                rows,
                cols,
                &inputs[token * cols..(token + 1) * cols],
            )
            .unwrap();
            for row in 0..rows {
                let actual = batched[token * rows + row];
                assert_eq!(
                    actual.to_bits(),
                    expected[row].to_bits(),
                    "tokens={n_tokens} token={token} row={row}"
                );
            }
        }
    }
}

#[test]
fn q4_0_matmul_matches_per_token_matvec() {
    use friedflaxia_core::quant::matmul_q4_0;

    let rows = 2usize;
    let cols = 64usize;
    let mut weights = Vec::new();
    for row in 0..rows {
        for block in 0..cols / 32 {
            push_f16(&mut weights, F16_1_0 + row as u16 * 2 + block as u16);
            for lane in 0..16u8 {
                weights.push(lane.wrapping_add((row + block) as u8));
            }
        }
    }
    for n_tokens in [4usize, 17, 33] {
        let inputs: Vec<f32> = (0..n_tokens * cols)
            .map(|i| pseudo(i as u64 + 17))
            .collect();
        let mut batched = vec![0.0f32; n_tokens * rows];
        matmul_q4_0(&weights, rows, cols, &inputs, n_tokens, &mut batched).expect("batched");
        for token in 0..n_tokens {
            let expected = matvec_q4_0(
                &weights,
                rows,
                cols,
                &inputs[token * cols..(token + 1) * cols],
            )
            .unwrap();
            for row in 0..rows {
                let actual = batched[token * rows + row];
                assert_eq!(
                    actual.to_bits(),
                    expected[row].to_bits(),
                    "tokens={n_tokens} token={token} row={row}"
                );
            }
        }
    }
}

#[test]
fn tensor_matmul_falls_back_for_f32() {
    use friedflaxia_core::gguf::{TensorInfo, TensorType};
    use friedflaxia_core::tensor::TensorView;

    let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let info = TensorInfo {
        name: "x".into(),
        dims: vec![2, 2],
        tensor_type: TensorType::F32,
        offset: 0,
    };
    let view = TensorView {
        info: &info,
        bytes: &bytes,
    };
    let out = view.matmul(&[1.0, 1.0, 2.0, 3.0], 2).expect("f32 matmul");
    // rows=2, cols=2:
    // token 0: row0 [1,2]·[1,1]=3, row1 [3,4]·[1,1]=7
    // token 1: row0 [1,2]·[2,3]=8, row1 [3,4]·[2,3]=18
    assert_eq!(out, vec![3.0, 7.0, 8.0, 18.0]);
}

#[test]
fn matmul_rejects_bad_shapes() {
    use friedflaxia_core::quant::matmul_q8_0;

    let mut out = vec![0.0f32; 2];
    assert!(matmul_q8_0(&[0u8; 33], 1, 32, &[0.0; 64], 2, &mut out).is_err());
    assert!(matmul_q8_0(&[0u8; 34], 1, 32, &[0.0; 64], 2, &mut [0.0; 1]).is_err());
    assert!(matmul_q8_0(&[0u8; 34], 1, 32, &[0.0; 32], 0, &mut []).is_err());
    // cols must be a whole number of blocks.
    assert!(matmul_q8_0(&[0u8; 68], 1, 48, &[0.0; 48], 1, &mut [0.0; 1]).is_err());
    assert!(
        friedflaxia_core::quant::matmul_q4_0(&[0u8; 36], 1, 40, &[0.0; 40], 1, &mut [0.0; 1])
            .is_err()
    );
}
