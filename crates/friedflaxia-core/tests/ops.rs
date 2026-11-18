use friedflaxia_core::ops::{
    gelu_tanh, l2_normalize, rms_norm, rope_neox_inplace, softmax_in_place,
};

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn rms_norm_scales_by_root_mean_square_and_weight() {
    let out = rms_norm(&[1.0, 1.0, 1.0, 1.0], Some(&[1.0, 2.0, 3.0, 4.0]), 0.0);
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

    let out = rms_norm(&[3.0, 4.0], None, 0.0);
    assert_close(out[0], 3.0 / 12.5f32.sqrt(), 1e-6);
    assert_close(out[1], 4.0 / 12.5f32.sqrt(), 1e-6);
}

#[test]
fn gelu_tanh_matches_the_reference_values() {
    assert_eq!(gelu_tanh(0.0), 0.0);
    assert_close(gelu_tanh(1.0), 0.841_192, 1e-5);
    assert_close(gelu_tanh(-1.0), -0.158_808, 1e-5);
    assert_close(gelu_tanh(8.0), 8.0, 1e-3);
}

#[test]
fn softmax_is_stable_and_normalized() {
    let mut values = vec![1.0, 2.0, 3.0];
    softmax_in_place(&mut values);
    let sum: f32 = values.iter().sum();
    assert_close(sum, 1.0, 1e-6);
    assert!(values[0] < values[1] && values[1] < values[2]);

    let mut large = vec![1000.0, 1001.0, 1002.0];
    softmax_in_place(&mut large);
    assert!(large.iter().all(|value| value.is_finite()));
    assert_close(large.iter().sum(), 1.0, 1e-6);
}

#[test]
fn rope_neox_rotates_at_the_requested_position() {
    let mut at_zero = vec![1.0, 0.0];
    rope_neox_inplace(&mut at_zero, 1, 2, 0, 10_000.0);
    assert_eq!(at_zero, vec![1.0, 0.0]);

    let mut at_one = vec![1.0, 0.0];
    rope_neox_inplace(&mut at_one, 1, 2, 1, 10_000.0);
    assert_close(at_one[0], 1.0f32.cos(), 1e-6);
    assert_close(at_one[1], 1.0f32.sin(), 1e-6);
}

#[test]
fn l2_normalize_returns_a_unit_vector() {
    let out = l2_normalize(&[3.0, 4.0]);
    assert_eq!(out, vec![0.6, 0.8]);
    assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
}
