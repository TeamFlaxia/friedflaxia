//! Vectorized kernels must stay within a tight tolerance of the scalar path.

use friedflaxia_core::quant::{matvec_q4_0, matvec_q8_0};
use friedflaxia_core::simd;
use std::sync::Mutex;

/// `set_enabled` is process-global, so the parity tests must not overlap.
static SIMD_LOCK: Mutex<()> = Mutex::new(());

fn q8_0_block(scale_bits: u16, values: &[i8; 32]) -> Vec<u8> {
    let mut block = Vec::from(scale_bits.to_le_bytes());
    block.extend(values.iter().map(|value| *value as u8));
    block
}

fn q4_0_block(scale_bits: u16, packed: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::from(scale_bits.to_le_bytes());
    block.extend_from_slice(packed);
    block
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        let tolerance = 1e-3 * (1.0 + a.abs().max(b.abs()));
        assert!((a - b).abs() <= tolerance, "row {index}: {a} != {b}");
    }
}

#[test]
fn q8_0_simd_matches_scalar() {
    let _guard = SIMD_LOCK.lock().unwrap();
    let input: Vec<f32> = (0..64).map(|index| (index as f32 - 32.0) / 32.0).collect();
    let mut weights = Vec::new();
    for block in 0..6 {
        let values: Vec<i8> = (0..32).map(|i| ((block * 32 + i) % 17) as i8 - 8).collect();
        weights.extend(q8_0_block(0x3c00, &values.try_into().unwrap()));
    }

    simd::set_enabled(true);
    let vectorized = matvec_q8_0(&weights, 3, 64, &input).unwrap();
    simd::set_enabled(false);
    let scalar = matvec_q8_0(&weights, 3, 64, &input).unwrap();
    simd::set_enabled(true);

    assert_close(&vectorized, &scalar);
}

#[test]
fn q4_0_simd_matches_scalar() {
    let _guard = SIMD_LOCK.lock().unwrap();
    let input: Vec<f32> = (0..64).map(|index| (index as f32 - 32.0) / 32.0).collect();
    let mut weights = Vec::new();
    for _ in 0..8 {
        weights.extend(q4_0_block(0x3c00, &[0x79; 16]));
    }

    simd::set_enabled(true);
    let vectorized = matvec_q4_0(&weights, 4, 64, &input).unwrap();
    simd::set_enabled(false);
    let scalar = matvec_q4_0(&weights, 4, 64, &input).unwrap();
    simd::set_enabled(true);

    assert_close(&vectorized, &scalar);
}

#[test]
fn dot_handles_tail_elements() {
    let _guard = SIMD_LOCK.lock().unwrap();
    let a: Vec<f32> = (0..19).map(|index| index as f32 * 0.25).collect();
    let b: Vec<f32> = (0..19).map(|index| 1.0 - index as f32 * 0.01).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    simd::set_enabled(true);
    let vectorized = friedflaxia_core::simd::dot(&a, &b);
    simd::set_enabled(false);
    let scalar = friedflaxia_core::simd::dot(&a, &b);
    simd::set_enabled(true);

    assert!((vectorized - expected).abs() < 1e-4);
    assert!((scalar - expected).abs() < 1e-4);
}
