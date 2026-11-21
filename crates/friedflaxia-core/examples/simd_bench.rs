//! Micro-benchmarks for the SIMD kernels:
//! `cargo run --release --example simd_bench`.

use friedflaxia_core::quant::{matvec_bf16, matvec_f32, matvec_q4_0, matvec_q8_0};
use friedflaxia_core::simd;
use std::time::Instant;

fn dot_bench() {
    let len = 1 << 20;
    let a: Vec<f32> = (0..len).map(|i| (i % 97) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..len).map(|i| (i % 31) as f32 * 0.02).collect();
    let iterations = 200;

    for enabled in [false, true] {
        simd::set_enabled(enabled);
        // Warm up and keep the compiler from eliding the loop.
        let mut sink = 0.0f32;
        for _ in 0..10 {
            sink += simd::dot(&a, &b);
        }
        let started = Instant::now();
        for _ in 0..iterations {
            sink += simd::dot(&a, &b);
        }
        let elapsed = started.elapsed();
        let label = if enabled { "simd  " } else { "scalar" };
        println!(
            "dot {label}: {:.2?} for {iterations} x {len} ({:.2?}/dot, sink {sink:.1})",
            elapsed,
            elapsed / iterations
        );
    }
    simd::set_enabled(true);
}

fn matvec_bench(
    name: &str,
    rows: usize,
    cols: usize,
    weights: &[u8],
    run: fn(&[u8], usize, usize, &[f32]) -> Vec<f32>,
) {
    let input: Vec<f32> = (0..cols).map(|i| ((i % 29) as f32 - 14.0) / 14.0).collect();
    let iterations = 20;
    for enabled in [false, true] {
        simd::set_enabled(enabled);
        let mut sink = 0.0f32;
        for _ in 0..3 {
            sink += run(weights, rows, cols, &input).iter().sum::<f32>();
        }
        let started = Instant::now();
        for _ in 0..iterations {
            sink += run(weights, rows, cols, &input).iter().sum::<f32>();
        }
        let elapsed = started.elapsed();
        let label = if enabled { "simd  " } else { "scalar" };
        println!(
            "{name} {label}: {:.2?} for {iterations} x {rows}x{cols} ({:.2?}/matvec, sink {sink:.1})",
            elapsed,
            elapsed / iterations
        );
    }
    simd::set_enabled(true);
}

fn main() {
    dot_bench();

    let rows = 768;
    let cols = 768;

    let f32_weights: Vec<u8> = (0..rows * cols)
        .flat_map(|i| ((i % 61) as f32 * 0.01 - 0.3).to_le_bytes())
        .collect();
    matvec_bench("f32 ", rows, cols, &f32_weights, |w, r, c, i| {
        matvec_f32(w, r, c, i).unwrap()
    });

    let bf16_weights: Vec<u8> = (0..rows * cols)
        .flat_map(|i| {
            let value = (i % 61) as f32 * 0.01 - 0.3;
            let bits = (value.to_bits() >> 16) as u16;
            bits.to_le_bytes()
        })
        .collect();
    matvec_bench("bf16", rows, cols, &bf16_weights, |w, r, c, i| {
        matvec_bf16(w, r, c, i).unwrap()
    });

    let q8_weights: Vec<u8> = (0..rows * (cols / 32))
        .flat_map(|block| {
            let mut bytes = Vec::from(0x3c00u16.to_le_bytes());
            for i in 0..32 {
                bytes.push((((block * 32 + i) % 33) as i8 - 16) as u8);
            }
            bytes
        })
        .collect();
    matvec_bench("q8_0", rows, cols, &q8_weights, |w, r, c, i| {
        matvec_q8_0(w, r, c, i).unwrap()
    });

    let q4_weights: Vec<u8> = (0..rows * (cols / 32))
        .flat_map(|block| {
            let mut bytes = Vec::from(0x3c00u16.to_le_bytes());
            for i in 0..16 {
                bytes.push(((block * 16 + i) % 256) as u8);
            }
            bytes
        })
        .collect();
    matvec_bench("q4_0", rows, cols, &q4_weights, |w, r, c, i| {
        matvec_q4_0(w, r, c, i).unwrap()
    });
}
