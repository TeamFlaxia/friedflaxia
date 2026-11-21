//! Vectorized dot products with a scalar fallback.
//!
//! The public kernel signatures in `quant::matvec` and `ops` stay unchanged.
//! SIMD is used for the reduction-heavy loops on x86_64 (AVX2), aarch64 (NEON)
//! and wasm32 (`simd128`). A runtime switch lets hosts fall back to the exact
//! scalar path when a golden comparison needs bit-identical reductions.
//!
//! Note: the wasm build script adds `-C target-feature=+simd128`. A wasm module
//! built with SIMD instructions will not run in engines without the feature,
//! so the shipped artifact and the feature flag must always be updated together.

use std::sync::atomic::{AtomicBool, Ordering};

static SIMD_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable vectorized kernels at runtime.
pub fn set_enabled(enabled: bool) {
    SIMD_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether vectorized kernels may be used.
pub fn is_enabled() -> bool {
    SIMD_ENABLED.load(Ordering::Relaxed)
}

/// Dot product of two equal-length slices.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if !is_enabled() || a.len() < 8 {
        return dot_scalar(a, b);
    }
    #[cfg(target_arch = "wasm32")]
    {
        // `simd128` is enabled by build-wasm.sh and checked in CI.
        unsafe { dot_wasm(a.as_ptr(), b, a.len()) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { dot_avx2(a.as_ptr(), b, a.len()) }
        } else {
            dot_scalar(a, b)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a.as_ptr(), b, a.len()) }
    }
    #[cfg(not(any(
        target_arch = "wasm32",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        dot_scalar(a, b)
    }
}

/// Dot product of little-endian `f32` bytes with an `f32` slice.
///
/// GGUF stores tensors as bytes, so loading directly from the unaligned weight
/// buffer avoids a full dequantization pass. Big-endian hosts use the scalar
/// path.
pub fn dot_f32_le_bytes(bytes: &[u8], b: &[f32]) -> f32 {
    debug_assert_eq!(bytes.len(), b.len() * 4);
    let len = bytes.len() / 4;
    if !is_enabled() || len < 8 || cfg!(target_endian = "big") {
        return bytes
            .chunks_exact(4)
            .zip(b)
            .map(|(chunk, value)| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * value
            })
            .sum();
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { dot_wasm(bytes.as_ptr().cast(), b, len) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { dot_avx2(bytes.as_ptr().cast(), b, len) }
        } else {
            dot_scalar_bytes(bytes, b)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(bytes.as_ptr().cast(), b, len) }
    }
    #[cfg(not(any(
        target_arch = "wasm32",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        dot_scalar_bytes(bytes, b)
    }
}

/// Sum of a slice, using the same vector width as [`dot`].
pub fn sum(a: &[f32]) -> f32 {
    if !is_enabled() || a.len() < 8 {
        return a.iter().sum();
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { sum_wasm(a) }
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { sum_avx2(a) }
        } else {
            a.iter().sum()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sum_neon(a) }
    }
    #[cfg(not(any(
        target_arch = "wasm32",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        a.iter().sum()
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(any(
    not(any(
        target_arch = "wasm32",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )),
    all(target_arch = "x86_64", not(target_arch = "wasm32"))
))]
fn dot_scalar_bytes(bytes: &[u8], b: &[f32]) -> f32 {
    bytes
        .chunks_exact(4)
        .zip(b)
        .map(|(chunk, value)| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) * value)
        .sum()
}

#[cfg(target_arch = "wasm32")]
unsafe fn dot_wasm(a: *const f32, b: &[f32], len: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, v128_load};

    unsafe {
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        // Four independent accumulators hide the horizontal-reduction latency.
        let mut index = 0;
        while index + 16 <= len {
            let a0 = f32x4_mul(
                v128_load(a.add(index).cast()),
                v128_load(b.as_ptr().add(index).cast()),
            );
            let a1 = f32x4_mul(
                v128_load(a.add(index + 4).cast()),
                v128_load(b.as_ptr().add(index + 4).cast()),
            );
            let a2 = f32x4_mul(
                v128_load(a.add(index + 8).cast()),
                v128_load(b.as_ptr().add(index + 8).cast()),
            );
            let a3 = f32x4_mul(
                v128_load(a.add(index + 12).cast()),
                v128_load(b.as_ptr().add(index + 12).cast()),
            );
            let sum = f32x4_add(f32x4_add(a0, a1), f32x4_add(a2, a3));
            acc0 += f32x4_extract_lane::<0>(sum);
            acc1 += f32x4_extract_lane::<1>(sum);
            acc2 += f32x4_extract_lane::<2>(sum);
            acc3 += f32x4_extract_lane::<3>(sum);
            index += 16;
        }
        let mut total = (acc0 + acc1) + (acc2 + acc3);
        while index < len {
            total += *a.add(index) * *b.get_unchecked(index);
            index += 1;
        }
        total
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn sum_wasm(a: &[f32]) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, v128_load};

    unsafe {
        let len = a.len();
        let mut acc = v128_load(a.as_ptr().cast());
        let mut index = 4;
        while index + 4 <= len {
            acc = f32x4_add(acc, v128_load(a.as_ptr().add(index).cast()));
            index += 4;
        }
        let mut total = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while index < len {
            total += *a.get_unchecked(index);
            index += 1;
        }
        total
    }
}

#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: *const f32, b: &[f32], len: usize) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    unsafe {
        let chunks = len / 8;
        let mut acc = _mm256_setzero_ps();
        for index in 0..chunks {
            let offset = index * 8;
            let left = _mm256_loadu_ps(a.add(offset));
            let right = _mm256_loadu_ps(b.as_ptr().add(offset));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(left, right));
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        let mut index = chunks * 8;
        while index < len {
            total += *a.add(index) * *b.get_unchecked(index);
            index += 1;
        }
        total
    }
}

#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(a: &[f32]) -> f32 {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let mut acc = _mm256_setzero_ps();
        for index in 0..chunks {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(a.as_ptr().add(index * 8)));
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        let mut index = chunks * 8;
        while index < len {
            total += *a.get_unchecked(index);
            index += 1;
        }
        total
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: *const f32, b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vld1q_f32, vmulq_f32};

    unsafe {
        let chunks = len / 4;
        let mut acc = vld1q_f32([0.0; 4].as_ptr());
        for index in 0..chunks {
            let offset = index * 4;
            let left = vld1q_f32(a.add(offset));
            let right = vld1q_f32(b.as_ptr().add(offset));
            acc = vaddq_f32(acc, vmulq_f32(left, right));
        }
        let mut total = vaddvq_f32(acc);
        let mut index = chunks * 4;
        while index < len {
            total += *a.add(index) * *b.get_unchecked(index);
            index += 1;
        }
        total
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sum_neon(a: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vld1q_f32};

    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let mut acc = vld1q_f32([0.0; 4].as_ptr());
        for index in 0..chunks {
            acc = vaddq_f32(acc, vld1q_f32(a.as_ptr().add(index * 4)));
        }
        let mut total = vaddvq_f32(acc);
        let mut index = chunks * 4;
        while index < len {
            total += *a.get_unchecked(index);
            index += 1;
        }
        total
    }
}
