/// Decode an IEEE-754 binary16 value to binary32.
///
/// The GGML block formats store their scale as an f16, so this is on the hot
/// path for every dequantized block. The implementation is branch-light for
/// normal values and only does the normalization loop for subnormals.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // f16 subnormal: value = mantissa * 2^-24.
            let mut e = 113u32;
            let mut m = mantissa;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        // Infinity or NaN: widen the mantissa and keep the payload.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

/// Encode an f32 as IEEE-754 binary16 bits, rounding to nearest (ties to even).
///
/// Finite values beyond the f16 range saturate to the largest finite magnitude
/// instead of becoming infinity. GGUF block scales use this so a finite input
/// can never produce an inf/NaN scale. NaN/inf inputs are passed through as f16
/// NaN/inf; callers that require finite scales must validate them first.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        return sign | 0x7e00 | ((mantissa >> 13) as u16 & 0x01ff);
    }

    let mut biased = exponent - 127 + 15;
    if biased >= 31 {
        return sign | 0x7bff;
    }

    if biased >= 1 {
        // Round the 23-bit mantissa to 10 bits, carrying into the exponent on
        // overflow (e.g. 0.99999 rounds up to 1.0).
        let rounded = (mantissa + 0x1000 + ((mantissa >> 13) & 1)) >> 13;
        if rounded == 0x400 {
            biased += 1;
            if biased >= 31 {
                return sign | 0x7bff;
            }
            return sign | ((biased as u16) << 10);
        }
        return sign | ((biased as u16) << 10) | rounded as u16;
    }

    // Subnormal / zero: value = (0x800000 | mantissa) * 2^(biased - 38) and
    // one f16 subnormal unit is 2^-24, so shift right by 14 - biased.
    let shift = 14 - biased;
    if shift >= 25 {
        return sign;
    }
    let shift = shift as u32;
    let significand = 0x0080_0000 | mantissa;
    let mut half = (significand >> shift) as u16;
    let remainder = significand & ((1u32 << shift) - 1);
    let midpoint = 1u32 << (shift - 1);
    if remainder > midpoint || (remainder == midpoint && half & 1 != 0) {
        half += 1;
    }
    sign | half
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    #[test]
    fn f16_round_trip_keeps_representable_values() {
        for value in [0.0, -0.0, 0.5, -0.125, 1.0, 65504.0, 2f32.powi(-14)] {
            let bits = f32_to_f16(value);
            assert_eq!(f16_to_f32(bits), value, "bits {bits:#06x}");
        }
    }

    #[test]
    fn f16_encoding_saturates_instead_of_producing_infinity() {
        for value in [f32::MAX, -f32::MAX, 1e30] {
            let bits = f32_to_f16(value);
            assert_eq!(bits & 0x7c00, 0x7bff & 0x7c00, "bits {bits:#06x}");
            assert!(f16_to_f32(bits).is_finite());
        }
        assert_eq!(f32_to_f16(f32::MIN_POSITIVE), 0);
    }
}
