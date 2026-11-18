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
