//! Deterministic counter-based RNG for offline ES / federated LoRA training.
//!
//! `Math.random()` and `crypto.getRandomValues()` are deliberately avoided:
//! the host must reconstruct the exact same perturbation vector from a seed,
//! possibly in a different runtime. SplitMix64 is a small integer algorithm
//! that is trivial to implement identically in Rust, JS and WASM.

/// SplitMix64 state. All arithmetic is wrapping u64.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, derived from the top 24 bits of a u64.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    /// Standard normal via Box–Muller. Two u64 draws per sample, so the
    /// sequence is reproducible across runtimes.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1.0 / 16_777_216.0);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (core::f64::consts::TAU as f32 * u2).cos()
    }

    /// Fill `out` with N(0, 1) samples. The seed and length fully determine it.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for value in out {
            *value = self.next_normal();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn known_splitmix64_vectors() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        let mut rng = SplitMix64::new(42);
        assert_eq!(rng.next_u64(), 0xBDD7_3226_2FEB_6E95);
    }

    #[test]
    fn same_seed_same_perturbation() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        let mut va = vec![0.0; 128];
        let mut vb = vec![0.0; 128];
        a.fill_normal(&mut va);
        b.fill_normal(&mut vb);
        assert_eq!(va, vb);
        assert!(va.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn different_seed_different_perturbation() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let mut va = vec![0.0; 64];
        let mut vb = vec![0.0; 64];
        a.fill_normal(&mut va);
        b.fill_normal(&mut vb);
        assert_ne!(va, vb);
    }
}
