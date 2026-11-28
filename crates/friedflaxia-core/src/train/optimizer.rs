//! Self-contained AdamW optimizer for offline training.
//!
//! No external optimizer crate: plain `f32` vectors are updated in place with
//! bias-corrected first/second moments and decoupled weight decay. Fully
//! deterministic — the same inputs always produce the same parameters.

/// Reasons a single optimizer step can be rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimError {
    /// `params`, `grads` and the optimizer's moment buffers disagree in length.
    LengthMismatch,
    /// At least one gradient is NaN or infinite.
    NonFiniteGradient,
}

impl core::fmt::Display for OptimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::LengthMismatch => "optimizer: params/grads/moment lengths differ",
            Self::NonFiniteGradient => "optimizer: gradient must be finite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for OptimError {}

/// AdamW state for a flat parameter vector.
#[derive(Debug, Clone, PartialEq)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Number of successfully applied steps (1-based inside `step`).
    pub step_count: u64,
    /// First moment, same length as the parameter vector.
    pub m: Vec<f32>,
    /// Second moment, same length as the parameter vector.
    pub v: Vec<f32>,
}

impl AdamW {
    /// Create an optimizer for `len` parameters with the standard defaults:
    /// `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.
    pub fn new(len: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    /// Apply one update in place:
    /// `w -= lr · m̂ / (sqrt(v̂) + eps) + lr · weight_decay · w`.
    ///
    /// Validation happens before any mutation, so a rejected step leaves
    /// `params`, the moment buffers and `step_count` untouched.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), OptimError> {
        let len = self.m.len();
        if params.len() != len || grads.len() != len || self.v.len() != len {
            return Err(OptimError::LengthMismatch);
        }
        if grads.iter().any(|g| !g.is_finite()) {
            return Err(OptimError::NonFiniteGradient);
        }

        self.step_count += 1;
        let t = self.step_count as f32;
        let (lr, beta1, beta2, eps, weight_decay) =
            (self.lr, self.beta1, self.beta2, self.eps, self.weight_decay);
        let bias1 = 1.0 - beta1.powf(t);
        let bias2 = 1.0 - beta2.powf(t);
        let moments = self.m.iter_mut().zip(self.v.iter_mut());

        for ((param, grad), (m_slot, v_slot)) in params.iter_mut().zip(grads).zip(moments) {
            let m = beta1 * *m_slot + (1.0 - beta1) * grad;
            let v = beta2 * *v_slot + (1.0 - beta2) * grad * grad;
            *m_slot = m;
            *v_slot = v;

            let m_hat = m / bias1;
            let v_hat = v / bias2;
            *param -= lr * m_hat / (v_hat.sqrt() + eps) + lr * weight_decay * *param;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{AdamW, OptimError};

    fn quadratic_loss(params: &[f32], target: &[f32]) -> f32 {
        params
            .iter()
            .zip(target)
            .map(|(p, t)| {
                let d = p - t;
                d * d
            })
            .sum()
    }

    #[test]
    fn new_uses_defaults() {
        let opt = AdamW::new(3, 0.05, 0.01);
        assert_eq!(opt.lr, 0.05);
        assert_eq!(opt.beta1, 0.9);
        assert_eq!(opt.beta2, 0.999);
        assert_eq!(opt.eps, 1e-8);
        assert_eq!(opt.weight_decay, 0.01);
        assert_eq!(opt.step_count, 0);
        assert_eq!(opt.m, vec![0.0; 3]);
        assert_eq!(opt.v, vec![0.0; 3]);
    }

    #[test]
    fn converges_on_quadratic_in_200_steps() {
        // f(w) = Σ (w - target)², gradient = 2 (w - target).
        let target = [0.5_f32, -1.0, 2.0, 1.5];
        let mut params = vec![0.0; target.len()];
        let mut opt = AdamW::new(target.len(), 0.05, 0.0);
        for _ in 0..200 {
            let grads: Vec<f32> = params
                .iter()
                .zip(&target)
                .map(|(p, t)| 2.0 * (p - t))
                .collect();
            opt.step(&mut params, &grads).unwrap();
        }
        assert!(
            quadratic_loss(&params, &target) <= 1e-3,
            "loss too high: {:?} (loss {})",
            params,
            quadratic_loss(&params, &target)
        );
    }

    #[test]
    fn weight_decay_shrinks_params_toward_zero() {
        let mut params = vec![1.0_f32, -2.0, 0.5];
        let grads = vec![0.0; params.len()];
        let mut opt = AdamW::new(params.len(), 0.1, 0.5);
        for _ in 0..200 {
            opt.step(&mut params, &grads).unwrap();
        }
        assert!(
            params.iter().all(|p| p.abs() < 1e-3),
            "weight decay did not shrink params: {:?}",
            params
        );
    }

    #[test]
    fn non_finite_gradient_is_rejected_without_mutation() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut params = vec![0.5_f32, -0.5];
            let grads = vec![1.0_f32, bad];
            let mut opt = AdamW::new(params.len(), 0.05, 0.0);
            opt.step(&mut vec![0.0; 2], &[1.0, 1.0]).unwrap(); // seed state

            let before_params = params.clone();
            let before_m = opt.m.clone();
            let before_v = opt.v.clone();
            assert_eq!(
                opt.step(&mut params, &grads),
                Err(OptimError::NonFiniteGradient)
            );
            assert_eq!(params, before_params);
            assert_eq!(opt.m, before_m);
            assert_eq!(opt.v, before_v);
            assert_eq!(opt.step_count, 1);
        }
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let mut opt = AdamW::new(3, 0.05, 0.0);

        // grads shorter than params.
        assert_eq!(
            opt.step(&mut [0.0, 0.0, 0.0], &[1.0, 1.0]),
            Err(OptimError::LengthMismatch)
        );
        // grads longer than params.
        assert_eq!(
            opt.step(&mut [0.0], &[1.0, 1.0]),
            Err(OptimError::LengthMismatch)
        );
        // params disagree with the optimizer's internal state length.
        assert_eq!(
            opt.step(&mut [0.0, 0.0], &[1.0, 1.0]),
            Err(OptimError::LengthMismatch)
        );
        assert_eq!(opt.step_count, 0);
    }

    #[test]
    fn same_inputs_give_identical_results() {
        let initial = vec![0.1_f32, -0.2, 0.3];
        let grads: Vec<f32> = (1..=3).map(|i| (i as f32) * 0.01).collect();
        let mut a = AdamW::new(initial.len(), 0.02, 0.05);
        let mut b = AdamW::new(initial.len(), 0.02, 0.05);
        let mut pa = initial.clone();
        let mut pb = initial.clone();
        for _ in 0..20 {
            a.step(&mut pa, &grads).unwrap();
            b.step(&mut pb, &grads).unwrap();
        }
        assert_eq!(pa, pb);
        assert_eq!(a.m, b.m);
        assert_eq!(a.v, b.v);
        assert_eq!(a.step_count, b.step_count);
        assert!(pa.iter().any(|p| *p != 0.0));
    }
}
