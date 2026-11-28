//! LoRA 特有の解析微分。汎用 autograd tape は使わない。
//!
//! frozen な base の隠れ活性 `x` は呼び出し側でキャッシュして渡す想定。
//! `LoraPair` は A `[in_dim, rank]` と B `[rank, out_dim]` を row-major の
//! `Vec<f32>` で持ち、順伝播は `scale * B(A x)`。

use std::fmt;

/// 学習対象の低ランク行列の組。`a` は `[in_dim, rank]`、`b` は `[rank, out_dim]`。
#[derive(Debug, Clone, PartialEq)]
pub struct LoraPair {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub scale: f32,
}

/// shape 不正や非有限な scale を表すエラー。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradError {
    /// `field` の長さが期待と異なる（または次元が 0）。
    Shape {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidScale(f32),
}

fn shape_error(field: &'static str, expected: usize, actual: usize) -> GradError {
    GradError::Shape {
        field,
        expected,
        actual,
    }
}

fn check_len(field: &'static str, expected: usize, actual: usize) -> Result<(), GradError> {
    if actual == expected {
        Ok(())
    } else {
        Err(shape_error(field, expected, actual))
    }
}

impl fmt::Display for GradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                field,
                expected,
                actual,
            } => write!(
                f,
                "lora_grad: {field} has {actual} elements, expected {expected}"
            ),
            Self::InvalidScale(scale) => {
                write!(f, "lora_grad: scale must be finite, got {scale}")
            }
        }
    }
}

impl std::error::Error for GradError {}

impl LoraPair {
    /// 各配列の長さと scale を検証する。`forward` / `backward` から自動で呼ばれる。
    pub fn validate(&self) -> Result<(), GradError> {
        if self.rank == 0 {
            return Err(shape_error("rank", 1, 0));
        }
        if self.in_dim == 0 {
            return Err(shape_error("in_dim", 1, 0));
        }
        if self.out_dim == 0 {
            return Err(shape_error("out_dim", 1, 0));
        }
        check_len("a", self.in_dim.saturating_mul(self.rank), self.a.len())?;
        check_len("b", self.rank.saturating_mul(self.out_dim), self.b.len())?;
        if !self.scale.is_finite() {
            return Err(GradError::InvalidScale(self.scale));
        }
        Ok(())
    }

    /// `A x`。検証済みの `self` に対してだけ呼ぶ。
    fn project_a(&self, x: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0; self.rank];
        for (i, xi) in x.iter().enumerate() {
            let row = &self.a[i * self.rank..(i + 1) * self.rank];
            for (h, ar) in hidden.iter_mut().zip(row) {
                *h += ar * xi;
            }
        }
        hidden
    }

    /// `Bᵀ dout`。検証済みの `self` に対してだけ呼ぶ。
    fn project_bt(&self, dout: &[f32]) -> Vec<f32> {
        let mut back = vec![0.0; self.rank];
        for (r, value) in back.iter_mut().enumerate() {
            let row = &self.b[r * self.out_dim..(r + 1) * self.out_dim];
            for (b, d) in row.iter().zip(dout) {
                *value += b * d;
            }
        }
        back
    }

    /// `scale * B(A x)`。
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, GradError> {
        self.validate()?;
        check_len("x", self.in_dim, x.len())?;

        let hidden = self.project_a(x);
        let mut out = vec![0.0; self.out_dim];
        for (r, h) in hidden.iter().enumerate() {
            let row = &self.b[r * self.out_dim..(r + 1) * self.out_dim];
            for (y, b) in out.iter_mut().zip(row) {
                *y += b * h;
            }
        }
        for y in &mut out {
            *y *= self.scale;
        }
        Ok(out)
    }

    /// `⟨forward(x), dout⟩` の勾配。
    ///
    /// - `dA[i, r] = scale * (Bᵀ dout)[r] * x[i]`
    /// - `dB[r, o] = scale * dout[o] * (A x)[r]`
    /// - `dx[i]    = scale * Σ_r A[i, r] * (Bᵀ dout)[r]`
    pub fn backward(&self, x: &[f32], dout: &[f32]) -> Result<LoraGradients, GradError> {
        self.validate()?;
        check_len("x", self.in_dim, x.len())?;
        check_len("dout", self.out_dim, dout.len())?;

        let hidden = self.project_a(x);
        let back = self.project_bt(dout);

        let mut da = vec![0.0; self.in_dim * self.rank];
        for (i, xi) in x.iter().enumerate() {
            for (slot, br) in da[i * self.rank..(i + 1) * self.rank].iter_mut().zip(&back) {
                *slot = self.scale * br * xi;
            }
        }

        let mut db = vec![0.0; self.rank * self.out_dim];
        for (r, h) in hidden.iter().enumerate() {
            for (slot, d) in db[r * self.out_dim..(r + 1) * self.out_dim]
                .iter_mut()
                .zip(dout)
            {
                *slot = self.scale * d * h;
            }
        }

        let mut dx = vec![0.0; self.in_dim];
        for (i, value) in dx.iter_mut().enumerate() {
            let row = &self.a[i * self.rank..(i + 1) * self.rank];
            let mut sum = 0.0;
            for (ar, br) in row.iter().zip(&back) {
                sum += ar * br;
            }
            *value = self.scale * sum;
        }

        Ok(LoraGradients { da, db, dx })
    }
}

/// Analytic gradients of one LoRA projection.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraGradients {
    /// `[in_dim, rank]`, row-major.
    pub da: Vec<f32>,
    /// `[rank, out_dim]`, row-major.
    pub db: Vec<f32>,
    /// `[in_dim]`; only needed when propagating through a frozen base.
    pub dx: Vec<f32>,
}

/// log-sum-exp 安定化した softmax。非有限な logits は一様分布にフォールバックする。
fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max = if max.is_finite() { max } else { 0.0 };
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// `-log softmax(logits)[target]` の勾配 `softmax(logits) - onehot(target)`。
/// `target` が範囲外ならゼロ勾配を返す。
pub fn cross_entropy_grad(logits: &[f32], target: usize) -> Vec<f32> {
    let mut grad = stable_softmax(logits);
    let Some(slot) = grad.get_mut(target) else {
        return vec![0.0; logits.len()];
    };
    *slot -= 1.0;
    grad
}

/// GRPO のサンプル勾配 `advantage * (softmax(logits) - onehot(sampled))`。
/// `sampled` が範囲外、または `advantage` が 0 / 非有限ならゼロ勾配を返す。
pub fn grpo_grad(logits: &[f32], sampled: usize, advantage: f32) -> Vec<f32> {
    if !advantage.is_finite() || advantage == 0.0 {
        return vec![0.0; logits.len()];
    }
    let mut grad = stable_softmax(logits);
    let Some(slot) = grad.get_mut(sampled) else {
        return vec![0.0; logits.len()];
    };
    *slot -= 1.0;
    grad.into_iter().map(|g| g * advantage).collect()
}

#[cfg(test)]
mod tests {
    use super::{GradError, LoraPair, cross_entropy_grad, grpo_grad};

    /// 手計算で求められる小さな A/B。A は [in=3, rank=2]、B は [rank=2, out=2]。
    fn sample() -> LoraPair {
        LoraPair {
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![0.5, -1.0, 2.0, 0.25],
            in_dim: 3,
            out_dim: 2,
            rank: 2,
            scale: 0.5,
        }
    }

    fn with_a(pair: &LoraPair, a: Vec<f32>) -> LoraPair {
        LoraPair { a, ..pair.clone() }
    }

    fn with_b(pair: &LoraPair, b: Vec<f32>) -> LoraPair {
        LoraPair { b, ..pair.clone() }
    }

    /// dout を固定したスカラー損失 ⟨forward(x), dout⟩。
    fn loss(pair: &LoraPair, x: &[f32], dout: &[f32]) -> f32 {
        pair.forward(x)
            .unwrap()
            .iter()
            .zip(dout)
            .map(|(y, d)| y * d)
            .sum()
    }

    fn assert_close(actual: f32, expected: f32, label: &str) {
        assert!(
            (actual - expected).abs() < 1e-2,
            "{label}: analytic={actual}, expected={expected}"
        );
    }

    #[test]
    fn forward_matches_hand_computation() {
        let pair = sample();
        let x = [0.2, -0.3, 0.4];
        // A は row-major: Ax = [1*0.2+3*(-0.3)+5*0.4, 2*0.2+4*(-0.3)+6*0.4]
        //                   = [1.3, 1.6]
        // B も row-major: B(Ax) = [0.5*1.3+2*1.6, -1*1.3+0.25*1.6]
        //                      = [3.85, -0.9], scale = 0.5
        let expected = [1.925, -0.45];
        let got = pair.forward(&x).unwrap();
        assert_eq!(got.len(), pair.out_dim);
        for (actual, expected) in got.iter().zip(expected) {
            assert_close(*actual, expected, "forward");
        }
    }

    #[test]
    fn backward_parameter_grads_match_central_difference() {
        let pair = sample();
        let x = [0.2, -0.3, 0.4];
        let dout = [0.3, -0.8];
        let grads = pair.backward(&x, &dout).unwrap();
        let da = grads.da;
        let db = grads.db;
        assert_eq!(da.len(), pair.in_dim * pair.rank);
        assert_eq!(db.len(), pair.rank * pair.out_dim);

        let eps = 1e-3;
        for (i, analytic) in da.iter().enumerate() {
            let mut plus = pair.a.clone();
            plus[i] += eps;
            let mut minus = pair.a.clone();
            minus[i] -= eps;
            let numeric = (loss(&with_a(&pair, plus), &x, &dout)
                - loss(&with_a(&pair, minus), &x, &dout))
                / (2.0 * eps);
            assert_close(*analytic, numeric, &format!("dA[{i}]"));
        }
        for (i, analytic) in db.iter().enumerate() {
            let mut plus = pair.b.clone();
            plus[i] += eps;
            let mut minus = pair.b.clone();
            minus[i] -= eps;
            let numeric = (loss(&with_b(&pair, plus), &x, &dout)
                - loss(&with_b(&pair, minus), &x, &dout))
                / (2.0 * eps);
            assert_close(*analytic, numeric, &format!("dB[{i}]"));
        }
    }

    #[test]
    fn backward_input_grad_matches_central_difference() {
        let pair = sample();
        let x = [0.2, -0.3, 0.4];
        let dout = [0.3, -0.8];
        let dx = pair.backward(&x, &dout).unwrap().dx;
        assert_eq!(dx.len(), pair.in_dim);

        let eps = 1e-3;
        for (i, analytic) in dx.iter().enumerate() {
            let mut plus = x.to_vec();
            plus[i] += eps;
            let mut minus = x.to_vec();
            minus[i] -= eps;
            let numeric = (loss(&pair, &plus, &dout) - loss(&pair, &minus, &dout)) / (2.0 * eps);
            assert_close(*analytic, numeric, &format!("dx[{i}]"));
        }
    }

    #[test]
    fn cross_entropy_grad_is_finite_and_sums_to_zero() {
        let grad = cross_entropy_grad(&[2.0, 1.0, -1.0], 1);
        assert_eq!(grad.len(), 3);
        assert!(grad.iter().all(|g| g.is_finite()), "grad={grad:?}");
        assert!(grad.iter().sum::<f32>().abs() < 1e-5, "grad={grad:?}");
        assert!(grad[1] < 0.0, "target must get a negative gradient");

        // log-sum-exp 安定化: 巨大な logits でも有限で合計 0。
        let huge = cross_entropy_grad(&[1.0e30, -1.0e30, 1000.0, -1000.0], 0);
        assert!(huge.iter().all(|g| g.is_finite()), "grad={huge:?}");
        assert!(huge.iter().sum::<f32>().abs() < 1e-5, "grad={huge:?}");
    }

    #[test]
    fn grpo_grad_scales_by_advantage() {
        let logits = [2.0, 1.0, -1.0];
        let zero = grpo_grad(&logits, 0, 0.0);
        assert_eq!(zero, vec![0.0; logits.len()]);

        let positive = grpo_grad(&logits, 1, 2.0);
        assert_eq!(positive.len(), logits.len());
        assert!(positive.iter().all(|g| g.is_finite()));
        assert!(positive[1] < 0.0, "sampled action must be pushed down");
        assert!(positive.iter().sum::<f32>().abs() < 1e-5);

        let negative = grpo_grad(&logits, 1, -2.0);
        assert!(negative[1] > 0.0, "negative advantage pushes sampled up");
        for (p, n) in positive.iter().zip(&negative) {
            assert_close(*p, -*n, "advantage sign");
        }
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        let pair = sample();
        let x = [0.2, -0.3, 0.4];
        let dout = [0.3, -0.8];

        let bad_a = with_a(&pair, vec![1.0, 2.0]);
        assert!(matches!(
            bad_a.forward(&x),
            Err(GradError::Shape { field: "a", .. })
        ));
        let bad_b = with_b(&pair, vec![1.0, 2.0]);
        assert!(matches!(
            bad_b.backward(&x, &dout),
            Err(GradError::Shape { field: "b", .. })
        ));
        assert!(matches!(
            pair.forward(&x[..2]),
            Err(GradError::Shape { field: "x", .. })
        ));
        assert!(matches!(
            pair.backward(&x, &dout[..1]),
            Err(GradError::Shape { field: "dout", .. })
        ));
        let zero_rank = LoraPair {
            rank: 0,
            ..sample()
        };
        assert!(matches!(
            zero_rank.forward(&x),
            Err(GradError::Shape { field: "rank", .. })
        ));
    }

    #[test]
    fn non_finite_scale_is_rejected() {
        let pair = LoraPair {
            scale: f32::NAN,
            ..sample()
        };
        assert!(matches!(
            pair.forward(&[0.2, -0.3, 0.4]),
            Err(GradError::InvalidScale(_))
        ));
    }

    #[test]
    fn loss_grads_are_ignored_for_out_of_range_target() {
        let grad = cross_entropy_grad(&[1.0, 2.0], 9);
        assert_eq!(grad, vec![0.0, 0.0]);
    }
}
