//! 自前の最小 autograd tape。外部 ML クレートは使わない。
//!
//! `Tensor` は f32 のフラットな Vec と shape を持つリーフ／中間値、`Tape` は
//! forward を記録して逆伝播と SGD 更新を行う。GGUF の frozen 重みは
//! `Tensor::constant` として渡し、dequant した f32 の LoRA A/B だけを
//! `Tensor::from_vec` / `zeros` で学習対象にする想定。

use core::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn fresh_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// `zeros` / `reshape` が `usize` と `[usize; N]` の両方を受け取れるようにする。
pub trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// f32 テンソル。データは `Rc<RefCell<_>>` で共有され、`backward` が
/// リーフの勾配とパラメータ値を更新する。
#[derive(Debug, Clone)]
pub struct Tensor {
    id: usize,
    data: Rc<RefCell<Vec<f32>>>,
    grad: Rc<RefCell<Option<Vec<f32>>>>,
    shape: Vec<usize>,
    requires_grad: bool,
    is_leaf: bool,
}

impl Tensor {
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool, is_leaf: bool) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "tensor data length must match shape"
        );
        Self {
            id: fresh_id(),
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            shape,
            requires_grad,
            is_leaf,
        }
    }

    /// 学習対象のリーフ（例: dequant した LoRA A/B）。
    pub fn from_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        Self::new(data, vec![len], true, true)
    }

    /// 学習対象のゼロ初期化リーフ（例: bias）。shape は `3` でも `[2, 3]` でも良い。
    pub fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.into_shape();
        Self::new(vec![0.0; numel(&shape)], shape, true, true)
    }

    /// frozen なリーフ（入力 / GGUF から dequant した固定重み）。
    pub fn constant(data: Vec<f32>) -> Self {
        let len = data.len();
        Self::new(data, vec![len], false, true)
    }

    /// shape を切り替える（データ長は不変）。重み行列は `[rows, cols]` を指定する。
    pub fn reshape(mut self, shape: impl IntoShape) -> Self {
        let shape = shape.into_shape();
        assert_eq!(
            numel(&shape),
            self.len(),
            "reshape must preserve the number of elements"
        );
        self.shape = shape;
        self
    }

    pub fn data(&self) -> Vec<f32> {
        self.data.borrow().clone()
    }

    pub fn set_data(&self, data: Vec<f32>) {
        assert_eq!(data.len(), self.len(), "set_data length mismatch");
        *self.data.borrow_mut() = data;
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.grad.borrow().clone()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn intermediate(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Self::new(data, shape, requires_grad, false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardError {
    LossNotScalar,
    NonFiniteLearningRate,
}

impl core::fmt::Display for BackwardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LossNotScalar => f.write_str("backward: loss must be a scalar"),
            Self::NonFiniteLearningRate => f.write_str("backward: learning rate must be finite"),
        }
    }
}

impl std::error::Error for BackwardError {}

/// forward 操作中に必要な値だけをスナップショットとして保持する。
#[derive(Debug)]
enum Op {
    MatVec {
        w_id: usize,
        x_id: usize,
        out_id: usize,
        w: Vec<f32>,
        x: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Add {
        a_id: usize,
        b_id: usize,
        out_id: usize,
        a_len: usize,
        b_len: usize,
    },
    MulScalar {
        a_id: usize,
        out_id: usize,
        a_len: usize,
        scalar: f32,
    },
    Relu {
        a_id: usize,
        out_id: usize,
        a: Vec<f32>,
    },
    Sum {
        a_id: usize,
        out_id: usize,
        a_len: usize,
    },
    Square {
        a_id: usize,
        out_id: usize,
        a: Vec<f32>,
    },
    CrossEntropy {
        logits_id: usize,
        out_id: usize,
        probs: Vec<f32>,
        target: usize,
    },
}

/// log-sum-exp で softmax と log(sum(exp(x))) を同時に求める。
fn stable_softmax_and_lse(logits: &[f32]) -> (f32, Vec<f32>) {
    if logits.is_empty() {
        return (0.0, Vec::new());
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    // 非有限な logits が来ても exp(x - max) が NaN にならないようにする。
    let max = if max.is_finite() { max } else { 0.0 };
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        let probs = exps.iter().map(|e| e / sum).collect();
        (max + sum.ln(), probs)
    } else {
        // すべて -inf などの退化ケース。有限の値にフォールバックする。
        let uniform = 1.0 / logits.len() as f32;
        (0.0, vec![uniform; logits.len()])
    }
}

fn accumulate(grads: &mut HashMap<usize, Vec<f32>>, id: usize, incoming: Vec<f32>) {
    match grads.get_mut(&id) {
        Some(current) => {
            for (dst, src) in current.iter_mut().zip(incoming) {
                *dst += src;
            }
        }
        None => {
            grads.insert(id, incoming);
        }
    }
}

fn take(grads: &mut HashMap<usize, Vec<f32>>, id: usize, len: usize) -> Vec<f32> {
    grads.remove(&id).unwrap_or_else(|| vec![0.0; len])
}

/// 長さ 1 の勾配をブロードキャストする。それ以外はそのまま返す。
fn unbroadcast(g: &[f32], len: usize) -> Vec<f32> {
    if len == 1 && g.len() > 1 {
        vec![g.iter().sum()]
    } else {
        g.to_vec()
    }
}

#[derive(Debug, Default)]
struct TapeInner {
    ops: Vec<Op>,
    nodes: HashMap<usize, Tensor>,
}

/// forward を記録する tape。1 ステップごとに `Tape::new()` を作る想定。
/// 内部可変にしているので `tape.relu(&tape.add(..))` のような入れ子呼び出しができる。
#[derive(Debug, Default)]
pub struct Tape {
    inner: RefCell<TapeInner>,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    /// 記録した操作数。
    pub fn len(&self) -> usize {
        self.inner.borrow().ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.borrow().ops.is_empty()
    }

    fn track(&self, tensor: &Tensor) {
        self.inner
            .borrow_mut()
            .nodes
            .entry(tensor.id)
            .or_insert_with(|| tensor.clone());
    }

    fn output(&self, data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let out = Tensor::intermediate(data, shape, requires_grad);
        self.track(&out);
        out
    }

    fn record(&self, op: Op) {
        self.inner.borrow_mut().ops.push(op);
    }

    /// `weight` は `[rows, cols]`（または長さ rows*cols の 1D）、`input` は `[cols]`。
    pub fn matvec(&self, weight: &Tensor, input: &Tensor) -> Tensor {
        let cols = input.len();
        assert!(cols > 0, "matvec: input must not be empty");
        let rows = if weight.shape().len() == 2 {
            assert_eq!(
                weight.shape()[1], cols,
                "matvec: weight columns must match input length"
            );
            weight.shape()[0]
        } else {
            assert_eq!(
                weight.len() % cols,
                0,
                "matvec: flat weight length must be a multiple of input length"
            );
            weight.len() / cols
        };
        assert_eq!(
            weight.len(),
            rows * cols,
            "matvec: weight length mismatch"
        );

        let w = weight.data();
        let x = input.data();
        let mut y = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                y[i] += w[i * cols + j] * x[j];
            }
        }
        self.track(weight);
        self.track(input);
        let out = self.output(y, vec![rows], weight.requires_grad() || input.requires_grad());
        self.record(Op::MatVec {
            w_id: weight.id,
            x_id: input.id,
            out_id: out.id,
            w,
            x,
            rows,
            cols,
        });
        out
    }

    /// 要素ごとの加算。片方が長さ 1 ならブロードキャストする。
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let (a_data, b_data) = (a.data(), b.data());
        let n = a_data.len().max(b_data.len());
        assert!(
            a_data.len() == b_data.len() || a_data.len() == 1 || b_data.len() == 1,
            "add: shapes must match or one side must be a scalar"
        );
        let mut out_data = vec![0.0; n];
        for i in 0..n {
            out_data[i] = a_data[i % a_data.len()] + b_data[i % b_data.len()];
        }
        self.track(a);
        self.track(b);
        let shape = if a.len() == b.len() {
            a.shape().to_vec()
        } else {
            vec![n]
        };
        let out = self.output(out_data, shape, a.requires_grad() || b.requires_grad());
        self.record(Op::Add {
            a_id: a.id,
            b_id: b.id,
            out_id: out.id,
            a_len: a_data.len(),
            b_len: b_data.len(),
        });
        out
    }

    pub fn mul_scalar(&self, a: &Tensor, scalar: f32) -> Tensor {
        let data: Vec<f32> = a.data().iter().map(|v| v * scalar).collect();
        self.track(a);
        let out = self.output(data, a.shape().to_vec(), a.requires_grad());
        self.record(Op::MulScalar {
            a_id: a.id,
            out_id: out.id,
            a_len: a.len(),
            scalar,
        });
        out
    }

    pub fn relu(&self, a: &Tensor) -> Tensor {
        let data = a.data();
        let out_data = data.iter().map(|v| v.max(0.0)).collect();
        self.track(a);
        let out = self.output(out_data, a.shape().to_vec(), a.requires_grad());
        self.record(Op::Relu {
            a_id: a.id,
            out_id: out.id,
            a: data,
        });
        out
    }

    /// 全要素の和（スカラーを返す）。
    pub fn sum(&self, a: &Tensor) -> Tensor {
        let value = a.data().iter().sum::<f32>();
        self.track(a);
        let out = self.output(vec![value], Vec::new(), a.requires_grad());
        self.record(Op::Sum {
            a_id: a.id,
            out_id: out.id,
            a_len: a.len(),
        });
        out
    }

    pub fn square(&self, a: &Tensor) -> Tensor {
        let data = a.data();
        let out_data = data.iter().map(|v| v * v).collect();
        self.track(a);
        let out = self.output(out_data, a.shape().to_vec(), a.requires_grad());
        self.record(Op::Square {
            a_id: a.id,
            out_id: out.id,
            a: data,
        });
        out
    }

    /// `-log(softmax(logits)[target])`。log-sum-exp で安定化している。
    pub fn cross_entropy(&self, logits: &Tensor, target: usize) -> Tensor {
        assert!(!logits.is_empty(), "cross_entropy: empty logits");
        assert!(target < logits.len(), "cross_entropy: target out of range");
        let values = logits.data();
        let (lse, probs) = stable_softmax_and_lse(&values);
        let value = lse - values[target];
        let value = if value.is_finite() { value } else { 0.0 };
        self.track(logits);
        let out = self.output(vec![value], Vec::new(), logits.requires_grad());
        self.record(Op::CrossEntropy {
            logits_id: logits.id,
            out_id: out.id,
            probs,
            target,
        });
        out
    }

    /// loss から逆向きに勾配を計算し、リーフの `grad` を更新する。
    /// `lr != 0` ならさらに `param -= lr * grad` で SGD 更新する。
    pub fn backward(&self, loss: &Tensor, lr: f32) -> Result<(), BackwardError> {
        if loss.len() != 1 {
            return Err(BackwardError::LossNotScalar);
        }
        if !lr.is_finite() {
            return Err(BackwardError::NonFiniteLearningRate);
        }

        let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
        grads.insert(loss.id, vec![1.0]);

        let inner = self.inner.borrow();
        for op in inner.ops.iter().rev() {
            match op {
                Op::MatVec {
                    w_id,
                    x_id,
                    out_id,
                    w,
                    x,
                    rows,
                    cols,
                } => {
                    let g = take(&mut grads, *out_id, *rows);
                    let mut gw = vec![0.0; rows * cols];
                    let mut gx = vec![0.0; *cols];
                    for i in 0..*rows {
                        for j in 0..*cols {
                            let d = g[i];
                            gw[i * cols + j] += d * x[j];
                            gx[j] += d * w[i * cols + j];
                        }
                    }
                    accumulate(&mut grads, *w_id, gw);
                    accumulate(&mut grads, *x_id, gx);
                }
                Op::Add {
                    a_id,
                    b_id,
                    out_id,
                    a_len,
                    b_len,
                } => {
                    let g = take(&mut grads, *out_id, (*a_len).max(*b_len));
                    let ga = unbroadcast(&g, *a_len);
                    let gb = unbroadcast(&g, *b_len);
                    accumulate(&mut grads, *a_id, ga);
                    accumulate(&mut grads, *b_id, gb);
                }
                Op::MulScalar {
                    a_id,
                    out_id,
                    a_len,
                    scalar,
                } => {
                    let g = take(&mut grads, *out_id, *a_len);
                    accumulate(
                        &mut grads,
                        *a_id,
                        g.iter().map(|v| v * scalar).collect(),
                    );
                }
                Op::Relu { a_id, out_id, a } => {
                    let g = take(&mut grads, *out_id, a.len());
                    let ga = g
                        .iter()
                        .zip(a)
                        .map(|(d, v)| if *v > 0.0 { *d } else { 0.0 })
                        .collect();
                    accumulate(&mut grads, *a_id, ga);
                }
                Op::Sum {
                    a_id,
                    out_id,
                    a_len,
                } => {
                    let g = take(&mut grads, *out_id, 1);
                    accumulate(&mut grads, *a_id, vec![g[0]; *a_len]);
                }
                Op::Square { a_id, out_id, a } => {
                    let g = take(&mut grads, *out_id, a.len());
                    let ga = g
                        .iter()
                        .zip(a)
                        .map(|(d, v)| d * 2.0 * v)
                        .collect();
                    accumulate(&mut grads, *a_id, ga);
                }
                Op::CrossEntropy {
                    logits_id,
                    out_id,
                    probs,
                    target,
                } => {
                    let _ = take(&mut grads, *out_id, 1);
                    let g: Vec<f32> = probs
                        .iter()
                        .enumerate()
                        .map(|(i, p)| p - if i == *target { 1.0 } else { 0.0 })
                        .collect();
                    accumulate(&mut grads, *logits_id, g);
                }
            }
        }

        for (id, tensor) in &inner.nodes {
            if !tensor.is_leaf {
                continue;
            }
            let g = grads
                .get(id)
                .cloned()
                .unwrap_or_else(|| vec![0.0; tensor.len()]);
            *tensor.grad.borrow_mut() = Some(g.clone());
            if tensor.requires_grad && lr != 0.0 {
                let mut data = tensor.data.borrow_mut();
                for (value, grad) in data.iter_mut().zip(g) {
                    *value -= lr * grad;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, Tape};

    /// 2 層 MLP を 1 ステップだけ更新すると cross-entropy が減ること。
    #[test]
    fn mlp_one_step_reduces_loss() {
        let x = Tensor::constant(vec![0.4, -0.2, 0.3]);
        let w1 = Tensor::from_vec(vec![
            0.10, -0.05, 0.08, //
            0.02, 0.12, -0.04, //
            -0.03, 0.07, 0.11, //
            0.05, -0.02, 0.09, //
        ])
        .reshape([4, 3]);
        let b1 = Tensor::zeros(4);
        let w2 = Tensor::from_vec(vec![
            0.10, -0.08, 0.05, 0.02, //
            -0.04, 0.11, 0.03, -0.06, //
        ])
        .reshape([2, 4]);
        let b2 = Tensor::zeros(2);

        let forward = || {
            let tape = Tape::new();
            let hidden = tape.relu(&tape.add(&tape.matvec(&w1, &x), &b1));
            let logits = tape.add(&tape.matvec(&w2, &hidden), &b2);
            let loss = tape.cross_entropy(&logits, 1);
            (tape, loss)
        };

        let (tape, loss) = forward();
        let before = loss.data()[0];
        tape.backward(&loss, 0.1).unwrap();
        let (_, updated_loss) = forward();
        let after = updated_loss.data()[0];

        assert!(
            after < before,
            "loss should decrease: before={before}, after={after}"
        );
        for grad in [w1.grad().unwrap(), w2.grad().unwrap()].concat() {
            assert!(grad.is_finite(), "gradient must be finite: {grad}");
        }
    }

    /// matvec の重み勾配が中心差分と一致すること。
    #[test]
    fn matvec_gradient_matches_central_difference() {
        let x = Tensor::constant(vec![0.3, -0.7, 0.2]);
        let w = Tensor::from_vec(vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6]).reshape([2, 3]);

        let tape = Tape::new();
        let y = tape.matvec(&w, &x);
        let loss = tape.sum(&tape.square(&y));
        // lr = 0 なら勾配のみ計算し、重みは変更しない。
        tape.backward(&loss, 0.0).unwrap();
        let analytic = w.grad().unwrap();

        let eps = 1e-3;
        let original = w.data();
        for i in 0..original.len() {
            let mut plus = original.clone();
            plus[i] += eps;
            w.set_data(plus);
            let t = Tape::new();
            let loss_plus = t.sum(&t.square(&t.matvec(&w, &x))).data()[0];

            let mut minus = original.clone();
            minus[i] -= eps;
            w.set_data(minus);
            let t = Tape::new();
            let loss_minus = t.sum(&t.square(&t.matvec(&w, &x))).data()[0];

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (analytic[i] - numeric).abs() < 1e-2,
                "w[{i}]: analytic={}, numeric={numeric}",
                analytic[i]
            );
        }
        w.set_data(original);
    }

    /// 巨大な logits でも log-sum-exp により cross-entropy が NaN にならないこと。
    #[test]
    fn cross_entropy_is_finite_for_large_logits() {
        let logits = Tensor::from_vec(vec![1.0e30, -1.0e30, 1000.0, -1000.0]);
        let tape = Tape::new();
        let loss = tape.cross_entropy(&logits, 0);
        assert!(loss.data()[0].is_finite(), "loss={}", loss.data()[0]);

        // softmax の勾配も有限で、総和が 0 に近いこと。
        tape.backward(&loss, 0.0).unwrap();
        let grad = logits.grad().unwrap();
        assert!(grad.iter().all(|g| g.is_finite()));
        assert!((grad.iter().sum::<f32>()).abs() < 1e-5);
    }

    /// 全操作を含むグラフの勾配が finite であること。
    #[test]
    fn gradients_are_finite() {
        let x = Tensor::constant(vec![0.2, -0.4, 0.6, 0.1]);
        let w1 = Tensor::from_vec(vec![
            0.1, 0.2, -0.3, 0.4, //
            -0.5, 0.6, 0.7, -0.8, //
            0.9, -0.1, 0.2, 0.3, //
        ])
        .reshape([3, 4]);
        let b1 = Tensor::zeros(3);
        let w2 = Tensor::from_vec(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6]).reshape([2, 3]);
        let b2 = Tensor::zeros(2);

        let tape = Tape::new();
        let hidden = tape.relu(&tape.mul_scalar(&tape.add(&tape.matvec(&w1, &x), &b1), 2.0));
        let logits = tape.add(&tape.matvec(&w2, &hidden), &b2);
        let loss = tape.cross_entropy(&logits, 1);
        tape.backward(&loss, 0.0).unwrap();

        for grad in [
            w1.grad().unwrap(),
            b1.grad().unwrap(),
            w2.grad().unwrap(),
            b2.grad().unwrap(),
        ]
        .concat()
        {
            assert!(grad.is_finite(), "gradient must be finite: {grad}");
        }
        assert!(loss.data()[0].is_finite());
    }
}
