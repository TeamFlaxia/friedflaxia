//! Offline Evolution Strategies updates for LoRA parameters.
//!
//! No backpropagation and no gradient exchange: the host regenerates the
//! perturbation from the seed and applies `Δ = α/N · Σ Z_n ε_n`. Nodes only
//! ever send `(seed, scalar reward)`.

use super::rng::SplitMix64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsSample {
    pub seed: u64,
    pub reward: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct EsConfig {
    /// Learning rate; `1/σ` is already absorbed into it (arXiv:2509.24372 §3.1).
    pub alpha: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EsError {
    EmptyWeights,
    EmptySamples,
    NonFiniteAlpha,
    NonFiniteReward,
}

impl core::fmt::Display for EsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::EmptyWeights => "es: empty parameter vector",
            Self::EmptySamples => "es: no samples",
            Self::NonFiniteAlpha => "es: alpha must be finite and positive",
            Self::NonFiniteReward => "es: reward must be finite",
        };
        f.write_str(message)
    }
}

/// Regenerate the exact N(0, 1) perturbation for a seed and parameter length.
pub fn perturbation(seed: u64, len: usize) -> Vec<f32> {
    let mut values = vec![0.0; len];
    SplitMix64::new(seed).fill_normal(&mut values);
    values
}

/// Reward normalization. With one sample the raw reward is kept (there is no
/// relative signal to center); with equal rewards there is no signal at all.
pub fn zscore(rewards: &[f32]) -> Vec<f32> {
    if rewards.len() == 1 {
        return vec![rewards[0]];
    }
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance = rewards
        .iter()
        .map(|r| {
            let d = r - mean;
            d * d
        })
        .sum::<f32>()
        / rewards.len() as f32;
    let std = variance.sqrt();
    if !std.is_finite() || std <= 1e-8 {
        return vec![0.0; rewards.len()];
    }
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// Apply one ES step in place. Deterministic for a given seed/reward set.
pub fn es_step(
    weights: &mut [f32],
    samples: &[EsSample],
    config: EsConfig,
) -> Result<(), EsError> {
    if weights.is_empty() {
        return Err(EsError::EmptyWeights);
    }
    if samples.is_empty() {
        return Err(EsError::EmptySamples);
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(EsError::NonFiniteAlpha);
    }
    let rewards: Vec<f32> = samples.iter().map(|s| s.reward).collect();
    if rewards.iter().any(|r| !r.is_finite()) {
        return Err(EsError::NonFiniteReward);
    }

    let scores = zscore(&rewards);
    let scale = config.alpha / samples.len() as f32;
    for (sample, score) in samples.iter().zip(scores) {
        if score == 0.0 {
            continue;
        }
        let eps = perturbation(sample.seed, weights.len());
        for (weight, noise) in weights.iter_mut().zip(eps) {
            *weight += scale * score * noise;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{EsConfig, EsError, EsSample, es_step, perturbation, zscore};

    #[test]
    fn zero_rewards_do_not_change_weights() {
        let mut weights = vec![0.1; 8];
        let samples = [
            EsSample { seed: 1, reward: 0.0 },
            EsSample { seed: 2, reward: 0.0 },
        ];
        es_step(&mut weights, &samples, EsConfig { alpha: 0.01 }).unwrap();
        assert_eq!(weights, vec![0.1; 8]);
    }

    #[test]
    fn single_positive_sample_moves_along_its_perturbation() {
        let seed = 99;
        let eps = perturbation(seed, 16);
        let mut weights = vec![0.0; 16];
        es_step(
            &mut weights,
            &[EsSample { seed, reward: 1.0 }],
            EsConfig { alpha: 0.01 },
        )
        .unwrap();
        for (got, expected) in weights.iter().zip(eps) {
            assert!((got - expected * 0.01).abs() < 1e-6);
        }
    }

    #[test]
    fn better_sample_gets_positive_score_and_worse_negative() {
        let scores = zscore(&[2.0, 0.0, -2.0]);
        assert!(scores[0] > 0.0);
        assert!(scores[2] < 0.0);
        assert!((scores.iter().sum::<f32>()).abs() < 1e-5);
    }

    #[test]
    fn equal_rewards_have_no_signal() {
        assert_eq!(zscore(&[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn update_is_reproducible() {
        let samples = [
            EsSample { seed: 3, reward: 1.0 },
            EsSample { seed: 9, reward: -0.5 },
            EsSample { seed: 12, reward: 0.25 },
        ];
        let mut a = vec![0.0; 32];
        let mut b = vec![0.0; 32];
        let cfg = EsConfig { alpha: 0.001 };
        es_step(&mut a, &samples, cfg).unwrap();
        es_step(&mut b, &samples, cfg).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(
            es_step(&mut [], &[EsSample { seed: 1, reward: 1.0 }], EsConfig { alpha: 0.1 }),
            Err(EsError::EmptyWeights)
        );
        assert_eq!(
            es_step(&mut [0.0], &[], EsConfig { alpha: 0.1 }),
            Err(EsError::EmptySamples)
        );
        assert_eq!(
            es_step(&mut [0.0], &[EsSample { seed: 1, reward: 1.0 }], EsConfig { alpha: 0.0 }),
            Err(EsError::NonFiniteAlpha)
        );
        assert_eq!(
            es_step(
                &mut [0.0],
                &[EsSample { seed: 1, reward: f32::NAN }],
                EsConfig { alpha: 0.1 }
            ),
            Err(EsError::NonFiniteReward)
        );
    }
}
