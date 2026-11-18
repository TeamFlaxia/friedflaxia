//! Model forward passes built on GGUF tensors.

pub mod d1omni;
pub mod embeddinggemma;
pub mod plamo3;

use crate::gguf::ConfigError;
use crate::tensor::TensorError;
use std::fmt;

#[derive(Debug)]
pub enum ModelError {
    Tensor(TensorError),
    Config(ConfigError),
    Invalid(&'static str),
    Shape(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
            Self::Config(error) => write!(f, "config error: {error}"),
            Self::Invalid(what) => write!(f, "invalid model input: {what}"),
            Self::Shape(what) => write!(f, "model shape error: {what}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<TensorError> for ModelError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<ConfigError> for ModelError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}
