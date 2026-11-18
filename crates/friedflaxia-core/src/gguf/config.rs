//! Typed access to `{architecture}.*` GGUF metadata keys.

use super::{Gguf, Value};
use std::fmt;

/// Architecture-scoped configuration values.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig<'a> {
    gguf: &'a Gguf,
    architecture: &'a str,
}

impl<'a> ModelConfig<'a> {
    pub fn from_gguf(gguf: &'a Gguf) -> Result<Self, ConfigError> {
        let architecture = match gguf.metadata("general.architecture") {
            Some(Value::String(value)) => value.as_str(),
            Some(_) => {
                return Err(ConfigError::WrongType(
                    "general.architecture".to_owned(),
                    "string",
                ));
            }
            None => return Err(ConfigError::MissingArchitecture),
        };
        Ok(Self { gguf, architecture })
    }

    pub fn architecture(&self) -> &'a str {
        self.architecture
    }

    pub fn u32(&self, suffix: &str) -> Result<Option<u32>, ConfigError> {
        self.value(suffix, "u32", Value::as_u32)
    }

    pub fn f32(&self, suffix: &str) -> Result<Option<f32>, ConfigError> {
        self.value(suffix, "f32", Value::as_f32)
    }

    pub fn bool(&self, suffix: &str) -> Result<Option<bool>, ConfigError> {
        self.value(suffix, "bool", |value| match value {
            Value::Bool(value) => Some(*value),
            _ => None,
        })
    }

    pub fn string(&self, suffix: &str) -> Result<Option<&'a str>, ConfigError> {
        let Some(value) = self.raw(suffix) else {
            return Ok(None);
        };
        match value {
            Value::String(value) => Ok(Some(value.as_str())),
            _ => Err(ConfigError::WrongType(self.key(suffix), "string")),
        }
    }

    fn value<T>(
        &self,
        suffix: &str,
        expected: &'static str,
        read: impl FnOnce(&Value) -> Option<T>,
    ) -> Result<Option<T>, ConfigError> {
        let Some(value) = self.raw(suffix) else {
            return Ok(None);
        };
        read(value)
            .map(Some)
            .ok_or_else(|| ConfigError::WrongType(self.key(suffix), expected))
    }

    fn raw(&self, suffix: &str) -> Option<&'a Value> {
        self.gguf.metadata(&self.key(suffix))
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}.{}", self.architecture, suffix)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    MissingArchitecture,
    WrongType(String, &'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingArchitecture => f.write_str("GGUF has no general.architecture"),
            Self::WrongType(key, expected) => {
                write!(f, "GGUF metadata {key} is not a {expected}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}
