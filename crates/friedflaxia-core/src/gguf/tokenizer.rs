//! Tokenizer metadata embedded in GGUF files.
//!
//! GGUF carries the vocabulary, scores, token types, merge rules and special
//! token ids. Reading them here keeps tokenization out of Transformers.js and
//! ONNX Runtime.

use super::{Gguf, Value};
use std::fmt;

/// GGUF `tokenizer.ggml.token_type` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            1 => Some(Self::Normal),
            2 => Some(Self::Unknown),
            3 => Some(Self::Control),
            4 => Some(Self::UserDefined),
            5 => Some(Self::Unused),
            6 => Some(Self::Byte),
            _ => None,
        }
    }
}

/// The tokenizer fields a GGUF model may provide.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerSpec {
    /// `tokenizer.ggml.model` (for example `gpt2`, `gemma4`, `plamo3`).
    pub model: String,
    pub tokens: Vec<String>,
    /// Unigram scores; empty when the format has no scores.
    pub scores: Vec<f32>,
    pub token_types: Vec<TokenType>,
    /// BPE merge rules; empty for Unigram models.
    pub merges: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub chat_template: Option<String>,
}

impl TokenizerSpec {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, TokenizerError> {
        let model = required_str(gguf, "tokenizer.ggml.model")?.to_owned();
        let tokens = string_array(gguf, "tokenizer.ggml.tokens")?
            .ok_or(TokenizerError::Missing("tokenizer.ggml.tokens"))?;
        if tokens.is_empty() {
            return Err(TokenizerError::Missing("tokenizer.ggml.tokens"));
        }

        let scores = match array(gguf, "tokenizer.ggml.scores")? {
            Some(values) => values
                .iter()
                .map(|value| {
                    value
                        .as_f32()
                        .ok_or(TokenizerError::WrongType("tokenizer.ggml.scores"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        if !scores.is_empty() && scores.len() != tokens.len() {
            return Err(TokenizerError::LengthMismatch("tokenizer.ggml.scores"));
        }

        let token_types = match array(gguf, "tokenizer.ggml.token_type")? {
            Some(values) => values
                .iter()
                .map(|value| {
                    let raw = value
                        .as_i32()
                        .ok_or(TokenizerError::WrongType("tokenizer.ggml.token_type"))?;
                    TokenType::from_i32(raw).ok_or(TokenizerError::InvalidTokenType(raw))
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![TokenType::Normal; tokens.len()],
        };
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::LengthMismatch("tokenizer.ggml.token_type"));
        }

        let merges = string_array(gguf, "tokenizer.ggml.merges")?.unwrap_or_default();

        Ok(Self {
            model,
            tokens,
            scores,
            token_types,
            merges,
            bos_token_id: optional_u32(gguf, "tokenizer.ggml.bos_token_id")?.unwrap_or(0),
            eos_token_id: optional_u32(gguf, "tokenizer.ggml.eos_token_id")?.unwrap_or(0),
            add_bos_token: optional_bool(gguf, "tokenizer.ggml.add_bos_token")?.unwrap_or(true),
            add_eos_token: optional_bool(gguf, "tokenizer.ggml.add_eos_token")?.unwrap_or(false),
            chat_template: match gguf.metadata("tokenizer.chat_template") {
                Some(Value::String(template)) => Some(template.clone()),
                Some(_) => return Err(TokenizerError::WrongType("tokenizer.chat_template")),
                None => None,
            },
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

fn required_str<'a>(gguf: &'a Gguf, key: &'static str) -> Result<&'a str, TokenizerError> {
    match gguf.metadata(key) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(TokenizerError::WrongType(key)),
        None => Err(TokenizerError::Missing(key)),
    }
}

fn array<'a>(gguf: &'a Gguf, key: &'static str) -> Result<Option<&'a [Value]>, TokenizerError> {
    match gguf.metadata(key) {
        Some(Value::Array(values)) => Ok(Some(values)),
        Some(_) => Err(TokenizerError::WrongType(key)),
        None => Ok(None),
    }
}

fn string_array(gguf: &Gguf, key: &'static str) -> Result<Option<Vec<String>>, TokenizerError> {
    let Some(values) = array(gguf, key)? else {
        return Ok(None);
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or(TokenizerError::WrongType(key))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn optional_u32(gguf: &Gguf, key: &'static str) -> Result<Option<u32>, TokenizerError> {
    match gguf.metadata(key) {
        Some(value) => value
            .as_u32()
            .map(Some)
            .ok_or(TokenizerError::WrongType(key)),
        None => Ok(None),
    }
}

fn optional_bool(gguf: &Gguf, key: &'static str) -> Result<Option<bool>, TokenizerError> {
    match gguf.metadata(key) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::WrongType(key)),
        None => Ok(None),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    Missing(&'static str),
    WrongType(&'static str),
    LengthMismatch(&'static str),
    InvalidTokenType(i32),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(key) => write!(f, "missing tokenizer metadata {key}"),
            Self::WrongType(key) => write!(f, "tokenizer metadata {key} has the wrong type"),
            Self::LengthMismatch(key) => write!(f, "tokenizer metadata {key} has the wrong length"),
            Self::InvalidTokenType(value) => write!(f, "unknown GGUF token type {value}"),
        }
    }
}

impl std::error::Error for TokenizerError {}
