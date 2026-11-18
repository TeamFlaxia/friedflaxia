//! Browser-facing WebAssembly API.
//!
//! `Engine` contains the testable, native logic. `Model` is the thin
//! wasm-bindgen wrapper that converts errors to `JsValue`.

use friedflaxia_core::gguf::tokenizer::{TokenizerError, TokenizerSpec};
use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1omni::decision_logits;
use friedflaxia_core::models::embeddinggemma::embed_text;
use friedflaxia_core::models::plamo3::next_token;
use friedflaxia_core::tokenizer::{BpeTokenizer, UnigramTokenizer};
use std::fmt;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    message: String,
}

impl EngineError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EngineError {}

impl From<TokenizerError> for EngineError {
    fn from(error: TokenizerError) -> Self {
        Self::new(error.to_string())
    }
}

enum Tokenizer {
    Bpe(Box<BpeTokenizer>),
    Unigram(Box<UnigramTokenizer>),
}

impl Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Self::Bpe(tokenizer) => tokenizer.encode(text, add_special_tokens),
            Self::Unigram(tokenizer) => tokenizer.encode(text, add_special_tokens),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Self::Bpe(tokenizer) => tokenizer.decode(ids),
            Self::Unigram(tokenizer) => tokenizer.decode(ids),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    EmbeddingGemma,
    D1Omni,
    Plamo3,
}

impl ModelKind {
    fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            "gemma-embedding2" => Some(Self::EmbeddingGemma),
            "lfm2" => Some(Self::D1Omni),
            "plamo3" => Some(Self::Plamo3),
            _ => None,
        }
    }
}

/// Native, testable engine handle.
pub struct Engine {
    bytes: Vec<u8>,
    gguf: Gguf,
    kind: ModelKind,
    tokenizer: Option<Tokenizer>,
}

impl Engine {
    pub fn new(bytes: Vec<u8>) -> Result<Self, EngineError> {
        let gguf = Gguf::parse_with_options(&bytes, ParseOptions::default())
            .map_err(|error| EngineError::new(error.to_string()))?;
        let architecture = gguf
            .metadata("general.architecture")
            .and_then(|value| value.as_str())
            .ok_or_else(|| EngineError::new("GGUF has no general.architecture"))?;
        let kind = ModelKind::from_architecture(architecture)
            .ok_or_else(|| EngineError::new(format!("unsupported architecture: {architecture}")))?;
        let tokenizer =
            TokenizerSpec::from_gguf(&gguf)
                .ok()
                .and_then(|spec| match spec.model.as_str() {
                    "gpt2" | "gemma4" => BpeTokenizer::from_spec(&spec)
                        .ok()
                        .map(|tokenizer| Tokenizer::Bpe(Box::new(tokenizer))),
                    "plamo3" => UnigramTokenizer::from_spec(&spec)
                        .ok()
                        .map(|tokenizer| Tokenizer::Unigram(Box::new(tokenizer))),
                    _ => None,
                });
        Ok(Self {
            bytes,
            gguf,
            kind,
            tokenizer,
        })
    }

    pub fn architecture(&self) -> &str {
        self.gguf
            .metadata("general.architecture")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
    }

    pub fn tensor_count(&self) -> usize {
        self.gguf.tensor_count()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, EngineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::new("GGUF has no supported tokenizer"))?;
        Ok(tokenizer.encode(text, add_special_tokens)?)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, EngineError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::new("GGUF has no supported tokenizer"))?;
        Ok(tokenizer.decode(ids)?)
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        if self.kind != ModelKind::EmbeddingGemma {
            return Err(EngineError::new("model is not gemma-embedding2"));
        }
        embed_text(&self.gguf, &self.bytes, token_ids)
            .map_err(|error| EngineError::new(error.to_string()))
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        let ids = self.encode(text, true)?;
        self.embed(&ids)
    }

    pub fn decision(
        &self,
        token_ids: &[u32],
        marker_positions: &[u32],
        qtype: u32,
    ) -> Result<Vec<f32>, EngineError> {
        if self.kind != ModelKind::D1Omni {
            return Err(EngineError::new("model is not lfm2"));
        }
        let markers: Vec<usize> = marker_positions
            .iter()
            .map(|value| *value as usize)
            .collect();
        decision_logits(&self.gguf, &self.bytes, token_ids, &markers, qtype as usize)
            .map_err(|error| EngineError::new(error.to_string()))
    }

    pub fn next_token(&self, token_ids: &[u32]) -> Result<u32, EngineError> {
        if self.kind != ModelKind::Plamo3 {
            return Err(EngineError::new("model is not plamo3"));
        }
        next_token(&self.gguf, &self.bytes, token_ids)
            .map_err(|error| EngineError::new(error.to_string()))
    }
}

#[wasm_bindgen]
pub struct Model {
    engine: Engine,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>) -> Result<Model, JsValue> {
        Engine::new(bytes)
            .map(|engine| Model { engine })
            .map_err(js_error)
    }

    pub fn architecture(&self) -> String {
        self.engine.architecture().to_owned()
    }

    pub fn tensor_count(&self) -> usize {
        self.engine.tensor_count()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
        self.engine
            .encode(text, add_special_tokens)
            .map_err(js_error)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, JsValue> {
        self.engine.decode(ids).map_err(js_error)
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, JsValue> {
        self.engine.embed(token_ids).map_err(js_error)
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        self.engine.embed_text(text).map_err(js_error)
    }

    pub fn decision(
        &self,
        token_ids: &[u32],
        marker_positions: &[u32],
        qtype: u32,
    ) -> Result<Vec<f32>, JsValue> {
        self.engine
            .decision(token_ids, marker_positions, qtype)
            .map_err(js_error)
    }

    pub fn next_token(&self, token_ids: &[u32]) -> Result<u32, JsValue> {
        self.engine.next_token(token_ids).map_err(js_error)
    }
}

fn js_error(error: impl fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
