//! Browser-facing WebAssembly API.
//!
//! `Engine` contains the testable, native logic. `Model` is the thin
//! wasm-bindgen wrapper that converts errors to `JsValue`.

use friedflaxia_core::gguf::tokenizer::{TokenizerError, TokenizerSpec};
use friedflaxia_core::gguf::{Gguf, ParseOptions};
use friedflaxia_core::models::d1omni::decision_logits;
use friedflaxia_core::models::embeddinggemma::{embed_text, embed_text_with_features};
use friedflaxia_core::models::gemma_audio::encode_audio;
use friedflaxia_core::models::gemma_vision::encode_image;
use friedflaxia_core::models::nudenet::NudeNet;
use friedflaxia_core::models::plamo3::{PlamoSession, next_token};
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

    fn token_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Bpe(tokenizer) => tokenizer.token_id(token),
            Self::Unigram(tokenizer) => tokenizer.token_id(token),
        }
    }

    fn is_control(&self, id: u32) -> bool {
        match self {
            Self::Bpe(tokenizer) => tokenizer.is_control(id),
            Self::Unigram(tokenizer) => tokenizer.is_control(id),
        }
    }

    fn bos_token_id(&self) -> u32 {
        match self {
            Self::Bpe(tokenizer) => tokenizer.bos_token_id(),
            Self::Unigram(tokenizer) => tokenizer.bos_token_id(),
        }
    }

    fn eos_token_id(&self) -> u32 {
        match self {
            Self::Bpe(tokenizer) => tokenizer.eos_token_id(),
            Self::Unigram(tokenizer) => tokenizer.eos_token_id(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    EmbeddingGemma,
    D1Omni,
    Plamo3,
    NudeNet,
}

impl ModelKind {
    fn from_architecture(architecture: &str) -> Option<Self> {
        match architecture {
            "gemma-embedding2" => Some(Self::EmbeddingGemma),
            "lfm2" => Some(Self::D1Omni),
            "plamo3" => Some(Self::Plamo3),
            "yolov8-nudenet" => Some(Self::NudeNet),
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
    nudenet: Option<NudeNet>,
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
        let nudenet = if kind == ModelKind::NudeNet {
            Some(
                NudeNet::load(&gguf, &bytes)
                    .map_err(|error| EngineError::new(error.to_string()))?,
            )
        } else {
            None
        };
        Ok(Self {
            bytes,
            gguf,
            kind,
            tokenizer,
            nudenet,
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

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.gguf
            .metadata(key)
            .and_then(friedflaxia_core::gguf::Value::as_f32)
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.gguf
            .metadata(key)
            .and_then(friedflaxia_core::gguf::Value::as_u32)
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

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.as_ref()?.token_id(token)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer.as_ref().map(Tokenizer::bos_token_id)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.as_ref().map(Tokenizer::eos_token_id)
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

    /// NudeNet detections as flattened `[x1, y1, x2, y2, score, label]` rows.
    pub fn detect_nudenet(
        &self,
        input: &[f32],
        top_k: u32,
        iou_threshold: f32,
        score_threshold: f32,
    ) -> Result<Vec<f32>, EngineError> {
        if self.kind != ModelKind::NudeNet {
            return Err(EngineError::new("model is not yolov8-nudenet"));
        }
        let net = self
            .nudenet
            .as_ref()
            .ok_or_else(|| EngineError::new("NudeNet model was not loaded"))?;
        let detections = net
            .detect(input, top_k as usize, iou_threshold, score_threshold)
            .map_err(|error| EngineError::new(error.to_string()))?;
        let mut flat = Vec::with_capacity(detections.len() * 6);
        for detection in detections {
            flat.extend_from_slice(&detection.box_xyxy);
            flat.push(detection.score);
            flat.push(detection.label_index as f32);
        }
        Ok(flat)
    }

    /// Generate up to `max_tokens` greedy tokens with the PLaMO KV cache.
    /// Returns only newly generated ids (the prompt is not included).
    pub fn generate(&self, prompt_ids: &[u32], max_tokens: u32) -> Result<Vec<u32>, EngineError> {
        if self.kind != ModelKind::Plamo3 {
            return Err(EngineError::new("model is not plamo3"));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::new("GGUF has no supported tokenizer"))?;
        let limit = max_tokens.min(4096);
        let mut session = PlamoSession::new(&self.gguf, &self.bytes)
            .map_err(|error| EngineError::new(error.to_string()))?;
        let mut logits = session
            .prefill(prompt_ids)
            .map_err(|error| EngineError::new(error.to_string()))?;
        let eos = tokenizer.eos_token_id();
        let mut generated = Vec::new();
        for _ in 0..limit {
            let next = argmax(&logits)?;
            if next == eos || tokenizer.is_control(next) {
                break;
            }
            generated.push(next);
            logits = session
                .decode(next)
                .map_err(|error| EngineError::new(error.to_string()))?;
        }
        Ok(generated)
    }
}

fn argmax(values: &[f32]) -> Result<u32, EngineError> {
    let (index, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| EngineError::new("model returned empty logits"))?;
    Ok(index as u32)
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

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.engine.metadata_f32(key)
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.engine.metadata_u32(key)
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

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.engine.token_id(token)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.engine.bos_token_id()
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.engine.eos_token_id()
    }

    pub fn generate(&self, prompt_ids: &[u32], max_tokens: u32) -> Result<Vec<u32>, JsValue> {
        self.engine
            .generate(prompt_ids, max_tokens)
            .map_err(js_error)
    }

    pub fn detect(
        &self,
        input: &[f32],
        top_k: u32,
        iou_threshold: f32,
        score_threshold: f32,
    ) -> Result<Vec<f32>, JsValue> {
        self.engine
            .detect_nudenet(input, top_k, iou_threshold, score_threshold)
            .map_err(js_error)
    }
}

/// Multimodal EmbeddingGemma 2 model: text GGUF plus the vision/audio mmproj.
#[wasm_bindgen]
pub struct MultimodalModel {
    text_bytes: Vec<u8>,
    text_gguf: Gguf,
    mmproj_bytes: Vec<u8>,
    mmproj: Gguf,
    tokenizer: Option<Tokenizer>,
}

#[wasm_bindgen]
impl MultimodalModel {
    #[wasm_bindgen(constructor)]
    pub fn new(text_bytes: Vec<u8>, mmproj_bytes: Vec<u8>) -> Result<MultimodalModel, JsValue> {
        let text_gguf = Gguf::parse_with_options(&text_bytes, ParseOptions::default())
            .map_err(|error| js_error(error.to_string()))?;
        let architecture = text_gguf
            .metadata("general.architecture")
            .and_then(|value| value.as_str())
            .ok_or_else(|| js_error("text GGUF has no general.architecture"))?;
        if architecture != "gemma-embedding2" {
            return Err(js_error(format!(
                "expected gemma-embedding2 text GGUF, got {architecture}"
            )));
        }
        let mmproj = Gguf::parse_with_options(&mmproj_bytes, ParseOptions::default())
            .map_err(|error| js_error(error.to_string()))?;
        let mmproj_arch = mmproj
            .metadata("general.architecture")
            .and_then(|value| value.as_str())
            .ok_or_else(|| js_error("mmproj GGUF has no general.architecture"))?;
        if mmproj_arch != "clip" {
            return Err(js_error(format!(
                "expected clip mmproj GGUF, got {mmproj_arch}"
            )));
        }
        let tokenizer =
            TokenizerSpec::from_gguf(&text_gguf)
                .ok()
                .and_then(|spec| match spec.model.as_str() {
                    "gpt2" | "gemma4" => BpeTokenizer::from_spec(&spec)
                        .ok()
                        .map(|tokenizer| Tokenizer::Bpe(Box::new(tokenizer))),
                    _ => None,
                });
        Ok(Self {
            text_bytes,
            text_gguf,
            mmproj_bytes,
            mmproj,
            tokenizer,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| js_error("text GGUF has no supported tokenizer"))?
            .encode(text, add_special_tokens)
            .map_err(js_error)
    }

    pub fn encode_image(
        &self,
        patches: &[f32],
        grid_h: u32,
        grid_w: u32,
    ) -> Result<Vec<f32>, JsValue> {
        encode_image(
            &self.mmproj,
            &self.mmproj_bytes,
            patches,
            grid_h as usize,
            grid_w as usize,
        )
        .map_err(js_error)
    }

    pub fn encode_audio(&self, mel: &[f32], frames: u32) -> Result<Vec<f32>, JsValue> {
        encode_audio(&self.mmproj, &self.mmproj_bytes, mel, frames as usize).map_err(js_error)
    }

    pub fn embed(
        &self,
        token_ids: &[u32],
        image_features: &[f32],
        video_features: &[f32],
        audio_features: &[f32],
    ) -> Result<Vec<f32>, JsValue> {
        embed_text_with_features(
            &self.text_gguf,
            &self.text_bytes,
            token_ids,
            image_features,
            video_features,
            audio_features,
        )
        .map_err(js_error)
    }
}

fn js_error(error: impl fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
