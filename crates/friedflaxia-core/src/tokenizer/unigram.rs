use super::{Result, byte_token};
use crate::gguf::tokenizer::{TokenType, TokenizerError, TokenizerSpec};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// PLaMO uses a Unigram model with special tokens and byte fallback.
#[derive(Debug)]
struct Node {
    children: HashMap<char, usize>,
    token_id: Option<u32>,
    score: f32,
}

pub struct UnigramTokenizer {
    nodes: Vec<Node>,
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<TokenType>,
    vocab: HashMap<String, u32>,
    byte_tokens: [Option<u32>; 256],
    bos_token_id: u32,
    eos_token_id: u32,
    add_bos: bool,
    add_eos: bool,
}

impl UnigramTokenizer {
    pub fn from_spec(spec: &TokenizerSpec) -> Result<Self> {
        if spec.model != "plamo3" {
            return Err(TokenizerError::UnsupportedModel(spec.model.clone()));
        }
        if !spec.scores.is_empty() && spec.scores.len() != spec.tokens.len() {
            return Err(TokenizerError::LengthMismatch("tokenizer.ggml.scores"));
        }
        let mut vocab = HashMap::with_capacity(spec.tokens.len());
        let mut byte_tokens = [None; 256];
        let mut nodes = vec![Node {
            children: HashMap::new(),
            token_id: None,
            score: 0.0,
        }];

        for (id, token) in spec.tokens.iter().enumerate() {
            let id = id as u32;
            vocab.insert(token.clone(), id);
            let token_type = spec.token_types[id as usize];
            if token_type == TokenType::Byte {
                if let Some(byte) = byte_token(token) {
                    byte_tokens[byte as usize] = Some(id);
                }
                continue;
            }
            if token_type == TokenType::Control || token_type == TokenType::Unused {
                continue;
            }
            let score = spec.scores.get(id as usize).copied().unwrap_or(0.0);
            let mut node = 0usize;
            for character in token.chars() {
                let next = if let Some(next) = nodes[node].children.get(&character) {
                    *next
                } else {
                    let next = nodes.len();
                    nodes.push(Node {
                        children: HashMap::new(),
                        token_id: None,
                        score: 0.0,
                    });
                    nodes[node].children.insert(character, next);
                    next
                };
                node = next;
            }
            nodes[node].token_id = Some(id);
            nodes[node].score = score;
        }

        Ok(Self {
            nodes,
            tokens: spec.tokens.clone(),
            scores: spec.scores.clone(),
            token_types: spec.token_types.clone(),
            vocab,
            byte_tokens,
            bos_token_id: spec.bos_token_id,
            eos_token_id: spec.eos_token_id,
            add_bos: spec.add_bos_token,
            add_eos: spec.add_eos_token,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_special_tokens && self.add_bos {
            ids.push(self.bos_token_id);
        }
        let special = special_regex()?;
        let boundary = boundary_regex()?;
        let mut cursor = 0usize;
        for special_match in special.find_iter(text) {
            let special_match =
                special_match.map_err(|error| TokenizerError::InvalidPattern(error.to_string()))?;
            if special_match.start() > cursor {
                self.encode_plain(&text[cursor..special_match.start()], boundary, &mut ids);
            }
            let token = special_match.as_str();
            let id = self
                .vocab
                .get(token)
                .copied()
                .ok_or_else(|| TokenizerError::InvalidMerge(token.to_owned()))?;
            ids.push(id);
            cursor = special_match.end();
        }
        if cursor < text.len() {
            self.encode_plain(&text[cursor..], boundary, &mut ids);
        }
        if add_special_tokens && self.add_eos {
            ids.push(self.eos_token_id);
        }
        Ok(ids)
    }

    fn encode_plain(&self, text: &str, boundary: &Regex, ids: &mut Vec<u32>) {
        let mut cursor = 0usize;
        for boundary_match in boundary.find_iter(text) {
            let Ok(boundary_match) = boundary_match else {
                break;
            };
            if boundary_match.start() > cursor {
                self.encode_segment(&text[cursor..boundary_match.start()], ids);
            }
            self.encode_segment(boundary_match.as_str(), ids);
            cursor = boundary_match.end();
        }
        if cursor < text.len() {
            self.encode_segment(&text[cursor..], ids);
        }
    }

    fn encode_segment(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut prev: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best[0] = 0.0;

        for start in 0..n {
            if !best[start].is_finite() {
                continue;
            }
            let mut node = 0usize;
            for end in start..n {
                let Some(next) = self.nodes[node].children.get(&chars[end]) else {
                    break;
                };
                node = *next;
                if let Some(token_id) = self.nodes[node].token_id {
                    let score = best[start] + self.nodes[node].score;
                    if score > best[end + 1] {
                        best[end + 1] = score;
                        prev[end + 1] = Some((start, token_id));
                    }
                }
            }
        }

        if best[n].is_finite() {
            let mut path = Vec::new();
            let mut position = n;
            while position > 0 {
                if let Some((start, token_id)) = prev[position] {
                    path.push(token_id);
                    position = start;
                } else {
                    break;
                }
            }
            path.reverse();
            ids.extend(path);
            return;
        }

        // Byte fallback for text that the Unigram lattice cannot cover.
        for character in chars {
            let mut encoded = [0u8; 4];
            for byte in character.encode_utf8(&mut encoded).as_bytes() {
                if let Some(token_id) = self.byte_tokens[*byte as usize] {
                    ids.push(token_id);
                } else {
                    ids.push(0);
                }
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in ids {
            if *id == self.bos_token_id || *id == self.eos_token_id {
                continue;
            }
            let token = self
                .tokens
                .get(*id as usize)
                .ok_or(TokenizerError::InvalidTokenId(*id))?;
            if self
                .token_types
                .get(*id as usize)
                .is_some_and(|token_type| *token_type == TokenType::Control)
            {
                continue;
            }
            if let Some(byte) = byte_token(token) {
                bytes.push(byte);
            } else {
                bytes.extend_from_slice(token.as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn token_scores(&self) -> &[f32] {
        &self.scores
    }
}

fn special_regex() -> Result<&'static Regex> {
    static REGEX: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();
    match REGEX.get_or_init(|| Regex::new(r"(<\|plamo:[^|\s]{0,64}\|>)").map_err(|e| e.to_string()))
    {
        Ok(regex) => Ok(regex),
        Err(error) => Err(TokenizerError::InvalidPattern(error.clone())),
    }
}

fn boundary_regex() -> Result<&'static Regex> {
    static REGEX: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();
    match REGEX.get_or_init(|| Regex::new(r" {2,}|(.)\1{3,}").map_err(|e| e.to_string())) {
        Ok(regex) => Ok(regex),
        Err(error) => Err(TokenizerError::InvalidPattern(error.clone())),
    }
}
