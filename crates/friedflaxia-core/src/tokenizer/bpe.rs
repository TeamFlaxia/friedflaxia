use super::{Result, byte_token};
use crate::gguf::tokenizer::{TokenType, TokenizerError, TokenizerSpec};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// GPT-2-style word split used by the d1-omni tokenizer.
const GPT2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpeMode {
    Gpt2,
    Gemma,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    token_types: Vec<TokenType>,
    tokens: Vec<String>,
    mode: BpeMode,
    bos_token_id: u32,
    eos_token_id: u32,
    add_bos: bool,
    add_eos: bool,
}

impl BpeTokenizer {
    pub fn from_spec(spec: &TokenizerSpec) -> Result<Self> {
        let mode = match spec.model.as_str() {
            "gpt2" => BpeMode::Gpt2,
            model if model.starts_with("gemma") => BpeMode::Gemma,
            _ => return Err(TokenizerError::UnsupportedModel(spec.model.clone())),
        };
        let mut vocab = HashMap::with_capacity(spec.tokens.len());
        for (id, token) in spec.tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }
        let mut merges = HashMap::with_capacity(spec.merges.len());
        for (rank, merge) in spec.merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(TokenizerError::InvalidMerge(merge.clone()));
            };
            merges.insert((left.to_owned(), right.to_owned()), rank as u32);
        }
        Ok(Self {
            vocab,
            merges,
            token_types: spec.token_types.clone(),
            tokens: spec.tokens.clone(),
            mode,
            bos_token_id: spec.bos_token_id,
            eos_token_id: spec.eos_token_id,
            // The d1 tokenizer's post-processor always prepends <|startoftext|>,
            // even though the GGUF add_bos flag is false.
            add_bos: spec.add_bos_token || mode == BpeMode::Gpt2,
            add_eos: spec.add_eos_token,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_special_tokens && self.add_bos {
            ids.push(self.bos_token_id);
        }
        match self.mode {
            BpeMode::Gpt2 => {
                let regex = gpt2_regex()?;
                for word in regex.find_iter(text) {
                    let word =
                        word.map_err(|error| TokenizerError::InvalidPattern(error.to_string()))?;
                    let mapped = bytes_to_unicode_string(word.as_str().as_bytes());
                    self.encode_piece(&mapped, &mut ids);
                }
            }
            BpeMode::Gemma => {
                let normalized = text.replace(' ', "▁");
                self.encode_piece(&normalized, &mut ids);
            }
        }
        if add_special_tokens && self.add_eos {
            ids.push(self.eos_token_id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self.mode {
            BpeMode::Gpt2 => {
                let mut bytes = Vec::new();
                for id in ids {
                    if self.is_special(*id) {
                        continue;
                    }
                    let token = self.token(*id)?;
                    for character in token.chars() {
                        if let Some(byte) = unicode_to_byte(character) {
                            bytes.push(byte);
                        } else {
                            bytes.extend_from_slice(character.to_string().as_bytes());
                        }
                    }
                }
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            BpeMode::Gemma => {
                let mut bytes = Vec::new();
                for id in ids {
                    if self.is_special(*id) {
                        continue;
                    }
                    let token = self.token(*id)?;
                    if let Some(byte) = byte_token(token) {
                        bytes.push(byte);
                    } else {
                        bytes.extend_from_slice(token.as_bytes());
                    }
                }
                let text = String::from_utf8_lossy(&bytes).into_owned();
                Ok(text.replace('▁', " "))
            }
        }
    }

    fn encode_piece(&self, piece: &str, ids: &mut Vec<u32>) {
        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        while let Some((best_index, _rank)) = self.best_merge(&symbols) {
            let left = symbols[best_index].clone();
            let right = symbols[best_index + 1].clone();
            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(symbols[index].clone());
                    index += 1;
                }
            }
            symbols = merged;
        }

        for symbol in symbols {
            if let Some(id) = self.vocab.get(&symbol) {
                ids.push(*id);
            } else {
                // Byte fallback: encode the UTF-8 bytes with <0xXX> tokens.
                let mut wrote_byte = false;
                for byte in symbol.as_bytes() {
                    let token = format!("<0x{byte:02X}>");
                    if let Some(id) = self.vocab.get(&token) {
                        ids.push(*id);
                        wrote_byte = true;
                    }
                }
                if !wrote_byte {
                    ids.push(0);
                }
            }
        }
    }

    fn best_merge(&self, symbols: &[String]) -> Option<(usize, u32)> {
        let mut best: Option<(usize, u32)> = None;
        for index in 0..symbols.len().saturating_sub(1) {
            let key = (symbols[index].clone(), symbols[index + 1].clone());
            if let Some(rank) = self.merges.get(&key) {
                if best.is_none_or(|(_, best_rank)| *rank < best_rank) {
                    best = Some((index, *rank));
                }
            }
        }
        best
    }

    fn is_special(&self, id: u32) -> bool {
        id == self.bos_token_id
            || id == self.eos_token_id
            || self
                .token_types
                .get(id as usize)
                .is_some_and(|token_type| *token_type == TokenType::Control)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.token_types
            .get(id as usize)
            .is_some_and(|token_type| *token_type == TokenType::Control)
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn token(&self, id: u32) -> Result<&str> {
        self.tokens
            .get(id as usize)
            .map(String::as_str)
            .ok_or(TokenizerError::InvalidTokenId(id))
    }
}

fn gpt2_regex() -> Result<&'static Regex> {
    static REGEX: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();
    match REGEX.get_or_init(|| Regex::new(GPT2_PATTERN).map_err(|error| error.to_string())) {
        Ok(regex) => Ok(regex),
        Err(error) => Err(TokenizerError::InvalidPattern(error.clone())),
    }
}

/// GPT-2's reversible byte-to-unicode map.
fn byte_to_unicode(byte: u8) -> char {
    let mut table = [0u16; 256];
    let mut next = 0u16;
    for value in 0u16..=255 {
        let is_printable = (33..=126).contains(&value)
            || (161..=172).contains(&value)
            || (174..=255).contains(&value);
        if is_printable {
            table[value as usize] = value;
        } else {
            table[value as usize] = 256 + next;
            next += 1;
        }
    }
    char::from_u32(table[byte as usize] as u32).unwrap_or('\u{fffd}')
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let code = character as u32;
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return Some(code as u8);
    }
    if (256..=511).contains(&code) {
        let mut next = 0u32;
        for value in 0u32..=255 {
            let is_printable = (33..=126).contains(&value)
                || (161..=172).contains(&value)
                || (174..=255).contains(&value);
            if !is_printable {
                if 256 + next == code {
                    return Some(value as u8);
                }
                next += 1;
            }
        }
    }
    None
}

fn bytes_to_unicode_string(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| byte_to_unicode(*byte)).collect()
}
