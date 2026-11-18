//! Tokenizers reconstructed from GGUF metadata.

mod bpe;
mod unigram;

pub use bpe::BpeTokenizer;
pub use unigram::UnigramTokenizer;

use crate::gguf::tokenizer::TokenizerError;

/// Decode a `<0xXX>` byte-fallback token.
pub(crate) fn byte_token(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || !token.starts_with("<0x") || !token.ends_with('>') {
        return None;
    }
    u8::from_str_radix(&token[3..5], 16).ok()
}

pub(crate) type Result<T> = std::result::Result<T, TokenizerError>;
