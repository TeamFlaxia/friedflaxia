use friedflaxia_core::gguf::tokenizer::{TokenType, TokenizerSpec};
use friedflaxia_core::tokenizer::UnigramTokenizer;

fn spec() -> TokenizerSpec {
    TokenizerSpec {
        model: "plamo3".to_owned(),
        tokens: [
            "a",
            "b",
            "ab",
            "abc",
            "<|plamo:bos|>",
            "<0xC3>",
            "<0xA9>",
            " ",
        ]
        .iter()
        .map(|token| (*token).to_owned())
        .collect(),
        scores: vec![0.0, -1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        token_types: vec![
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Control,
            TokenType::Byte,
            TokenType::Byte,
            TokenType::Normal,
        ],
        merges: Vec::new(),
        bos_token_id: 4,
        eos_token_id: 4,
        add_bos_token: true,
        add_eos_token: false,
        chat_template: None,
    }
}

#[test]
fn unigram_picks_the_highest_scoring_path() {
    let tokenizer = UnigramTokenizer::from_spec(&spec()).unwrap();
    assert_eq!(tokenizer.encode("ab", false).unwrap(), vec![2]);
    assert_eq!(tokenizer.encode("abc", false).unwrap(), vec![3]);
    assert_eq!(tokenizer.encode(" ab", false).unwrap(), vec![7, 2]);
}

#[test]
fn unigram_adds_bos_and_handles_special_tokens() {
    let tokenizer = UnigramTokenizer::from_spec(&spec()).unwrap();
    assert_eq!(tokenizer.encode("ab", true).unwrap(), vec![4, 2]);
    assert_eq!(
        tokenizer.encode("<|plamo:bos|>ab", false).unwrap(),
        vec![4, 2]
    );
    assert_eq!(tokenizer.decode(&[4, 2]).unwrap(), "ab");
}

#[test]
fn unigram_falls_back_to_byte_tokens() {
    let tokenizer = UnigramTokenizer::from_spec(&spec()).unwrap();
    assert_eq!(tokenizer.encode("é", false).unwrap(), vec![5, 6]);
    assert_eq!(tokenizer.decode(&[5, 6]).unwrap(), "é");
}
