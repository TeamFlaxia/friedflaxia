use friedflaxia_core::gguf::tokenizer::{TokenType, TokenizerSpec};
use friedflaxia_core::tokenizer::BpeTokenizer;

fn spec(model: &str, tokens: &[&str], merges: &[&str]) -> TokenizerSpec {
    TokenizerSpec {
        model: model.to_owned(),
        tokens: tokens.iter().map(|token| (*token).to_owned()).collect(),
        scores: Vec::new(),
        token_types: vec![TokenType::Normal; tokens.len()],
        merges: merges.iter().map(|merge| (*merge).to_owned()).collect(),
        bos_token_id: u32::MAX,
        eos_token_id: u32::MAX,
        add_bos_token: false,
        add_eos_token: false,
        chat_template: None,
    }
}

#[test]
fn gemma_bpe_merges_and_decodes_metaspace() {
    let spec = spec("gemma4", &["▁", "a", "b", "ab"], &["a b"]);
    let tokenizer = BpeTokenizer::from_spec(&spec).expect("gemma BPE");

    assert_eq!(tokenizer.encode("ab", false).unwrap(), vec![3]);
    assert_eq!(tokenizer.encode(" ab", false).unwrap(), vec![0, 3]);
    assert_eq!(tokenizer.decode(&[0, 3]).unwrap(), " ab");
}

#[test]
fn gemma_adds_bos_and_eos_when_requested() {
    let mut spec = spec("gemma4", &["▁", "a", "b", "ab", "<s>", "</s>"], &["a b"]);
    spec.bos_token_id = 4;
    spec.eos_token_id = 5;
    spec.add_bos_token = true;
    spec.add_eos_token = true;
    let tokenizer = BpeTokenizer::from_spec(&spec).expect("gemma BPE");

    assert_eq!(tokenizer.encode("a", true).unwrap(), vec![4, 1, 5]);
    assert_eq!(tokenizer.decode(&[4, 1, 5]).unwrap(), "a");
}

#[test]
fn gpt2_byte_level_round_trips_spaces() {
    let spec = spec("gpt2", &["Ġ", "a", "Ġa", "b"], &["Ġ a"]);
    let tokenizer = BpeTokenizer::from_spec(&spec).expect("gpt2 BPE");

    assert_eq!(tokenizer.encode(" a", false).unwrap(), vec![2]);
    assert_eq!(tokenizer.decode(&[2, 3]).unwrap(), " ab");
}

#[test]
fn unknown_gemma_characters_fall_back_to_byte_tokens() {
    let mut tokens = vec!["▁", "a", "b"];
    let mut spec = spec("gemma4", &tokens, &[]);
    spec.token_types = vec![TokenType::Normal; tokens.len()];
    // Add byte tokens for the UTF-8 encoding of "é" (0xC3 0xA9).
    tokens.extend(["<0xC3>", "<0xA9>"]);
    spec.tokens
        .extend(["<0xC3>".to_owned(), "<0xA9>".to_owned()]);
    spec.token_types.extend([TokenType::Byte, TokenType::Byte]);

    let tokenizer = BpeTokenizer::from_spec(&spec).expect("gemma BPE");
    assert_eq!(tokenizer.encode("é", false).unwrap(), vec![3, 4]);
    assert_eq!(tokenizer.decode(&[3, 4]).unwrap(), "é");
}
