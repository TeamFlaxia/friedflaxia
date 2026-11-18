use friedflaxia_core::gguf::tokenizer::TokenizerSpec;
use friedflaxia_core::gguf::{Gguf, TensorType, Value};
use std::env;
use std::fs;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: inspect <gguf header or file>");
    let bytes = fs::read(&path).expect("read GGUF");
    let gguf = Gguf::parse(&bytes).expect("parse GGUF");

    println!(
        "version={} tensors={} data_offset={} alignment={}",
        gguf.version,
        gguf.tensor_count(),
        gguf.data_offset(),
        gguf.alignment()
    );
    for key in [
        "general.architecture",
        "general.name",
        "general.alignment",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.chat_template",
        "general.file_type",
        "pooling_type",
        "decision_type",
    ] {
        if let Some(value) = gguf.metadata(key) {
            println!("meta {key} = {}", summarize(value));
        }
    }

    if let Ok(tokenizer) = TokenizerSpec::from_gguf(&gguf) {
        println!(
            "tokenizer model={} vocab={} scores={} merges={} bos={} eos={} add_bos={} add_eos={}",
            tokenizer.model,
            tokenizer.vocab_size(),
            tokenizer.scores.len(),
            tokenizer.merges.len(),
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.add_bos_token,
            tokenizer.add_eos_token
        );
    } else if gguf.metadata("tokenizer.ggml.tokens").is_some() {
        println!("tokenizer metadata present but not readable");
    }

    let mut entries: Vec<(&str, &Value)> = gguf.metadata_entries().collect();
    entries.sort_by_key(|(key, _)| *key);
    for (key, value) in entries {
        if !matches!(value, Value::Array(_)) {
            println!("kv {key} = {}", summarize(value));
        }
    }

    for tensor in gguf.tensors() {
        println!(
            "tensor {} type={:?} dims={:?} elements={} bytes={:?} offset={}",
            tensor.name,
            tensor.tensor_type,
            tensor.dims,
            tensor.element_count(),
            tensor.byte_len(),
            tensor.offset
        );
    }

    let types: Vec<TensorType> = gguf
        .tensors()
        .iter()
        .map(|tensor| tensor.tensor_type)
        .collect();
    let mut unique = types;
    unique.sort_by_key(|tensor_type| tensor_type.as_u32());
    unique.dedup();
    println!("tensor types: {unique:?}");
}

fn summarize(value: &Value) -> String {
    match value {
        Value::String(text) => {
            let preview: String = text.chars().take(80).collect();
            format!("string(len={}, preview={:?})", text.len(), preview)
        }
        Value::Array(values) => format!("array(len={})", values.len()),
        Value::Bool(value) => format!("bool({value})"),
        Value::U8(value) => format!("u8({value})"),
        Value::I8(value) => format!("i8({value})"),
        Value::U16(value) => format!("u16({value})"),
        Value::I16(value) => format!("i16({value})"),
        Value::U32(value) => format!("u32({value})"),
        Value::I32(value) => format!("i32({value})"),
        Value::U64(value) => format!("u64({value})"),
        Value::I64(value) => format!("i64({value})"),
        Value::F32(value) => format!("f32({value})"),
        Value::F64(value) => format!("f64({value})"),
    }
}
