use friedflaxia_wasm::Engine;
use std::env;
use std::fs;

const PROMPT: &str = "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello world<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>";

#[test]
#[ignore = "requires FRIEDFLAXIA_PLAMO_GGUF pointing at plamo-3-translate-0.8b-Q4_0.gguf"]
fn plamo_generate_uses_the_kv_cache() {
    let Ok(path) = env::var("FRIEDFLAXIA_PLAMO_GGUF") else {
        eprintln!("skipping: FRIEDFLAXIA_PLAMO_GGUF is not set");
        return;
    };
    let engine = Engine::new(fs::read(path).expect("read GGUF")).expect("Engine::new");
    let prompt = engine.encode(PROMPT, true).expect("encode");
    assert_eq!(prompt.len(), 21);
    let generated = engine.generate(&prompt, 4).expect("generate");
    let text = engine.decode(&generated).expect("decode");
    assert_eq!(text, "ハローワールド", "unexpected PLaMO output: {text:?}");
}
