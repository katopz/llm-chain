use std::{env, path::Path};

use llm_chain::{traits::StepExt, Parameters};
// use llm_chain_llama_rs::{new_instruct_template, Executor, Step};
use llm_chain_llama_rs::{
    cli_args::{self, Generate, Infer, ModelLoad, PromptFile},
    infer,
};

fn main() {
    // let res = Step::new(new_instruct_template(
    //     "Write \"{something}\" code in {language}.",
    // ))
    // .to_chain()
    // .run(
    //     Parameters::new()
    //         .with("something", "hello world")
    //         .with("language", "Rust"),
    //     Executor::new(
    //         Path::new("ggml-alpaca-7b-q4.bin")
    //             .to_str()
    //             .unwrap()
    //             .to_string(),
    //     ),
    // )
    // .await
    // .unwrap();
    // println!("{:?}", res.to_string());
}

#[tokio::test]
async fn hello() {
    let _infer = Infer {
        model_load: ModelLoad {
            model_path: "/Users/katopz/git/katopz/ggml-alpaca-7b-q4.bin".to_string(),
            num_ctx_tokens: 2048usize,
        },
        prompt_file: PromptFile { prompt_file: None },
        generate: Generate {
            num_threads: None,
            num_predict: None,
            batch_size: 8usize,
            repeat_last_n: 64usize,
            repeat_penalty: 1.3f32,
            temperature: 0.1f32,
            top_k: 40usize,
            top_p: 0.95f32,
            load_session: None,
            seed: None,
            float16: false,
            token_bias: None,
            ignore_eos: false,
        },
        prompt: Some("Say 1 to 3.".to_string()),
        save_session: None,
        persist_session: None,
    };
    let res = infer(&_infer);
}
