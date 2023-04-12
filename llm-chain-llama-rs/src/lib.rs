//! # llm-chain-llama
//!
//! Welcome to the world of `llm-chain-llama`! This crate supercharges your applications with the power of LLaMA (Large Language Model Applications), providing a robust framework for creating chains of LLaMA models to generate human-like text.
//!
//! Designed to work seamlessly with LLaMA models, `llm-chain-llama` makes it a breeze to build and execute complex text generation workflows, unlocking the potential of Large Language Models for your projects.
//!
//! # What's Inside? 🎁
//!
//! With `llm-chain-llama`, you'll be able to:
//!
//! - Generate text using LLaMA models
//! - Create custom text summarization workflows
//! - Perform complex tasks by chaining together different prompts and models 🧠
//!
//!
//! # Examples 📚
//!
//! Dive into the examples folder to discover how to harness the power of this crate. You'll find detailed examples that showcase how to generate text using LLaMA models, as well as how to chain the prompts together to create more complex workflows.
//!
//! So gear up, and let llm-chain-llama elevate your applications to new heights! With the combined powers of Large Language Models and the LLaMA framework, there's no limit to what you can achieve. 🌠🎊
//!
//! Happy coding, and enjoy the amazing world of LLMs with llm-chain-llama! 🥳🚀

// mod executor;
// mod instruct;
// mod output;
// mod step;
// mod tokenizer;

// pub use executor::Executor;
// pub use instruct::new_instruct_template;
// pub use output::Output;
// pub use step::{LlamaConfig, Step};

use std::{convert::Infallible, io::Write, path::Path};

use llama_rs::{InferenceSession, InferenceSessionParameters, Model};

pub mod cli_args;
mod snapshot;

pub fn infer(args: &cli_args::Infer) {
    let prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

{{PROMPT}}

### Response:
";
    let _prompt = args.prompt.clone().unwrap_or("".to_string());
    let prompt = process_prompt(prompt, &_prompt);
    let inference_session_params = args.generate.inference_session_parameters();
    let (model, vocabulary) = args.model_load.load();
    let (mut session, session_loaded) = load_session_from_disk(
        &model,
        args.persist_session.as_deref(),
        args.generate.load_session.as_deref(),
        inference_session_params,
    );
    let inference_params = args.generate.inference_parameters(session_loaded);

    let mut rng = args.generate.rng();
    let res = session.inference_with_prompt::<Infallible>(
        &model,
        &vocabulary,
        &inference_params,
        &prompt,
        args.generate.num_predict,
        &mut rng,
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();

            Ok(())
        },
    );
    println!();

    match res {
        Ok(_) => (),
        Err(llama_rs::InferenceError::ContextFull) => {
            println!("Context window full, stopping inference.")
        }
        Err(llama_rs::InferenceError::TokenizationFailed) => {
            println!("Failed to tokenize initial prompt.");
        }
        Err(llama_rs::InferenceError::UserCallback(_))
        | Err(llama_rs::InferenceError::EndOfText) => unreachable!("cannot fail"),
    }

    if let Some(session_path) = args.save_session.as_ref().or(args.persist_session.as_ref()) {
        // Write the memory to the cache file
        // SAFETY: no other model functions used inside the block
        unsafe {
            match snapshot::write_to_disk(&session.get_snapshot(), session_path) {
                Ok(_) => {
                    println!("Successfully wrote session to {session_path:?}");
                }
                Err(err) => {
                    println!("Could not write session at {session_path:?}: {err}");
                    std::process::exit(1);
                }
            }
        }
    }
}

fn load_prompt_file_with_prompt(
    prompt_file: &cli_args::PromptFile,
    prompt: Option<&str>,
) -> String {
    if let Some(prompt_file) = prompt_file.contents() {
        if let Some(prompt) = prompt {
            process_prompt(&prompt_file, prompt)
        } else {
            prompt_file
        }
    } else if let Some(prompt) = prompt {
        prompt.to_owned()
    } else {
        std::process::exit(1);
    }
}

pub fn load_session_from_disk(
    model: &Model,
    persist_session: Option<&Path>,
    load_session: Option<&Path>,
    inference_session_params: InferenceSessionParameters,
) -> (InferenceSession, bool) {
    fn load_snapshot_from_disk(model: &Model, path: &Path) -> InferenceSession {
        let snapshot = snapshot::load_from_disk(path);
        match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
            Ok(session) => {
                println!("Loaded inference session from {path:?}");
                session
            }
            Err(err) => {
                eprintln!("Could not load inference session. Error: {err}");
                std::process::exit(1);
            }
        }
    }

    match (persist_session, load_session) {
        (Some(path), _) if path.exists() => (load_snapshot_from_disk(model, path), true),
        (_, Some(path)) => (load_snapshot_from_disk(model, path), true),
        _ => (model.start_session(inference_session_params), false),
    }
}

fn process_prompt(raw_prompt: &str, prompt: &str) -> String {
    raw_prompt.replace("{{PROMPT}}", prompt)
}
