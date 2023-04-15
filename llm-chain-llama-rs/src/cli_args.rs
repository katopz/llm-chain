use llama_rs::{
    InferenceParameters, InferenceSessionParameters, ModelKVMemoryType, TokenBias, EOT_TOKEN_ID,
};
use rand::SeedableRng;
use std::path::PathBuf;

#[derive(Debug)]
pub struct Infer {
    pub model_load: ModelLoad,

    pub generate: Generate,

    /// The prompt to feed the generator.
    pub prompt: Option<String>,

    /// Saves an inference session at the given path. The same session can then be
    /// loaded from disk using `--load-session`.
    ///
    /// Use this with `-n 0` to save just the prompt
    pub save_session: Option<PathBuf>,

    /// Loads an inference session from the given path if present, and then saves
    /// the result to the same path after inference is completed.
    ///
    /// Equivalent to `--load-session` and `--save-session` with the same path,
    /// but will not error if the path does not exist
    pub persist_session: Option<PathBuf>,
}

#[derive(Debug)]
pub struct Generate {
    /// Sets the number of threads to use
    pub num_threads: Option<usize>,

    /// Sets how many tokens to predict
    pub num_predict: Option<usize>,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    pub batch_size: usize,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// option. In tokens.
    pub repeat_last_n: usize,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,

    /// Temperature
    pub temperature: f32,

    /// Top-K: The top K words by score are kept during sampling.
    pub top_k: usize,

    /// Top-p: The cumulative probability after which no more words are kept
    /// for sampling.
    pub top_p: f32,

    /// Loads a saved inference session from the given path, previously saved using
    /// `--save-session`
    pub load_session: Option<PathBuf>,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    pub seed: Option<u64>,

    /// Use 16-bit floats for model memory key and value. Ignored when restoring
    /// from the cache.
    pub float16: bool,

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    pub token_bias: Option<TokenBias>,

    /// Prevent the end of stream (EOS/EOD) token from being generated. This will allow the
    /// model to generate text until it runs out of context space. Note: The --token-bias
    /// option will override this if specified.
    pub ignore_eos: bool,
}

impl Generate {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn autodetect_num_threads(&self) -> usize {
        std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.perflevel0.physicalcpu")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok()?.trim().parse().ok())
            .unwrap_or(num_cpus::get_physical())
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn autodetect_num_threads(&self) -> usize {
        num_cpus::get_physical()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
            .unwrap_or_else(|| self.autodetect_num_threads())
    }

    pub fn inference_session_parameters(&self) -> InferenceSessionParameters {
        let mem_typ = if self.float16 {
            ModelKVMemoryType::Float16
        } else {
            ModelKVMemoryType::Float32
        };
        InferenceSessionParameters {
            memory_k_type: mem_typ,
            memory_v_type: mem_typ,
            repetition_penalty_last_n: self.repeat_last_n,
        }
    }

    pub fn rng(&self) -> rand::rngs::StdRng {
        if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        }
    }

    pub fn inference_parameters(&self, session_loaded: bool) -> InferenceParameters {
        InferenceParameters {
            n_threads: self.num_threads(),
            n_batch: self.batch_size,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
            temperature: self.temperature,
            bias_tokens: self.token_bias.clone().unwrap_or_else(|| {
                if self.ignore_eos {
                    TokenBias::new(vec![(EOT_TOKEN_ID, -1.0)])
                } else {
                    TokenBias::default()
                }
            }),
            play_back_previous_tokens: session_loaded,
        }
    }
}

#[derive(Debug)]
pub struct ModelLoad {
    /// Where to load the model path from
    pub model_path: String,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory.
    ///
    /// LLaMA models are trained with a context size of 2048 tokens. If you
    /// want to use a larger context size, you will need to retrain the model,
    /// or use a model that was trained with a larger context size.
    ///
    /// Alternate methods to extend the context, including
    /// [context clearing](https://github.com/rustformers/llama-rs/issues/77) are
    /// being investigated, but are not yet implemented. Additionally, these
    /// will likely not perform as well as a model with a larger context size.
    pub num_ctx_tokens: usize,
}
impl ModelLoad {
    pub fn load(&self) -> (llama_rs::Model, llama_rs::Vocabulary) {
        let (model, vocabulary) =
            llama_rs::Model::load(&self.model_path, self.num_ctx_tokens, |progress| {
                use llama_rs::LoadProgress;
                match progress {
                    LoadProgress::HyperparametersLoaded(hparams) => {
                        println!("Loaded hyperparameters {hparams:#?}")
                    }
                    LoadProgress::ContextSize { bytes } => println!(
                        "ggml ctx size = {:.2} MB\n",
                        bytes as f64 / (1024.0 * 1024.0)
                    ),
                    LoadProgress::PartLoading {
                        file,
                        current_part,
                        total_parts,
                    } => {
                        let current_part = current_part + 1;
                        println!(
                            "Loading model part {}/{} from '{}'\n",
                            current_part,
                            total_parts,
                            file.to_string_lossy(),
                        )
                    }
                    LoadProgress::PartTensorLoaded {
                        current_tensor,
                        tensor_count,
                        ..
                    } => {
                        let current_tensor = current_tensor + 1;
                        if current_tensor % 8 == 0 {
                            println!("Loaded tensor {current_tensor}/{tensor_count}");
                        }
                    }
                    LoadProgress::PartLoaded {
                        file,
                        byte_size,
                        tensor_count,
                    } => {
                        println!("Loading of '{}' complete", file.to_string_lossy());
                        println!(
                            "Model size = {:.2} MB / num tensors = {}",
                            byte_size as f64 / 1024.0 / 1024.0,
                            tensor_count
                        );
                    }
                }
            })
            .expect("Could not load model");

        println!("Model fully loaded!");

        (model, vocabulary)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ElementType {
    /// Quantized 4-bit (type 0).
    Q4_0,
    /// Quantized 4-bit (type 1); used by GPTQ.
    Q4_1,
    /// Float 16-bit.
    F16,
    /// Float 32-bit.
    F32,
}
impl From<ElementType> for llama_rs::ElementType {
    fn from(model_type: ElementType) -> Self {
        match model_type {
            ElementType::Q4_0 => llama_rs::ElementType::Q4_0,
            ElementType::Q4_1 => llama_rs::ElementType::Q4_1,
            ElementType::F16 => llama_rs::ElementType::F16,
            ElementType::F32 => llama_rs::ElementType::F32,
        }
    }
}
