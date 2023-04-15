use llama_rs::{InferenceSession, InferenceSessionParameters, Model};
use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};
use zstd::{
    stream::{read::Decoder, write::Encoder},
    zstd_safe::CompressionLevel,
};

const SNAPSHOT_COMPRESSION_LEVEL: CompressionLevel = 1;

pub fn read_or_create_session(
    model: &Model,
    persist_session: Option<&Path>,
    load_session: Option<&Path>,
    inference_session_params: InferenceSessionParameters,
) -> (InferenceSession, bool) {
    fn load(model: &Model, path: &Path) -> InferenceSession {
        let file = unwrap_or_exit(File::open(path), || format!("Could not open file {path:?}"));
        let decoder = unwrap_or_exit(Decoder::new(BufReader::new(file)), || {
            format!("Could not create decoder for {path:?}")
        });
        let snapshot = unwrap_or_exit(bincode::deserialize_from(decoder), || {
            format!("Could not deserialize inference session from {path:?}")
        });
        let session = unwrap_or_exit(model.session_from_snapshot(snapshot), || {
            format!("Could not convert snapshot from {path:?} to session")
        });
        println!("Loaded inference session from {path:?}");
        session
    }

    match (persist_session, load_session) {
        (Some(path), _) if path.exists() => (load(model, path), true),
        (_, Some(path)) => (load(model, path), true),
        _ => (model.start_session(inference_session_params), false),
    }
}

pub fn write_session(mut session: llama_rs::InferenceSession, path: &Path) {
    // SAFETY: the session is consumed here, so nothing else can access it.
    let snapshot = unsafe { session.get_snapshot() };
    let file = unwrap_or_exit(File::create(path), || {
        format!("Could not create file {path:?}")
    });
    let encoder = unwrap_or_exit(
        Encoder::new(BufWriter::new(file), SNAPSHOT_COMPRESSION_LEVEL),
        || format!("Could not create encoder for {path:?}"),
    );
    unwrap_or_exit(
        bincode::serialize_into(encoder.auto_finish(), &snapshot),
        || format!("Could not serialize inference session to {path:?}"),
    );
    println!("Successfully wrote session to {path:?}");
}

fn unwrap_or_exit<T, E: Error>(result: Result<T, E>, error_message: impl Fn() -> String) -> T {
    match result {
        Ok(t) => t,
        Err(err) => {
            println!("{}. Error: {err}", error_message());
            std::process::exit(1);
        }
    }
}
