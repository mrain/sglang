//! Round-trip tests: decode each golden fixture with the Rust schema codec
//! and verify re-encoded bytes match the original byte-for-byte.
//!
//! This is the primary Phase 0 assertion: if Rust-produced msgpack matches
//! Python-produced msgpack, the schema codec is correct.

use std::path::Path;

const FIXTURE_DIR: &str = "tools/rust-tkm-migration/fixtures";

/// Find repo root by walking up for .gitignore.
fn repo_root() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
    loop {
        if dir.join(".gitignore").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("could not find repo root (no .gitignore found in ancestors)");
        }
    }
}

/// Read a fixture, decode with the correct Rust struct, re-encode, compare bytes.
fn roundtrip_fixture(name: &str) {
    let repo_root = repo_root();
    let path = repo_root
        .join(FIXTURE_DIR)
        .join(format!("{}.msgpack", name));
    let fixture_bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {:?}: {}", path, e));

    let re_encoded = match name {
        "single_text_generate"
        | "input_ids_generate"
        | "empty_fields"
        | "reasoner_request"
        | "session_request" => {
            let obj = sglang_server::schema::TokenizedGenerateReqInput::decode(&fixture_bytes)
                .expect("decode TokenizedGenerateReqInput");
            obj.encode().expect("encode")
        }
        "abort_request" => {
            let obj =
                sglang_server::schema::AbortReq::decode(&fixture_bytes).expect("decode AbortReq");
            obj.encode().expect("encode")
        }
        "flush_cache_request" => {
            let obj = sglang_server::schema::FlushCacheReqInput::decode(&fixture_bytes)
                .expect("decode FlushCacheReqInput");
            obj.encode().expect("encode")
        }
        "profile_request" => {
            let obj = sglang_server::schema::ProfileReq::decode(&fixture_bytes)
                .expect("decode ProfileReq");
            obj.encode().expect("encode")
        }
        "streaming_output_chunk"
        | "final_streaming_output"
        | "logprobs_output"
        | "cached_tokens_details" => {
            let obj = sglang_server::schema::BatchStrOutput::decode(&fixture_bytes)
                .expect("decode BatchStrOutput");
            obj.encode().expect("encode")
        }
        "token_id_output" => {
            let obj = sglang_server::schema::BatchTokenIDOutput::decode(&fixture_bytes)
                .expect("decode BatchTokenIDOutput");
            obj.encode().expect("encode")
        }
        "embedding_request" => {
            let obj = sglang_server::schema::TokenizedEmbeddingReqInput::decode(&fixture_bytes)
                .expect("decode TokenizedEmbeddingReqInput");
            obj.encode().expect("encode")
        }
        "embedding_output" => {
            let obj = sglang_server::schema::BatchEmbeddingOutput::decode(&fixture_bytes)
                .expect("decode BatchEmbeddingOutput");
            obj.encode().expect("encode")
        }
        _ => panic!("unknown fixture: {}", name),
    };

    assert_eq!(
        fixture_bytes,
        re_encoded,
        "{}: bytes differ\n  original={} bytes\n  reencoded={} bytes",
        name,
        fixture_bytes.len(),
        re_encoded.len(),
    );
}

// ── Individual tests ──

#[test]
fn all_fixtures_exist() {
    let repo_root = repo_root();
    let fixture_dir = repo_root.join(FIXTURE_DIR);
    assert!(
        fixture_dir.exists(),
        "fixture dir not found: {:?}",
        fixture_dir
    );
    assert!(
        fixture_dir.join("manifest.json").exists(),
        "manifest.json not found"
    );
}

#[test]
fn roundtrip_single_text_generate() {
    roundtrip_fixture("single_text_generate");
}

#[test]
fn roundtrip_input_ids_generate() {
    roundtrip_fixture("input_ids_generate");
}

#[test]
fn roundtrip_abort_request() {
    roundtrip_fixture("abort_request");
}

#[test]
fn roundtrip_flush_cache() {
    roundtrip_fixture("flush_cache_request");
}

#[test]
fn roundtrip_profile() {
    roundtrip_fixture("profile_request");
}

#[test]
fn roundtrip_streaming_chunk() {
    roundtrip_fixture("streaming_output_chunk");
}

#[test]
fn roundtrip_final_streaming() {
    roundtrip_fixture("final_streaming_output");
}

#[test]
fn roundtrip_logprobs() {
    roundtrip_fixture("logprobs_output");
}

#[test]
fn roundtrip_token_id_output() {
    roundtrip_fixture("token_id_output");
}

#[test]
fn roundtrip_embedding_request() {
    roundtrip_fixture("embedding_request");
}

#[test]
fn roundtrip_embedding_output() {
    roundtrip_fixture("embedding_output");
}

#[test]
fn roundtrip_empty() {
    roundtrip_fixture("empty_fields");
}

#[test]
fn roundtrip_cached_tokens() {
    roundtrip_fixture("cached_tokens_details");
}

#[test]
fn roundtrip_reasoner() {
    roundtrip_fixture("reasoner_request");
}

#[test]
fn roundtrip_session() {
    roundtrip_fixture("session_request");
}

#[test]
fn roundtrip_pickle_wrapper() {
    let repo_root = repo_root();
    let path = repo_root.join(FIXTURE_DIR).join("pickle_wrapper.msgpack");
    let fixture_bytes = std::fs::read(&path).unwrap();

    let val: rmpv::Value = rmpv::decode::read_value(&mut &fixture_bytes[..]).unwrap();
    let arr = val.as_array().unwrap();
    assert!(arr.len() >= 2, "pickle_wrapper should have >=2 items");

    for item in arr {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, item).unwrap();
        let obj = sglang_server::schema::PickleWrapper::decode(&buf).expect("decode");
        let re_encoded = obj.encode().expect("encode");
        assert_eq!(buf, re_encoded, "PickleWrapper round-trip");
    }
}
