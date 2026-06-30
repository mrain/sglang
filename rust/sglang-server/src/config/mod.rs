//! Typed views into the Python ServerArgs / ModelConfig / PortArgs blobs.
//!
//! These deserialize from the JSON dump the Python scheduler creates at startup.
//! Only fields the Rust server currently accesses are typed — the rest live in
//! the `extra` bucket so nothing is silently lost.

use serde::Deserialize;

// ── ModelConfigView ──

/// Typed view of `model_config` (attached to `server_args` before the dump).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfigView {
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_context_len")]
    pub context_len: u64,
    #[serde(default)]
    pub is_generation: bool,
    #[serde(default)]
    pub image_token_id: Option<i64>,
    #[serde(default)]
    pub vocab_size: Option<i64>,
    /// Extra model_config fields kept for forward compat and custom accessors.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

fn default_context_len() -> u64 {
    4096
}

// ── ServerArgsView ──

/// Typed view of the Python `ServerArgs` dump.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerArgsView {
    // ── Identity ──
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub served_model_name: String,
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,

    // ── Tokenizer ──
    #[serde(default)]
    pub tokenizer_path: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub tokenizer_mode: String,
    #[serde(default)]
    pub skip_tokenizer_init: bool,
    #[serde(default = "default_one")]
    pub tokenizer_worker_num: usize,
    #[serde(default = "default_one")]
    pub detokenizer_worker_num: usize,

    // ── Model config (nested blob, deserialized separately) ──
    #[serde(default)]
    pub model_config: Option<ModelConfigView>,

    // ── Inference ──
    #[serde(default)]
    pub disable_radix_cache: bool,
    #[serde(default)]
    pub disable_regex_jump_forward: bool,
    #[serde(default)]
    pub enable_mixed_chunk: bool,
    #[serde(default)]
    pub max_bs_size: Option<usize>,

    // ── Speculative decoding ──
    #[serde(default)]
    pub speculative_algorithm: String,

    // ── Disaggregation ──
    #[serde(default)]
    pub disaggregation_mode: String,

    // ── Metrics / observability ──
    #[serde(default)]
    pub enable_metrics: bool,

    // ── LoRA ──
    #[serde(default)]
    pub enable_lora: bool,
    #[serde(default)]
    pub lora_paths: Vec<String>,

    // ── Chat / parser ──
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub tool_call_parser: Option<String>,

    // ── Extra (any field not listed above) ──
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

fn default_port() -> u16 {
    30000
}
fn default_one() -> usize {
    1
}

impl ServerArgsView {
    pub fn context_len(&self) -> u64 {
        self.model_config
            .as_ref()
            .map(|m| m.context_len)
            .unwrap_or(4096)
    }
}

// ── PortArgsView ──

/// Typed view of the ZMQ port layout for inter-process communication.
#[derive(Debug, Clone, Deserialize)]
pub struct PortArgsView {
    #[serde(default)]
    pub tokenizer_ipc_name: String,
    #[serde(default)]
    pub scheduler_input_ipc_name: String,
    #[serde(default)]
    pub detokenizer_ipc_name: String,
    #[serde(default)]
    pub tokenizer_worker_ipc_name: String,
    #[serde(default)]
    pub scheduler_ipc_name: String,
    /// Extra port fields kept for forward compat.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

// ── Top-level config wrapper ──

/// Deserialized from the `server_args_json` blob the Python side passes at startup.
#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfigView {
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub served_model_name: String,
    #[serde(default)]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub tokenizer_path: String,
    #[serde(default)]
    pub revision: String,
    #[serde(default)]
    pub skip_tokenizer_init: bool,
    #[serde(default = "default_one")]
    pub tokenizer_worker_num: usize,
    #[serde(default = "default_one")]
    pub detokenizer_worker_num: usize,
    #[serde(default)]
    pub model_config: Option<ModelConfigView>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

impl RuntimeConfigView {
    /// Parse from the JSON blob the Python scheduler passes at startup.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("config parse error: {e}"))
    }

    pub fn context_len(&self) -> u64 {
        self.model_config
            .as_ref()
            .map(|m| m.context_len)
            .unwrap_or(4096)
    }

    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn resolved_tokenizer_path(&self) -> Option<String> {
        let tp = self.tokenizer_path.trim();
        let mp = self.model_path.trim();
        if !tp.is_empty() {
            Some(tp.to_owned())
        } else if !mp.is_empty() {
            Some(mp.to_owned())
        } else {
            None
        }
    }
}
