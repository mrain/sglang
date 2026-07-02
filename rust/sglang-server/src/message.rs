//! Messages moved between stages. All payloads are *moved* through `flume`
//! channels (zero copy); variable-length buffers are `bytes::Bytes` so the
//! egress fan-out to detokenizer shards is a refcount bump, never a copy.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokio::sync::mpsc;

use crate::error::Error;
use crate::fsm::RequestState;
use crate::ids::RequestId;

/// Serde default-fn for fields where Python normalizes `None` → `-1`.
fn minus_one() -> i64 {
    -1
}

/// Sink the API-server connection handler reads from to emit SSE frames.
/// One per request, bounded for backpressure. The FSM owner holds the sender;
/// dropping it (disconnect) is observed by the handler as stream end.
pub type EgressSink = mpsc::Sender<EgressItem>;
#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type EgressSource = mpsc::Receiver<EgressItem>;

/// Protocol-neutral output for one generation step (cumulative values). The
/// detok shard produces these; each API handler formats them into its own wire
/// shape — SGLang `/generate`, OpenAI `/v1/completions`, `/v1/chat/completions`.
#[derive(Debug, Clone, Default)]
pub struct GenerationOutput {
    pub rid: String,
    /// Cumulative decoded text (empty in `skip_tokenizer_init` mode).
    pub text: String,
    /// Cumulative output token ids (`skip_tokenizer_init` mode; empty otherwise).
    pub output_ids: Vec<i32>,
    /// Prompt token count (from the scheduler; constant across the request).
    pub prompt_tokens: u32,
    /// Cumulative output token count.
    pub completion_tokens: u64,
    /// Number of output tokens already emitted in prior streaming frames. The
    /// handler slices cumulative logprobs at this offset to produce incremental
    /// deltas (matches Python `incremental_streaming_output` behavior).
    pub last_output_offset: usize,
    /// `Some(reason)` on the final step, `None` while streaming.
    pub finish_reason: Option<String>,
    /// Output token logprobs (cumulative, per-position).
    pub output_token_logprobs: Vec<f32>,
    /// Output token IDs for logprobs (position-aligned).
    pub output_token_logprob_ids: Vec<i32>,
    /// Top-k logprobs for output tokens.
    pub output_top_logprobs: Vec<Vec<(f32, i32)>>,
    /// Input token logprobs (cumulative, per-position; empty when return_logprob is off).
    pub input_token_logprobs: Vec<f32>,
    /// Input token IDs for logprobs.
    pub input_token_logprob_ids: Vec<i32>,
    /// Input top-k logprobs per position.
    pub input_top_logprobs: Vec<Vec<(f32, i32)>>,
    /// Prompt token IDs, included when `return_prompt_token_ids` is true.
    pub prompt_token_ids: Option<Vec<i32>>,
    /// Decoded token text for output logprobs (position-aligned with
    /// `output_token_logprobs`). Empty when `return_text_in_logprobs` is
    /// false, or in skip_tokenizer_init mode.
    pub output_token_logprob_texts: Vec<Option<String>>,
    /// Decoded token text for input logprobs (position-aligned with
    /// `input_token_logprobs`).
    pub input_token_logprob_texts: Vec<Option<String>>,
}

/// What the connection handler receives on the egress stream. Generation output
/// is protocol-neutral (the handler formats it); control results are a single
/// verbatim payload.
#[derive(Debug)]
pub enum EgressItem {
    /// An intermediate streamed generation step (only sent for streaming reqs).
    Frame(GenerationOutput),
    /// The final generation step.
    Done(GenerationOutput),
    /// A control-request result: one verbatim payload (e.g. `/server_info`),
    /// delivered as-is with no per-protocol formatting.
    Control(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// What kind of request this is — selects the ingress branch, the wire message
/// pushed to the scheduler, and the egress shape. Each variant owns its own
/// body, so the type system keeps generate fields off control requests (and
/// vice versa); a control endpoint migrated with parameters grows
/// `ControlRequest` rather than abusing the generate payload.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(GenerateRequest),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization, and
    /// the egress is a single non-streamed JSON result.
    Control(ControlRequest),
}

impl RequestKind {
    /// Whether the client asked for SSE streaming. Always false for control
    /// requests (their response is a single result, never streamed).
    pub fn is_stream(&self) -> bool {
        match self {
            RequestKind::Generate(g) => g.stream,
            RequestKind::Control(_) => false,
        }
    }
}

/// Body of a `/generate` request.
#[derive(Debug)]
pub struct GenerateRequest {
    /// Decoded HTTP body (the `GenerateReqInput` view we need for tokenization).
    pub payload: GeneratePayload,
    /// Token ids, populated by the Tokenizer stage (or already present from the
    /// client).
    pub input_ids: Option<Vec<i32>>,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
}

/// Body of a control request. `tag` is the scheduler request-struct name
/// (msgspec class, e.g. `"GetInternalStateReq"`) pushed as a bare
/// `[tag, rid, nil]`. Typed params for migrated control endpoints land here.
#[derive(Debug)]
pub struct ControlRequest {
    pub tag: &'static str,
}

/// The owned request as it travels ingress stages. Single owner at all times,
/// so the embedded `state` FSM is mutated without any lock. Fields common to
/// every request live here; variant-specific data lives in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    pub id: RequestId,
    pub state: RequestState,
    /// Back-channel to the client connection for egress frames.
    pub sink: EgressSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// Encode a bare `BaseReq` control message (just `rid` + `http_worker_ipc`) as
/// the msgspec tagged array `[tag, rid, nil]`. Used for control requests like
/// `GetInternalStateReq` that carry no extra fields.
pub fn control_req_msgpack(tag: &str, rid: &str) -> Result<Bytes, Error> {
    use rmpv::Value;
    let arr = Value::Array(vec![
        Value::from(tag), // struct tag
        Value::from(rid), // rid
        Value::Nil,       // http_worker_ipc
    ]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
    Ok(Bytes::from(buf))
}

/// Accept `text: "..."` (single) or `text: ["...", "..."]` (batch), matching
/// Python's `GenerateReqInput.text: Optional[Union[str, List[str]]]`.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum TextInput {
    Single(String),
    Batch(Vec<String>),
}

impl TextInput {
    /// Return the single text, or `None` if batch.
    pub fn into_single(self) -> Option<String> {
        match self {
            TextInput::Single(s) => Some(s),
            TextInput::Batch(_) => None,
        }
    }
}

// Custom (de)serialization so JSON `"hello"` → Single, `["a","b"]` → Batch.
impl serde::Serialize for TextInput {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            TextInput::Single(s) => s.serialize(serializer),
            TextInput::Batch(v) => v.serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for TextInput {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Try string first (most common), then array of strings.
        use serde::de::Error;
        let val = serde_json::Value::deserialize(deserializer)?;
        match val {
            serde_json::Value::String(s) => Ok(TextInput::Single(s)),
            serde_json::Value::Array(arr) => {
                let strs: Vec<String> = arr
                    .into_iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => Ok(s),
                        other => Err(D::Error::custom(format!(
                            "expected string in text array, got {}",
                            other
                        ))),
                    })
                    .collect::<Result<_, _>>()?;
                Ok(TextInput::Batch(strs))
            }
            _ => Err(D::Error::custom(format!(
                "expected string or array of strings for text, got {}",
                val
            ))),
        }
    }
}

/// Accept `[1, 2, 3]` (single) or `[[1, 2], [3, 4]]` (batch), matching
/// Python's `GenerateReqInput.input_ids: Optional[Union[List[int], List[List[int]]]]`.
#[derive(Debug, Clone)]
pub enum IdsInput {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

impl IdsInput {
    /// Return the single ids vec, or `None` if batch.
    #[allow(dead_code)]
    pub fn into_single(self) -> Option<Vec<i32>> {
        match self {
            IdsInput::Single(v) => Some(v),
            IdsInput::Batch(_) => None,
        }
    }

    /// Length of the inner ids, or 0 for batch.
    pub fn len(&self) -> usize {
        match self {
            IdsInput::Single(v) => v.len(),
            IdsInput::Batch(_) => 0,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the single ids vec, or `None` if batch.
    pub fn as_single(&self) -> Option<&Vec<i32>> {
        match self {
            IdsInput::Single(v) => Some(v),
            IdsInput::Batch(_) => None,
        }
    }
}

impl serde::Serialize for IdsInput {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            IdsInput::Single(v) => v.serialize(serializer),
            IdsInput::Batch(v) => v.serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for IdsInput {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let val = serde_json::Value::deserialize(deserializer)?;
        match val {
            serde_json::Value::Array(arr) => {
                // Check if first element is an array → batch, else single.
                if arr.first().and_then(|v| v.as_array()).is_some() {
                    let batches: Vec<Vec<i32>> = arr
                        .into_iter()
                        .map(|v| {
                            v.as_array()
                                .ok_or_else(|| {
                                    D::Error::custom(
                                        "expected array of integers in input_ids batch",
                                    )
                                })
                                .and_then(|inner| {
                                    inner
                                        .iter()
                                        .map(|x| {
                                            x.as_i64()
                                                .ok_or_else(|| {
                                                    D::Error::custom(
                                                        "expected integer in input_ids",
                                                    )
                                                })
                                                .map(|i| i as i32)
                                        })
                                        .collect()
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    Ok(IdsInput::Batch(batches))
                } else {
                    // Flat array → single sequence of token IDs.
                    let ids: Vec<i32> = arr
                        .into_iter()
                        .map(|v| {
                            v.as_i64()
                                .ok_or_else(|| D::Error::custom("expected integer in input_ids"))
                                .map(|i| i as i32)
                        })
                        .collect::<Result<_, _>>()?;
                    Ok(IdsInput::Single(ids))
                }
            }
            _ => Err(D::Error::custom(format!(
                "expected array of integers or array of arrays for input_ids, got {}",
                val
            ))),
        }
    }
}

/// Minimal decoded view of an incoming `/generate` body. Core fields are typed;
/// everything else round-trips through `extra` so we stay faithful to the full
/// Python schema (and the in-flight msgpack-migration) without enumerating it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratePayload {
    /// Accepts `"..."` (single) or `["...", "..."]` (batch), matching Python's
    /// `GenerateReqInput.text: Optional[Union[str, List[str]]]`.
    #[serde(default)]
    pub text: Option<TextInput>,
    /// Accepts `[1, 2, 3]` (single) or `[[1, 2], [3, 4]]` (batch), matching
    /// Python's `GenerateReqInput.input_ids: Optional[Union[List[int], List[List[int]]]]`.
    #[serde(default)]
    pub input_ids: Option<IdsInput>,
    #[serde(default)]
    pub stream: bool,
    /// Request priority (None = normal; used by scheduler priority scheduling).
    /// Python keeps this as Optional[int]; never emit 0 when omitted.
    #[serde(default)]
    pub priority: Option<i64>,
    /// Whether to return logprobs in the response.
    #[serde(default)]
    pub return_logprob: bool,
    /// Start position for returning prompt logprobs (-1 = output tokens only).
    /// `-1` is the Python default (normalized from None); serde gives 0 for
    /// omitted, so we default to -1 explicitly.
    #[serde(default = "minus_one")]
    pub logprob_start_len: i64,
    /// Number of top-logprobs to return per position.
    #[serde(default)]
    pub top_logprobs_num: i64,
    /// Token IDs to return logprobs for.
    #[serde(default)]
    pub token_ids_logprob: Option<Vec<i32>>,
    /// Whether to detokenize logprob token IDs to text.
    #[serde(default)]
    pub return_text_in_logprobs: bool,
    /// Whether to include the prompt token IDs in the response.
    #[serde(default)]
    pub return_prompt_token_ids: bool,
    /// Whether to return hidden states in the response.
    #[serde(default)]
    pub return_hidden_states: bool,
    /// Whether to return captured routed experts.
    #[serde(default)]
    pub return_routed_experts: bool,
    /// Absolute start position for returned routed experts.
    #[serde(default)]
    pub routed_experts_start_len: i64,
    /// Whether to return indexer top-k metadata.
    #[serde(default)]
    pub return_indexer_topk: bool,
    /// Whether the model should produce reasoning output.
    #[serde(default)]
    pub require_reasoning: bool,
    /// Opaque sampling params, carried through to the scheduler untouched.
    #[serde(default)]
    pub sampling_params: Option<rmpv::Value>,
    /// Any other fields on the request body, preserved for re-serialization.
    #[serde(flatten)]
    pub extra: BTreeMap<String, rmpv::Value>,
}

impl GeneratePayload {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| match v {
            IdsInput::Single(ids) => !ids.is_empty(),
            IdsInput::Batch(batches) => batches.iter().any(|b| !b.is_empty()),
        })
    }

    /// Multimodal detection hook. Deferred this iteration (Encoder stubbed):
    /// always false until mm fields are wired in.
    pub fn has_multimodal(&self) -> bool {
        false
    }

    /// Extract `max_new_tokens` from the opaque sampling params, if set.
    pub fn max_new_tokens(&self) -> Option<u64> {
        self.sampling_params.as_ref().and_then(|sp| {
            sp.as_map()?
                .iter()
                .find(|(k, _)| k.as_str() == Some("max_new_tokens"))
                .and_then(|(_, v)| v.as_u64())
        })
    }
}

/// One ingress-ring entry, split columnar: the scalar `header` (msgpack, with
/// `input_ids` omitted) plus the request's raw `ids` cell (little-endian int64,
/// empty for control requests). `recv_requests` concatenates the `ids` cells of
/// a drained batch into one buffer so the large tensor never goes through
/// msgpack; the scalar headers stay tiny.
#[derive(Debug)]
pub struct IngressMsg {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Wire form of `TokenizedGenerateReqInput`.
///
/// The scheduler decodes this with msgspec, whose IPC structs are
/// `array_like=True, tag=True` — so the wire format is a **tagged msgpack
/// array** `[tag, ...fields in declaration order]`, NOT a named map. msgspec
/// fills trailing fields (all of which carry defaults) from a short array, so
/// we emit only through `priority` (index 32) and let msgspec default the rest.
#[derive(Debug)]
pub struct TokenizedReqPayload {
    pub rid: String,
    pub input_text: Option<String>,
    pub input_ids: Vec<i32>,
    pub sampling_params: Option<rmpv::Value>,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    pub token_ids_logprob: Option<Vec<i32>>,
    pub stream: bool,
    /// Priority (None = normal). Emitted as Nil at index 32 when unset,
    /// matching Python's Optional[int] with default None.
    pub priority: Option<i64>,
    pub return_hidden_states: bool,
    pub return_routed_experts: bool,
    pub routed_experts_start_len: i64,
    pub return_indexer_topk: bool,
    pub require_reasoning: bool,
}

impl TokenizedReqPayload {
    /// Widen `input_ids` (i32) to the raw little-endian **int64** bytes the
    /// scheduler's `array("q")` expects. This is the *columnar tensor cell*: it
    /// travels the ingress ring as raw bytes (no msgpack) and is concatenated
    /// with the other requests' cells in `recv_requests`.
    pub fn input_ids_i64_le(&self) -> Bytes {
        let mut buf = Vec::with_capacity(self.input_ids.len() * 8);
        for &id in &self.input_ids {
            buf.extend_from_slice(&(id as i64).to_le_bytes());
        }
        Bytes::from(buf)
    }

    /// Serialize the *scalar header* to the msgspec-compatible tagged array,
    /// with `input_ids` left as `Nil` — the ids ride alongside as a raw columnar
    /// buffer (see [`input_ids_i64_le`](Self::input_ids_i64_le)) and are set on
    /// the decoded struct by the Python `drain`.
    pub fn to_header_msgpack(&self) -> Result<Bytes, Error> {
        use rmpv::Value;

        // input_ids omitted from the header; delivered as a columnar buffer.
        let input_ids_val = Value::Nil;

        let input_text_val = match &self.input_text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };

        // `sampling_params: SamplingParams` is required (not Optional) and
        // map-encoded; default to an empty map (all-defaults) when absent. Send
        // only what the client set: the scheduler normalizes these and turns an
        // absent `stop` / `stop_regex` into an empty list. Injecting `""` here
        // instead would make `normalize` expand it to `[""]`, which matches at
        // every position and ends generation on the first token.
        let sampling_params_val = match self.sampling_params.clone() {
            Some(v @ Value::Map(_)) => v,
            _ => Value::Map(Vec::new()),
        };

        // `token_ids_logprob` — serialize Optional<Vec<i32>> as array or Nil.
        // Python normalizes both None and [] to None; match that here.
        let token_ids_logprob_val = match &self.token_ids_logprob {
            Some(ids) if !ids.is_empty() => {
                Value::Array(ids.iter().map(|&id| Value::from(id)).collect())
            }
            _ => Value::Nil,
        };

        // Tagged array in TokenizedGenerateReqInput declaration order (BaseReq
        // fields first). msgspec fills trailing fields (all of which carry
        // defaults) from a short array, so we emit only through index 32
        // (priority) and let msgspec default indices 33+ (extra_key through
        // time_stats). Indices 18-30 are Nil (Optional fields, not populated
        // by the Rust server); 14-17 and 31-32 are default-filled below.
        let mut arr = vec![
            Value::from("TokenizedGenerateReqInput"), // 0  tag
            Value::from(self.rid.as_str()),           // 1  rid
            Value::Nil,                               // 2  http_worker_ipc
            input_text_val,                           // 3  input_text
            input_ids_val,                            // 4  input_ids
            Value::Nil,                               // 5  input_embeds
            Value::Nil,                               // 6  mm_inputs
            Value::Nil,                               // 7  token_type_ids
            sampling_params_val,                      // 8  sampling_params
            Value::from(self.return_logprob),         // 9  return_logprob
            Value::from(self.logprob_start_len),      // 10 logprob_start_len
            Value::from(self.top_logprobs_num),       // 11 top_logprobs_num
            token_ids_logprob_val,                    // 12 token_ids_logprob
            Value::from(self.stream),                 // 13 stream
        ];
        // Extend to index 32 (priority). Indices 14-17 and 31 are
        // non-optional bool/int fields — must fill with real defaults,
        // not Nil, or msgspec decode on the Python side will fail.
        arr.resize(33, Value::Nil); // 0-32 = 33 elements
        arr[14] = Value::Boolean(self.return_hidden_states); // 14 return_hidden_states
        arr[15] = Value::Boolean(self.return_routed_experts); // 15 return_routed_experts
        arr[16] = Value::from(self.routed_experts_start_len); // 16 routed_experts_start_len
        arr[17] = Value::Boolean(self.return_indexer_topk); // 17 return_indexer_topk
        arr[31] = Value::Boolean(self.require_reasoning); // 31 require_reasoning
        arr[32] = match self.priority {
            Some(p) => Value::from(p),
            None => Value::Nil,
        }; // 32 priority
        let arr = Value::Array(arr);

        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
    }
}

/// Egress-ring frame discriminator (first byte). Internal to the Rust egress
/// ring: Python pushes raw payloads via `push_chunk` / `push_result` and the
/// tag is prepended on the Rust side, so the Python wire format is unchanged.
pub const EGRESS_TAG_CHUNK: u8 = 0;
pub const EGRESS_TAG_RESULT: u8 = 1;

/// Frame a generation chunk for the egress ring (msgpack already built by
/// Python's `push_chunk`; just prepend the tag).
pub fn frame_egress_chunk(chunk: &[u8]) -> Bytes {
    let mut buf = Vec::with_capacity(1 + chunk.len());
    buf.push(EGRESS_TAG_CHUNK);
    buf.extend_from_slice(chunk);
    Bytes::from(buf)
}

/// Frame a control result `[rid, payload]` for the egress ring (tag prepended).
pub fn frame_egress_result(rid: &str, payload: &[u8]) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::Binary(payload.to_vec())]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 8);
    buf.push(EGRESS_TAG_RESULT);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// One scheduler output increment for a request, pushed from Python via
/// `push_chunk` into the egress ring. Decoded on a Rust detok shard.
///
/// Fields added after the initial wire format carry `#[serde(default)]` so the
/// Rust side stays backward-compatible with older Python producers that omit them.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkEvent {
    pub rid: String,
    pub seq: u64,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: Vec<i32>,
    /// `None` while streaming, `Some(reason)` on the final chunk.
    pub finish_reason: Option<String>,
    /// Prompt token count for this request (constant across its chunks).
    /// `#[serde(default)]` keeps the wire backward-compatible with 4-field frames.
    #[serde(default)]
    pub prompt_tokens: u32,

    // ── Logprob fields (default-empty, backward-compatible) ──
    /// Output token logprob values for this step's token_ids (index-aligned).
    #[serde(default)]
    pub output_token_logprobs_val: Vec<f32>,
    /// Output token IDs for logprobs (index-aligned with val).
    #[serde(default)]
    pub output_token_logprobs_idx: Vec<i32>,
    /// Input token logprob values (only on the first chunk when return_logprob is set).
    #[serde(default)]
    pub input_token_logprobs_val: Vec<f32>,
    /// Input token IDs for logprobs (index-aligned with val).
    #[serde(default)]
    pub input_token_logprobs_idx: Vec<i32>,
    /// Output top-k logprobs per position (outer: positions, inner: (value, id) pairs).
    #[serde(default)]
    pub output_top_logprobs_val: Vec<Vec<(f32, i32)>>,
    /// Input top-k logprobs per position (outer: positions, inner: (value, id) pairs).
    #[serde(default)]
    pub input_top_logprobs_val: Vec<Vec<(f32, i32)>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the handwritten `to_header_msgpack` encoder emits fields at the
    /// correct indices. This test catches the class of bug where fields were
    /// shifted by skipping `input_embeds` and `token_type_ids` — the schema
    /// round-trip tests (which use the generated codec) would not catch it.
    #[test]
    fn test_tokenized_req_header_field_positions() {
        let payload = TokenizedReqPayload {
            rid: "42".into(),
            input_text: Some("hello".into()),
            input_ids: vec![1, 2, 3],
            sampling_params: None,
            return_logprob: true,
            logprob_start_len: -1,
            top_logprobs_num: 3,
            token_ids_logprob: Some(vec![100, 200]),
            stream: false,
            priority: Some(5),
            return_hidden_states: false,
            return_routed_experts: false,
            routed_experts_start_len: 0,
            return_indexer_topk: false,
            require_reasoning: false,
        };

        let header = payload.to_header_msgpack().expect("encode header");
        let val = rmpv::decode::read_value(&mut &header[..]).expect("decode header");
        let arr = val.as_array().expect("expected array");

        // Index 0: tag
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));

        // Index 4: input_ids (always Nil in header — columnar)
        assert!(arr[4].is_nil());

        // Index 5: input_embeds (Nil — not populated)
        assert!(arr[5].is_nil());

        // Index 7: token_type_ids (Nil — not populated)
        assert!(arr[7].is_nil());

        // Index 8: sampling_params (default empty map)
        assert!(arr[8].as_map().is_some());

        // Index 9: return_logprob
        assert_eq!(arr[9].as_bool(), Some(true));

        // Index 10: logprob_start_len
        assert_eq!(arr[10].as_i64(), Some(-1));

        // Index 11: top_logprobs_num
        assert_eq!(arr[11].as_i64(), Some(3));

        // Index 12: token_ids_logprob
        assert!(arr[12].as_array().is_some());
        assert_eq!(arr[12].as_array().map(|a| a.len()), Some(2));

        // Index 13: stream
        assert_eq!(arr[13].as_bool(), Some(false));

        // Indices 14-17 are non-optional bool/int — must NOT be Nil
        // (msgspec would fail to decode Nil into these). Verify defaults.
        assert_eq!(arr[14].as_bool(), Some(false)); // return_hidden_states
        assert_eq!(arr[15].as_bool(), Some(false)); // return_routed_experts
        assert_eq!(arr[16].as_i64(), Some(0)); // routed_experts_start_len
        assert_eq!(arr[17].as_bool(), Some(false)); // return_indexer_topk
        assert_eq!(arr[31].as_bool(), Some(false)); // require_reasoning

        // Index 32: priority (also non-optional)
        assert_eq!(arr[32].as_i64(), Some(5));

        // Array length: 33 (tag@0 + 32 fields, truncated at priority)
        assert_eq!(arr.len(), 33);
    }

    /// Verify that omitted priority emits Nil at index 32 (matching Python's
    /// Optional[int] with default None).
    #[test]
    fn test_priority_nil_when_unset() {
        let payload = TokenizedReqPayload {
            rid: "0".into(),
            input_text: None,
            input_ids: vec![],
            sampling_params: None,
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
            priority: None,
            return_hidden_states: false,
            return_routed_experts: false,
            routed_experts_start_len: 0,
            return_indexer_topk: false,
            require_reasoning: false,
        };

        let header = payload.to_header_msgpack().expect("encode");
        let val = rmpv::decode::read_value(&mut &header[..]).expect("decode");
        let arr = val.as_array().expect("expected array");
        assert!(arr[32].is_nil());
    }

    /// Verify that `token_ids_logprob: Some(vec![])` normalizes to Nil at
    /// index 12 (matching Python's `[] → None` normalization).
    #[test]
    fn test_empty_token_ids_logprob_normalized_to_nil() {
        let payload = TokenizedReqPayload {
            rid: "0".into(),
            input_text: None,
            input_ids: vec![],
            sampling_params: None,
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: Some(vec![]),
            stream: false,
            priority: None,
            return_hidden_states: false,
            return_routed_experts: false,
            routed_experts_start_len: 0,
            return_indexer_topk: false,
            require_reasoning: false,
        };

        let header = payload.to_header_msgpack().expect("encode");
        let val = rmpv::decode::read_value(&mut &header[..]).expect("decode");
        let arr = val.as_array().expect("expected array");
        assert!(arr[12].is_nil());
    }
}
