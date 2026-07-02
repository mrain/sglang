//! API server (axum / tokio). I/O-bound; runs on its own pinned multi-thread
//! runtime. Designed so additional protocols (h2/h3/websocket/grpc) can mount
//! the same `AppState` later — only this module knows about HTTP.
//!
//! `/generate` opens a per-request egress channel, moves a `Request` into the
//! ingress pipeline, and then either awaits a single `Done` (unary) or relays
//! frames as Server-Sent Events (streaming), byte-compatible with the Python
//! `http_server.generate_request` (`data: {json}\n\n` … `data: [DONE]\n\n`).
//!
//! The OpenAI-compatible endpoints (`/v1/*`) live in the [`openai`] submodule;
//! they share this module's [`AppState`] and submit machinery. Future protocols
//! (e.g. Anthropic) get their own sibling submodule the same way.

mod openai;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use std::convert::Infallible;
use tokio::sync::mpsc;

use crate::fsm::RequestState;
use crate::ids::RequestIdGen;
use dynamo_renderer::PromptFormatter;

use crate::message::{
    ControlRequest, EgressItem, GeneratePayload, GenerateRequest, GenerationOutput, Request,
    RequestKind,
};
use crate::runtime::ServerArgs;
use crate::runtime::channels::{Senders, TmEvent};

/// Built once at startup from the model's `tokenizer_config.json` (`None` when
/// the model has no chat template, or under `skip_tokenizer_init`). `Clone` is a
/// refcount bump (the formatter is `Arc`-backed), so it rides on `AppState`.
#[derive(Clone)]
struct ChatFormatter(PromptFormatter);

/// Shared state for every handler. Holds the submit machinery (`senders`,
/// `id_gen`, `egress_buf`) plus the static `ServerArgs` read by `/v1/models`.
/// `server_args` is an `Arc`, so the per-request clone axum makes is a refcount
/// bump.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// `None` when the model has no chat template → `/v1/chat/completions` 400s.
    chat: Option<ChatFormatter>,
}

pub async fn serve(
    bind: SocketAddr,
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    startup_tx: Option<std::sync::mpsc::Sender<Result<(), String>>>,
) {
    let chat = openai::load_chat_formatter(&server_args).map(ChatFormatter);
    let state = AppState {
        senders,
        id_gen,
        egress_buf,
        server_args,
        chat,
    };
    let app = Router::new()
        .route("/generate", post(generate))
        // OpenAI-compatible: same tokenize→generate→detok pipeline, OpenAI shape.
        .route("/v1/completions", post(openai::openai_completions))
        .route(
            "/v1/chat/completions",
            post(openai::openai_chat_completions),
        )
        // Control-plane endpoints: each reuses the ingress FSM (no tokenization)
        // and returns a single, non-streamed JSON result from the scheduler.
        // Adding one = a route line passing its scheduler request-struct tag.
        .route("/server_info", get(server_info))
        // OpenAI-compatible embedding endpoint.
        .route("/v1/embeddings", post(openai::openai_embeddings))
        // Static config endpoint (OpenAI-compatible): no scheduler round-trip.
        .route("/v1/models", get(openai::available_models))
        .with_state(state);

    match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => {
            if let Some(tx) = startup_tx {
                let _ = tx.send(Ok(()));
            }
            tracing::info!(%bind, "sglang-server api listening");
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        Err(e) => {
            if let Some(tx) = startup_tx {
                let _ = tx.send(Err(format!("failed to bind {bind}: {e}")));
            }
            tracing::error!(error = %e, %bind, "failed to bind api server");
        }
    }
}

/// Submit a request into the ingress pipeline. Returns the per-request egress
/// receiver to read the result(s) from. The `kind` carries the variant body
/// (generate payload / control tag), so this stays generic over both.
async fn submit(state: &AppState, kind: RequestKind) -> Result<mpsc::Receiver<EgressItem>, ()> {
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let id = state.id_gen.next();
    let req = Request {
        id,
        state: RequestState::Received,
        sink: tx,
        kind,
    };
    // Async-aware send from this tokio task: under a full TM inbox it yields
    // (backpressure) instead of parking a worker thread, which flume's sync
    // `send` would do. Err only when the inbox is closed (runtime shutdown).
    match state.senders.tm.send_async(TmEvent::Ingress(req)).await {
        Ok(()) => Ok(rx),
        Err(_) => {
            tracing::error!("tm inbox closed; request dropped");
            Err(())
        }
    }
}

/// Submit a `Control(tag)` request through the same ingress FSM as `/generate`
/// (minus tokenization) and await the scheduler's single msgpack `Result`. The
/// scheduler pushes a *named map* (`structs.asdict` of the response struct).
/// Returns the raw msgpack bytes, or an error `Response` to return as-is.
async fn await_control_result(
    state: &AppState,
    tag: &'static str,
) -> Result<bytes::Bytes, Response> {
    let mut rx = submit(state, RequestKind::Control(ControlRequest { tag }))
        .await
        .map_err(|()| (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response())?;
    match rx.recv().await {
        Some(EgressItem::Control(bytes)) => Ok(bytes),
        Some(EgressItem::Error(e)) => {
            let code =
                StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err((code, e.to_string()).into_response())
        }
        // A control request never receives generation frames.
        Some(EgressItem::Frame(_)) | Some(EgressItem::Done(_)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected generation output for control request",
        )
            .into_response()),
        None => Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response()),
    }
}

/// Format a neutral [`GenerationOutput`] as one SGLang `/generate` frame. Lives
/// in the handler now that the detok shard is protocol-neutral. `output_ids` is
/// surfaced only in `skip_tokenizer_init` mode (cumulative, non-empty there).
fn sglang_frame(out: &GenerationOutput) -> Vec<u8> {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": {
            "id": out.rid,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "finish_reason": out.finish_reason.as_deref().map(|r| serde_json::json!({ "type": r })),
        },
    });
    if !out.output_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.output_ids);
    }

    // ── Logprobs (all optional, only included when non-empty) ──
    // Output-side arrays are sliced at `last_output_offset` for streaming
    // intermediate frames (matches Python `_slice_streaming_output_meta_info`).
    // `Done` frames carry the full cumulative list so unary clients get everything.
    let meta = v["meta_info"].as_object_mut().unwrap();
    let slice_offset = if out.finish_reason.is_some() {
        0 // Done → full cumulative list
    } else {
        out.last_output_offset // intermediate → delta only
    };
    if !out.output_token_logprobs.is_empty() {
        let text_slice = if out.output_token_logprob_texts.len() >= out.output_token_logprobs.len()
        {
            &out.output_token_logprob_texts[slice_offset..]
        } else {
            &[]
        };
        let pairs: Vec<serde_json::Value> = out.output_token_logprobs[slice_offset..]
            .iter()
            .zip(out.output_token_logprob_ids[slice_offset..].iter())
            .enumerate()
            .map(|(i, (&lp, &id))| {
                let text = text_slice.get(i).and_then(|t| t.as_deref());
                serde_json::json!([lp, id, text])
            })
            .collect();
        meta.insert(
            "output_token_logprobs".into(),
            serde_json::Value::Array(pairs),
        );
    }
    if !out.input_token_logprobs.is_empty() {
        let text_slice = if out.input_token_logprob_texts.len() >= out.input_token_logprobs.len() {
            &out.input_token_logprob_texts[..]
        } else {
            &[]
        };
        let pairs: Vec<serde_json::Value> = out
            .input_token_logprobs
            .iter()
            .zip(out.input_token_logprob_ids.iter())
            .enumerate()
            .map(|(i, (&lp, &id))| {
                let text = text_slice.get(i).and_then(|t| t.as_deref());
                serde_json::json!([lp, id, text])
            })
            .collect();
        meta.insert(
            "input_token_logprobs".into(),
            serde_json::Value::Array(pairs),
        );
    }
    if !out.output_top_logprobs.is_empty() {
        let positions: Vec<serde_json::Value> = out.output_top_logprobs[slice_offset..]
            .iter()
            .map(|pos| {
                serde_json::Value::Array(
                    pos.iter()
                        .map(|&(lp, id)| serde_json::json!([lp, id, null]))
                        .collect(),
                )
            })
            .collect();
        meta.insert(
            "output_top_logprobs".into(),
            serde_json::Value::Array(positions),
        );
    }
    if !out.input_top_logprobs.is_empty() {
        let positions: Vec<serde_json::Value> = out
            .input_top_logprobs
            .iter()
            .map(|pos| {
                serde_json::Value::Array(
                    pos.iter()
                        .map(|&(lp, id)| serde_json::json!([lp, id, null]))
                        .collect(),
                )
            })
            .collect();
        meta.insert(
            "input_top_logprobs".into(),
            serde_json::Value::Array(positions),
        );
    }

    // ── Prompt token IDs (top-level, only when return_prompt_token_ids is true) ──
    if let Some(pids) = &out.prompt_token_ids {
        let arr: Vec<serde_json::Value> = pids
            .iter()
            .map(|&id| serde_json::Value::from(id as i64))
            .collect();
        v["prompt_token_ids"] = serde_json::Value::Array(arr);
    }

    serde_json::to_vec(&v).unwrap_or_default()
}

/// Generic control endpoint: returns the scheduler's response rendered straight
/// to JSON (`tag` = the scheduler request-struct name). Used by control
/// endpoints whose response needs no shaping.
#[allow(dead_code)] // first non-/server_info control endpoint will use this
async fn control(State(state): State<AppState>, tag: &'static str) -> Response {
    match await_control_result(&state, tag).await {
        Ok(bytes) => match msgpack_to_json(&bytes) {
            Ok(json) => {
                (StatusCode::OK, [("content-type", "application/json")], json).into_response()
            }
            Err(e) => {
                tracing::error!(error = %e, "control: msgpack→json failed");
                (StatusCode::INTERNAL_SERVER_ERROR, "bad control response").into_response()
            }
        },
        Err(resp) => resp,
    }
}

/// `GET /server_info` — shapes the scheduler's `GetInternalStateReqOutput` the
/// way `tokenizer_control_mixin.get_internal_state` does: lift `server_args` to
/// the top, drop null fields, merge (internal-state fields win on collision).
///
/// TODO(server_info): the original Python endpoint also includes `version`,
/// `kv_events`, and scheduler init info (`max_total_num_tokens`,
/// `max_req_input_len`). Those are dropped for now — add them here once the
/// values are plumbed through (e.g. captured at `Server.start` / a richer
/// scheduler response).
async fn server_info(State(state): State<AppState>) -> Response {
    let bytes = match await_control_result(&state, "GetInternalStateReq").await {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    match shape_server_info(&bytes) {
        Ok(json) => (StatusCode::OK, [("content-type", "application/json")], json).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "server_info: shaping failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad server_info response",
            )
                .into_response()
        }
    }
}

fn shape_server_info(msgpack: &[u8]) -> Result<Vec<u8>, String> {
    // Decode the msgpack named map directly into a JSON object.
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;

    // server_args = res.pop("server_args", {})
    let mut merged = match obj.remove("server_args") {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    // res = {k: v for k, v in res.items() if v is not None}; merged = server_args | res
    for (k, v) in obj {
        if !v.is_null() {
            merged.insert(k, v);
        }
    }

    let response = serde_json::json!({ "internal_states": [serde_json::Value::Object(merged)] });
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

/// Convert a msgpack control response (the scheduler's native ring format) into
/// JSON bytes for the HTTP client.
fn msgpack_to_json(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let val = rmpv::decode::read_value(&mut &*bytes).map_err(|e| e.to_string())?;
    serde_json::to_vec(&val).map_err(|e| e.to_string())
}

// ── Batch-body normalization helpers ──

/// Batchable field keys — scalars are replicated, arrays are expanded.
/// `token_ids_logprob` is NOT included because Python only treats
/// `List[List[int]]` as batch; `List[int]` is a single request.
const BATCHABLE_KEYS: &[&str] = &[
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "sampling_params",
];

/// Check whether the raw JSON body contains any batchable field as an
/// array. Returns the common array length, or `None` if all batchable
/// fields are scalars/absent.
///
/// `token_ids_logprob` is handled separately — only `List[List[int]]`
/// is treated as batch, matching Python's shape distinction.
fn batch_body_len(body: &serde_json::Value) -> Option<usize> {
    let mut n: Option<usize> = None;
    // Check BATCHABLE_KEYS for array fields.
    for key in BATCHABLE_KEYS {
        if let Some(arr) = body.get(key).and_then(|v| v.as_array()) {
            if arr.is_empty() {
                continue;
            }
            match n {
                None => n = Some(arr.len()),
                Some(prev) if prev != arr.len() => return None,
                _ => {}
            }
        }
    }
    // token_ids_logprob: only List[List[int]] is batch.
    if let Some(arr) = body.get("token_ids_logprob").and_then(|v| v.as_array())
        && arr.first().and_then(|v| v.as_array()).is_some()
    {
        match n {
            None => n = Some(arr.len()),
            Some(prev) if prev != arr.len() => return None,
            _ => {}
        }
    }
    n
}

/// Build one `GeneratePayload` for the i-th item in a batch by extracting
/// the i-th element of each batchable array field from the raw body.
fn payload_for_index(body: &serde_json::Value, i: usize) -> serde_json::Value {
    let mut p = body.clone();
    if let Some(obj) = p.as_object_mut() {
        // Expand BATCHABLE_KEYS arrays.
        for key in BATCHABLE_KEYS {
            if let Some(arr) = obj.get(*key).and_then(|v| v.as_array()) {
                let elem = arr.get(i).cloned().unwrap_or(serde_json::Value::Null);
                obj.insert(key.to_string(), elem);
            }
        }
        // text batch: always expand (single-element arrays are batch).
        if let Some(arr) = obj.get("text").and_then(|v| v.as_array()) {
            obj.insert(
                "text".to_string(),
                arr.get(i).cloned().unwrap_or(serde_json::Value::Null),
            );
        }
        // input_ids batch: only if the first element is an array.
        if let Some(arr) = obj.get("input_ids").and_then(|v| v.as_array())
            && arr.first().and_then(|v| v.as_array()).is_some()
        {
            obj.insert(
                "input_ids".to_string(),
                arr.get(i).cloned().unwrap_or(serde_json::Value::Null),
            );
        }
        // token_ids_logprob batch: only if the first element is an array.
        if let Some(arr) = obj.get("token_ids_logprob").and_then(|v| v.as_array())
            && arr.first().and_then(|v| v.as_array()).is_some()
        {
            obj.insert(
                "token_ids_logprob".to_string(),
                arr.get(i).cloned().unwrap_or(serde_json::Value::Null),
            );
        }
    }
    p
}

async fn generate(State(state): State<AppState>, Json(body): Json<serde_json::Value>) -> Response {
    // Detect batch mode from array fields.
    let is_batch = batch_body_len(&body).unwrap_or(0) > 1;
    let has_batch_text = body
        .get("text")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty()) // Python wraps single strings in a list
        .unwrap_or(false);
    let has_batch_ids = body
        .get("input_ids")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first().and_then(|v| v.as_array()))
        .map(|_| true)
        .unwrap_or(false);
    let has_batch_tl = body
        .get("token_ids_logprob")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first().and_then(|v| v.as_array()))
        .map(|_| true)
        .unwrap_or(false);
    let is_batch = is_batch || has_batch_text || has_batch_ids || has_batch_tl;

    if !is_batch {
        // Single request: deserialize into GeneratePayload directly.
        let payload: GeneratePayload = match serde_json::from_value(body) {
            Ok(p) => p,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("invalid payload: {e}")).into_response();
            }
        };
        return handle_single_generate(state, payload, false).await;
    }

    // Batch mode: determine batch size.
    let n = batch_body_len(&body)
        .or_else(|| body.get("text").and_then(|v| v.as_array()).map(|a| a.len()))
        .or_else(|| {
            let ids_arr = body.get("input_ids").and_then(|v| v.as_array())?;
            let first_is_array = ids_arr.first().and_then(|v| v.as_array()).is_some();
            if first_is_array {
                Some(ids_arr.len())
            } else {
                None
            }
        })
        .or_else(|| {
            let tl_arr = body.get("token_ids_logprob").and_then(|v| v.as_array())?;
            let first_is_array = tl_arr.first().and_then(|v| v.as_array()).is_some();
            if first_is_array {
                Some(tl_arr.len())
            } else {
                None
            }
        })
        .unwrap_or(0);

    if n == 0 {
        return (StatusCode::BAD_REQUEST, "cannot determine batch size").into_response();
    }

    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if stream {
        return (
            StatusCode::BAD_REQUEST,
            "batch inputs are not supported for streaming",
        )
            .into_response();
    }

    // Build sub-payloads and submit all.
    let mut rxs = Vec::with_capacity(n);
    for i in 0..n {
        let item = payload_for_index(&body, i);
        let payload: GeneratePayload = match serde_json::from_value(item) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("invalid sub-payload at index {i}: {e}"),
                )
                    .into_response();
            }
        };
        let kind = RequestKind::Generate(GenerateRequest {
            payload,
            input_ids: None,
            stream: false,
        });
        match submit(&state, kind).await {
            Ok(rx) => rxs.push(rx),
            Err(()) => {
                return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
            }
        }
    }

    // Collect all responses.
    let mut frames = Vec::with_capacity(rxs.len());
    for mut rx in rxs {
        while let Some(item) = rx.recv().await {
            match item {
                EgressItem::Frame(_) => continue,
                EgressItem::Done(out) => {
                    frames.push(sglang_frame(&out));
                    break;
                }
                EgressItem::Error(e) => {
                    let body = serde_json::json!({
                        "error": { "message": e.to_string(), "code": e.http_status() }
                    });
                    frames.push(serde_json::to_vec(&body).unwrap_or_default());
                    break;
                }
                EgressItem::Control(_) => continue,
            }
        }
    }

    let json_frames: Vec<serde_json::Value> = frames
        .iter()
        .filter_map(|f| serde_json::from_slice(f).ok())
        .collect();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&json_frames).unwrap_or_default(),
    )
        .into_response()
}

/// Handle a single (non-batch) generate request.
async fn handle_single_generate(
    state: AppState,
    payload: GeneratePayload,
    stream: bool,
) -> Response {
    let stream = stream || payload.stream;
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream,
    });
    let mut rx = match submit(&state, kind).await {
        Ok(rx) => rx,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };

    if stream {
        let s = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                match item {
                    EgressItem::Frame(out) => {
                        let f = sglang_frame(&out);
                        yield Ok::<_, Infallible>(Event::default().data(String::from_utf8_lossy(&f)));
                    }
                    EgressItem::Done(out) => {
                        let f = sglang_frame(&out);
                        yield Ok(Event::default().data(String::from_utf8_lossy(&f)));
                        break;
                    }
                    EgressItem::Error(e) => {
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                    EgressItem::Control(_) => break,
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(s).into_response()
    } else {
        while let Some(item) = rx.recv().await {
            match item {
                EgressItem::Frame(_) => continue,
                EgressItem::Done(out) => {
                    return (
                        StatusCode::OK,
                        [("content-type", "application/json")],
                        sglang_frame(&out),
                    )
                        .into_response();
                }
                EgressItem::Error(e) => {
                    let code = StatusCode::from_u16(e.http_status())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let body = serde_json::json!({
                        "error": { "message": e.to_string(), "code": e.http_status() }
                    });
                    return (code, Json(body)).into_response();
                }
                EgressItem::Control(_) => continue,
            }
        }
        (StatusCode::from_u16(499).unwrap(), "request aborted").into_response()
    }
}
