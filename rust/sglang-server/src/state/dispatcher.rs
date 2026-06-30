//! Type-based response dispatcher.
//!
//! Routes incoming messages from the DetokenizerManager / Scheduler to the
//! correct handler: generation output → ReqState, control responses → pending
//! response table, admin messages (abort, health) → direct handlers.

use std::collections::HashMap;

use crate::ids::RequestId;
use crate::state::pending::PendingResponseTable;
use crate::state::req_state::{ReqState, ResponseChunk};

/// Dispatch target for an incoming batch output or control message.
pub enum DispatchTarget {
    /// Route to a ReqState (generation output).
    ReqState(RequestId),
    /// Route to a pending control response.
    PendingResponse(RequestId),
    /// Swallow (e.g., health check echo).
    Swallow,
    /// Deferred to a dedicated handler (abort, admin).
    Deferred,
}

/// Response dispatcher: takes decoded DetokenizerManager output and routes to
/// the correct handler.
#[derive(Default)]
pub struct Dispatcher {
    pending: PendingResponseTable,
    /// Active generation requests: rid → ReqState.
    req_states: HashMap<RequestId, ReqState>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new generation request.
    pub fn insert_req(&mut self, id: RequestId, state: ReqState) {
        self.req_states.insert(id, state);
    }

    /// Register a pending control response.
    pub fn insert_pending(&mut self, id: RequestId) -> tokio::sync::oneshot::Receiver<rmpv::Value> {
        self.pending.insert(id)
    }

    /// Route a decoded batch output to the correct handler.
    /// Inspects `finished_reasons` (array index 3) to distinguish Frame vs Done.
    pub fn dispatch(&mut self, id: &RequestId, tag: &str, value: rmpv::Value) {
        let is_finished = match &value {
            rmpv::Value::Array(arr) if tag == "BatchStrOutput" || tag == "BatchTokenIDOutput" => {
                // finished_reasons is at index 3: if any entry is not Nil, request is done
                arr.get(3)
                    .and_then(|v| v.as_array())
                    .map(|reasons| reasons.iter().any(|r| !r.is_nil()))
                    .unwrap_or(false)
            }
            rmpv::Value::Array(arr) if tag == "BatchEmbeddingOutput" => arr
                .get(3)
                .and_then(|v| v.as_array())
                .map(|reasons| reasons.iter().any(|r| !r.is_nil()))
                .unwrap_or(false),
            _ => false,
        };

        match tag {
            // Generation output → ReqState
            "BatchStrOutput" | "BatchTokenIDOutput" | "BatchEmbeddingOutput" => {
                if let Some(state) = self.req_states.get_mut(id) {
                    state.observe_first_token();
                    let chunk = if is_finished {
                        ResponseChunk::Done(value)
                    } else {
                        ResponseChunk::Frame(value)
                    };
                    state.push_chunk(chunk);
                }
            }
            // Control response → pending table
            _ => {
                self.pending.resolve(id, value);
            }
        }
    }

    /// Remove a completed/aborted request.
    pub fn remove_req(&mut self, id: &RequestId) -> Option<ReqState> {
        self.req_states.remove(id)
    }

    /// Number of active generation requests.
    pub fn active_count(&self) -> usize {
        self.req_states.len()
    }
}
