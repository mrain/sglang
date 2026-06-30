//! Pending response table — maps request IDs to channels awaiting scheduler
//! responses. Used for both generation and control request/response pairs.

use std::collections::HashMap;

use tokio::sync::oneshot;

use crate::ids::RequestId;

/// A pending response future — the sender side of a oneshot channel.
/// Dropped if the request is aborted before a response arrives.
pub type PendingResponse = oneshot::Sender<rmpv::Value>;

/// Handle for awaiting a pending response.
pub type PendingResponseFuture = oneshot::Receiver<rmpv::Value>;

/// Table of pending responses, keyed by request ID.
///
/// Lock-free in the common case: insertions happen from the HTTP handler
/// (tokio task), removals happen from the response handler (same tokio task
/// after the response is received). Only the tokio runtime accesses this.
#[derive(Debug, Default)]
pub struct PendingResponseTable {
    inner: HashMap<RequestId, PendingResponse>,
}

impl PendingResponseTable {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Register a pending response and return the receiver end.
    pub fn insert(&mut self, id: RequestId) -> PendingResponseFuture {
        let (tx, rx) = oneshot::channel();
        self.inner.insert(id, tx);
        rx
    }

    /// Complete a pending response — delivers the value to the waiter.
    pub fn resolve(&mut self, id: &RequestId, value: rmpv::Value) {
        if let Some(tx) = self.inner.remove(id) {
            let _ = tx.send(value);
        }
    }

    /// Remove and discard a pending response (abort path).
    pub fn discard(&mut self, id: &RequestId) {
        self.inner.remove(id);
    }

    /// Number of pending responses.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
