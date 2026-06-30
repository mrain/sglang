//! Generic fan-out communicator for control request/response patterns.
//!
//! Sends a control request to the scheduler, awaits a response (single or
//! multi-DP), and returns the result. Mirrors the Python `FanOutCommunicator`.

use std::time::Duration;

use tokio::sync::oneshot;
use tokio::time::timeout;

use crate::error::Error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_outcome_ok() {
        let o = ControlOutcome::ok(None);
        assert!(o.success);
        assert!(o.message.is_none());
    }

    #[test]
    fn test_control_outcome_ok_with_message() {
        let o = ControlOutcome::ok(Some("done".into()));
        assert!(o.success);
        assert_eq!(o.message.unwrap(), "done");
    }

    #[test]
    fn test_control_outcome_err() {
        let o = ControlOutcome::err("failed");
        assert!(!o.success);
        assert_eq!(o.message.unwrap(), "failed");
    }

    #[test]
    fn test_control_outcome_ok_with_payload() {
        let o = ControlOutcome::ok_with(rmpv::Value::Nil);
        assert!(o.success);
        assert!(o.payload.is_some());
    }

    #[test]
    fn test_register_pending_roundtrip() {
        let (tx, mut rx) = register_pending();
        let expected = rmpv::Value::from(42i64);
        tx.send(Ok(expected.clone())).unwrap();
        let result = rx.try_recv().unwrap().unwrap();
        assert_eq!(result.as_i64(), Some(42));
    }
}

/// The receiver half — used by the caller to await the response.
pub type PendingResponseRx = oneshot::Receiver<Result<rmpv::Value, Error>>;

/// Outcome of a control request.
#[derive(Debug, Clone)]
pub struct ControlOutcome {
    pub success: bool,
    pub message: Option<String>,
    pub payload: Option<rmpv::Value>,
}

impl ControlOutcome {
    pub fn ok(msg: Option<String>) -> Self {
        Self {
            success: true,
            message: msg,
            payload: None,
        }
    }

    pub fn ok_with(payload: rmpv::Value) -> Self {
        Self {
            success: true,
            message: None,
            payload: Some(payload),
        }
    }

    pub fn err(msg: &str) -> Self {
        Self {
            success: false,
            message: Some(msg.to_owned()),
            payload: None,
        }
    }
}

/// Fan-out mode for multi-rank responses.
#[derive(Debug, Clone, Copy)]
pub enum FanOutMode {
    /// Single response expected (no DP or single DP rank).
    Singleton,
    /// One response per DP rank, merged into `(success, message)`.
    /// `dp_size` is the number of DP ranks.
    AllRanks { dp_size: usize },
}

/// Register a pending response slot, returning the receiver.
/// The caller is responsible for:
///   1. Encoding the control request
///   2. Dispatching it to the scheduler
///   3. Calling `resolve_callback` later when the response arrives
pub fn register_pending() -> (
    oneshot::Sender<Result<rmpv::Value, Error>>,
    PendingResponseRx,
) {
    oneshot::channel()
}

/// Await a pending response with a timeout.
pub async fn await_response(
    rx: PendingResponseRx,
    timeout_dur: Duration,
) -> Result<rmpv::Value, Error> {
    match timeout(timeout_dur, rx).await {
        Ok(Ok(Ok(val))) => Ok(val),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(_)) => Err(Error::Codec("control response cancelled".into())),
        Err(_) => Err(Error::Codec("control response timeout".into())),
    }
}
