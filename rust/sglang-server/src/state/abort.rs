//! Abort handling for generation requests.
//!
//! Two abort paths:
//!   1. Explicit `abort_request(rid)` call from the HTTP handler →
//!      pushes control msg through ingress ring to scheduler
//!   2. Scheduler-initiated abort (over-max-tokens, internal error) →
//!      echoed back via egress ring

/// Outcome of an abort signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortOutcome {
    /// Request was found and aborted.
    Aborted,
    /// Request already finished — nothing to do.
    AlreadyFinished,
    /// No such request.
    NotFound,
}

/// Validate that an abort is needed: the request must still be tracked
/// and not yet finished.
pub fn should_abort(finished: bool, has_state: bool) -> AbortOutcome {
    if !has_state {
        AbortOutcome::NotFound
    } else if finished {
        AbortOutcome::AlreadyFinished
    } else {
        AbortOutcome::Aborted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abort_outcome_found() {
        assert_eq!(AbortOutcome::Aborted, should_abort(false, true));
    }

    #[test]
    fn test_abort_outcome_already_finished() {
        assert_eq!(AbortOutcome::AlreadyFinished, should_abort(true, true));
    }

    #[test]
    fn test_abort_outcome_not_found() {
        assert_eq!(AbortOutcome::NotFound, should_abort(false, false));
    }

    #[test]
    fn test_abort_cleanup_after_remove() {
        use std::collections::HashMap;
        let mut states: HashMap<u64, bool> = HashMap::new();
        states.insert(1, false);
        states.remove(&1);
        assert_eq!(
            AbortOutcome::NotFound,
            should_abort(false, states.contains_key(&1))
        );
    }

    #[test]
    fn test_abort_finished_then_removed() {
        use std::collections::HashMap;
        let mut states: HashMap<u64, bool> = HashMap::new();
        states.insert(1, true);
        assert_eq!(
            AbortOutcome::AlreadyFinished,
            should_abort(true, states.contains_key(&1))
        );
        states.remove(&1);
        assert!(states.is_empty());
    }
}
