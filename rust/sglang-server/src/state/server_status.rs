//! Server lifecycle status, matching the Python `ServerStatus` enum.

/// Server startup/running phase.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerStatus {
    /// Initializing — not ready to accept requests.
    #[default]
    Starting,
    /// Fully operational.
    Ready,
    /// Draining — finishing in-flight, rejecting new.
    Draining,
    /// Shutting down.
    Stopped,
}
