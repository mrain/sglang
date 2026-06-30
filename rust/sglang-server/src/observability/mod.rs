//! No-op/default observability carriers for Phase 1.
//!
//! These stubs implement the logging, metrics, and tracing interfaces the
//! TokenizerManager needs without pulling in heavy dependencies. They are
//! replaced with real implementations in Phase 8.

use std::time::{Duration, Instant};

/// Per-request time stats, matching `APIServerReqTimeStats` in Python.
#[derive(Debug, Clone, Default)]
pub struct TimeStats {
    pub created_at: Option<Instant>,
    pub tokenize_finish_at: Option<Instant>,
    pub dispatch_start_at: Option<Instant>,
    pub dispatch_finish_at: Option<Instant>,
    pub first_token_at: Option<Instant>,
    pub finished_at: Option<Instant>,
    pub response_sent_at: Option<Instant>,
}

impl TimeStats {
    pub fn now() -> Self {
        Self {
            created_at: Some(Instant::now()),
            ..Default::default()
        }
    }

    pub fn mark_tokenize_done(&mut self) {
        self.tokenize_finish_at = Some(Instant::now());
    }

    pub fn mark_dispatched(&mut self) {
        self.dispatch_start_at = Some(Instant::now());
        self.dispatch_finish_at = Some(Instant::now());
    }

    pub fn mark_first_token(&mut self) {
        self.first_token_at.get_or_insert_with(Instant::now);
    }

    pub fn mark_finished(&mut self) {
        self.finished_at = Some(Instant::now());
    }

    pub fn mark_response_sent(&mut self) {
        self.response_sent_at = Some(Instant::now());
    }

    pub fn ttft(&self) -> Option<Duration> {
        let first = self.first_token_at?;
        let created = self.created_at?;
        Some(first.duration_since(created))
    }

    pub fn e2e_latency(&self) -> Option<Duration> {
        let finished = self.finished_at.or(self.response_sent_at)?;
        let created = self.created_at?;
        Some(finished.duration_since(created))
    }
}

/// Request logger stub — replaced with real implementation in Phase 8.
#[derive(Debug, Default)]
pub struct RequestLogger {
    pub enabled: bool,
}

impl RequestLogger {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn log_received(&self, _rid: &str) {
        if self.enabled {
            tracing::info!(rid = %_rid, "request received");
        }
    }

    pub fn log_finished(&self, _rid: &str, _status: &str) {
        if self.enabled {
            tracing::info!(rid = %_rid, status = %_status, "request finished");
        }
    }
}

/// Metrics stub — replaced with real implementation in Phase 8.
#[derive(Debug, Default)]
pub struct MetricsCollector;

impl MetricsCollector {
    pub fn new() -> Self {
        Self
    }

    pub fn observe_ttft(&self, _rid: &str, _latency: Duration) {}
    pub fn observe_itl(&self, _rid: &str, _latency: Duration) {}
    pub fn observe_finished(&self, _rid: &str, _latency: Duration, _status: &str) {}
}
