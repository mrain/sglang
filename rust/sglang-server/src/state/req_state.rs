//! Per-request accumulator, matching the Python `ReqState` dataclass.
//!
//! Each active request has a `ReqState` entry that lives from `generate_request`
//! until the final response chunk is dispatched. It accumulates output tokens,
//! decoded text, logprobs, and tracks streaming state.

use std::collections::VecDeque;
use std::time::Instant;

use tokio::sync::Notify;

/// One pending output chunk from the scheduler/detokenizer.
#[derive(Debug, Clone)]
pub enum ResponseChunk {
    /// Intermediate streaming output (text or token-ids).
    Frame(rmpv::Value),
    /// Final output with finish reason.
    Done(rmpv::Value),
    /// Embedding output.
    Embedding(rmpv::Value),
}

/// Per-request state, mirroring the Python `ReqState` dataclass.
#[derive(Debug)]
pub struct ReqState {
    /// Queue of pending output chunks (filled by handle_loop, drained by wait_one_response).
    pub out_queue: VecDeque<ResponseChunk>,

    /// True when the final chunk has been enqueued.
    pub finished: bool,

    /// Waker for the waiting async handler.
    pub notify: Notify,

    /// Request body reference (opaque rmpv).
    pub request: rmpv::Value,

    // ── Timing ──
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub finished_at: Option<Instant>,

    // ── Streaming state ──
    pub last_output_offset: usize,
    pub output_ids: Vec<i64>,
    pub text: String,
    pub text_chunks: Vec<String>,

    // ── Logprobs (accumulated across streaming chunks) ──
    pub input_token_logprobs_val: Vec<f64>,
    pub input_token_logprobs_idx: Vec<i64>,
    pub output_token_logprobs_val: Vec<f64>,
    pub output_token_logprobs_idx: Vec<i64>,
    pub input_top_logprobs_val: Vec<Vec<(f64, i64)>>,
    pub output_top_logprobs_val: Vec<Vec<(f64, i64)>>,

    // ── Prompt tokens ──
    pub prompt_token_ids: Option<Vec<i64>>,
}

impl ReqState {
    pub fn new(request: rmpv::Value) -> Self {
        Self {
            out_queue: VecDeque::new(),
            finished: false,
            notify: Notify::new(),
            request,
            created_at: Instant::now(),
            first_token_at: None,
            finished_at: None,
            output_ids: Vec::new(),
            text: String::new(),
            text_chunks: Vec::new(),
            last_output_offset: 0,
            input_token_logprobs_val: Vec::new(),
            input_token_logprobs_idx: Vec::new(),
            output_token_logprobs_val: Vec::new(),
            output_token_logprobs_idx: Vec::new(),
            input_top_logprobs_val: Vec::new(),
            output_top_logprobs_val: Vec::new(),
            prompt_token_ids: None,
        }
    }

    /// Append a text delta (lazy accumulation — strings are joined on finalize).
    pub fn append_text(&mut self, chunk: &str) {
        if !chunk.is_empty() {
            self.text_chunks.push(chunk.to_owned());
        }
    }

    /// Materialize the accumulated text.
    pub fn get_text(&mut self) -> String {
        if !self.text_chunks.is_empty() {
            self.text += &self.text_chunks.concat();
            self.text_chunks.clear();
        }
        self.text.clone()
    }

    /// Enqueue a response chunk and wake the waiting handler.
    pub fn push_chunk(&mut self, chunk: ResponseChunk) {
        if matches!(&chunk, ResponseChunk::Done(_)) {
            self.finished = true;
        }
        self.out_queue.push_back(chunk);
        self.notify.notify_one();
    }

    /// Set first-token timestamp on the first output.
    pub fn observe_first_token(&mut self) {
        self.first_token_at.get_or_insert_with(Instant::now);
    }

    /// Accumulate input token logprobs from one chunk.
    pub fn accumulate_input_logprobs(&mut self, vals: &[f64], idxs: &[i64]) {
        self.input_token_logprobs_val.extend_from_slice(vals);
        self.input_token_logprobs_idx.extend_from_slice(idxs);
    }

    /// Accumulate output token logprobs from one chunk.
    pub fn accumulate_output_logprobs(&mut self, vals: &[f64], idxs: &[i64]) {
        self.output_token_logprobs_val.extend_from_slice(vals);
        self.output_token_logprobs_idx.extend_from_slice(idxs);
    }

    /// Accumulate top logprobs from one chunk (value, token_id pairs).
    pub fn accumulate_top_logprobs(
        &mut self,
        input: &[Vec<(f64, i64)>],
        output: &[Vec<(f64, i64)>],
    ) {
        self.input_top_logprobs_val.extend_from_slice(input);
        self.output_top_logprobs_val.extend_from_slice(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> ReqState {
        ReqState::new(rmpv::Value::Nil)
    }

    #[test]
    fn test_push_frame_notifies_and_not_finished() {
        let mut s = make_state();
        s.push_chunk(ResponseChunk::Frame(rmpv::Value::from(1)));
        assert!(!s.finished);
        assert_eq!(s.out_queue.len(), 1);
    }

    #[test]
    fn test_push_done_marks_finished() {
        let mut s = make_state();
        s.push_chunk(ResponseChunk::Done(rmpv::Value::from(2)));
        assert!(s.finished);
    }

    #[test]
    fn test_drain_and_remove_after_finished() {
        let mut s = make_state();
        s.push_chunk(ResponseChunk::Frame(rmpv::Value::from(10)));
        s.push_chunk(ResponseChunk::Done(rmpv::Value::from(20)));
        assert_eq!(s.out_queue.len(), 2);
        assert!(s.finished);

        // Drain all
        while s.out_queue.pop_front().is_some() {}
        assert!(s.out_queue.is_empty());
        // State is cleaned up — finished flag stays true for observability
        assert!(s.finished);
    }

    #[test]
    fn test_append_and_get_text() {
        let mut s = make_state();
        s.append_text("Hello ");
        s.append_text("World");
        let text = s.get_text();
        assert_eq!(text, "Hello World");
        assert!(s.text_chunks.is_empty());
    }

    #[test]
    fn test_get_text_idempotent() {
        let mut s = make_state();
        s.append_text("abc");
        let t1 = s.get_text();
        let t2 = s.get_text();
        assert_eq!(t1, "abc");
        assert_eq!(t2, "abc");
    }

    #[test]
    fn test_first_token_timestamp() {
        let mut s = make_state();
        assert!(s.first_token_at.is_none());
        s.observe_first_token();
        assert!(s.first_token_at.is_some());
        // Second call should not change timestamp
        let ts = s.first_token_at;
        s.observe_first_token();
        assert_eq!(s.first_token_at, ts);
    }

    #[test]
    fn test_empty_queue() {
        let s = make_state();
        assert!(s.out_queue.is_empty());
        assert!(!s.finished);
    }
}
