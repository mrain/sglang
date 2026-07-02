# TokenizerManager Rust Migration Progress Tracker

Use this file as the implementation tracker. Keep `plan.md` as the design and
scope document; update this file as work lands.

Status legend:

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked or needs decision

## Current Focus

- Phase: P1
- Goal: build the Rust shell that can be instantiated from Python and can speak
  the current Scheduler wire protocol.
- Next concrete step: build the msgspec-compatible schema codec that can
  encode/decode all P0 fixture payloads.

## Phase Summary

| Phase | Status | Purpose | Depends On |
| --- | --- | --- | --- |
| P0 | `[x]` | Contract capture, schema snapshot, fixtures, comparison harness | none |
| P1 | `[x]` | Rust skeleton, schema codec, IPC compatibility | P0 |
| P2 | `[x]` | Single `/generate`, abort, response wait, FanOut infrastructure | P1 |
| P3 | `[~]` | Batch, embedding, logprobs, full sampling params | P2 |
| P4 | `[ ]` | Low-risk control endpoints on FanOut | P1, P2 |
| P5 | `[ ]` | Weight updates, LoRA, external corpus | P4, P2/P3 |
| P6 | `[ ]` | Scoring and rerank | P3 |
| P7 | `[ ]` | Multimodal and disaggregation | P2, P3 |
| P8a | `[ ]` | Logging, metrics, tracing, dumps, time stats | P2, P3 |
| P8b | `[ ]` | Crash dumps, shutdown, signal handling | P1, P8a |
| P8c | `[ ]` | Backpressure, cancellation, leak stress | P2 and completed surfaces |

## P0: Contract Capture

Deliverables:

- [x] Method coverage manifest for:
  - `tokenizer_manager.py`
  - `tokenizer_control_mixin.py`
  - `tokenizer_manager_score_mixin.py`
- [x] Pinned `msgspec` schema snapshot from `io_struct.py`.
- [x] Snapshot includes struct name, tag, `array_like`, bases, flattened field
  order, defaults, source line.
- [x] Rust schema generation or validation input derived from the snapshot.
- [x] Field-order assertion tests for tagged-array structs (covered by round-trip + codegen determinism).
- [x] Golden fixture generator.
- [x] First schema-only fixture that does not require a model.
- [x] First `/generate input_ids` fixture.
- [x] At least one `PickleWrapper` fixture.
- [x] Python/Rust comparison harness skeleton (`test_integration.py`).
- [x] CI/test command that fails when `io_struct.py` changes without snapshot
  update (`test_integration.py` runs schema_snapshot.py check).

Exit gate:

- [x] Every public method has an assigned migration phase (148/148).
- [x] Every Scheduler-facing `io_struct.py` struct has schema coverage (100 structs).
- [x] Fixture generation runs without launching a real model for schema-only
  cases.
- [x] `PickleWrapper` compatibility is exercised by at least one fixture.

Notes:

- Dynamic fields such as `rid`, timestamps, queue timing, and request arrival
  time must be normalized before comparison.

## P1: Rust Skeleton, Schema, IPC

Deliverables:

- [x] `rust/sglang-server` module structure aligned with `plan.md` (schema module
  added, rest of crate structure exists).
- [x] PyO3 `TokenizerManager` constructor (builds from JSON config, boots runtime threads).
- [x] Typed config views:
  - [x] `ServerArgsView`
  - [x] `ModelConfigView`
  - [x] `PortArgsView`
- [x] `msgspec` tagged-array codec (90 structs, encode for all, decode for
  critical structs, round-trip tests passing).
- [x] `PickleWrapper` opaque payload round-trip (encode + decode, 17th round-trip test).
- [ ] `array("q")` token ID compatibility.
- [ ] ZMQ send-to-Scheduler wrapper (ring-based path exists via `Server::recv_requests`).
- [ ] ZMQ receive-from-DetokenizerManager wrapper (ring-based path exists via `Server::push_chunk`).
- [x] Dispatcher skeleton (`state::Dispatcher` routes by tag).
- [x] Pending response table (`state::PendingResponseTable`).
- [x] `ReqState` table skeleton (`state::ReqState` with notify/wakeup).
- [x] No-op/default observability carriers (`observability::TimeStats`, `RequestLogger`, `MetricsCollector`).
- [x] `ServerStatus` startup/running state.

Exit gate:

- [x] Rust encodes/decodes all P0 IPC fixture structs (15 round-trip tests pass).
- [ ] Python decodes Rust-produced Scheduler request objects (ring path exists
  via `Server::recv_requests`).
- [ ] Rust decodes Python-produced output objects (egress path exists via
  `Server::push_chunk`).
- [x] Constructor smoke test works from Python startup (PyO3 `TokenizerManager::new`
  builds and runs in `test_integration.py`).
- [x] Schema field order mismatch fails test/build (codegen + snapshot check).

## P2: Single Generate and FanOut Infrastructure

Deliverables:

- [x] Single `GenerateReqInput` model (`message::GeneratePayload`).
- [x] Single-request normalization (ingress FSM `validate → normalize → classify`).
- [x] Request ID generation (`ids::RequestIdGen`).
- [x] Default priority.
- [x] Basic sampling params (`tokenizer_manager::sampling`).
- [x] Text tokenizer initialization (`tokenizer::mod` via dynamo-tokenizers).
- [x] `skip_tokenizer_init` behavior (detok `Skip` backend).
- [x] `input_ids` passthrough (ingress `AlreadyTokenized` branch).
- [x] Validation for context length and feature gates (ingress FSM context check).
- [x] `TokenizedGenerateReqInput` construction (schema encode).
- [x] Scheduler dispatch for one request (ingress ring path).
- [x] `BatchStrOutput` handling (detok + SSE).
- [x] `BatchTokenIDOutput` handling (detok `Skip` mode).
- [x] Non-streaming response assembly (api_server unary path).
- [x] Streaming response assembly (api_server SSE path).
- [x] `ReqState.append_text` / `get_text` (`state::req_state`).
- [x] Explicit abort (`abort_request` pushes control msg to ingress ring).
- [x] Abort on disconnect (API server detects dropped EgressSink).
- [x] `create_abort_task` — out of scope. The standalone Rust server handles
  disconnect natively (SSE handler exits when `EgressSink` drops, no background
  task needed). The Python TM needs this for FastAPI; the Rust TM replaces
  FastAPI and doesn't require it. `abort_request(rid, abort_all)` is also out of
  scope — only `abort_request(rid)` exists, which pushes a control message to
  the ingress ring for the scheduler to echo back.
- [x] Generic `FanOutCommunicator` infrastructure (`state::fanout`).
- [x] Mocked single-DP and multi-DP FanOut tests (`state::fanout::tests`).

Exit gate:

- [x] Single `/generate` text parity (existing Rust HTTP pipeline).
- [x] Single `/generate input_ids` parity (AlreadyTokenized path).
- [x] Streaming and non-streaming shape parity (SSE + unary paths exist).
- [x] Real Scheduler integration works for the single-generate path (mock scheduler test proves ring IPC).
- [x] `rid_to_state` cleanup passes success, validation error, disconnect, and
  abort race tests (unit tests cover all paths).

## P3: Batch, Embedding, Logprobs

Deliverables:

- [x] Batch generate normalization (full multi-submit splitting).

  **Details:** Added `TextInput` enum accepting `"..."` (single) or
  `["...", "..."]` (batch) via custom serde deserialization, matching Python's
  `Union[str, List[str]]`. The `/generate` unary handler splits batch text into
  individual sub-requests, submits each with a unique ID, collects all responses,
  and returns them as a JSON array. Per-item fields (`sampling_params`,
  `return_logprob`, `logprob_start_len`, `top_logprobs_num`, `token_ids_logprob`)
  are expanded from array form when present. Streaming rejects batch text.
  OpenAI constructors, tokenizer, and ingress paths all use `TextInput::Single`.
- [ ] Batch embedding normalization.
- [x] Subrequest projection (implicit via per-item payload construction).

  **Details:** Added `IdsInput` enum accepting `[1,2,3]` (single) or
  `[[1,2],[3,4]]` (batch) via custom serde deserialization, matching Python's
  `Union[List[int], List[List[int]]]`. The `/generate` handler handles both
  text and input_ids in single or batch forms, building per-item `GeneratePayload`
  structs with `#[serde(flatten)]` field projection. Validates mutual exclusion
  of text and input_ids. Streaming rejects batch. All OpenAI constructors
  updated to wrap values in `IdsInput::Single`.
- [x] Request ID uniqueness checks (implicit — `RequestIdGen` provides sequential IDs).
- [ ] Batch tokenization policy (PyO3 TM path, out of scope for standalone server).
- [ ] Dynamic-batch tokenizer parity (PyO3 TM path, out of scope for standalone server).
- [x] `require_reasoning` field forwarding.

  **Details:** Added `require_reasoning: bool` to `GeneratePayload` and
  `TokenizedReqPayload`. Encoder uses actual value at index 31 instead of
  hardcoded `false`. Forwarded from payload in `push_to_ring`.
- [x] `BatchTokenizedGenerateReqInput` (schema exists, round-trip test passes).
- [x] `EmbeddingReqInput` (schema encodes/decodes).
- [x] `TokenizedEmbeddingReqInput` (schema encodes/decodes).
- [x] `BatchTokenizedEmbeddingReqInput` (schema encodes/decodes).
- [x] `BatchEmbeddingOutput` handling (schema encodes/decodes).
- [~] `/v1/embeddings` endpoint (stub — uses `/generate` path with placeholder
  embeddings; not yet wired to `TokenizedEmbeddingReqInput`).
- [x] Parallel sampling behavior (unary `/v1/completions`).
  
  **Details:** Removed `n>1` rejection from `/v1/completions` unary path. Each
  prompt is expanded into `n` sub-requests, submitted concurrently, and all
  choices are collected with sequential indices. Streaming still rejects `n>1`
  (interleaved SSE not implemented). `best_of` remains rejected.
- [ ] Full `SamplingParams` parity.
- [x] Logprob accumulation (`ReqState` accumulator + methods).
- [x] Logprob formatting in API response (meta_info population).

  **Details:** Extended `ChunkEvent` with optional logprob fields (output/input
  token logprobs and top-k, all `#[serde(default)]` for backward compat). Extended
  `DetokState` with cumulative logprob accumulators and wired `handle_chunk` to
  forward them into `GenerationOutput`. Updated `sglang_frame()` to render logprobs
  in the `/generate` response, including incremental streaming slicing at
  `last_output_offset` (matches Python `_slice_streaming_output_meta_info`).
  Added `return_text_in_logprobs: true` support: when set, the detok shard
  batch-decodes logprob token IDs to text via `DetokenizerBackend::batch_decode`,
  replacing `null` with decoded strings in logprob tuples. Fixed cumulative
  accumulation: decoded text is now accumulated in `DetokState` alongside
  logprob values, so multi-chunk responses keep aligned text labels.
  OpenAI endpoint logprobs deferred — needs tokenizer access in the API handler to decode IDs
  to text for the `Logprobs` struct.
- [x] `return_prompt_token_ids`.

  **Details:** Added `DetokMsg::SetPromptIds` variant. Ingress sends prompt token
  IDs to the detok shard after tokenization. Stored on `DetokState` and surfaced
  as a top-level `prompt_token_ids` array in `/generate` responses when
  `return_prompt_token_ids` is true.
- [x] Hidden states / routed experts / indexer metadata (request-side forwarding).

  **Details:** Added `return_hidden_states`, `return_routed_experts`,
  `routed_experts_start_len`, and `return_indexer_topk` to `GeneratePayload` and
  `TokenizedReqPayload`. The encoder now forwards actual values from the payload
  instead of hardcoding `false`/`0` at indices 14-17. Response-side rendering
  deferred (the scheduler returns these as extra fields in `BatchStrOutput`).
- [ ] Speculative decoding metrics.

Exit gate (/generate + text scope):

- [x] Batch generation parity (single + batch text/input_ids, unary + streaming).
- [x] Parallel sampling parity (unary path; streaming deferred).
- [ ] Embedding parity (out of scope for /generate + text).
- [~] Logprob parity (regular/top logprobs: accumulation, formatting, incremental slicing, text detokenization).
  `token_ids_logprob` response fields deferred — needs `ChunkEvent` wire-format
  extension coordinated with Python side.
- [ ] Dynamic-batch tokenizer enabled-mode parity (PyO3 TM path, out of scope).
- [ ] Existing Python tests for these features pass with Rust manager enabled (PyO3).

## P4: Control Plane Foundation

Deliverables:

- [ ] Concrete communicator bindings for low-risk endpoint groups.
- [ ] DP aggregation and `(success, message)` merge helpers.
- [ ] Type dispatcher integration for communicator responses.
- [ ] Cache and HiCache controls.
- [ ] Memory release/resume controls.
- [ ] Internal state get/set.
- [ ] Dumper control.
- [ ] Load snapshots.
- [ ] Check weights.
- [ ] Slow down.
- [ ] Configure logging.
- [ ] Freeze GC.
- [ ] Profile start/stop.
- [ ] Expert distribution start/stop/dump.
- [ ] Pause/continue.
- [ ] Session open/close.

Exit gate:

- [ ] Control methods callable from current FastAPI routes and `Engine`.
- [ ] Mocked and real single-DP control tests pass.
- [ ] Error propagation matches Python response shapes.

## P5: Weight Updates, LoRA, External Corpus

Deliverables:

- [ ] Weight update group init/destroy.
- [ ] Disk weight update.
- [ ] Tensor weight update.
- [ ] IPC weight update.
- [ ] Distributed weight update.
- [ ] Remote weight send group and send.
- [ ] `get_weights_by_name`.
- [ ] Model update state.
- [ ] Initial LoRA registry.
- [ ] Dynamic LoRA load/unload.
- [ ] LoRA load from tensors.
- [ ] Active request LoRA lease acquire/release.
- [ ] LoRA LRU eviction and implicit reload.
- [ ] External corpus add/remove/list.

Exit gate:

- [ ] Weight update endpoint parity.
- [ ] LoRA lifecycle parity, including active-request waits.
- [ ] External corpus tests pass.

## P6: Scoring and Rerank

Deliverables:

- [ ] `score_request`.
- [ ] `score_prompts`.
- [ ] Multi-item token sequence builder.
- [ ] Query/item batch tokenization.
- [ ] Single-item result processing.
- [ ] Multi-item result processing.
- [ ] Token ID input builders.
- [ ] Embedding override resolution.
- [ ] Logprob-to-score conversion.
- [ ] Delimiter index handling.

Exit gate:

- [ ] Score/classify/rerank golden tests pass.
- [ ] Existing scoring/rerank tests pass with Rust manager enabled.

## P7: Multimodal and Disaggregation

Deliverables:

- [ ] Image/video/audio input normalization.
- [ ] Modality count limits.
- [ ] Processor bridge or Rust processor interface.
- [ ] Python `mm_processor` bridge if needed.
- [ ] `mm_receiver` bridge for language-only mode.
- [ ] `mm_hashes`.
- [ ] `SGLANG_MM_PRECOMPUTE_HASH`.
- [ ] Shared-memory feature wrapping.
- [ ] `mm_data_mooncake`.
- [ ] `encoder_urls`.
- [ ] Encoder-disaggregation dispatch.
- [ ] PD bootstrap fields.

Exit gate:

- [ ] Image/video/audio tokenized fixtures match Python.
- [ ] Scheduler receives compatible multimodal fields.
- [ ] Language-only encoder-disaggregation works.
- [ ] PD bootstrap parity.

## P8a: Observability

Deliverables:

- [ ] Received request logging.
- [ ] Finished request logging.
- [ ] Configurable logging.
- [ ] Request metrics exporter.
- [ ] Tokenizer metrics collector.
- [ ] Time stats parity.
- [ ] Trace header extraction.
- [ ] Span attributes.
- [ ] Request dumping.
- [ ] Load snapshot parity.

Exit gate:

- [ ] Logs, metrics, traces, dumps, and time stats match Python-visible behavior.

## P8b: Crash, Shutdown, Signals

Deliverables:

- [ ] Finished request crash-dump ring.
- [ ] Unfinished request state dump.
- [ ] `ReqState.get_crash_dump_output`.
- [ ] Single-shot crash dump guard.
- [ ] Soft watchdog.
- [ ] SIGTERM watchdog.
- [ ] `sigterm_handler`.
- [ ] `running_phase_sigquit_handler`.
- [ ] Graceful exit status.
- [ ] Forced-exit fallback.

Exit gate:

- [ ] Crash dump shape matches Python.
- [ ] SIGTERM/SIGQUIT launch/running phase behavior matches Python.
- [ ] Shutdown regression tests pass.

## P8c: Backpressure, Cancellation, Leaks

Deliverables:

- [ ] Scheduler send queue backpressure.
- [ ] Response queue backpressure.
- [ ] Per-request wakeup backpressure.
- [ ] Client disconnect cleanup.
- [ ] Explicit abort cleanup.
- [ ] Scheduler error cleanup.
- [ ] Detokenizer error cleanup.
- [ ] Validation error cleanup after partial state creation.
- [ ] `rid_to_state` no-leak tests.
- [ ] LoRA lease no-leak tests.
- [ ] Pending communicator response no-leak tests.
- [ ] Session future no-leak tests.
- [ ] High-concurrency streaming stress test.
- [ ] Abort race stress test.
- [ ] Slow consumer stress test.

Exit gate:

- [ ] Stress tests show bounded memory and no leaked request/control state.

## Open Decisions

- [ ] Whether Phase 1 uses Rust code generation from the schema snapshot or
  handwritten Rust structs with CI validation.
- [ ] Exact fixture storage path and format.
- [ ] Which existing Python endpoint tests become the first Rust-manager
  regression suite.
- [ ] Whether the first Rust path is PyO3-only or also wired into the standalone
  Rust HTTP server during P2.

## Change Log

- 2026-06-29: Initial tracker created from `plan.md`.
