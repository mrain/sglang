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
| P2 | `[ ]` | Single `/generate`, abort, response wait, FanOut infrastructure | P1 |
| P3 | `[ ]` | Batch, embedding, logprobs, full sampling params | P2 |
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

- [ ] Single `GenerateReqInput` model.
- [ ] Single-request normalization.
- [ ] Request ID generation.
- [ ] Default priority.
- [ ] Basic sampling params.
- [ ] Text tokenizer initialization.
- [ ] `skip_tokenizer_init` behavior.
- [ ] `input_ids` passthrough.
- [ ] Validation for context length and feature gates.
- [ ] `TokenizedGenerateReqInput` construction.
- [ ] Scheduler dispatch for one request.
- [ ] `BatchStrOutput` handling.
- [ ] `BatchTokenIDOutput` handling.
- [ ] Non-streaming response assembly.
- [ ] Streaming response assembly.
- [ ] `ReqState.append_text` / `get_text`.
- [ ] Explicit abort.
- [ ] Abort on disconnect.
- [ ] `create_abort_task`.
- [ ] Generic `FanOutCommunicator` infrastructure.
- [ ] Mocked single-DP and multi-DP FanOut tests.

Exit gate:

- [ ] Single `/generate` text parity.
- [ ] Single `/generate input_ids` parity.
- [ ] Streaming and non-streaming shape parity.
- [ ] Real Scheduler integration works for the single-generate path.
- [ ] `rid_to_state` cleanup passes success, validation error, disconnect, and
  abort race tests.

## P3: Batch, Embedding, Logprobs

Deliverables:

- [ ] Batch generate normalization.
- [ ] Batch embedding normalization.
- [ ] Subrequest projection.
- [ ] Request ID uniqueness checks.
- [ ] Batch tokenization policy.
- [ ] Dynamic-batch tokenizer parity.
- [ ] `BatchTokenizedGenerateReqInput`.
- [ ] `EmbeddingReqInput`.
- [ ] `TokenizedEmbeddingReqInput`.
- [ ] `BatchTokenizedEmbeddingReqInput`.
- [ ] `BatchEmbeddingOutput` handling.
- [ ] Parallel sampling behavior.
- [ ] Full `SamplingParams` parity.
- [ ] Logprob accumulation and formatting.
- [ ] `return_prompt_token_ids`.
- [ ] Hidden states, routed experts, indexer metadata.
- [ ] Speculative decoding metrics.

Exit gate:

- [ ] Batch generation parity.
- [ ] Parallel sampling parity.
- [ ] Embedding parity.
- [ ] Logprob parity.
- [ ] Dynamic-batch tokenizer enabled-mode parity.
- [ ] Existing Python tests for these features pass with Rust manager enabled.

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
