# TokenizerManager Rust Migration Plan

This plan targets a complete Rust replacement for the current Python
`TokenizerManager` behavior described in `tokenizer_manager.md` and implemented
across:

- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/tokenizer_control_mixin.py`
- `python/sglang/srt/managers/tokenizer_manager_score_mixin.py`
- `python/sglang/srt/managers/io_struct.py`

The migration should preserve current user-facing behavior and Scheduler-facing
wire semantics before any cleanup or protocol redesign.

## Scope

In scope:

- Request admission and request ID state.
- `GenerateReqInput` and `EmbeddingReqInput` normalization.
- Text tokenization.
- Multimodal preprocessing dispatch and metadata plumbing.
- Sampling parameter construction, normalization, and validation.
- Tokenized request construction.
- Scheduler dispatch.
- Response routing from DetokenizerManager/Scheduler outputs.
- Streaming and non-streaming response assembly.
- Abort and disconnect handling.
- Pause/continue.
- LoRA resolution, load/unload control, reference tracking.
- Sessions.
- Weight updates.
- Cache, memory, profile, internal state, expert distribution, HiCache, external
  corpus, dumper, and admin controls.
- Request logging, tracing, metrics, request dumps, crash dumps, watchdogs.
- Scoring/rerank helper behavior currently in `TokenizerManagerScoreMixin`.

Out of scope for this specific plan:

- Scheduler batching and model execution.
- KV cache allocation.
- ModelRunner.
- Rewriting HTTP routes, except where a route adapter is needed to call the Rust
  TokenizerManager.
- Rewriting DetokenizerManager, except for the compatibility surface needed to
  receive its current outputs.

## Migration Principle

Use a compatibility-first migration.

The first complete Rust version should behave like the current Python
TokenizerManager from both sides:

```text
HTTP / Engine / protocol adapters
  -> Rust TokenizerManager API
  -> current Scheduler request structs / current IPC semantics
  -> current Scheduler and DetokenizerManager
  -> Rust TokenizerManager response generator API
```

Do not change Scheduler semantics while porting TokenizerManager. Once behavior
is covered by parity tests, wire-format and API cleanup can happen separately.

## Implementation Target and Rust Module Layout

Implement this migration inside `rust/sglang-server`.

Do not create a separate crate for the first migration pass. The current
`sglang-server` crate already owns the frontend runtime shape: HTTP/API ingress,
TokenizerManager-style request processing, tokenizer workers, scheduler boundary,
egress routing, and detokenization. Keeping the migration in this crate avoids
premature public API design while the Scheduler-facing protocol is still being
proven.

Keep the module boundary clean enough that stable pieces can be extracted later
into a shared crate if needed. Good later extraction candidates are schema,
msgpack codec, sampling params, request normalization, and LoRA registry. Keep
runtime-specific pieces in `sglang-server`: HTTP routes, PyO3 entry points,
thread/runtime layout, request sinks, and scheduler/detokenizer bridge code.

```text
rust/sglang-server/
  Cargo.toml
  pyproject.toml
  src/
    lib.rs
    error.rs
    config/
      mod.rs
      server_args.rs
      model_config.rs
      port_args.rs
    api/
      mod.rs
      py.rs
      engine.rs
      stream.rs
    state/
      mod.rs
      manager.rs
      req_state.rs
      session.rs
      pause.rs
      watchdog.rs
    schema/
      mod.rs
      base.rs
      generate.rs
      embedding.rs
      tokenized.rs
      output.rs
      control.rs
      pickle.rs
    normalize/
      mod.rs
      generate.rs
      embedding.rs
      batch.rs
      scoring.rs
    tokenizer/
      mod.rs
      backend.rs
      dynamic_batch.rs
      input_format.rs
    sampling/
      mod.rs
      params.rs
      stop.rs
      verify.rs
    multimodal/
      mod.rs
      processor.rs
      receiver.rs
      shm.rs
      hashes.rs
      limits.rs
    ipc/
      mod.rs
      zmq.rs
      codec.rs
      dispatcher.rs
      fanout.rs
      scheduler.rs
      detokenizer.rs
    response/
      mod.rs
      batch_output.rs
      wait.rs
      logprob.rs
      metrics.rs
      dump.rs
    control/
      mod.rs
      weights.rs
      lora.rs
      session.rs
      cache.rs
      memory.rs
      profile.rs
      internal_state.rs
      external_corpus.rs
      admin.rs
    lora/
      mod.rs
      registry.rs
      refs.rs
    disaggregation/
      mod.rs
      pd.rs
      encoder.rs
    observability/
      mod.rs
      logging.rs
      tracing.rs
      metrics.rs
      crash_dump.rs
      load_snapshot.rs
    scoring/
      mod.rs
      request.rs
      postprocess.rs
```

The exact file names can adapt to the current `rust/sglang-server/src` layout,
but the dependency direction should stay clear:

```text
api_server -> tokenizer_manager -> schema/ipc/tokenizer/sampling/multimodal
tokenizer_manager -> response/control/lora/scoring/observability
runtime -> stages and channels
lower-level modules must not depend on api_server
```

## Core Public Surface

Expose a Rust-owned manager through PyO3 first, so current Python routes and
`Engine` can call it with minimal churn.

The binding must expose every public method mapped in the phase plan. The sketch
below shows the expected shape; the phase mappings are the source of truth for
the complete method list.

```rust
#[pyclass]
pub struct TokenizerManager { ... }

#[pymethods]
impl TokenizerManager {
    #[new]
    fn new(server_args: PyObject, port_args: PyObject) -> PyResult<Self>;

    fn generate_request(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn abort_request(&self, rid: Option<String>, abort_all: bool) -> PyResult<()>;
    fn create_abort_task(&self, obj: PyObject) -> PyResult<PyObject>;

    fn pause_generation(&self, obj: PyObject) -> PyResult<PyObject>;
    fn continue_generation(&self, obj: PyObject) -> PyResult<PyObject>;

    fn init_weights_update_group(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn destroy_weights_update_group(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn update_weights_from_disk(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn update_weights_from_tensor(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn update_weights_from_ipc(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn update_weights_from_distributed(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn init_weights_send_group_for_remote_instance(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn send_weights_to_remote_instance(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn get_weights_by_name(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;

    fn load_lora_adapter(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn load_lora_adapter_from_tensors(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn unload_lora_adapter(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;

    fn open_session(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn close_session(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;

    fn flush_cache(&self, timeout_s: Option<f64>) -> PyResult<PyObject>;
    fn clear_hicache_storage(&self) -> PyResult<PyObject>;
    fn attach_hicache_storage(&self, ...) -> PyResult<PyObject>;
    fn detach_hicache_storage(&self) -> PyResult<PyObject>;
    fn release_memory_occupation(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn resume_memory_occupation(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;

    fn start_profile(&self, req: Option<PyObject>) -> PyResult<PyObject>;
    fn stop_profile(&self) -> PyResult<PyObject>;
    fn start_expert_distribution_record(&self) -> PyResult<PyObject>;
    fn stop_expert_distribution_record(&self) -> PyResult<PyObject>;
    fn dump_expert_distribution_record(&self) -> PyResult<PyObject>;
    fn get_internal_state(&self) -> PyResult<PyObject>;
    fn set_internal_state(&self, obj: PyObject) -> PyResult<PyObject>;
    fn dumper_control(&self, obj: PyObject) -> PyResult<PyObject>;
    fn get_loads(&self, include: Option<Vec<String>>, dp_rank: Option<usize>) -> PyResult<PyObject>;
    fn check_weights(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn slow_down(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn configure_logging(&self, obj: PyObject) -> PyResult<()>;
    fn freeze_gc(&self) -> PyResult<PyObject>;

    fn add_external_corpus(&self, obj: PyObject) -> PyResult<PyObject>;
    fn remove_external_corpus(&self, corpus_id: String) -> PyResult<PyObject>;
    fn list_external_corpora(&self) -> PyResult<PyObject>;

    fn score_request(&self, obj: PyObject, request: Option<PyObject>) -> PyResult<PyObject>;
    fn score_prompts(&self, ...) -> PyResult<PyObject>;
}
```

Long term, direct Rust HTTP handlers can call the same internal Rust manager
without PyO3 object shims.

## Data Structures to Port

### Configuration

Rust structs:

- `ServerArgsView`
- `ModelConfigView`
- `PortArgsView`
- `DisaggregationMode`
- `SpeculativeAlgorithm`

Required fields:

- `model_path`, `served_model_name`, `tokenizer_path`, tokenizer mode/backend.
- `context_len`, `image_token_id`, vocab size, embedding/generation flags.
- `skip_tokenizer_init`, dynamic batch tokenizer flags.
- `tokenizer_worker_num`, DP/TP settings.
- metrics/logging/tracing flags.
- LoRA settings and preloaded LoRA refs.
- multimodal limits and processor settings.
- disaggregation and encoder settings.
- cache, HiCache, profile, and watchdog settings.

Implementation choice:

- Initially parse from Python `ServerArgs` and `ModelConfig` objects through
  PyO3 into typed Rust views.
- Avoid holding arbitrary `PyObject` in hot-path state unless a feature has not
  been ported yet.

### Request Inputs

Rust structs:

- `GenerateReqInput`
- `EmbeddingReqInput`
- `SessionParams`
- `RequestId`
- `BatchView<T>`

Must cover:

- Single and batch shapes.
- `text`, `input_ids`, `input_embeds`.
- image/video/audio data handles.
- sampling params.
- logprob fields.
- streaming flag.
- hidden states, routed experts, indexer flags.
- LoRA path and resolved LoRA ID.
- session fields.
- disaggregation/bootstrap fields.
- DP routing fields.
- priority, extra key, routing key.
- observability fields.

The Rust model should distinguish:

```rust
enum RequestBody {
    Generate(GenerateReqInput),
    Embedding(EmbeddingReqInput),
}

enum InputSource {
    Text(TextInput),
    TokenIds(Vec<i64>),
    Embeds(Vec<Vec<f32>>),
}
```

### Normalized Requests

Rust structs:

- `NormalizedGenerateReq`
- `NormalizedEmbeddingReq`
- `NormalizedBatch<T>`
- `SubRequestRef`

Responsibilities:

- Exactly one input mode validation.
- Batch-size detection.
- Single-value to batch-value expansion.
- Request ID generation.
- Parallel sampling expansion.
- Per-subrequest projection.
- Request ID uniqueness validation.
- Field shape validation for logprobs, multimodal, LoRA, and custom options.

### Tokenized Scheduler Inputs

Rust structs must match current Python msgspec array/tag order:

- `TokenizedGenerateReqInput`
- `TokenizedEmbeddingReqInput`
- `BatchTokenizedGenerateReqInput`
- `BatchTokenizedEmbeddingReqInput`

Fields to preserve:

- Base fields: `rid`, `http_worker_ipc`.
- Inputs: `input_text`, `input_ids`, `input_embeds`.
- Multimodal: `mm_inputs`, `token_type_ids`, `mm_data_mooncake`,
  `encoder_urls`, `need_wait_for_mm_inputs`, `num_items_assigned`.
- Sampling: `SamplingParams`.
- Return controls: logprobs, hidden states, routed experts, indexer output,
  bytes, entropy, prompt token IDs.
- Sessions.
- LoRA ID.
- Custom logit processor.
- Positional embedding overrides.
- Bootstrap/disaggregation/routing fields.
- Priority, extra key, routing key.
- Observability: `time_stats`.

Compatibility requirements:

- Encode msgpack exactly like `msgspec.Struct(tag=True, array_like=True)`.
- Preserve field order from `io_struct.py`.
- Preserve `array("q")` semantics for token IDs when crossing into Python.
- Implement a Rust equivalent of `PickleWrapper` for fields that remain Python
  objects.
- Implement or call into existing shared-memory feature wrapping.

### Outputs

Rust structs:

- `BatchStrOutput`
- `BatchTokenIDOutput`
- `BatchEmbeddingOutput`
- `AbortReq`
- Control outputs for communicator responses.

Output processing must preserve:

- `meta_info` shape.
- finish reason shape.
- prompt/completion/reasoning/cache token counts.
- logprob output format.
- hidden state output behavior.
- routed experts and indexer base64 behavior.
- placeholder token metadata.
- detailed cache breakdown.
- DP rank.
- time stats.
- multimodal token counts.
- speculative decoding metrics.
- pooled hidden states for embeddings.

### Internal State

Rust structs:

```rust
struct TokenizerManagerState {
    rid_to_state: DashMap<RequestId, ReqState>, // or Mutex<HashMap<...>>
    session_futures: SessionTable,
    pause: PauseState,
    model_update: ModelUpdateState,
    lora: LoraState,
    dispatchers: DispatcherTable,
    observability: ObservabilityState,
    disaggregation: DisaggregationState,
}

struct ReqState {
    out_queue: VecDeque<ResponseChunk>,
    finished: bool,
    notify: Notify,
    request: RequestBody,
    time_stats: ApiServerReqTimeStats,
    last_completion_tokens: u64,
    ttft_observed: bool,
    last_output_offset: usize,
    text: String,
    text_chunks: Vec<String>,
    output_ids: Vec<i64>,
    logprob_state: LogprobState,
    customized_info_accumulated: HashMap<String, Vec<Value>>,
    prompt_token_ids: Option<Vec<i64>>,
}
```

Use `tokio::sync::Notify` or channels for per-request wakeups. Keep the
request-state table API explicit; it is the core concurrency boundary.

### LoRA State

Rust structs:

- `LoraRegistry`
- `LoraRef`
- `LoraLease`
- `LoraUpdateState`

Must support:

- Initial registry from `server_args.lora_paths`.
- Dynamic register/unregister.
- Active request reference counting.
- LRU eviction excluding pinned adapters.
- Reloading previously evicted adapters.
- Waiting for active leases before unload.

### Communicators

Rust equivalent of `FanOutCommunicator`:

```rust
struct FanOutCommunicator<Req, Resp> {
    dispatch: SchedulerDispatch,
    dp_size: usize,
    mode: FanOutMode,
    pending: PendingResponseTable<Resp>,
}
```

Modes:

- queueing: ordinary request/response collection.
- watching: ongoing watcher-style outputs such as load snapshots.

Must support:

- One request sent to Scheduler.
- One response per DP rank where applicable.
- Merge helpers for `(success, message)`.
- Timeout/cancellation behavior compatible with current Python call paths.

## Plan Review

The inventory above is accurate, but the first draft mixed two different views:

- Functional areas: what has to be ported.
- Milestones: what should be implemented together to create a usable, testable
  system.

For execution, use phases that each produce a runnable or testable artifact. Do
not start with the full control plane. The critical path is:

```text
schema compatibility
  -> request normalization
  -> tokenizer initialization
  -> single generate request
  -> response assembly
  -> fan-out control infrastructure
  -> batch / embedding / logprobs
  -> control plane
  -> LoRA / scoring
  -> multimodal / disaggregation
  -> observability / shutdown / stress hardening
```

The most important engineering constraint is to keep the Scheduler-facing
contract stable while porting. The highest-risk parts are msgspec field order,
opaque Python fields, async-generator behavior, and multimodal processors.

Feature coverage audit:

- The Python method surface is covered by the phases below. Mechanical comparison
  against `tokenizer_manager.py`, `tokenizer_control_mixin.py`, and
  `tokenizer_manager_score_mixin.py` should be part of Phase 0 so newly added
  methods cannot silently fall out of scope.
- `io_struct.py` contains some structs that TokenizerManager does not implement
  behavior for directly, but the Rust schema layer must still be able to encode
  or decode every struct that can cross the TokenizerManager/Scheduler/
  DetokenizerManager boundary.
- The plan treats these as two deliverable classes:
  - behavior parity: Rust implements the method/state machine;
  - schema parity: Rust can serialize/deserialize the IPC struct even if the
    behavior is owned elsewhere.

Phase dependency graph:

- Phase 0 blocks all implementation phases. It owns the pinned Python contract:
  method manifest, schema field-order snapshot, golden fixtures, and comparison
  harness.
- Phase 1 depends on Phase 0 and blocks all runtime phases. It owns constructor,
  schema codec, ZMQ primitives, dispatcher tables, no-op observability fields,
  and request-state skeletons.
- Phase 2 depends on Phase 1. It owns the first real data-plane path and the
  reusable `FanOutCommunicator` infrastructure, but not the full control API
  set.
- Phase 3 depends on Phase 2. It extends the already-working generate path to
  batch, embedding, logprobs, and full sampling behavior.
- Phase 4 depends on Phase 1 IPC/dispatcher and Phase 2 FanOut infrastructure.
  It turns the generic control transport into concrete low-risk endpoints.
- Phase 5 depends on Phase 4 for control request machinery and Phase 2/3 for
  request-lifetime hooks used by LoRA leases.
- Phase 6 depends on Phase 3 tokenization, embedding, and logprob behavior.
- Phase 7 depends on Phase 2 tokenization flow and Phase 3 batch behavior; its
  processor bridge may depend on temporary PyO3 object handling from Phase 1.
- Phase 8a depends on Phase 2/3 response timing checkpoints.
- Phase 8b depends on Phase 1 lifecycle state and Phase 8a request records.
- Phase 8c depends on Phase 2 abort/cancellation paths and the completed
  control/data-plane flows being stress-tested.

Cross-cutting stubs required before their full phase:

- `ApiServerReqTimeStats` fields must exist in Phase 1 structs as compatible
  no-op/default values, then get real timestamps in Phase 2 and complete parity
  in Phase 8a.
- Logging, tracing, metrics, request dump, and crash dump hooks should be cheap
  no-op traits or structs in Phase 1 so hot-path code does not need a broad
  retrofit in Phase 8a/8b.
- `PickleWrapper` must be supported by the Phase 1 codec even when the wrapped
  object is still opaque to Rust.

Schema coverage groups:

- Data-plane requests:
  - `GenerateReqInput`, `EmbeddingReqInput`
  - `TokenizedGenerateReqInput`, `TokenizedEmbeddingReqInput`
  - `BatchTokenizedGenerateReqInput`, `BatchTokenizedEmbeddingReqInput`
- Data-plane outputs:
  - `BatchTokenIDOutput`, `BatchStrOutput`, `BatchEmbeddingOutput`
- Core control/lifecycle:
  - `AbortReq`, `ActiveRanksOutput`, `HealthCheckOutput`, `FreezeGCReq`,
    `ShutdownReq`, `ConfigureLoggingReq`
  - `PauseGenerationReqInput`, `ContinueGenerationReqInput`
  - `TokenizerWorkerRegistrationReq`, `PauseContinueBroadcastReq`
- Cache, memory, loads, dump:
  - `FlushCacheReqInput`, `FlushCacheReqOutput`
  - `ClearHiCacheReqInput`, `ClearHiCacheReqOutput`
  - `AttachHiCacheStorageReqInput`, `AttachHiCacheStorageReqOutput`
  - `DetachHiCacheStorageReqInput`, `DetachHiCacheStorageReqOutput`
  - `ReleaseMemoryOccupationReqInput`, `ReleaseMemoryOccupationReqOutput`
  - `ResumeMemoryOccupationReqInput`, `ResumeMemoryOccupationReqOutput`
  - `GetLoadsReqInput`, `GetLoadsReqOutput`
  - `DumperControlReqInput`, `DumperControlReqOutput`
  - `SetInjectDumpMetadataReqInput`, `SetInjectDumpMetadataReqOutput`
  - `LazyDumpTensorsReqInput`, `LazyDumpTensorsReqOutput`
  - `BlockReqInput`
- Weights:
  - `UpdateWeightFromDiskReqInput`, `UpdateWeightFromDiskReqOutput`
  - `UpdateWeightsFromDistributedReqInput`,
    `UpdateWeightsFromDistributedReqOutput`
  - `UpdateWeightsFromTensorReqInput`, `UpdateWeightsFromTensorReqOutput`
  - `UpdateWeightsFromIPCReqInput`, `UpdateWeightsFromIPCReqOutput`
  - `InitWeightsUpdateGroupReqInput`, `InitWeightsUpdateGroupReqOutput`
  - `DestroyWeightsUpdateGroupReqInput`, `DestroyWeightsUpdateGroupReqOutput`
  - `InitWeightsSendGroupForRemoteInstanceReqInput`,
    `InitWeightsSendGroupForRemoteInstanceReqOutput`
  - `SendWeightsToRemoteInstanceReqInput`, `SendWeightsToRemoteInstanceReqOutput`
  - `UpdateWeightVersionReqInput`
  - `GetWeightsByNameReqInput`, `GetWeightsByNameReqOutput`
  - `UpdateExpertBackupReq`, `BackupDramReq`
- Sessions:
  - `OpenSessionReqInput`, `CloseSessionReqInput`, `OpenSessionReqOutput`
- Profiling and expert distribution:
  - `ProfileReq`, `ProfileReqOutput`
  - `ExpertDistributionReq`, `ExpertDistributionReqOutput`
- Internal state/admin:
  - `GetInternalStateReq`, `GetInternalStateReqOutput`
  - `SetInternalStateReq`, `SetInternalStateReqOutput`
  - `CheckWeightsReqInput`, `CheckWeightsReqOutput`
  - `SlowDownReqInput`, `SlowDownReqOutput`
  - `RpcReqInput`, `RpcReqOutput`
- LoRA:
  - `LoadLoRAAdapterReqInput`, `LoadLoRAAdapterFromTensorsReqInput`
  - `UnloadLoRAAdapterReqInput`, `LoRAUpdateOutput`
- External corpus:
  - `AddExternalCorpusReqInput`, `AddExternalCorpusReqOutput`
  - `RemoveExternalCorpusReqInput`, `RemoveExternalCorpusReqOutput`
  - `ListExternalCorporaReqInput`, `ListExternalCorporaReqOutput`
- Protocol-adapter structs that are schema compatibility items, not core
  TokenizerManager behavior:
  - `ParseFunctionCallReq`
  - `SeparateReasoningReqInput`
  - `VertexGenerateReqInput`

## Phased Migration Plan

### Phase 0: Contract Capture and Test Harness

Goal: freeze the current Python behavior before porting it.

Deliverables:

- Golden fixture generator that runs the current Python TokenizerManager path and
  records:
  - raw API/Engine request object,
  - normalized request object,
  - tokenized request object,
  - decoded msgpack object sent to Scheduler,
  - representative DetokenizerManager output,
  - final response chunk(s).
- Pinned msgspec schema snapshot generated from `io_struct.py`, including:
  - struct name,
  - tag value,
  - `array_like` flag,
  - base structs,
  - flattened field order,
  - default/default-factory metadata,
  - source file and line number.
- Rust schema generation or validation input derived from the pinned snapshot.
  The preferred path is to generate Rust field-order constants/codecs from the
  snapshot. If full code generation is not used initially, CI must fail when the
  Python snapshot and Rust schema declarations diverge.
- Field-order assertion tests for every `msgspec.Struct(tag=True,
  array_like=True)` that crosses the TokenizerManager/Scheduler/
  DetokenizerManager boundary. These tests must compare raw tagged-array
  msgpack bytes for at least one fixture per struct family.
- Build-time or test-time schema guard:
  - `build.rs`/codegen may fail compilation if the pinned snapshot does not
    match Rust declarations; or
  - a mandatory CI test may fail before integration tests run.
- Fixture categories:
  - single text generation,
  - `input_ids` generation,
  - `skip_tokenizer_init`,
  - streaming and non-streaming,
  - batch generation,
  - parallel sampling,
  - embedding,
  - score/rerank,
  - logprobs,
  - sessions,
  - LoRA,
  - abort,
  - weight update,
  - at least one request with a `PickleWrapper`-wrapped field,
  - multimodal,
  - encoder-disaggregation metadata.
- A schema inspection tool that extracts `msgspec.Struct` field order from
  `io_struct.py` and emits the pinned snapshot above.
- A Rust/Python test harness that can compare Rust-produced decoded msgpack to
  Python-produced decoded msgpack.

Exit criteria:

- At least one fixture exists for every public interface category.
- At least one fixture exercises `PickleWrapper` encode/decode compatibility.
- Field order snapshots exist for all Scheduler-facing request and output
  structs.
- The Rust schema layer is generated from, or mechanically checked against, the
  pinned Python snapshot.
- A change to `io_struct.py` field order fails CI unless the snapshot and Rust
  schema are intentionally updated together.
- CI can run the fixture comparison without launching a real model.

Python source mapping:

- `io_struct.py`
  - `BaseReq`, `BaseBatchReq`, `PickleWrapper`
  - `GenerateReqInput`, `EmbeddingReqInput`
  - `TokenizedGenerateReqInput`, `TokenizedEmbeddingReqInput`
  - `BatchTokenizedGenerateReqInput`, `BatchTokenizedEmbeddingReqInput`
  - `BatchStrOutput`, `BatchTokenIDOutput`, `BatchEmbeddingOutput`
  - all control request/output structs
- `tokenizer_manager.py`
  - `generate_request`
  - `_tokenize_one_request`
  - `_create_tokenized_object`
  - `_send_one_request`
  - `_send_batch_request`
  - `_handle_batch_output`
  - `_wait_one_response`
- `tokenizer_control_mixin.py`
  - `_COMMUNICATOR_SPECS`
  - public control methods for fixture coverage
- `tokenizer_manager_score_mixin.py`
  - `score_request`
  - `score_prompts`

### Phase 1: Rust Skeleton, Schema, and IPC Compatibility

Goal: build the Rust shell that can be instantiated from Python and can speak
the current Scheduler wire protocol.

Deliverables:

- `rust/sglang-server` module structure created or reorganized.
- PyO3 constructor for `TokenizerManager`.
- Typed config views:
  - `ServerArgsView`
  - `ModelConfigView`
  - `PortArgsView`
  - `DisaggregationMode`
  - `SpeculativeAlgorithm`
- `schema/` module with Rust structs or generated codecs for:
  - base request structs,
  - generate/embedding API inputs,
  - tokenized Scheduler inputs,
  - normal outputs,
  - control inputs/outputs,
  - `PickleWrapper`.
- Schema declarations generated from, or mechanically checked against, the Phase
  0 pinned msgspec snapshot.
- msgspec-compatible tagged-array codec.
- `array("q")` token ID conversion compatibility.
- Opaque Python-object payload compatibility through `PickleWrapper`, including
  round-trip tests for fields that Rust does not understand yet.
- ZMQ socket wrappers for:
  - send to Scheduler,
  - receive from DetokenizerManager.
- Dispatcher skeleton:
  - type/tag-based output dispatcher,
  - control response routing table,
  - pending response table.
- `ReqState` table skeleton with per-request wakeup primitive.
- No-op/default observability carriers for time stats, request logging, tracing,
  metrics, dumps, and crash dumps. These must preserve schema fields early even
  though Phase 8a/8b own full behavior.
- `serving_chat_class` equivalent for HTTP/OpenAI template-serving code paths.
- `ServerStatus` state and minimal startup/running status transitions.

Exit criteria:

- Rust can encode and decode all Phase 0 IPC fixture structs.
- Rust field-order declarations match the Phase 0 pinned schema snapshot.
- Python can decode Rust-produced Scheduler request objects.
- Rust can decode Python-produced DetokenizerManager output objects.
- `PickleWrapper` fixtures round-trip without Rust inspecting the wrapped Python
  object.
- Constructor can be called from Python startup without changing request flow.
- Mechanical source-surface check shows no unmapped initialization, schema, or
  IPC helper methods from `tokenizer_manager.py`.

Python source mapping:

- `tokenizer_manager.py`
  - `__init__`
  - `serving_chat_class`
  - `init_model_config`
  - `init_ipc_channels`
  - `init_running_status`
  - `init_request_dispatcher`
  - `_dispatch_to_scheduler`
  - `_async_dispatch_to_scheduler`
  - `stamp_http_worker_ipc`
  - `ServerStatus`
- `io_struct.py`
  - all `msgspec.Struct` IPC definitions
  - `wrap_as_pickle`, `unwrap_from_pickle`, `PickleWrapper`
- `tokenizer_control_mixin.py`
  - `_COMMUNICATOR_SPECS`
  - `init_communicators`

### Phase 2: Single Generate Path

Goal: make one text or `input_ids` generation request flow through the Rust
manager and current Python Scheduler/Detokenizer path.

Deliverables:

- Rust `GenerateReqInput` model for single requests.
- Single-request normalization:
  - exactly-one-input validation,
  - request ID generation,
  - default sampling params,
  - default priority,
  - routed DP validation.
- Text tokenization path:
  - tokenizer initialization from `server_args.tokenizer_path` /
    `server_args.model_path`,
  - tokenizer mode/backend/revision/trust-remote-code handling needed for
    text-only models,
  - dynamic-batch tokenizer may be feature-gated off in this phase,
  - `skip_tokenizer_init` behavior,
  - normal tokenizer encode,
  - `input_ids` passthrough.
- Sampling params subset sufficient for normal generation:
  - defaults,
  - preferred sampling params merge,
  - `max_new_tokens`,
  - temperature/top-p/top-k/min-p,
  - stop/stop_regex,
  - penalties,
  - `is_normalized`.
- Validation:
  - context length,
  - input plus max-new-token length,
  - hidden-state and custom-logit feature gates.
- `TokenizedGenerateReqInput` construction.
- `_send_one_request` equivalent.
- `handle_loop` for `BatchStrOutput` and `BatchTokenIDOutput`.
- `_wait_one_response` equivalent for:
  - non-streaming response,
  - streaming response,
  - incremental streaming,
  - response-sent timestamp.
- `ReqState` text accumulation helpers:
  - `append_text`,
  - `get_text`,
  - lazy text materialization.
- Basic abort on disconnect or explicit `abort_request`.
- `create_abort_task` for FastAPI streaming disconnects.
- `background` request behavior for disconnect checks.
- Generic `FanOutCommunicator` infrastructure:
  - typed request/response envelope,
  - pending response table,
  - DP fan-out/fan-in,
  - timeout and cancellation behavior,
  - mocked Scheduler response tests.
  Concrete control endpoints remain Phase 4+, but the transport abstraction
  should exist here so later control work only adds request types and handlers.

Exit criteria:

- Single `/generate` text request parity against Python.
- Single `/generate input_ids` request parity against Python.
- Streaming and non-streaming output shapes match Python.
- Real Scheduler integration works for the single-generate path.
- `FanOutCommunicator` passes mocked single-DP and multi-DP request/response
  tests.
- `rid_to_state` entries are created, woken, finished, and removed exactly once
  for success and abort cases.
- Text-only tokenizer initialization works without requiring multimodal
  processor support.

Python source mapping:

- `io_struct.py`
  - `GenerateReqInput`
  - `TokenizedGenerateReqInput`
  - `BatchStrOutput`
  - `BatchTokenIDOutput`
  - `AbortReq`
- `tokenizer_manager.py`
  - `generate_request`
  - `_set_default_priority`
  - `init_tokenizer_and_processor` text-tokenizer branch
  - `_init_req_state`
  - `_detect_input_format`
  - `_prepare_tokenizer_input`
  - `_extract_tokenizer_results`
  - `_tokenize_texts`
  - `_tokenize_one_request`
  - `_validate_one_request`
  - `_validate_token_ids_logprob`
  - `_create_tokenized_object`
  - `_send_one_request`
  - `auto_create_handle_loop`
  - `handle_loop`
  - `_handle_batch_output`
  - `_wait_one_response`
  - `_coalesce_streaming_chunks`
  - `_slice_streaming_output_meta_info`
  - `_handle_abort_finish_reason`
  - `abort_request`
  - `create_abort_task`
  - `_handle_abort_req`
  - `ReqState.append_text`
  - `ReqState.get_text`
- `tokenizer_control_mixin.py`
  - `init_communicators` generic infrastructure behavior
  - `_COMMUNICATOR_SPECS` transport shape only

### Phase 3: Batch, Parallel Sampling, Embedding, and Logprobs

Goal: complete the data-plane request variants except multimodal and scoring.

Deliverables:

- Batch normalization for generate and embedding requests.
- `__getitem__`-equivalent subrequest projection.
- Request ID uniqueness checks.
- Batch tokenization policy:
  - `_should_use_batch_tokenization`,
  - `_batch_has_text`,
  - `_validate_batch_tokenization_constraints`,
  - `_batch_tokenize_and_process`.
- `BatchTokenizedGenerateReqInput`.
- Parallel sampling behavior:
  - prefix-cache warmup requests,
  - regenerated request IDs,
  - final sampled request expansion.
- `EmbeddingReqInput`.
- `TokenizedEmbeddingReqInput`.
- `BatchTokenizedEmbeddingReqInput`.
- `BatchEmbeddingOutput` response handling.
- Matryoshka dimension validation.
- Pooled hidden state output handling.
- Logprob accumulation and response formatting:
  - input/output token logprobs,
  - top logprobs,
  - token IDs logprobs,
  - text detokenization in logprobs where enabled,
  - entropy output,
  - incremental streaming metadata slicing.
- `return_prompt_token_ids`.
- Complete `SamplingParams` parity beyond the Phase 2 subset:
  - tokenizer-dependent stop token handling,
  - stop token IDs,
  - grammar fields,
  - regex/json_schema/ebnf exclusivity,
  - logit bias,
  - custom params carried through to Scheduler.
- `AsyncDynamicbatchTokenizer` parity when
  `enable_dynamic_batch_tokenizer=True`.
- Hidden states, routed experts, indexer top-k response metadata.
- Speculative decoding metrics in `meta_info`.

Exit criteria:

- Batch generation parity.
- Parallel sampling parity.
- Embedding request parity.
- Logprob response parity.
- `return_prompt_token_ids` parity.
- Full sampling params parity and dynamic-batch tokenizer enabled-mode parity.
- Existing Python tests covering generate/embedding/logprob pass with the Rust
  manager.

Python source mapping:

- `io_struct.py`
  - `EmbeddingReqInput`
  - `TokenizedEmbeddingReqInput`
  - `BatchTokenizedGenerateReqInput`
  - `BatchTokenizedEmbeddingReqInput`
  - `BatchEmbeddingOutput`
  - logprob type aliases and output fields
- `tokenizer_manager.py`
  - `_handle_batch_request`
  - `_batch_tokenize_and_process`
  - `init_tokenizer_and_processor` dynamic-batch tokenizer branch
  - `_validate_batch_tokenization_constraints`
  - `_batch_has_text`
  - `_should_use_batch_tokenization`
  - `_send_batch_request`
  - `_validate_for_matryoshka_dim`
  - `_validate_input_ids_in_vocab`
  - `_resolve_embed_overrides`
  - `add_logprob_to_meta_info`
  - `convert_logprob_style`
  - `detokenize_logprob_tokens`
  - `detokenize_top_logprobs_tokens`
  - `_calculate_spec_decoding_metrics`
  - `_request_has_grammar`

### Phase 4: Control Plane Foundation

Goal: use the Phase 2 control transport to port low-risk control endpoints and
their response aggregation behavior.

Deliverables:

- DP response aggregation and `(success, message)` merge helpers.
- Type dispatcher integration for all communicator response types.
- Concrete communicator bindings for the low-risk endpoint groups below.
- Low-risk control methods:
  - `flush_cache`,
  - `clear_hicache_storage`,
  - `attach_hicache_storage`,
  - `detach_hicache_storage`,
  - `release_memory_occupation`,
  - `resume_memory_occupation`,
  - `get_internal_state`,
  - `set_internal_state`,
  - `dumper_control`,
  - `get_loads`,
  - `check_weights`,
  - `slow_down`,
  - `configure_logging`,
  - `freeze_gc`.
- Profile methods:
  - `start_profile`,
  - `stop_profile`,
  - `_execute_profile`.
- Expert-distribution methods:
  - `start_expert_distribution_record`,
  - `stop_expert_distribution_record`,
  - `dump_expert_distribution_record`.
- Pause/continue:
  - local pause state,
  - condition/wakeup behavior,
  - abort-mode pause behavior.
- Session lifecycle:
  - `open_session`,
  - `close_session`,
  - `session_futures`,
  - `OpenSessionReqOutput` handling.

Exit criteria:

- Control methods callable from current FastAPI routes and `Engine`.
- DP fan-out/fan-in works with mocked and real scheduler responses through the
  Phase 2 `FanOutCommunicator`.
- Pause/continue/session behavior matches Python.
- Profile and expert-distribution requests round-trip through
  `FanOutCommunicator` and surface scheduler errors like Python.

Python source mapping:

- `tokenizer_control_mixin.py`
  - `flush_cache`
  - `clear_hicache_storage`
  - `attach_hicache_storage`
  - `detach_hicache_storage`
  - `release_memory_occupation`
  - `resume_memory_occupation`
  - `get_internal_state`
  - `set_internal_state`
  - `dumper_control`
  - `get_loads`
  - `check_weights`
  - `slow_down`
  - `start_profile`
  - `stop_profile`
  - `_execute_profile`
  - `start_expert_distribution_record`
  - `stop_expert_distribution_record`
  - `dump_expert_distribution_record`
  - `open_session`
  - `close_session`
- `tokenizer_manager.py`
  - `pause_generation`
  - `continue_generation`
  - `configure_logging`
  - `freeze_gc`
  - `_handle_open_session_req_output`
  - `update_active_ranks`
- `io_struct.py`
  - corresponding control request/output structs

### Phase 5: Weight Updates, LoRA, and External Corpus

Goal: port control paths that mutate model/runtime state or require local
reference tracking.

Deliverables:

- Weight update methods:
  - `init_weights_update_group`,
  - `destroy_weights_update_group`,
  - `update_weights_from_distributed`,
  - `init_weights_send_group_for_remote_instance`,
  - `send_weights_to_remote_instance`,
  - `update_weights_from_tensor`,
  - `update_weights_from_ipc`,
  - `update_weights_from_disk`,
  - `get_weights_by_name`.
- Model update state:
  - `initial_weights_loaded`,
  - `model_update_lock`,
  - `model_update_result`,
  - `model_update_tmp`,
  - local model path/load format updates,
  - weight version updates.
- LoRA registry:
  - `LoRARef`,
  - load from initial server args,
  - dynamic register/unregister,
  - active request acquire/release,
  - LRU eviction excluding pinned adapters,
  - implicit reload of evicted adapters,
  - wait-for-unload behavior.
- LoRA public methods:
  - `load_lora_adapter`,
  - `load_lora_adapter_from_tensors`,
  - `unload_lora_adapter`.
- Inference-time LoRA:
  - `_validate_and_resolve_lora`,
  - `_resolve_lora_path`,
  - release on finish/abort.
- External corpus for n-gram speculative decoding:
  - `add_external_corpus`,
  - `remove_external_corpus`,
  - `list_external_corpora`.

Exit criteria:

- Weight update endpoint parity.
- LoRA lifecycle parity including eviction and active-request waits.
- Requests with LoRA paths acquire and release references correctly.
- External corpus control tests pass.

Python source mapping:

- `tokenizer_manager.py`
  - `init_weight_update`
  - `init_lora`
  - `update_weights_from_disk`
  - `_wait_for_model_update_from_disk`
  - `_update_model_path_info`
  - `_handle_update_weights_from_disk_req_output`
  - `_validate_and_resolve_lora`
  - `_resolve_lora_path`
- `tokenizer_control_mixin.py`
  - `init_weights_update_group`
  - `destroy_weights_update_group`
  - `update_weights_from_distributed`
  - `init_weights_send_group_for_remote_instance`
  - `send_weights_to_remote_instance`
  - `update_weights_from_tensor`
  - `update_weights_from_ipc`
  - `get_weights_by_name`
  - `load_lora_adapter`
  - `load_lora_adapter_from_tensors`
  - `unload_lora_adapter`
  - `_unload_lora_adapter_locked`
  - `_update_weight_version_if_provided`
  - `add_external_corpus`
  - `remove_external_corpus`
  - `list_external_corpora`
- `io_struct.py`
  - all weight, LoRA, and external corpus request/output structs

### Phase 6: Scoring and Rerank

Goal: port `TokenizerManagerScoreMixin` after generate, embedding, and logprobs
are already stable.

Deliverables:

- `score_request`.
- `score_prompts`.
- Multi-item token sequence builder.
- Query/item batch tokenization.
- Single-item and multi-item result processing.
- Token ID input builders.
- Embedding override resolution for scoring requests.
- Logprob-to-score conversion.
- Delimiter index handling.

Exit criteria:

- classify/score/rerank outputs match Python fixtures.
- Multi-item scoring parity.
- Existing scoring/rerank tests pass with the Rust manager.

Python source mapping:

- `tokenizer_manager_score_mixin.py`
  - `score_prompts`
  - `_build_multi_item_token_sequence`
  - `_batch_tokenize_query_and_items`
  - `_process_multi_item_scoring_results`
  - `_process_single_item_scoring_results`
  - `_resolve_overrides_for_sequence`
  - `_resolve_embed_overrides_for_request`
  - `_build_token_id_inputs`
  - `score_request`
  - `_convert_logprobs_to_scores`
  - `_extract_logprobs_for_tokens`
- `tokenizer_manager.py`
  - shared tokenization and embedding paths from earlier phases
- `io_struct.py`
  - scoring uses `GenerateReqInput`, `EmbeddingReqInput`, tokenized request
    structs, and logprob output fields

### Phase 7: Multimodal and Disaggregation

Goal: cover the multimodal and encoder/PD-disaggregation paths without changing
Scheduler expectations.

Deliverables:

- Multimodal input normalization:
  - image/video/audio list normalization,
  - modality count limits,
  - tiling controls,
  - `use_audio_in_video`.
- Multimodal processor bridge or Rust processor interface.
- `mm_processor.process_mm_data_async` equivalent or PyO3 bridge.
- `mm_receiver.recv_mm_data` bridge for language-only mode.
- `mm_hashes` application.
- `SGLANG_MM_PRECOMPUTE_HASH` behavior.
- shared-memory feature wrapping compatible with `wrap_shm_features`.
- `mm_data_mooncake`.
- `encoder_urls` snapshot behavior.
- encoder-disaggregation request dispatch:
  - `_should_dispatch_to_encoder`,
  - `_handle_epd_disaggregation_encode_request`.
- PD disaggregation bootstrap fields:
  - `bootstrap_host`,
  - `bootstrap_port`,
  - `bootstrap_room`,
  - `bootstrap_pair_key`,
  - fake bootstrap room counter.

Exit criteria:

- Image/video/audio tokenized fixtures match Python.
- Scheduler receives compatible `mm_inputs` and shared-memory handles.
- Language-only encoder-disaggregation paths continue to work.
- PD disaggregation fields match Python behavior.

Python source mapping:

- `tokenizer_manager.py`
  - `init_tokenizer_and_processor` multimodal processor branch
  - `init_disaggregation`
  - `_tokenize_one_request` multimodal branch
  - `_validate_mm_limits`
  - `_should_dispatch_to_encoder`
  - `_handle_epd_disaggregation_encode_request`
  - `_determine_tensor_transport_mode`
  - `_get_processor_wrapper`
- `io_struct.py`
  - multimodal fields on `GenerateReqInput`, `EmbeddingReqInput`,
    `TokenizedGenerateReqInput`, `TokenizedEmbeddingReqInput`
  - `mm_data_mooncake`
  - `encoder_urls`
  - `num_items_assigned`
- `managers/mm_utils.py`
  - `wrap_shm_features`
  - shared-memory feature helpers

### Phase 8a: Observability and Operator-Visible Telemetry

Goal: make normal successful and failed requests visible the same way they are
in Python.

Deliverables:

- Request logging:
  - received request logging,
  - finished request logging,
  - configurable logging.
- Request metrics exporter.
- Tokenizer metrics collector.
- Time stats:
  - created,
  - tokenize finish,
  - dispatch start/finish,
  - first token,
  - finished,
  - response sent.
- Tracing:
  - external trace header extraction,
  - root span attrs,
  - `convert_to_span_attrs`.
- Request dumping for ordinary debugging flows.
- Load snapshots exposed through the already-ported control path.

Exit criteria:

- Logs/metrics/traces used by tests and production dashboards remain compatible.
- Time stats in tokenized requests and output batches match Python semantics.
- Request dump outputs match Python for running and finished requests.
- Operational endpoint outputs that depend on loads/dumper state match Python.

Python source mapping:

- `tokenizer_manager.py`
  - `init_request_logging_and_dumping`
  - `init_metric_collector_watchdog`
  - `collect_metrics`
  - `dump_requests`
  - `_dump_data_to_file`
  - `convert_to_span_attrs`
- `tokenizer_control_mixin.py`
  - `get_loads`
  - `dumper_control`
- `io_struct.py`
  - time stats wrapped in tokenized requests and output batches

### Phase 8b: Crash Dumps, Shutdown, and Signal Handling

Goal: preserve Python behavior during failures and process lifecycle events.

Deliverables:

- Crash dumping:
  - finished request ring,
  - unfinished request state dump,
  - `ReqState.get_crash_dump_output`,
  - single-shot crash dump guard.
- Watchdogs:
  - soft watchdog,
  - SIGTERM watchdog.
- Signal handlers:
  - `sigterm_handler`,
  - `running_phase_sigquit_handler`,
  - launch-phase versus running-phase behavior.
- Graceful exit status and forced-exit fallback.

Exit criteria:

- Crash dumps include finished requests and active unfinished requests with the
  same partial-output shape as Python.
- SIGTERM/SIGQUIT handlers preserve Python launch/running phase behavior.
- Watchdog-triggered dumps and exits are deterministic in tests.
- Existing lifecycle and shutdown regression tests pass with Rust manager
  enabled.

Python source mapping:

- `tokenizer_manager.py`
  - `record_request_for_crash_dump`
  - `dump_requests_before_crash`
  - `sigterm_watchdog`
  - `force_exit_handler`
  - `ReqState.get_crash_dump_output`
  - `SignalHandler`
  - `SignalHandler.sigterm_handler`
  - `SignalHandler.running_phase_sigquit_handler`

### Phase 8c: Backpressure, Cancellation, and Leak Stress Testing

Goal: prove the Rust manager stays correct under load, cancellation, queue
pressure, and error races.

Deliverables:

- Backpressure behavior for scheduler send queues.
- Backpressure behavior for response queues and per-request wakeups.
- Cancellation cleanup for:
  - client disconnect,
  - explicit abort,
  - scheduler error,
  - detokenizer error,
  - validation error after partial state creation.
- No-leak guarantees for `rid_to_state`, LoRA leases, pending communicator
  responses, and session futures.
- Stress tests for high-concurrency streaming, abort races, and slow consumer
  behavior.

Exit criteria:

- Stress tests show no request-state leaks after aborts, disconnects, and errors.
- Pending communicator responses, session futures, and LoRA leases are released
  on all cancellation paths.
- Queue pressure produces bounded memory growth and Python-compatible error or
  waiting behavior.
- Existing endpoint regression suite passes with Rust manager enabled under
  repeated runs.

Python source mapping:

- `tokenizer_manager.py`
  - `_discard_pending_req_states`
  - `abort_request`
  - `_handle_abort_req`
  - `_wait_one_response`
  - `_handle_batch_output`
- `tokenizer_control_mixin.py`
  - communicator timeout/cancellation paths
- `io_struct.py`
  - abort and control response structs involved in cancellation cleanup

## Exit Criteria Validation Matrix

The phase exit criteria are valid if they can be checked automatically. Use this
matrix as the minimum acceptance gate for each phase.

| Phase | Required validation |
| --- | --- |
| Phase 0 | A generated manifest lists every public method from `tokenizer_manager.py`, `tokenizer_control_mixin.py`, and `tokenizer_manager_score_mixin.py`, every Scheduler-facing `io_struct.py` struct, and the phase assigned to each item. A pinned msgspec schema snapshot records tag, `array_like`, base structs, flattened field order, defaults, and source lines. Fixture generation passes without a model for schema-only cases, including at least one `PickleWrapper` fixture. |
| Phase 1 | Rust/Python round-trip tests pass for every `msgspec.Struct` in the schema coverage groups. Rust schema declarations are generated from, or mechanically checked against, the Phase 0 snapshot. A field-order snapshot test fails if `io_struct.py` changes without updating Rust. `PickleWrapper` fixtures round-trip. Constructor smoke test creates and shuts down the Rust manager. |
| Phase 2 | Real Scheduler integration passes for single text, single `input_ids`, streaming, non-streaming, and explicit abort. `FanOutCommunicator` mocked single-DP and multi-DP tests pass. `rid_to_state` leak tests pass for success, validation error, disconnect, and abort echo race. |
| Phase 3 | Golden tests pass for batch generate, parallel sampling, embedding, full sampling params, dynamic-batch tokenizer enabled mode, logprobs, prompt token IDs, hidden states/routed experts/indexer metadata, and speculative metrics. Existing Python endpoint tests for those features pass with Rust manager enabled. |
| Phase 4 | Mocked DP fan-out tests and real single-DP tests pass for low-risk control methods, pause/continue, profile, expert distribution, sessions, internal state, load snapshots, and dumper control using the Phase 2 `FanOutCommunicator`. Error propagation matches Python response shapes. |
| Phase 5 | Weight update tests cover disk/tensor/IPC/distributed/group/remote-send paths. LoRA tests cover initial load, dynamic load, tensor load, unload, active request lease, LRU eviction, implicit reload, and failure paths. External corpus tests cover document and file inputs. |
| Phase 6 | Score/classify/rerank golden tests pass for single-item, multi-item, token-ID input, embedding override, delimiter, and logprob-to-score conversion paths. |
| Phase 7 | Multimodal golden tests pass for image, multi-image, video, audio, hashes, shared-memory wrapping, local processing, language-only encoder receive, encoder dispatch, Mooncake metadata, and PD bootstrap fields. |
| Phase 8a | Operational regression tests pass for request logging, metrics, tracing, request dumps, load snapshots, and time-stat parity. |
| Phase 8b | Failure-mode tests pass for crash dumps, watchdogs, SIGTERM/SIGQUIT, launch/running phase behavior, graceful exit, and forced-exit fallback. |
| Phase 8c | Stress tests pass for backpressure, cancellation, slow consumers, abort races, and no-leak behavior for `rid_to_state`, communicator pending tables, session futures, and LoRA leases. |

If a phase cannot satisfy its validation row, it should not be considered
complete even if the visible happy-path endpoint works.

## Test Plan

### Unit Tests

- request normalization.
- sampling params.
- input format detection.
- tokenization strategy selection.
- validation.
- LoRA registry.
- fanout communicator.
- msgpack codec field order against the pinned Phase 0 schema snapshot.
- `PickleWrapper` opaque payload round-trip.
- response coalescing.
- abort state transitions.

### Golden Tests

For each fixture, compare Python and Rust at these checkpoints:

```text
API input
  -> normalized request fields
  -> tokenized request fields
  -> msgpack bytes or decoded msgpack values
  -> response chunks
```

Use decoded msgpack values instead of raw bytes where Python/Rust map ordering
can differ, but use raw byte tests for tagged-array structs. Any fixture that
contains an opaque Python field must also verify that the `PickleWrapper` bytes
survive Rust round-trip unchanged.

### Integration Tests

- Rust TokenizerManager + current Python Scheduler + current DetokenizerManager.
- Single request.
- Streaming request.
- Batch request.
- Abort before scheduling.
- Abort while running.
- Control request through `FanOutCommunicator`.

### Regression Tests

Run selected existing tests with Rust TokenizerManager enabled:

- core SRT endpoint tests.
- OpenAI route tests that go through `generate_request`.
- embedding/rerank tests.
- LoRA tests.
- session tests.
- multimodal tests.
- control endpoint tests.

## Key Risks

### msgspec Compatibility

`msgspec.Struct(tag=True, array_like=True)` is order-sensitive. Any mismatch in
Rust field order can silently corrupt requests.

Mitigation:

- Generate or validate Rust schema from the Phase 0 pinned snapshot.
- Add field-order tests for every IPC struct.
- Fail build or CI when `io_struct.py` field order changes without a matching
  snapshot and Rust schema update.

### Python Object Fields

Some fields are arbitrary Python objects today: multimodal processor outputs,
time stats, tensors, custom info, and positional embedding overrides.

Mitigation:

- Keep `PickleWrapper` compatibility first.
- Add at least one Phase 0 fixture with a pickle-wrapped field.
- Port each opaque field only after golden tests exist.

### Async Generator Semantics

Python `generate_request` is an async generator. Rust must present a compatible
interface to existing FastAPI and Engine call sites.

Mitigation:

- Start with a PyO3 wrapper that returns a Python async iterator backed by Rust
  channels.
- Later, expose native Rust streams for Rust HTTP paths.

### Multimodal Processor Surface

Multimodal processors are model-specific and Python-heavy.

Mitigation:

- Bridge to Python first.
- Keep text-only and token-ID-only paths fully Rust.

### Control Plane Breadth

TokenizerManager owns many operational endpoints beyond generation.

Mitigation:

- Implement `FanOutCommunicator` infrastructure in Phase 2.
- Port control methods by category with mocked Scheduler responses.

### Observability Parity

Users depend on logs, metrics, dumps, and trace metadata.

Mitigation:

- Treat observability fields as part of response/schema parity, not optional
  polish.

## Deliverables Checklist

- [ ] P0: method coverage manifest.
- [ ] P0: pinned msgspec schema snapshot from `io_struct.py`.
- [ ] P0: golden fixture generator.
- [ ] P0: `PickleWrapper` fixture.
- [ ] P0: Python/Rust comparison harness.
- [ ] P1: `rust/sglang-server` module structure.
- [ ] P1: PyO3 `TokenizerManager` constructor.
- [ ] P1: typed `ServerArgsView`, `ModelConfigView`, `PortArgsView`.
- [ ] P1: msgspec-compatible schema codec generated from or checked against the
  pinned snapshot.
- [ ] P1: `PickleWrapper` opaque payload compatibility.
- [ ] P1: ZMQ scheduler dispatch and DetokenizerManager receive primitives.
- [ ] P1: dispatcher, pending-response, `ReqState`, and no-op observability
  skeletons.
- [ ] P2: single `GenerateReqInput` Rust model.
- [ ] P2: single-request normalization.
- [ ] P2: text tokenizer and `input_ids` passthrough.
- [ ] P2: basic sampling params.
- [ ] P2: tokenized scheduler request construction.
- [ ] P2: single-request response waiting.
- [ ] P2: streaming/non-streaming response assembly.
- [ ] P2: abort and disconnect flow.
- [ ] P2: `FanOutCommunicator` infrastructure.
- [ ] P3: batch generation and embedding normalization.
- [ ] P3: batch tokenization and dynamic-batch tokenizer parity.
- [ ] P3: full sampling params parity.
- [ ] P3: logprob, hidden-state, routed-expert, indexer, and speculative metadata
  parity.
- [ ] P4: full low-risk control-plane methods.
- [ ] P4: pause/continue behavior.
- [ ] P4: session lifecycle.
- [ ] P4: profile, expert distribution, internal state, load, dumper, cache, and
  memory controls.
- [ ] P5: weight update methods.
- [ ] P5: LoRA registry and lifecycle.
- [ ] P5: external corpus methods.
- [ ] P6: score/rerank behavior.
- [ ] P7: multimodal bridge/port.
- [ ] P7: encoder and PD disaggregation behavior.
- [ ] P8a: request logging, metrics, tracing, request dumps, load snapshots, and
  time-stat parity.
- [ ] P8b: crash dumps, watchdogs, signal handlers, graceful exit, and forced
  exit.
- [ ] P8c: backpressure, cancellation, leak, and stress tests.
- [ ] All phases: existing endpoint regression suite with Rust manager enabled
  for the completed surface.

## Suggested First Implementation Slice

Even though the final target is complete parity, start with the smallest slice
that exercises the real Scheduler boundary:

```text
single /generate
  text or input_ids
  basic SamplingParams
  TokenizedGenerateReqInput msgpack parity
  ZMQ dispatch to current Scheduler
  BatchStrOutput response handling
  non-streaming and streaming response assembly
  abort on disconnect
```

This slice validates the hardest boundary: current Python request semantics
converted into current Scheduler inputs and current DetokenizerManager outputs
converted back into current API responses.
