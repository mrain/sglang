# TokenizerManager

This document describes the current Python `TokenizerManager` implementation.

## Role

`TokenizerManager` is the frontend runtime manager that sits between the HTTP /
Engine API layer and the Scheduler. It is not only a tokenizer wrapper. In the
current Python server it owns:

- Request admission and request ID state.
- API-facing request normalization.
- Text tokenization and multimodal preprocessing dispatch.
- LoRA request validation and reference tracking.
- Scheduler request construction and ZMQ dispatch.
- Response fan-out from DetokenizerManager back to the waiting HTTP/Engine call.
- Abort, pause, sessions, weight updates, cache, profile, and admin control
  messages.
- Request logging, metrics, tracing, request dumping, crash dumping, and
  watchdog behavior.

High-level process placement:

```text
HTTP server / Engine API / TokenizerManager
  run in the main Python process

Scheduler subprocess(es)
  receive tokenized requests and control messages

DetokenizerManager subprocess(es)
  receive scheduler token outputs, detokenize text, and send frontend outputs
  back to TokenizerManager
```

## Main Modules

### `python/sglang/srt/managers/tokenizer_manager.py`

Core request lifecycle:

- Builds `ModelConfig`, tokenizer, optional multimodal processor, IPC sockets,
  state maps, logging, metrics, disaggregation helpers, and dispatch tables.
- Implements `generate_request`, tokenization, validation, scheduler dispatch,
  output handling, abort handling, and response generators.
- Defines `ReqState`, the in-memory accumulator for one active request.

Important class:

```python
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin)
```

### `python/sglang/srt/managers/tokenizer_control_mixin.py`

Control-plane methods mixed into `TokenizerManager`.

It creates `FanOutCommunicator` instances for scheduler operations that need
responses from one or more DP ranks. These cover weight updates, cache/memory
controls, LoRA loading, profiling, internal state, expert distribution, HiCache,
external corpus management, and request dumper control.

### `python/sglang/srt/managers/tokenizer_manager_score_mixin.py`

Scoring and reward-model helpers mixed into `TokenizerManager`.

It builds generation or embedding inputs for scoring/reranking-style APIs, uses
the normal tokenization/generation machinery, then post-processes logprobs or
model outputs into scores.

### `python/sglang/srt/managers/io_struct.py`

IPC and API data container definitions.

Important conventions:

- Scheduler IPC structs use `msgspec.Struct`.
- `BaseReq` and `BaseBatchReq` use `tag=True` and `array_like=True`.
- Opaque Python fields are wrapped with `PickleWrapper` before msgpack IPC.
- `GenerateReqInput` and `EmbeddingReqInput` also contain request normalization
  logic used before tokenization.

### Adjacent Modules

- `entrypoints/http_server.py`: FastAPI routes call `TokenizerManager` methods.
- `entrypoints/engine.py`: Python `Engine` API calls the same methods directly.
- `managers/detokenizer_manager.py`: produces `BatchStrOutput` from
  `BatchTokenIDOutput`, then sends frontend-ready batches to `TokenizerManager`.
- `managers/scheduler_components/request_receiver.py`: Scheduler-side receiver
  that consumes tokenized requests and control messages from TokenizerManager.
- `managers/mm_utils.py`: shared-memory wrapping for multimodal features.
- `utils/request_logger.py`, metrics, tracing, and load-snapshot modules:
  observability support used by TokenizerManager.

## Startup Flow

```text
launch_server()
  -> Engine._launch_subprocesses()
     -> start Scheduler subprocess(es)
     -> start DetokenizerManager subprocess(es)
     -> create TokenizerManager in the main process
     -> initialize TemplateManager
     -> wait for scheduler readiness
     -> set tokenizer_manager.max_req_input_len
  -> start FastAPI HTTP server
     -> route handlers call TokenizerManager
```

`TokenizerManager.__init__` is split into explicit initialization phases:

1. `init_model_config`
   - Stores model path/name, `ModelConfig`, `context_len`, `image_token_id`,
     priority scheduling settings, speculative reserved-token count, and total
     token validation flag.
   - Initializes `max_req_input_len` to `None`; `Engine` fills it after the
     Scheduler reports capacity.
2. `init_tokenizer_and_processor`
   - Creates tokenizer unless `skip_tokenizer_init` is set.
   - Creates multimodal processor for multimodal models.
   - Optionally creates `AsyncDynamicbatchTokenizer`.
3. `init_ipc_channels`
   - Creates ZMQ PULL from DetokenizerManager and ZMQ PUSH to Scheduler.
   - In multi-tokenizer mode, prepares per-worker routing metadata.
   - Creates load-snapshot reader.
4. `init_running_status`
   - Creates request state map, event-loop task set, server status, session
     future map, and subprocess watchdog slot.
5. `init_request_logging_and_dumping`
   - Creates `RequestLogger`, dump settings, crash dump buffers, and request
     metrics exporter.
6. `init_weight_update`
   - Creates model update lock, update result future slot, pause flag, and pause
     condition variable.
7. `init_lora`
   - Creates `LoRARegistry`, LoRA update lock, and LoRA ref cache.
8. `init_disaggregation`
   - Starts PD disaggregation bootstrap service.
   - For language-only encoder disaggregation, creates encoder URL registry,
     encoder bootstrap server, and multimodal receiver.
9. `init_metric_collector_watchdog`
   - Creates tokenizer metrics collector and soft watchdog when enabled.
10. `init_request_dispatcher`
   - Creates type-based response dispatcher and control-plane communicators.

## IPC Channels

```text
TokenizerManager --ZMQ PUSH / msgpack or pickle--> Scheduler
TokenizerManager <--ZMQ PULL / msgpack or pickle-- DetokenizerManager
```

Main socket attributes:

- `send_to_scheduler`: sends tokenized inference requests and control messages.
- `recv_from_detokenizer`: receives `BatchStrOutput`, `BatchTokenIDOutput`,
  `BatchEmbeddingOutput`, and control outputs.

In multi-tokenizer mode:

- Each tokenizer worker uses a worker IPC endpoint.
- `_dispatch_to_scheduler` stamps outgoing objects with `http_worker_ipc`.
- Scheduler/Detokenizer can route outputs back to the owning tokenizer worker.

## Out-Facing Interfaces

The public surface is called by FastAPI routes, the Python `Engine`, OpenAI
protocol adapters, and internal helper APIs.

### Inference

- `generate_request(obj, request=None)`
  - Primary request entry.
  - Accepts `GenerateReqInput` or `EmbeddingReqInput`.
  - Normalizes input, initializes `ReqState`, tokenizes, sends to Scheduler, and
    yields response chunks or a final response.

- `score_request(obj, request=None)`
  - Scoring/rerank entry from `TokenizerManagerScoreMixin`.
  - Builds internal generation/embedding requests and reuses `generate_request`.

- `score_prompts(...)`
  - Lower-level scoring helper that prepares prompt/token sequences and
    post-processes logprobs into scores.

### Abort and Client Disconnect

- `abort_request(rid="", abort_all=False)`
  - Sends `AbortReq` to Scheduler.
  - Local state is finalized when the Scheduler echoes an `AbortReq` back.

- `create_abort_task(obj)`
  - Returns FastAPI `BackgroundTasks` for streaming disconnect handling.

### Pause and Resume

- `pause_generation(obj)`
  - Sets local pause state.
  - Either sends pause request to Scheduler or aborts all requests until no model
    update lock is held, depending on mode.

- `continue_generation(obj)`
  - Clears pause state, sends continue request, and wakes paused request
    admission.

### Weight Updates

- `update_weights_from_disk(obj, request=None)`
- `update_weights_from_tensor(obj, request=None)`
- `update_weights_from_ipc(obj, request=None)`
- `update_weights_from_distributed(obj, request=None)`
- `init_weights_update_group(obj, request=None)`
- `destroy_weights_update_group(obj, request=None)`
- `init_weights_send_group_for_remote_instance(obj, request=None)`
- `send_weights_to_remote_instance(obj, request=None)`
- `get_weights_by_name(obj, request=None)`

These methods coordinate with `model_update_lock`, optional abort-all behavior,
and `FanOutCommunicator` response collection.

### LoRA

- `load_lora_adapter(obj, request=None)`
- `load_lora_adapter_from_tensors(obj, request=None)`
- `unload_lora_adapter(obj, request=None)`

These validate LoRA support, serialize updates with `lora_update_lock`, update
Scheduler workers, maintain `LoRARegistry`, and update `lora_ref_cache`.

Inference-time LoRA resolution is handled by `_validate_and_resolve_lora` and
`_resolve_lora_path`.

### Sessions

- `open_session(obj, request=None)`
  - Generates session ID if absent.
  - Stores a future in `session_futures`.
  - Dispatches `OpenSessionReqInput`.
  - Completes when `OpenSessionReqOutput` returns.

- `close_session(obj, request=None)`
  - Dispatches `CloseSessionReqInput` asynchronously.

### Cache, Memory, Profile, and Admin

- `flush_cache(timeout_s=None)`
- `clear_hicache_storage()`
- `attach_hicache_storage(...)`
- `detach_hicache_storage()`
- `release_memory_occupation(obj, request=None)`
- `resume_memory_occupation(obj, request=None)`
- `start_profile(req=None)`
- `stop_profile()`
- `start_expert_distribution_record()`
- `stop_expert_distribution_record()`
- `dump_expert_distribution_record()`
- `get_internal_state()`
- `set_internal_state(obj)`
- `dumper_control(obj)`
- `get_loads(include=None, dp_rank=None)`
- `check_weights(obj, request=None)`
- `slow_down(obj, request=None)`
- `configure_logging(obj)`
- `freeze_gc()`

### External Corpus

Used by n-gram speculative decoding:

- `add_external_corpus(obj)`
- `remove_external_corpus(corpus_id)`
- `list_external_corpora()`

`add_external_corpus` may tokenize documents or file chunks on the tokenizer
side before dispatching them to Scheduler workers.

### Background Loops and Handlers

- `auto_create_handle_loop()`
  - Lazily starts `handle_loop`, signal handlers, and SIGTERM watchdog.

- `handle_loop()`
  - Receives DetokenizerManager/Scheduler outputs and dispatches them.

- `_handle_batch_output(recv_obj)`
  - Handles `BatchStrOutput`, `BatchTokenIDOutput`, and `BatchEmbeddingOutput`.

- `_handle_abort_req(recv_obj)`
  - Handles Scheduler abort echo.

- `_handle_open_session_req_output(recv_obj)`
- `_handle_update_weights_from_disk_req_output(recv_obj)`
- `update_active_ranks(ranks)`

## Inbound Data Types

### API-Facing Inference Inputs

#### `GenerateReqInput`

Used for generation APIs.

Main fields:

- Identity: `rid`, `session_id`, `conversation_id`, `http_worker_ipc`.
- Inputs: `text`, `input_ids`, `input_embeds`.
- Multimodal: `image_data`, `video_data`, `audio_data`, `mm_hashes`,
  multimodal tiling controls.
- Sampling: `sampling_params`, `stream`, `parallel_sample_num` through
  normalization.
- Logprobs: `return_logprob`, `logprob_start_len`, `top_logprobs_num`,
  `token_ids_logprob`, `return_text_in_logprobs`.
- Optional outputs: `return_hidden_states`, `return_routed_experts`,
  `return_indexer_topk`, `return_bytes`, `return_entropy`,
  `return_prompt_token_ids`.
- LoRA: `lora_path`, `lora_id`.
- Sessions: `session_params`.
- Disaggregation/routing: bootstrap fields, `routed_dp_rank`,
  `disagg_prefill_dp_rank`, `routing_key`.
- Scheduling and cache: `priority`, `extra_key`.
- Observability: `log_metrics`, `no_logs`, `custom_labels`,
  `external_trace_header`, `received_time`.

`normalize_batch_and_arguments()` performs:

- Input exclusivity validation: exactly one of text, input IDs, or input embeds.
- Batch-size detection.
- Parallel sampling expansion.
- Single vs batch field normalization.
- Default request ID creation.
- Per-subrequest projection via `__getitem__`.
- Request ID uniqueness validation.

#### `EmbeddingReqInput`

Used for embedding/classification/rerank flows.

It overlaps with `GenerateReqInput` for `rid`, text/token/embed inputs,
multimodal inputs, LoRA, priority, and observability. It also carries embedding
specific fields such as:

- `dimensions` for Matryoshka embeddings.
- `return_pooled_hidden_states`.
- Cross-encoder request shape.
- Embedding override fields.

### API-Facing Control Inputs

Examples include:

- Abort/pause: `AbortReq`, `PauseGenerationReqInput`,
  `ContinueGenerationReqInput`.
- Weight updates: `UpdateWeightFromDiskReqInput`,
  `UpdateWeightsFromTensorReqInput`, `UpdateWeightsFromIPCReqInput`,
  `UpdateWeightsFromDistributedReqInput`, group init/destroy/send requests.
- LoRA: `LoadLoRAAdapterReqInput`, `LoadLoRAAdapterFromTensorsReqInput`,
  `UnloadLoRAAdapterReqInput`.
- Cache/memory/profile: `FlushCacheReqInput`, HiCache requests,
  `ProfileReq`, memory occupation requests.
- Sessions: `OpenSessionReqInput`, `CloseSessionReqInput`.
- Admin/debug: internal state requests, dumper control, check weights, slow
  down, expert distribution, external corpus requests.

### Detokenizer/Scheduler Outputs Received by TokenizerManager

Normal data-plane outputs:

- `BatchStrOutput`
  - Produced after DetokenizerManager decodes token IDs into text.
  - Carries `rids`, `finished_reasons`, `output_strs`, `output_ids`, token
    counts, logprob arrays, hidden states, routed experts, indexer results,
    placeholder metadata, cache details, DP ranks, time stats, multimodal token
    counts, and speculative metrics.

- `BatchTokenIDOutput`
  - Used when Scheduler bypasses DetokenizerManager or `skip_tokenizer_init`
    requires token IDs rather than text.
  - Carries decode IDs, output IDs, detokenization config, token counts,
    logprobs, hidden states, routed experts, cache details, DP ranks, time
    stats, multimodal token counts, and speculative metrics.

- `BatchEmbeddingOutput`
  - Used for embedding outputs.
  - Carries embeddings, token/cache counts, placeholder metadata, time stats,
    and optional pooled hidden states.

Control outputs handled by `TypeBasedDispatcher`:

- `AbortReq`
- `OpenSessionReqOutput`
- `UpdateWeightFromDiskReqOutput`
- `FreezeGCReq`
- `HealthCheckOutput`
- `ActiveRanksOutput`
- Communicator-registered outputs for weights, cache, LoRA, profile, internal
  state, HiCache, expert distribution, external corpus, loads, and dumper
  control.

## Outbound Data Types

### Scheduler Inference Requests

#### `TokenizedGenerateReqInput`

Sent to Scheduler after request normalization, tokenization, sampling param
normalization, validation, optional multimodal preprocessing, and optional LoRA
resolution.

Key fields:

- Base fields: `rid`, `http_worker_ipc`.
- Inputs: `input_text`, `input_ids` as `array("q")`, optional `input_embeds`.
- Multimodal: `mm_inputs`, `token_type_ids`, `mm_data_mooncake`,
  `encoder_urls`, `need_wait_for_mm_inputs`, `num_items_assigned`.
- Sampling: `sampling_params: SamplingParams`.
- Output controls: logprob fields, `stream`, hidden states, routed experts,
  indexer output, entropy/bytes flags.
- Sessions: `session_id`, `session_params`.
- LoRA: `lora_id`.
- Custom behavior: `custom_logit_processor`, positional embedding overrides,
  routing/disaggregation fields, `priority`, `extra_key`, `routing_key`,
  `require_reasoning`, `multi_item_delimiter_indices`.
- Observability: `time_stats`.

Before send:

- `wrap_shm_features()` may convert multimodal tensors/features to shared
  memory transport.
- `wrap_pickle_fields()` wraps opaque fields such as `mm_inputs`,
  `mm_data_mooncake`, and `time_stats`.

#### `TokenizedEmbeddingReqInput`

Sent to Scheduler for embedding/classification/rerank requests.

Key fields:

- `rid`, `http_worker_ipc`.
- `input_text`, `input_ids`, `mm_inputs`, `token_type_ids`.
- Dummy/compatible `sampling_params`.
- `lora_id`, positional embedding overrides.
- `routed_dp_rank`, `priority`.
- `dimensions`, `return_pooled_hidden_states`.
- `multi_item_delimiter_indices`.
- `time_stats`.

#### Batched Variants

- `BatchTokenizedGenerateReqInput`
- `BatchTokenizedEmbeddingReqInput`

These wrap a list of tokenized requests and are used when batch tokenization or
pre-tokenized batch dispatch is enabled.

### Scheduler Control Requests

Control methods send request structs such as:

- `AbortReq`
- `FlushCacheReqInput`
- HiCache request inputs.
- Weight update inputs.
- LoRA update inputs.
- Session open/close inputs.
- Profile/internal-state/debug/admin request inputs.

Most control-plane operations use `FanOutCommunicator`, which sends one request
and waits for all expected DP-rank outputs.

## Internally Kept State

### Configuration and Model State

- `server_args`
- `model_path`
- `served_model_name`
- `model_config`
- `is_generation`
- `context_len`
- `image_token_id`
- `max_req_input_len`
- `enable_priority_scheduling`
- `default_priority_value`
- `num_reserved_tokens`
- `validate_total_tokens`
- `preferred_sampling_params`

### Tokenizer and Multimodal State

- `tokenizer`
- `processor`
- `mm_processor`
- `async_dynamic_batch_tokenizer`
- `mm_receiver`
- `encoder_urls`
- `encoder_bootstrap_server`

### IPC State

- `recv_from_detokenizer`
- `send_to_scheduler`
- `tokenizer_ipc_name`
- `load_snapshot_reader`

### Running Status

- `rid_to_state: Dict[str, ReqState]`
- `event_loop`
- `asyncio_tasks`
- `server_status`
- `gracefully_exit`
- `last_receive_tstamp`
- `_subprocess_watchdog`

### Session State

- `session_futures: Dict[str, Future]`

### Logging, Metrics, and Dumping

- `request_logger`
- `dump_requests_folder`
- `dump_requests_threshold`
- `dump_requests_exclude_meta_keys`
- `dump_request_list`
- `crash_dump_request_list`
- `crash_dump_performed`
- `request_metrics_exporter_manager`
- `metrics_collector`
- `soft_watchdog`

### Weight Update and Pause State

- `initial_weights_loaded`
- `model_update_lock: RWLock`
- `model_update_result`
- `model_update_tmp`
- `is_pause`
- `is_pause_cond`

Inference acquires the reader side of `model_update_lock`; weight updates use
the writer side unless generation is already paused.

### LoRA State

- `lora_registry`
- `lora_update_lock`
- `lora_ref_cache`

`lora_registry` tracks registered adapters and active references. A request
acquires a LoRA ID before dispatch and releases it when the request finishes.

### Disaggregation State

- `disaggregation_mode`
- `bootstrap_server`
- `fake_bootstrap_room_counter`
- Language-only encoder disaggregation helpers when enabled.

### Dispatch State

- `_result_dispatcher: TypeBasedDispatcher`
- One communicator attribute per `_COMMUNICATOR_SPECS` entry, for example
  `flush_cache_communicator`, `profile_communicator`,
  `update_lora_adapter_communicator`, and `get_internal_state_communicator`.

## `ReqState`

One `ReqState` exists per active request ID in `rid_to_state`.

Fields:

- `out_list`: queued response dictionaries ready for `_wait_one_response`.
- `finished`: whether a terminal output has arrived.
- `event`: `asyncio.Event` used to wake the response generator.
- `obj`: the per-request `GenerateReqInput` or `EmbeddingReqInput`.
- `time_stats`: `APIServerReqTimeStats`.
- `last_completion_tokens`, `ttft_observed`.
- `last_output_offset`: cursor for incremental streaming metadata slicing.
- `text`, `text_chunks`: lazy output text accumulation.
- `output_ids`: cumulative output token IDs.
- Raw logprob accumulator arrays.
- Detokenized logprob accumulator arrays.
- `customized_info_accumulated`.
- `prompt_token_ids`: stored when `return_prompt_token_ids` is requested.

Text accumulation is lazy:

```text
_handle_batch_output appends text deltas to text_chunks
_wait_one_response calls get_text() only when a full prefix is needed
```

This avoids repeated full-string rebuilds during streaming.

## Generate Request Dataflow

```text
GenerateReqInput / EmbeddingReqInput
  -> generate_request()
  -> auto_create_handle_loop()
  -> normalize_batch_and_arguments()
  -> _set_default_priority()
  -> routed_dp_rank validation
  -> _init_req_state()
  -> optional encoder-disaggregation encode dispatch
  -> request_logger.log_received_request()
  -> wait while paused
  -> acquire model_update_lock.reader_lock
  -> _validate_and_resolve_lora()
  -> tokenize / process / validate
  -> send tokenized request(s) to Scheduler
  -> _wait_one_response()
  -> yield response chunk(s) or final result
```

Important failure behavior:

- `_init_req_state` creates all per-request states before tokenization.
- If a pre-dispatch failure occurs, `generate_request` calls
  `_discard_pending_req_states` to avoid leaking `rid_to_state` entries.

## Tokenization Flow

`_tokenize_one_request` supports three input modes:

1. `input_embeds`
   - Passed through as embeddings.
   - Requires compatible cache settings, notably `disable_radix_cache`.
2. `input_ids`
   - Passed through without tokenizer encode.
3. `text`
   - Encoded by `_tokenize_texts`.
   - Rejected when `skip_tokenizer_init=True`.

`_tokenize_texts` detects:

| Format | Example | Purpose |
| --- | --- | --- |
| `SINGLE_STRING` | `"hello"` | Normal single prompt |
| `BATCH_STRINGS` | `["a", "b"]` | Batched text encode |
| `CROSS_ENCODER_PAIRS` | `[["query", "doc"]]` | Cross-encoder segment IDs |

Tokenizer strategy:

- `AsyncDynamicbatchTokenizer` is used for single strings when enabled.
- Otherwise fast tokenizers are called in batch form.
- Non-fast tokenizers use per-string `tokenizer.encode` for non-cross-encoder
  text.

Multimodal processing:

- Normalizes image/video/audio fields to lists.
- Enforces per-request multimodal limits.
- In language-only encoder-disaggregation mode, may receive encoder-produced
  multimodal data from `mm_receiver`.
- Otherwise invokes `mm_processor.process_mm_data_async`.
- Replaces `input_ids` and `token_type_ids` when the multimodal processor
  supplies them.
- Applies caller-provided `mm_hashes` to multimodal items when valid.
- Optionally precomputes multimodal pad values/hashes.

After tokenization:

```text
_validate_one_request()
  -> context length validation
  -> max_new_tokens + prompt length validation
  -> embedding model / generation model compatibility
  -> Matryoshka dimension validation
  -> generation feature gates
  -> token_ids_logprob validation

_create_tokenized_object()
  -> build SamplingParams
  -> normalize and verify SamplingParams
  -> create TokenizedGenerateReqInput or TokenizedEmbeddingReqInput
  -> attach APIServerReqTimeStats
```

## Batch Request Flow

`_handle_batch_request` handles normalized batch inputs.

Main branches:

- If `parallel_sample_num == 1` and `_should_use_batch_tokenization` is true:
  - `_batch_tokenize_and_process` tokenizes the batch as one tokenizer call when
    possible.
  - `_send_batch_request` sends `BatchTokenizedGenerateReqInput` or
    `BatchTokenizedEmbeddingReqInput`.

- If `parallel_sample_num == 1` and batch tokenization is not used:
  - Each subrequest is tokenized and sent independently.

- If `parallel_sample_num > 1`:
  - First sends zero-new-token prefix-cache warmup requests with regenerated
    request IDs.
  - Then expands each prompt into multiple sampled requests.
  - Each sampled request gets its own `ReqState`.

Response behavior:

- Non-streaming batch requests gather one final output per generator and yield a
  list.
- Streaming batch requests race the per-request generators and yield chunks with
  an added `index` field.

## Scheduler Dispatch Flow

Single request:

```text
_send_one_request(tokenized_obj)
  -> set_api_server_dispatch_time()
  -> wrap_shm_features()
  -> wrap_pickle_fields()
  -> _dispatch_to_scheduler()
  -> restore local time_stats reference
  -> set_api_server_dispatch_finish_time()
```

Batch request:

```text
_send_batch_request(tokenized_objs)
  -> set dispatch time on all
  -> wrap_pickle_fields() on all
  -> wrap in BatchTokenizedGenerateReqInput or BatchTokenizedEmbeddingReqInput
  -> _dispatch_to_scheduler()
  -> restore local time_stats references
  -> set dispatch finish time on all
```

`_dispatch_to_scheduler` stamps `http_worker_ipc` when needed and sends the
object through `sock_send`.

## Response Handling Flow

`handle_loop` runs as a background task:

```text
while True:
  recv_obj = async_sock_recv(recv_from_detokenizer)
  if recv_obj is BatchStrOutput / BatchTokenIDOutput / BatchEmbeddingOutput:
    await _handle_batch_output(recv_obj)
  else:
    _result_dispatcher(recv_obj)
  update last_receive_tstamp
  feed watchdog
```

`_handle_batch_output` processes each item in the received batch:

1. Unwraps pickled time stats and customized info.
2. Looks up `ReqState` by `rid`.
3. Builds `meta_info` with:
   - request ID,
   - finish reason,
   - prompt/completion/reasoning/cache token counts,
   - weight version,
   - retraction count,
   - scheduler/API timing,
   - logprobs,
   - hidden states,
   - routed experts,
   - indexer output,
   - cache details,
   - DP rank,
   - multimodal token counts,
   - speculative decoding metrics.
4. Updates cumulative text/output IDs/logprob/custom info state.
5. Creates an output dict when an intermediate streamed chunk or final response
   should be exposed.
6. On first output, records first-token time.
7. On final output:
   - records finish time and E2E latency,
   - removes `rid_to_state[rid]`,
   - releases LoRA reference if used.
8. Appends output dict to `state.out_list`.
9. Wakes waiting response generators in batches controlled by
   `batch_notify_size`.
10. Collects metrics and request dumps when enabled.

Output dict shapes:

- Text generation:
  - `{"text": ..., "output_ids": ..., "meta_info": ...}`
- Token-ID-only generation:
  - `{"output_ids": ..., "meta_info": ...}`
- Embedding:
  - `{"embedding": ..., "meta_info": ...}`
- Optional:
  - `prompt_token_ids`
  - `pooled_hidden_state`

## Client-Facing Response Generator

`_wait_one_response` is the async generator consumed by HTTP/Engine callers.

```text
loop:
  wait for state.event
  drain state.out_list
  clear state.event
  if several incremental stream chunks queued:
    coalesce chunks
  if non-incremental streaming and text was deferred:
    materialize full text
  if finished:
    set response-sent timestamp
    log final request
    export request metrics if enabled
    handle scheduler abort/error finish reason
    yield final output
    break
  if streaming:
    set first response-sent timestamp if needed
    yield chunk
  else:
    check client disconnect and abort if disconnected
```

Streaming modes:

- Incremental streaming:
  - Output text and output IDs are deltas.
  - Logprob/custom metadata is sliced to match the delta.
  - Multiple queued deltas may be coalesced.

- Non-incremental streaming:
  - Intermediate chunks can carry cumulative output IDs and deferred text.
  - Full text is materialized only when needed.

## Abort Flow

```text
client disconnect / explicit abort / abort_all
  -> abort_request()
  -> send AbortReq to Scheduler
  -> Scheduler removes waiting request or marks running request aborted
  -> Scheduler echoes AbortReq back
  -> handle_loop dispatches to _handle_abort_req
  -> _handle_abort_req marks ReqState finished
  -> builds abort output with finish_reason
  -> deletes rid_to_state entry
  -> appends output and sets state.event
  -> _wait_one_response yields or raises according to stream/error status
```

Scheduler abort echoes can race with normal final outputs. `_handle_abort_req`
therefore treats missing `rid_to_state` as an already-finished request rather
than a hard error.

## Control-Plane Flow

Most control methods follow this shape:

```text
public method
  -> auto_create_handle_loop()
  -> local validation / state update
  -> communicator sends request to Scheduler
  -> handle_loop receives per-rank output
  -> TypeBasedDispatcher routes output to communicator.handle_recv
  -> public method awaits merged result
```

`FanOutCommunicator` hides DP fan-out/fan-in. Some methods merge results into a
single `(success, message)` pair; others return the first rank's output or a
list of per-rank outputs.

Exceptions:

- `abort_request` is fire-and-forget; completion is observed through the abort
  echo.
- `configure_logging` updates local logging first and only dispatches when a
  scheduler-side log-level change is required.
- `get_loads` reads load snapshots locally through `load_snapshot_reader`.
- `close_session` dispatches asynchronously without waiting for a response.

## Session Flow

```text
open_session()
  -> assign session_id if absent
  -> create Future in session_futures
  -> dispatch OpenSessionReqInput
  -> _handle_open_session_req_output completes future
  -> remove future from session_futures

generation with session fields
  -> TokenizedGenerateReqInput carries session_id/session_params
  -> Scheduler applies session KV-cache semantics

close_session()
  -> dispatch CloseSessionReqInput
```

## LoRA Flow

Loading:

```text
load_lora_adapter()
  -> validate --enable-lora and dp constraints
  -> acquire lora_update_lock
  -> create LoRARef
  -> send scheduler update through update_lora_adapter_communicator
  -> register adapter in LoRARegistry on success
  -> cache LoRARef in lora_ref_cache
  -> evict LRU non-pinned adapters if max_loaded_loras exceeded
```

Inference:

```text
generate_request()
  -> _validate_and_resolve_lora()
  -> reload previously evicted adapter if needed
  -> acquire lora_id from LoRARegistry
  -> include lora_id in tokenized request
  -> release lora_id when request finishes or aborts
```

Unloading:

```text
unload_lora_adapter()
  -> acquire lora_update_lock
  -> unregister adapter from LoRARegistry
  -> wait until active references finish
  -> send scheduler unload request
```

## Weight Update and Pause Flow

Inference waits on two gates:

1. `is_pause_cond` must observe `is_pause == False`.
2. `model_update_lock.reader_lock` must be available.

Weight updates:

```text
update_weights_from_*
  -> optionally abort all active requests
  -> if not paused, acquire model_update_lock.writer_lock
  -> dispatch scheduler update request
  -> await result through communicator or model_update_result
  -> update local model path / load format / weight_version on success
```

`pause_generation(mode="abort")` repeatedly sends `abort_all=True` until the
model update lock is free.

## Observability and Failure Handling

TokenizerManager records or exposes:

- Request received, tokenize finish, dispatch, first token, finish, and
  response-sent timestamps.
- Scheduler timing carried back in output structs.
- Request logs on receive and finish.
- Optional request metrics exporter records.
- Prometheus-style tokenizer metrics.
- Request dumps and crash dumps.
- Tracing span attributes when tracing is enabled.
- Load snapshots via `get_loads`.
- Soft watchdog status and SIGTERM watchdog status.

Crash dump support stores recent finished requests and active unfinished request
states. `ReqState.get_crash_dump_output()` exposes partial text/output IDs for
unfinished requests.

## Current Design Boundaries

TokenizerManager owns frontend request semantics and scheduler ingress/egress
coordination, but it does not:

- Run model forward passes.
- Batch GPU execution.
- Allocate KV cache.
- Decode token IDs into text in the normal non-skip path; that is done by
  DetokenizerManager.
- Decide final scheduling or prefill/decode batching; that is Scheduler logic.

Its core contract with Scheduler is the tokenized request/control struct stream
going out and the batched output/control struct stream coming back.
