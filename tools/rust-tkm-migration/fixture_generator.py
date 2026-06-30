#!/usr/bin/env python3
"""Generate golden msgpack fixtures for the Rust TokenizerManager migration.

Uses the real io_struct.py classes when possible, falls back to shadow copies.

Usage:
    python tools/rust-tkm-migration/fixture_generator.py --write
"""

import enum
import json
import pickle
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_PYTHON = _REPO / "python"
for p in (str(_REPO), str(_PYTHON)):
    if p not in sys.path:
        sys.path.insert(0, p)

import msgspec

# Try real io_struct classes first; fall back to replicas
# ── Production-accurate encoding ──
# Use array("q") + enc_hook matching io_struct.py's msgpack_encode,
# so fixture bytes match production wire format without importing the
# full SGLang environment (torch, zmq, etc).

from array import array


def _enc_hook(obj):
    """Production-matching enc_hook: handle array.array as (typecode, tobytes)."""
    if isinstance(obj, array):
        return (obj.typecode, obj.tobytes())
    raise TypeError(f"Cannot encode {type(obj)}")


_shadow_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)

# ── Try real io_struct imports for struct types ──
_using_real_imports = False
_encode_fn = _shadow_encoder.encode
try:
    from array import array
    from sglang.srt.managers.io_struct import (
        AbortReq,
        BatchEmbeddingOutput,
        BatchStrOutput,
        BatchTokenIDOutput,
        BatchTokenizedEmbeddingReqInput,
        BatchTokenizedGenerateReqInput,
        FlushCacheReqInput,
        PickleWrapper,
        ProfileReq,
        ProfileReqType,
        SessionParams,
        TokenizedEmbeddingReqInput,
        TokenizedGenerateReqInput,
        msgpack_encode as _encode_fn,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams as _RealSamplingParams
    _using_real_imports = True
except ImportError as _exc:
    if "--allow-shadow-fixtures" not in sys.argv:
        print(f"ERROR: Cannot import sglang.srt.managers.io_struct ({_exc}).")
        print("Use --allow-shadow-fixtures to generate with shadow copies (may drift from production).")
        sys.exit(1)
    warnings.warn("Using shadow copies (may drift from production wire format)")
    _encode_fn = _shadow_encoder.encode

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _sp(obj):
    """Build sampling params — SamplingParams object when using real imports, dict otherwise."""
    if _using_real_imports:
        return _RealSamplingParams(**obj)
    return obj


def _ids(obj):
    """Build input_ids as array('q') matching production TokenizedGenerateReqInput.input_ids type."""
    return array("q", obj)


def _oids(rows):
    """Build output_ids as list[array('q')] matching production output_ids type."""
    return [array("q", r) for r in rows]


ALL_FIXTURES = []


def fixture(name, is_list=False):
    def wrapper(fn):
        ALL_FIXTURES.append((name, fn, is_list))
        return fn
    return wrapper


if not _using_real_imports:
    # Shadow copies matching io_struct.py field order — drift risk if io_struct
    # changes without regenerating the snapshot.
    class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
        rid: str | None = None
        http_worker_ipc: str | None = None

    class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
        rids: list[str] | None = None
        http_worker_ipcs: list[str | None] | None = None

    class PickleWrapper(msgspec.Struct, tag=True, array_like=True):
        data: bytes

    class SessionParams(msgspec.Struct, kw_only=True, array_like=True):
        id: str | None = None
        rid: str | None = None
        offset: int | None = None
        replace: bool | None = None
        drop_previous_output: bool | None = None

    class TokenizedGenerateReqInput(BaseReq, kw_only=True):
        rid: str | None = None
        http_worker_ipc: str | None = None
        input_text: str | None = None
        input_ids: array | None = None
        input_embeds: list | None = None
        mm_inputs: list | None = None
        token_type_ids: list | None = None
        sampling_params: dict | None = None
        return_logprob: bool = False
        logprob_start_len: int = -1
        top_logprobs_num: int = 0
        token_ids_logprob: list | None = None
        stream: bool = False
        return_hidden_states: bool = False
        return_routed_experts: bool = False
        routed_experts_start_len: int = 0
        return_indexer_topk: bool = False
        session_id: str | None = None
        session_params: SessionParams | None = None
        lora_id: int | None = None
        custom_logit_processor: bytes | None = None
        positional_embed_overrides: dict | None = None
        bootstrap_host: str | None = None
        bootstrap_port: int | None = None
        bootstrap_room: str | None = None
        bootstrap_pair_key: str | None = None
        decode_tp_size: int | None = None
        routed_dp_rank: int | None = None
        disagg_prefill_dp_rank: int | None = None
        routing_key: str | None = None
        require_reasoning: bool = False
        priority: int | None = None
        extra_key: str | None = None
        no_logs: bool = False
        return_bytes: bool = False
        return_entropy: bool = False
        need_wait_for_mm_inputs: bool = False
        num_items_assigned: int | None = None
        mm_data_mooncake: bytes | None = None
        encoder_urls: list[str] | None = None
        multi_item_delimiter_indices: list[int] | None = None
        time_stats: bytes | None = None

    class TokenizedEmbeddingReqInput(BaseReq, kw_only=True):
        rid: str | None = None
        http_worker_ipc: str | None = None
        input_text: str | None = None
        input_ids: array | None = None
        mm_inputs: list | None = None
        token_type_ids: list | None = None
        sampling_params: dict | None = None
        lora_id: int | None = None
        positional_embed_overrides: dict | None = None
        routed_dp_rank: int | None = None
        priority: int | None = None
        dimensions: int | None = None
        return_pooled_hidden_states: bool = False
        multi_item_delimiter_indices: list[int] | None = None
        time_stats: bytes | None = None

    class BatchTokenizedGenerateReqInput(BaseBatchReq, kw_only=True):
        rids: list[str] | None = None
        http_worker_ipcs: list[str | None] | None = None
        batch: list | None = None

    class BatchStrOutput(BaseBatchReq, kw_only=True):
        rids: list[str] | None = None
        http_worker_ipcs: list[str | None] | None = None
        finished_reasons: list | None = None
        output_strs: list[str | None] | None = None
        output_ids: list | None = None
        prompt_tokens: list[int] | None = None
        completion_tokens: list[int] | None = None
        reasoning_tokens: list[int] | None = None
        cached_tokens: list[int] | None = None
        input_token_logprobs_val: list | None = None
        input_token_logprobs_idx: list | None = None
        output_token_logprobs_val: list | None = None
        output_token_logprobs_idx: list | None = None
        input_top_logprobs_val: list | None = None
        input_top_logprobs_idx: list | None = None
        output_top_logprobs_val: list | None = None
        output_top_logprobs_idx: list | None = None
        input_token_ids_logprobs_val: list | None = None
        input_token_ids_logprobs_idx: list | None = None
        output_token_ids_logprobs_val: list | None = None
        output_token_ids_logprobs_idx: list | None = None
        output_token_entropy_val: list | None = None
        output_hidden_states: list | None = None
        routed_experts: list | None = None
        indexer_topk: list | None = None
        placeholder_tokens_idx: list | None = None
        placeholder_tokens_val: list | None = None
        retraction_counts: list[int] | None = None
        token_steps: list[int] | None = None
        customized_info: list | None = None
        cached_tokens_details: list | None = None
        dp_ranks: list[int] | None = None
        time_stats: bytes | None = None
        image_tokens: list[int] | None = None
        audio_tokens: list[int] | None = None
        video_tokens: list[int] | None = None
        spec_verify_ct: list[int] | None = None
        spec_num_correct_drafts: list[int] | None = None
        spec_correct_drafts_histogram: list | None = None

    class BatchTokenIDOutput(BaseBatchReq, kw_only=True):
        rids: list[str] | None = None
        http_worker_ipcs: list[str | None] | None = None
        finished_reasons: list | None = None
        decoded_texts: list[str | None] | None = None
        decode_ids: list | None = None
        read_offsets: list[int | None] | None = None
        output_ids: list | None = None
        skip_special_tokens: list[bool] | None = None
        spaces_between_special_tokens: list[bool] | None = None
        no_stop_trim: list[bool] | None = None
        prompt_tokens: list[int] | None = None
        reasoning_tokens: list[int] | None = None
        completion_tokens: list[int] | None = None
        cached_tokens: list[int] | None = None
        input_token_logprobs_val: list | None = None
        input_token_logprobs_idx: list | None = None
        output_token_logprobs_val: list | None = None
        output_token_logprobs_idx: list | None = None
        input_top_logprobs_val: list | None = None
        input_top_logprobs_idx: list | None = None
        output_top_logprobs_val: list | None = None
        output_top_logprobs_idx: list | None = None
        input_token_ids_logprobs_val: list | None = None
        input_token_ids_logprobs_idx: list | None = None
        output_token_ids_logprobs_val: list | None = None
        output_token_ids_logprobs_idx: list | None = None
        output_token_entropy_val: list | None = None
        output_hidden_states: list | None = None
        routed_experts: list | None = None
        indexer_topk: list | None = None
        placeholder_tokens_idx: list | None = None
        placeholder_tokens_val: list | None = None
        retraction_counts: list[int] | None = None
        token_steps: list[int] | None = None
        customized_info: list | None = None
        cached_tokens_details: list | None = None
        dp_ranks: list[int] | None = None
        time_stats: bytes | None = None
        image_tokens: list[int] | None = None
        audio_tokens: list[int] | None = None
        video_tokens: list[int] | None = None
        spec_verify_ct: list[int] | None = None
        spec_num_correct_drafts: list[int] | None = None
        spec_correct_drafts_histogram: list | None = None

    class BatchEmbeddingOutput(BaseBatchReq, kw_only=True):
        rids: list[str] | None = None
        http_worker_ipcs: list[str | None] | None = None
        finished_reasons: list | None = None
        embeddings: list | None = None
        prompt_tokens: list[int] | None = None
        cached_tokens: list[int] | None = None
        placeholder_tokens_idx: list | None = None
        placeholder_tokens_val: list | None = None
        retraction_counts: list[int] | None = None
        cached_tokens_details: list | None = None
        time_stats: bytes | None = None
        pooled_hidden_states: list | None = None

    class AbortReq(BaseReq, kw_only=True):
        rid: str | None = None
        http_worker_ipc: str | None = None
        abort_all: bool = False
        finished_reason: dict | None = None
        abort_message: str | None = None

    class FlushCacheReqInput(BaseReq, kw_only=True):
        rid: str | None = None
        http_worker_ipc: str | None = None
        timeout_s: float | None = None

    class ProfileReqType(enum.IntEnum):
        START_PROFILE = 1
        STOP_PROFILE = 2

    class ProfileReq(BaseReq, kw_only=True):
        rid: str | None = None
        http_worker_ipc: str | None = None
        req_type: ProfileReqType | None = None
        output_dir: str | None = None
        start_step: int | None = None
        num_steps: int | None = None
        activities: list | None = None
        profile_by_stage: bool = False
        with_stack: bool = False
        record_shapes: bool = False
        profile_id: str | None = None
        merge_profiles: bool = False
        profile_prefix: str | None = None
        profile_stages: list | None = None


# ── Fixture factories ──

@fixture("single_text_generate")
def _():
    return TokenizedGenerateReqInput(
        rid="test-rid-001", input_text="Hello, world!",
        input_ids=_ids([15496, 11, 1917, 0]),
        sampling_params=_sp({"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100}),
        stream=False,
    )


@fixture("input_ids_generate")
def _():
    return TokenizedGenerateReqInput(
        rid="test-rid-002", input_ids=_ids(list(range(1, 101))),
        sampling_params=_sp({"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 50}),
        stream=True, return_logprob=True, logprob_start_len=0, top_logprobs_num=5,
    )


@fixture("streaming_output_chunk")
def _():
    return BatchStrOutput(
        rids=["test-rid-001"], finished_reasons=[None], output_strs=[" Hello"],
        output_ids=_oids([[22550]]),
        prompt_tokens=[5], completion_tokens=[1],
        reasoning_tokens=[0], cached_tokens=[3],
    )


@fixture("final_streaming_output")
def _():
    return BatchStrOutput(
        rids=["test-rid-001"], finished_reasons=["stop"], output_strs=[" world!"],
        output_ids=_oids([[456]]),
        prompt_tokens=[5], completion_tokens=[2],
        reasoning_tokens=[0], cached_tokens=[3],
    )


@fixture("token_id_output")
def _():
    return BatchTokenIDOutput(
        rids=["test-rid-002"], finished_reasons=[None],
        decoded_texts=[None], decode_ids=[None], read_offsets=[None],
        output_ids=_oids([[42, 123, 456]]),
        skip_special_tokens=[True], spaces_between_special_tokens=[True],
        no_stop_trim=[True],
        prompt_tokens=[100], completion_tokens=[3], reasoning_tokens=[0], cached_tokens=[0],
    )


@fixture("batch_generate")
def _():
    return BatchTokenizedGenerateReqInput(batch=[
        TokenizedGenerateReqInput(
            rid="batch-rid-001", input_text="What is Rust?",
            input_ids=_ids([2051, 318, 3478, 30]),
            sampling_params=_sp({"temperature": 0.5, "max_new_tokens": 200}), stream=False,
        ),
        TokenizedGenerateReqInput(
            rid="batch-rid-002", input_text="What is Python?",
            input_ids=_ids([2051, 318, 2191, 30]),
            sampling_params=_sp({"temperature": 0.8, "max_new_tokens": 150}), stream=True,
        ),
    ])


@fixture("embedding_request")
def _():
    return TokenizedEmbeddingReqInput(
        rid="embed-rid-001", input_text="Embed this sentence.",
        input_ids=_ids([4973, 554, 5173, 13]),
    )


@fixture("embedding_output")
def _():
    return BatchEmbeddingOutput(
        rids=["embed-rid-001"], finished_reasons=[None],
        embeddings=[b"\x00\x00\x00\x00" * 4],
        prompt_tokens=[4], cached_tokens=[0],
    )


@fixture("pickle_wrapper", is_list=True)
def _():
    return [PickleWrapper(data=pickle.dumps(p)) for p in [
        {"custom_info": {"key": "value"}, "tokens": [1, 2, 3]},
        {"scores": [0.5, 0.3, 0.2]},
    ]]


@fixture("abort_request")
def _():
    return AbortReq(rid="test-rid-001")


@fixture("logprobs_output")
def _():
    return BatchStrOutput(
        rids=["test-rid-003"], finished_reasons=["length"],
        output_strs=[" The capital of France is Paris."],
        output_ids=_oids([[791, 11241, 315, 1187, 338, 3367, 13]]),
        prompt_tokens=[8], completion_tokens=[7], reasoning_tokens=[0], cached_tokens=[0],
        input_token_logprobs_val=[[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]],
        input_token_logprobs_idx=[[0, 1, 2, 3, 4, 5, 6, 7]],
        output_token_logprobs_val=[[-0.05, -0.01, -0.15, -0.02, -0.08, -0.11, -0.03]],
        output_token_logprobs_idx=[[8, 9, 10, 11, 12, 13, 14]],
    )


@fixture("cached_tokens_details")
def _():
    return BatchStrOutput(
        rids=["test-rid-004"], finished_reasons=["stop"], output_strs=[" result"],
        output_ids=_oids([[22550]]),
        prompt_tokens=[100], completion_tokens=[1], reasoning_tokens=[0], cached_tokens=[85],
        cached_tokens_details=[{"hit": 80, "miss": 5, "prefix": 85, "spare": 0}],
    )


@fixture("empty_fields")
def _():
    return TokenizedGenerateReqInput(
        rid="test-rid-005", input_text="", input_ids=_ids([]),
        sampling_params=_sp({}), stream=False,
    )


@fixture("reasoner_request")
def _():
    return TokenizedGenerateReqInput(
        rid="test-rid-006", input_text="Solve this step by step: 2 + 2",
        input_ids=_ids([13347, 554, 2678, 2035, 518, 11, 17, 287, 17]),
        sampling_params=_sp({"temperature": 0.6, "max_new_tokens": 500}),
        stream=True, require_reasoning=True,
    )


@fixture("session_request")
def _():
    return TokenizedGenerateReqInput(
        rid="test-rid-007", input_text="Continue the conversation.",
        input_ids=_ids([1002, 262, 5752, 13]),
        sampling_params=_sp({"temperature": 0.7, "max_new_tokens": 100}),
        stream=True, session_id="session-abc-123",
        session_params=SessionParams(
            id="session-abc-123", rid="test-rid-006",
            replace=False, drop_previous_output=False,
        ),
    )


@fixture("flush_cache_request")
def _():
    return FlushCacheReqInput(rid="control-rid-001", timeout_s=30.0)


@fixture("profile_request")
def _():
    return ProfileReq(rid="control-rid-002", req_type=ProfileReqType.START_PROFILE)


def write_fixtures(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, fn, is_list in ALL_FIXTURES:
        obj = fn()
        encoded = _encode_fn(obj if not is_list else obj)
        (output_dir / f"{name}.msgpack").write_bytes(encoded)
        try:
            (output_dir / f"{name}.json").write_text(
                json.dumps(msgspec.msgpack.decode(encoded), indent=2, default=str))
        except Exception:
            pass
        manifest.append({"name": name, "file": f"{name}.msgpack", "size_bytes": len(encoded)})
        print(f"  {name}.msgpack ({len(encoded)} bytes)")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(ALL_FIXTURES)} fixtures to {output_dir}")


if __name__ == "__main__":
    if "--check-real-imports" in sys.argv:
        # Smoke test: construct and encode one minimal struct using real classes.
        # Must pass ALL fields explicitly — production uses dataclasses.field()
        # defaults that msgspec doesn't evaluate until encoding.
        obj = TokenizedGenerateReqInput(
            rid="smoke-rid",
            input_text="test",
            input_ids=array("q", [1, 2, 3]),
            input_embeds=None, mm_inputs=None, token_type_ids=None,
            sampling_params=_sp({"temperature": 0.5, "max_new_tokens": 10}),
            return_logprob=False, logprob_start_len=-1, top_logprobs_num=0,
            token_ids_logprob=None, stream=False,
            return_hidden_states=False, return_routed_experts=False,
            routed_experts_start_len=0, return_indexer_topk=False,
            session_id=None, session_params=None,
            lora_id=None, custom_logit_processor=None,
            positional_embed_overrides=None,
            bootstrap_host=None, bootstrap_port=None, bootstrap_room=None,
            bootstrap_pair_key=None, decode_tp_size=None,
            routed_dp_rank=None, disagg_prefill_dp_rank=None,
            routing_key=None, require_reasoning=False,
            priority=None, extra_key=None, no_logs=False,
            return_bytes=False, return_entropy=False,
            need_wait_for_mm_inputs=False, num_items_assigned=None,
            mm_data_mooncake=None, encoder_urls=None,
            multi_item_delimiter_indices=None, time_stats=None,
        )
        _encode_fn(obj)
        print("OK: Real io_struct.py imports + encode work")
        sys.exit(0)
    elif "--write" in sys.argv:
        write_fixtures(FIXTURE_DIR)
    else:
        print("Use --write to generate fixtures")
