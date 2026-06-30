#!/usr/bin/env python3
"""Generate the method coverage manifest for the Rust TokenizerManager migration.

Scans tokenizer_manager.py, tokenizer_control_mixin.py, and
tokenizer_manager_score_mixin.py for all public methods and assigns each a
migration phase from the plan.

Usage:
    python method_manifest.py [--markdown] [--json]
"""

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOKENIZER_MANAGER_PY = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "tokenizer_manager.py"
CONTROL_MIXIN_PY = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "tokenizer_control_mixin.py"
SCORE_MIXIN_PY = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "tokenizer_manager_score_mixin.py"
MANIFEST_PATH = REPO_ROOT / "tools" / "rust-tkm-migration" / "method_manifest.json"

# Phase assignments from plan.md
PHASE_ASSIGNMENTS = {
    # Phase 0: contract capture (no implementation)
    # Phase 1: skeleton, schema, IPC
    "__init__": 1,
    "init_model_config": 1,
    "init_ipc_channels": 1,
    "init_running_status": 1,
    "init_request_dispatcher": 1,
    "serving_chat_class": 1,
    "init_communicators": 1,
    "_dispatch_to_scheduler": 1,
    "_async_dispatch_to_scheduler": 1,
    "stamp_http_worker_ipc": 1,

    # Phase 2: single generate + FanOut infrastructure
    "generate_request": 2,
    "_set_default_priority": 2,
    "_init_req_state": 2,
    "_detect_input_format": 2,
    "_prepare_tokenizer_input": 2,
    "_extract_tokenizer_results": 2,
    "_tokenize_texts": 2,
    "_tokenize_one_request": 2,
    "_validate_one_request": 2,
    "_validate_token_ids_logprob": 2,
    "_create_tokenized_object": 2,
    "_send_one_request": 2,
    "auto_create_handle_loop": 2,
    "handle_loop": 2,
    "_handle_batch_output": 2,
    "_wait_one_response": 2,
    "_coalesce_streaming_chunks": 2,
    "_slice_streaming_output_meta_info": 2,
    "_handle_abort_finish_reason": 2,
    "abort_request": 2,
    "create_abort_task": 2,
    "_handle_abort_req": 2,
    "ReqState": 2,
    "append_text": 2,
    "get_text": 2,
    "get_crash_dump_output": 8,
    "background_task": 2,
    "init_tokenizer_and_processor": 2,
    "_fan_out_communicator": 2,  # FanOutCommunicator infrastructure

    # Phase 3: batch, embedding, logprobs
    "_handle_batch_request": 3,
    "_batch_tokenize_and_process": 3,
    "_validate_batch_tokenization_constraints": 3,
    "_batch_has_text": 3,
    "_should_use_batch_tokenization": 3,
    "_send_batch_request": 3,
    "_validate_for_matryoshka_dim": 3,
    "_validate_input_ids_in_vocab": 3,
    "_resolve_embed_overrides": 3,
    "add_logprob_to_meta_info": 3,
    "convert_logprob_style": 3,
    "detokenize_logprob_tokens": 3,
    "detokenize_top_logprobs_tokens": 3,
    "_calculate_spec_decoding_metrics": 3,
    "_request_has_grammar": 3,

    # Phase 4: control plane foundation
    "flush_cache": 4,
    "clear_hicache_storage": 4,
    "attach_hicache_storage": 4,
    "detach_hicache_storage": 4,
    "release_memory_occupation": 4,
    "resume_memory_occupation": 4,
    "get_internal_state": 4,
    "set_internal_state": 4,
    "dumper_control": 4,
    "get_loads": 4,
    "check_weights": 4,
    "slow_down": 4,
    "configure_logging": 4,
    "freeze_gc": 4,
    "start_profile": 4,
    "stop_profile": 4,
    "_execute_profile": 4,
    "start_expert_distribution_record": 4,
    "stop_expert_distribution_record": 4,
    "dump_expert_distribution_record": 4,
    "pause_generation": 4,
    "continue_generation": 4,
    "open_session": 4,
    "close_session": 4,
    "_handle_open_session_req_output": 4,
    "update_active_ranks": 4,

    # Phase 5: weights, LoRA, external corpus
    "init_weight_update": 5,
    "init_lora": 5,
    "update_weights_from_disk": 5,
    "_wait_for_model_update_from_disk": 5,
    "_update_model_path_info": 5,
    "_handle_update_weights_from_disk_req_output": 5,
    "_validate_and_resolve_lora": 5,
    "_resolve_lora_path": 5,
    "init_weights_update_group": 5,
    "destroy_weights_update_group": 5,
    "update_weights_from_distributed": 5,
    "init_weights_send_group_for_remote_instance": 5,
    "send_weights_to_remote_instance": 5,
    "update_weights_from_tensor": 5,
    "update_weights_from_ipc": 5,
    "get_weights_by_name": 5,
    "load_lora_adapter": 5,
    "load_lora_adapter_from_tensors": 5,
    "unload_lora_adapter": 5,
    "_unload_lora_adapter_locked": 5,
    "_update_weight_version_if_provided": 5,
    "add_external_corpus": 5,
    "remove_external_corpus": 5,
    "list_external_corpora": 5,

    # Phase 6: scoring
    "score_request": 6,
    "score_prompts": 6,
    "_build_multi_item_token_sequence": 6,
    "_batch_tokenize_query_and_items": 6,
    "_process_multi_item_scoring_results": 6,
    "_process_single_item_scoring_results": 6,
    "_resolve_overrides_for_sequence": 6,
    "_resolve_embed_overrides_for_request": 6,
    "_build_token_id_inputs": 6,
    "_convert_logprobs_to_scores": 6,
    "_extract_logprobs_for_tokens": 6,

    # Phase 7: multimodal and disaggregation
    "init_disaggregation": 7,
    "_validate_mm_limits": 7,
    "_should_dispatch_to_encoder": 7,
    "_handle_epd_disaggregation_encode_request": 7,
    "_determine_tensor_transport_mode": 7,
    "_get_processor_wrapper": 7,
    "_count_mm_items": 7,

    # Phase 8: observability, shutdown
    "init_request_logging_and_dumping": 8,
    "init_metric_collector_watchdog": 8,
    "collect_metrics": 8,
    "dump_requests": 8,
    "record_request_for_crash_dump": 8,
    "_dump_data_to_file": 8,
    "dump_requests_before_crash": 8,
    "sigterm_watchdog": 8,
    "force_exit_handler": 8,
    "convert_to_span_attrs": 8,
    "_discard_pending_req_states": 8,
    "ReqState.get_crash_dump_output": 8,
    "SignalHandler": 8,
    "SignalHandler.__init__": 8,
    "sigterm_handler": 8,
    "running_phase_sigquit_handler": 8,
    "print_exception_wrapper": 8,
}


def get_public_methods(filepath: Path) -> list:
    """Extract all public methods (sync and async) from a Python file."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = []

    def is_func(node):
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    def is_dunder(name):
        return name.startswith("__") and name.endswith("__") and name != "__init__"

    for node in ast.walk(tree):
        # Top-level functions
        if is_func(node) and not is_dunder(node.name):
            methods.append({"name": node.name, "lineno": node.lineno, "file": filepath.name})

        # Methods inside classes — namespace as ClassName.methodname
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if is_func(item) and not is_dunder(item.name):
                    method_name = f"{node.name}.{item.name}"
                    methods.append({"name": method_name, "lineno": item.lineno, "file": filepath.name})

    return methods


def bare_name(fullname: str) -> str:
    """Strip class prefix for phase lookup."""
    return fullname.split(".", 1)[-1] if "." in fullname else fullname


def main():
    files = [
        ("tokenizer_manager.py", TOKENIZER_MANAGER_PY),
        ("tokenizer_control_mixin.py", CONTROL_MIXIN_PY),
        ("tokenizer_manager_score_mixin.py", SCORE_MIXIN_PY),
    ]

    manifest = {"methods": [], "unassigned": []}

    for fname, fpath in files:
        if not fpath.exists():
            print(f"WARNING: {fpath} not found", file=sys.stderr)
            continue
        methods = get_public_methods(fpath)
        for m in methods:
            entry = {
                "name": m["name"],
                "file": m["file"],
                "lineno": m["lineno"],
                "phase": PHASE_ASSIGNMENTS.get(m["name"], None) or PHASE_ASSIGNMENTS.get(bare_name(m["name"]), None),
            }
            manifest["methods"].append(entry)
            if entry["phase"] is None:
                manifest["unassigned"].append(m["name"])

    # Persist manifest JSON
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # Output
    if "--markdown" in sys.argv:
        print(f"# Method Coverage Manifest\n")
        print(f"| Method | File | Phase |")
        print(f"|---|---|---|")
        by_phase = sorted(manifest["methods"], key=lambda x: (x.get("phase") or 99, x["file"], x["name"]))
        for m in by_phase:
            phase = str(m["phase"]) if m["phase"] else "TBD"
            print(f"| {m['name']} | {m['file']}:{m['lineno']} | {phase} |")
    else:
        total = len(manifest["methods"])
        assigned = total - len(manifest["unassigned"])
        print(f"Total methods: {total}")
        print(f"Assigned: {assigned}")
        print(f"Unassigned: {len(manifest['unassigned'])}")
        print(f"Persisted to {MANIFEST_PATH}")
        if manifest["unassigned"]:
            print("\nWARNING: Unassigned methods:")
            for name in manifest["unassigned"]:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
