#!/usr/bin/env python3
"""PyO3 constructor smoke test — verifies TokenizerManager can be built from Python."""

import importlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUST_TARGET = REPO / "rust" / "sglang-server" / "target" / "release"

# PyO3 produces libsglang_server.so; Python needs sglang_server.so
so_path = RUST_TARGET / "libsglang_server.so"
if so_path.exists():
    import shutil
    target_so = RUST_TARGET / "sglang_server.so"
    if not target_so.exists():
        shutil.copy2(so_path, target_so)

sys.path.insert(0, str(RUST_TARGET))

import sglang_server


def test_constructor():
    """Build a TokenizerManager with minimal config."""
    config = {
        "model_path": "test-model",
        "served_model_name": "test-model",
        "host": "127.0.0.1",
        "port": 30000,
        "skip_tokenizer_init": True,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "model_config": {
            "model": "test-model",
            "context_len": 4096,
            "is_generation": True,
        },
    }
    config_json = json.dumps(config)

    tm = sglang_server.TokenizerManager(config_json)
    assert tm is not None
    print(f"  TokenizerManager constructed OK (type={type(tm).__name__})")
    return tm


def test_server_constructor():
    """Build a Server with minimal config (existing entry point)."""
    config = {
        "model_path": "test-model",
        "served_model_name": "test-model",
        "host": "127.0.0.1",
        "port": 30000,
        "skip_tokenizer_init": True,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "model_config": {
            "model": "test-model",
            "context_len": 4096,
            "is_generation": True,
        },
    }
    config_json = json.dumps(config)

    server = sglang_server.Server(
        bind="127.0.0.1:30000",
        server_args_json=config_json,
        tokenizer_path=None,
    )
    assert server is not None
    print(f"  Server constructed OK (type={type(server).__name__})")
    return server


def main():
    print("PyO3 smoke tests...\n")
    tests = [
        ("TokenizerManager constructor", test_constructor),
        ("Server constructor", test_server_constructor),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}\n")
        except Exception as e:
            print(f"  FAIL: {name} — {e}\n")
            failures += 1

    if failures:
        print(f"{failures} test(s) FAILED")
        sys.exit(1)
    print("All smoke tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
