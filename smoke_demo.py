#!/usr/bin/env python3
"""Smoke demo for the Rust text-serving frontend.

This is a controlled serving demo, not a model-quality demo. It starts the Rust
TokenizerManager HTTP server with `skip_tokenizer_init=True`, runs a tiny mock
Scheduler that echoes synthetic token chunks, and exercises the text inference
surface that is currently in scope:

- `/generate` single `input_ids`
- `/generate` batch `input_ids`
- `/generate` streaming
- `/v1/completions` with token-id prompt

Run from the repo root after building/installing `sglang_server`, for example:

    .venv/bin/python smoke_demo.py
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec


REPO = Path(__file__).resolve().parent


def import_sglang_server():
    try:
        import sglang_server  # type: ignore

        return sglang_server
    except ModuleNotFoundError:
        target_release = REPO / "rust" / "sglang-server" / "target" / "release"
        target_debug = REPO / "rust" / "sglang-server" / "target" / "debug"
        for path in (target_release, target_debug):
            if (path / "sglang_server.so").exists():
                sys.path.insert(0, str(path))
                import sglang_server  # type: ignore

                return sglang_server

    raise SystemExit(
        "Cannot import sglang_server. Build/install it first, then rerun:\n"
        "  cd rust/sglang-server && maturin develop --release\n"
        "  cd ../.. && .venv/bin/python smoke_demo.py"
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(port: int, path: str, body: dict[str, Any], timeout: float = 10.0) -> Any:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode()
    return json.loads(data)


def get_json(port: int, path: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_sse(port: int, path: str, body: dict[str, Any], timeout: float = 10.0) -> list[str]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode()

    events: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(line.removeprefix("data:").strip())
    return events


def decode_i64_cells(ids_buf: bytes, lengths: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    offset = 0
    for n_tokens in lengths:
        ids = []
        for _ in range(n_tokens):
            ids.append(struct.unpack_from("<q", ids_buf, offset)[0])
            offset += 8
        out.append(ids)
    return out


@dataclass
class HeaderView:
    rid: str
    tag: str
    prompt_tokens: int
    stream: bool
    return_logprob: bool
    top_logprobs_num: int


def header_view(header: list[Any], prompt_tokens: int) -> HeaderView:
    return HeaderView(
        rid=str(header[1]),
        tag=str(header[0]),
        prompt_tokens=prompt_tokens,
        stream=bool(header[13]) if len(header) > 13 else False,
        return_logprob=bool(header[9]) if len(header) > 9 else False,
        top_logprobs_num=int(header[11]) if len(header) > 11 and header[11] is not None else 0,
    )


def chunk(
    view: HeaderView,
    seq: int,
    token_id: int,
    finish_reason: str | None,
) -> bytes:
    payload: dict[str, Any] = {
        "rid": view.rid,
        "seq": seq,
        "token_ids": [token_id],
        "finish_reason": finish_reason,
        "prompt_tokens": view.prompt_tokens,
    }
    if view.return_logprob:
        payload["output_token_logprobs_val"] = [-0.10 - seq / 10.0]
        payload["output_token_logprobs_idx"] = [token_id]
        if seq == 0:
            payload["input_token_logprobs_val"] = [-0.01] * view.prompt_tokens
            payload["input_token_logprobs_idx"] = list(range(view.prompt_tokens))
        if view.top_logprobs_num > 0:
            payload["output_top_logprobs_val"] = [[[-0.10 - seq / 10.0, token_id]]]
    return msgspec.msgpack.encode(payload)


def mock_scheduler(tm: Any, stop: threading.Event) -> None:
    """Drain Rust ingress and push synthetic scheduler chunks."""
    while not stop.is_set():
        headers, ids_buf, lengths = tm.recv_requests(32)
        if not headers:
            time.sleep(0.005)
            continue

        input_batches = decode_i64_cells(ids_buf, list(lengths))
        for i, header_bytes in enumerate(headers):
            header = msgspec.msgpack.decode(header_bytes)
            if not isinstance(header, list) or not header:
                continue

            view = header_view(header, len(input_batches[i]))
            if view.tag != "TokenizedGenerateReqInput":
                continue

            print(
                f"mock scheduler: rid={view.rid} prompt={input_batches[i]} "
                f"stream={view.stream} return_logprob={view.return_logprob}"
            )
            tm.push_chunk(chunk(view, seq=0, token_id=101, finish_reason=None))
            tm.push_chunk(chunk(view, seq=1, token_id=102, finish_reason="stop"))


def assert_true(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name}{suffix}")
    print(f"PASS: {name}")


def main() -> None:
    sglang_server = import_sglang_server()
    port = free_port()
    config = {
        "model_path": "smoke-model",
        "served_model_name": "smoke-model",
        "host": "127.0.0.1",
        "port": port,
        "skip_tokenizer_init": True,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "model_config": {
            "model": "smoke-model",
            "context_len": 4096,
            "is_generation": True,
        },
    }

    tm = sglang_server.TokenizerManager(json.dumps(config), startup_timeout_ms=10000)
    stop = threading.Event()
    scheduler = threading.Thread(target=mock_scheduler, args=(tm, stop), daemon=True)
    scheduler.start()

    try:
        time.sleep(0.2)
        print(f"Rust server: http://127.0.0.1:{port}")

        models = get_json(port, "/v1/models")
        assert_true("/v1/models", models["data"][0]["id"] == "smoke-model")

        single = post_json(
            port,
            "/generate",
            {
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 2},
                "return_prompt_token_ids": True,
            },
        )
        assert_true("/generate single", single["output_ids"] == [101, 102])
        assert_true("prompt_token_ids", single["prompt_token_ids"] == [1, 2, 3])

        batch = post_json(
            port,
            "/generate",
            {
                "input_ids": [[4, 5], [6, 7, 8]],
                "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 2}],
                "return_logprob": [True, False],
                "top_logprobs_num": [1, 0],
            },
        )
        assert_true("/generate batch length", len(batch) == 2)
        assert_true("/generate batch outputs", [x["output_ids"] for x in batch] == [[101, 102], [101, 102]])
        assert_true("batch logprobs projected", "output_token_logprobs" in batch[0]["meta_info"])
        assert_true("batch logprobs per-item", "output_token_logprobs" not in batch[1]["meta_info"])

        events = post_sse(
            port,
            "/generate",
            {
                "input_ids": [9, 10],
                "sampling_params": {"max_new_tokens": 2},
                "stream": True,
            },
        )
        assert_true("/generate streaming events", len(events) >= 3)
        assert_true("/generate streaming done", events[-1] == "[DONE]")

        completion = post_json(
            port,
            "/v1/completions",
            {
                "model": "smoke-model",
                "prompt": [11, 12],
                "max_tokens": 2,
            },
        )
        assert_true("/v1/completions", completion["choices"][0]["finish_reason"] == "stop")

        print("\nSmoke demo completed.")
        print("This validates the Rust frontend and scheduler boundary with a mock scheduler.")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc
    finally:
        stop.set()
        tm.shutdown()
        scheduler.join(timeout=3)


if __name__ == "__main__":
    main()
