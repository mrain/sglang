#!/usr/bin/env python3
"""Mock Scheduler integration test for the Rust TokenizerManager.

Starts the Rust TM, runs a mock Scheduler thread that drains the ingress ring
and pushes response chunks, then verifies the full HTTP → TM → mock-Scheduler
→ TM → HTTP pipeline works.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import msgspec

REPO = Path(__file__).resolve().parent.parent.parent
RUST_CRATE = REPO / "rust" / "sglang-server"
TARGET = RUST_CRATE / "target" / "release"

# Build the Rust extension
print("Building sglang-server...")
subprocess.run(["cargo", "build", "--release"], cwd=RUST_CRATE, check=True,
               capture_output=True)

# Copy .so for Python import
so = TARGET / "libsglang_server.so"
target_so = TARGET / "sglang_server.so"
shutil.copy2(so, target_so)
sys.path.insert(0, str(TARGET))

import sglang_server


def mock_scheduler(tm, stop_event):
    """Mock scheduler: drain ingress ring, echo requests back as responses."""
    while not stop_event.is_set():
        try:
            headers, ids_buf, lengths = tm.recv_requests(16)
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"[mock-sched] recv error: {e}", file=sys.stderr)
            break

        if not headers:
            time.sleep(0.01)
            continue

        offset = 0
        for i, header_bytes in enumerate(headers):
            if stop_event.is_set():
                return

            # Decode the header (msgspec tagged array)
            header = msgspec.msgpack.decode(header_bytes)
            tag = header[0] if isinstance(header, list) and len(header) > 0 else None
            if tag is None:
                print(f"[mock-sched] skipping header with no tag", file=sys.stderr)
                continue

            # Request ID at index 1
            rid = header[1] if len(header) > 1 else "unknown"
            if rid is None:
                continue

            # Token count for this request
            n_tokens = lengths[i] if i < len(lengths) else 0

            # Read input_ids from ids_buf (n_tokens * 8 bytes each, little-endian i64)
            token_bytes = ids_buf[offset:offset + n_tokens * 8]
            offset += n_tokens * 8
            input_ids = []
            for j in range(n_tokens):
                raw = token_bytes[j * 8:(j + 1) * 8]
                if len(raw) == 8:
                    input_ids.append(struct.unpack("<q", raw)[0])

            print(f"[mock-sched] got rid={rid} tag={tag} n_tokens={n_tokens}")

            # Build response chunks: echo back input_ids as output_ids
            if input_ids:
                # Intermediate chunk (streaming)
                chunk = msgspec.msgpack.encode(
                    [rid, 0, input_ids[:2], None, n_tokens]
                )
                tm.push_chunk(chunk)

            # Final chunk with finish_reason
            chunk = msgspec.msgpack.encode(
                [rid, 1, input_ids, "stop", n_tokens]
            )
            tm.push_chunk(chunk)


def test_generate_endpoint(tm, port):
    """Send a request to the Rust HTTP server and verify the response."""
    body = json.dumps({
        "input_ids": [1, 2, 3, 4, 5],
        "sampling_params": {"max_new_tokens": 10, "temperature": 0.5},
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"  Response: {json.dumps(result, indent=2)[:300]}")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body[:300]}")
        return None
    except Exception as e:
        print(f"  HTTP error: {e}")
        return None


def find_free_port():
    """Return a free TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = find_free_port()

    config = {
        "model_path": "test-model",
        "served_model_name": "test-model",
        "host": "127.0.0.1",
        "port": port,
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

    print(f"Starting TokenizerManager on port {port}...")
    tm = sglang_server.TokenizerManager(config_json, startup_timeout_ms=10000)
    actual_port = port

    print(f"Server starting on port {actual_port}...")

    stop_event = threading.Event()
    sched_thread = threading.Thread(
        target=mock_scheduler, args=(tm, stop_event), daemon=True
    )
    sched_thread.start()

    time.sleep(0.5)  # Let server bind and mock scheduler settle

    print("\n--- Test 1: /generate non-streaming ---")
    result = test_generate_endpoint(tm, actual_port)
    if result:
        print("  PASS: /generate returned response")
    else:
        print("  FAIL: /generate failed")
        tm.shutdown()
        stop_event.set()
        sys.exit(1)

    print("\n--- Test 2: /v1/models ---")
    req = urllib.request.Request(f"http://127.0.0.1:{actual_port}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            models = json.loads(resp.read())
            print(f"  Response: {json.dumps(models, indent=2)[:200]}")
            print("  PASS: /v1/models")
    except Exception as e:
        print(f"  FAIL: /v1/models error: {e}")

    print("\n--- Test 3: /generate streaming ---")
    body = json.dumps({
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 5, "temperature": 1.0},
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{actual_port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode()
            if "data:" in data:
                print(f"  Got SSE response ({len(data)} bytes)")
                print("  PASS: streaming works")
            else:
                print(f"  Unexpected response: {data[:200]}")
                print("  FAIL: not SSE")
    except Exception as e:
        print(f"  FAIL: streaming error: {e}")

    # Cleanup
    tm.shutdown()
    stop_event.set()
    sched_thread.join(timeout=3)

    print("\nAll integration tests completed.")


if __name__ == "__main__":
    main()
