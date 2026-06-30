#!/usr/bin/env python3
"""Validate Python and Rust integration for the TokenizerManager migration.

Tests:
1. Fixture generator uses real io_struct.py imports (not shadow copies)
2. All fixtures are valid msgspec (decode + re-encode = same bytes)
3. Schema snapshot matches current io_struct.py
4. Rust round-trip tests pass
5. Rust codegen is deterministic
"""

import json
import subprocess
import sys
from pathlib import Path

import msgspec

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_repo = Path(__file__).resolve().parent
while not (_repo / ".gitignore").exists():
    _repo = _repo.parent
REPO_ROOT = _repo
RUST_CRATE = REPO_ROOT / "rust" / "sglang-server"

_FAILURES = 0


def check(name: str, ok: bool, detail: str = ""):
    global _FAILURES
    if ok:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}")
        _FAILURES += 1
        if detail:
            for line in detail.strip().split("\n"):
                print(f"    {line}")


def test_fixture_generator_production():
    """Fixture generator must work with real io_struct.py imports."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "fixture_generator.py"), "--check-real-imports"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0:
        print("  PASS: Production import path — real io_struct.py imports work")
        return

    if "ERROR: Cannot import sglang.srt.managers.io_struct" in output:
        print("  INFO: Production import path skipped (no SGLang env)")
        return

    # Any other failure is a real bug
    check("Production import path", False, f"Fixture generator crashed:\n{output[-500:]}")


def test_fixtures_valid_msgspec():
    """All fixtures must decode+re-encode to identical bytes."""
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    errors = []
    for entry in manifest:
        path = FIXTURE_DIR / entry["file"]
        original = path.read_bytes()
        try:
            decoded = msgspec.msgpack.decode(original)
            re_encoded = msgspec.msgpack.encode(decoded)
            if original != re_encoded:
                errors.append(f"{entry['file']}: {len(original)}b -> {len(re_encoded)}b (size mismatch)")
        except Exception as e:
            errors.append(f"{entry['file']}: decode failed — {e}")
    ok = not errors
    detail = "\n".join(errors) if errors else ""
    check(f"All {len(manifest)} fixtures valid msgspec", ok, detail)


def test_schema_snapshot():
    """Schema snapshot must match current io_struct.py."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "schema_snapshot.py")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    err = (result.stdout + result.stderr)[-500:]
    check("Schema snapshot matches io_struct.py", result.returncode == 0, err)


def test_rust_roundtrip():
    """Rust round-trip tests must pass."""
    result = subprocess.run(
        ["cargo", "test", "--test", "schema_roundtrip"],
        capture_output=True, text=True, cwd=RUST_CRATE,
    )
    err = (result.stdout + result.stderr)[-500:]
    check("Rust round-trip tests pass", result.returncode == 0, err)
    for line in result.stdout.split("\n"):
        if "passed" in line and "failed" not in line:
            print(f"         {line.strip()}")


def test_scheduler_integration():
    """Full pipeline: HTTP→TM→mock Scheduler→TM→response with /generate."""
    import struct
    import threading
    import time
    import urllib.error
    import urllib.request

    import struct
    import threading
    import time

    try:
        m = _ensure_sglang()
    except RuntimeError as e:
        check("Scheduler integration", False, str(e))
        return

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = {
        "model_path": "test-model", "served_model_name": "test-model",
        "host": "127.0.0.1", "port": port,
        "skip_tokenizer_init": True,
        "tokenizer_worker_num": 1, "detokenizer_worker_num": 1,
        "model_config": {"model": "test-model", "context_len": 4096, "is_generation": True},
    }
    tm = m.TokenizerManager(json.dumps(config), startup_timeout_ms=10000)

    stop = threading.Event()
    mock_errors = []

    def mock_scheduler():
        while not stop.is_set():
            try:
                headers, ids_buf, lengths = tm.recv_requests(16)
            except Exception:
                if stop.is_set():
                    return
                raise
            if not headers:
                time.sleep(0.005)
                continue
            offset = 0
            for i, hdr in enumerate(headers):
                decoded = msgspec.msgpack.decode(hdr)
                rid = decoded[1] if isinstance(decoded, list) and len(decoded) > 1 else "0"
                n = lengths[i] if i < len(lengths) else 0
                ids_bytes = ids_buf[offset:offset + n * 8]
                offset += n * 8
                ids = [struct.unpack("<q", ids_bytes[j*8:(j+1)*8])[0] for j in range(n)]
                # Intermediate chunk
                chunk = msgspec.msgpack.encode([rid, 0, ids[:2] if len(ids) >= 2 else ids, None, n])
                tm.push_chunk(chunk)
                # Final chunk
                chunk = msgspec.msgpack.encode([rid, 1, ids, "stop", n])
                tm.push_chunk(chunk)

    t = threading.Thread(target=mock_scheduler, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        body = json.dumps({
            "input_ids": [10, 20, 30],
            "sampling_params": {"max_new_tokens": 5, "temperature": 0.5},
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            meta = result.get("meta_info", {})
            fr = meta.get("finish_reason", {})
            errors = []
            if fr.get("type") != "stop":
                errors.append(f"finish_reason type={fr.get('type')!r} != 'stop'")
            if meta.get("prompt_tokens") != 3:
                errors.append(f"prompt_tokens={meta.get('prompt_tokens')} != 3")
            if meta.get("completion_tokens", 0) < 1:
                errors.append(f"completion_tokens={meta.get('completion_tokens')} < 1")
            out_ids = result.get("output_ids", [])
            if not out_ids:
                errors.append("output_ids is empty")
            if errors:
                check("Scheduler integration (shape)", False, "; ".join(errors))
            else:
                check("Scheduler integration (shape)", True)
    except Exception as e:
        check("Scheduler integration (shape)", False, str(e))

    tm.shutdown()
    stop.set()
    t.join(timeout=3)


_SGLANG_MODULE = None


def _ensure_sglang():
    """Lazy-import sglang_server once; rebuilds only on the first call."""
    global _SGLANG_MODULE
    if _SGLANG_MODULE is not None:
        return _SGLANG_MODULE

    import shutil

    build = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True, text=True, cwd=RUST_CRATE,
    )
    if build.returncode != 0:
        raise RuntimeError(f"cargo build failed: {build.stderr[-300:]}")

    so_path = RUST_CRATE / "target" / "release" / "libsglang_server.so"
    target_so = RUST_CRATE / "target" / "release" / "sglang_server.so"
    shutil.copy2(so_path, target_so)
    sys.path.insert(0, str(RUST_CRATE / "target" / "release"))

    import sglang_server as m
    _SGLANG_MODULE = m
    return m


def test_pyo3_constructor():
    """PyO3 TokenizerManager can be constructed from Python."""
    import json
    try:
        m = _ensure_sglang()
    except RuntimeError as e:
        check("PyO3 constructor", False, str(e))
        return

    config = {
        "model_path": "test-model", "served_model_name": "test-model",
        "host": "127.0.0.1", "port": 0,
        "skip_tokenizer_init": True,
        "tokenizer_worker_num": 1, "detokenizer_worker_num": 1,
        "model_config": {"model": "test-model", "context_len": 4096, "is_generation": True},
    }
    try:
        tm = m.TokenizerManager(json.dumps(config))
        check("PyO3 constructor", True)
        tm.shutdown()
    except Exception as e:
        check("PyO3 constructor", False, str(e))


def test_codegen_deterministic():
    """Re-running codegen must produce identical output."""
    old_path = RUST_CRATE / "src" / "schema" / "mod.rs"
    old = old_path.read_text()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "codegen_rust.py")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    new = old_path.read_text()
    ok = result.returncode == 0 and old == new
    detail = result.stderr if not result.returncode else ("Output changed" if old != new else "")
    check("Codegen deterministic", ok, detail)


def main():
    print("Validating Python integration...\n")

    tests = [
        ("Production import path", test_fixture_generator_production),
        ("Fixture msgspec validity", test_fixtures_valid_msgspec),
        ("Schema snapshot up to date", test_schema_snapshot),
        ("Rust round-trip tests", test_rust_roundtrip),
        ("Scheduler integration", test_scheduler_integration),
        ("PyO3 constructor", test_pyo3_constructor),
        ("Codegen determinism", test_codegen_deterministic),
    ]
    for name, fn in tests:
        try:
            fn()
            print()
        except Exception as e:
            check(name, False, str(e))
            print()

    if _FAILURES:
        print(f"{_FAILURES} test(s) FAILED")
        sys.exit(1)
    print("All integration checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
