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
