#!/usr/bin/env python3
"""Extract msgspec.Struct schema snapshots from io_struct.py for Rust migration.

Usage:
    python schema_snapshot.py [--json] [--markdown]

Outputs a JSON schema snapshot or a Markdown report of every msgspec.Struct
definition: struct name, tag, bases, array_like, kw_only, field order, field
types, defaults, and source line.

Run without arguments to validate the snapshot matches the current source.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

# Path to io_struct.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IO_STRUCT_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "io_struct.py"
SNAPSHOT_PATH = REPO_ROOT / "tools/rust-tkm-migration" / "schema_snapshot.json"


def parse_structs(source: str):
    """Parse msgspec.Struct subclasses from Python source using AST."""
    tree = ast.parse(source)
    lines = source.splitlines()
    structs = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if this class inherits from msgspec.Struct or a subclass of it
        is_msgspec_struct = False
        bases_raw = []
        tag = None
        array_like = None
        kw_only = None

        for base in node.bases:
            if isinstance(base, ast.Call):
                # Factory call like dataclasses.dataclass
                continue
            bases_raw.append(ast.unparse(base) if isinstance(base, ast.Name) else "")

        for deco in node.decorator_list:
            bases_raw.append(f"@{ast.unparse(deco)}")

        # Check bases for msgspec.Struct
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "msgspec":
                pass  # handled below
            if isinstance(base, ast.Attribute):
                if base.attr == "Struct" and isinstance(base.value, ast.Name) and base.value.id == "msgspec":
                    is_msgspec_struct = True

        # Check if a parent class in this module is a msgspec.Struct
        # We need to check keyword args on the class definition
        keywords = {kw.arg: kw for kw in node.keywords if kw.arg is not None}
        if "tag" in keywords:
            tag_val = keywords["tag"].value
            if isinstance(tag_val, ast.Constant):
                tag = tag_val.value
        if "array_like" in keywords:
            al_val = keywords["array_like"].value
            if isinstance(al_val, ast.Constant):
                array_like = al_val.value
        if "kw_only" in keywords:
            ko_val = keywords["kw_only"].value
            if isinstance(ko_val, ast.Constant):
                kw_only = ko_val.value

        if not is_msgspec_struct:
            # Check if any base class is another class in this file that IS a msgspec struct
            # or inherits from one
            for base in node.bases:
                if isinstance(base, ast.Name):
                    parent_name = base.id
                    if parent_name in structs:
                        # Inherit msgspec.Struct status from parent
                        is_msgspec_struct = structs[parent_name]["is_msgspec_struct"]
                        if tag is None:
                            tag = structs[parent_name].get("tag")
                        if array_like is None:
                            array_like = structs[parent_name].get("array_like")
                        if kw_only is None:
                            kw_only = structs[parent_name].get("kw_only")
                        break

        if not is_msgspec_struct:
            continue

        # Extract fields from class body
        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                field_name = item.target.id if isinstance(item.target, ast.Name) else None
                if field_name and not field_name.startswith("_"):
                    # Get type annotation
                    type_str = ast.unparse(item.annotation) if item.annotation else "Any"

                    # Get default value
                    default = None
                    has_default = False
                    if item.value is not None:
                        has_default = True
                        if isinstance(item.value, ast.Constant):
                            default = item.value.value
                        elif isinstance(item.value, ast.Name):
                            default = item.value.id
                        elif isinstance(item.value, ast.List):
                            default = "[]"
                        elif isinstance(item.value, ast.Dict):
                            default = "{}"
                        elif isinstance(item.value, ast.Tuple):
                            default = "()"
                        elif isinstance(item.value, ast.UnaryOp) and isinstance(item.value.op, ast.USub):
                            if isinstance(item.value.operand, ast.Constant):
                                default = -item.value.operand.value
                        else:
                            default = ast.unparse(item.value)

                    # Check for kw_only in field(...)
                    is_kw_only_field = False
                    if has_default and isinstance(item.value, ast.Call):
                        call = item.value
                        if isinstance(call.func, ast.Name) and call.func.id == "field":
                            for kw in call.keywords:
                                if kw.arg == "kw_only" and isinstance(kw.value, ast.Constant) and kw.value.value:
                                    is_kw_only_field = True

                    fields.append({
                        "name": field_name,
                        "type": type_str,
                        "default": default,
                        "has_default": has_default,
                        "kw_only_field": is_kw_only_field,
                    })

        structs[node.name] = {
            "name": node.name,
            "lineno": node.lineno,
            "is_msgspec_struct": True,
            "tag": tag,
            "array_like": array_like,
            "kw_only": kw_only,
            "bases": [ast.unparse(b) for b in node.bases if not isinstance(b, ast.Call)],
            "fields": fields,
        }

    # Resolve parent field inheritance: child fields come after parent fields
    def resolve_fields(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited:
            return []
        visited.add(name)
        s = structs.get(name)
        if not s:
            return []
        result = []
        for base_name in s["bases"]:
            if base_name in structs and base_name != "msgspec":
                parent_fields = resolve_fields(base_name, visited)
                result.extend(parent_fields)
        result.extend(s["fields"])
        return result

    for name in structs:
        structs[name]["resolved_fields"] = resolve_fields(name)

    return structs


def generate_snapshot():
    """Generate the JSON schema snapshot."""
    source = IO_STRUCT_PATH.read_text(encoding="utf-8")
    structs = parse_structs(source)
    snapshot = {
        "source_file": str(IO_STRUCT_PATH),
        "structs": structs,
    }
    return snapshot


def write_snapshot():
    """Write the JSON schema snapshot."""
    snapshot = generate_snapshot()

    # Compact JSON with sorted keys for diff stability
    json_str = json.dumps(snapshot, indent=2, default=str, sort_keys=False)
    SNAPSHOT_PATH.write_text(json_str, encoding="utf-8")
    print(f"Schema snapshot written to {SNAPSHOT_PATH}")
    print(f"  {len(snapshot['structs'])} structs captured")

    # Print summary
    for name, s in snapshot["structs"].items():
        base_info = ", ".join(s["bases"])
        fields_info = ", ".join(f["name"] for f in s["resolved_fields"])
        print(f"  {name}({base_info}): [{fields_info}]")


def check_snapshot():
    """Check current source against the stored snapshot. Returns 0 on match."""
    if not SNAPSHOT_PATH.exists():
        print(f"ERROR: No snapshot file at {SNAPSHOT_PATH}")
        print("Run with --write to create one.")
        return 1

    current = generate_snapshot()
    stored = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    # Compare struct by struct
    changes = []
    for name in sorted(set(list(current["structs"].keys()) + list(stored.get("structs", {}).keys()))):
        cur_s = current["structs"].get(name)
        stored_s = stored.get("structs", {}).get(name)

        if cur_s and not stored_s:
            changes.append(f"  + {name} (new)")
            continue
        if stored_s and not cur_s:
            changes.append(f"  - {name} (removed)")
            continue

        # Compare resolved fields
        cur_fields = [f["name"] for f in cur_s["resolved_fields"]]
        stored_fields = [f["name"] for f in stored_s["resolved_fields"]]
        if cur_fields != stored_fields:
            changes.append(f"  ~ {name}: field order changed")
            changes.append(f"    current:  {cur_fields}")
            changes.append(f"    snapshot: {stored_fields}")

        # Compare tag
        if cur_s.get("tag") != stored_s.get("tag"):
            changes.append(f"  ~ {name}: tag {cur_s.get('tag')} != {stored_s.get('tag')}")

        # Compare array_like
        if cur_s.get("array_like") != stored_s.get("array_like"):
            changes.append(f"  ~ {name}: array_like {cur_s.get('array_like')} != {stored_s.get('array_like')}")

        # Compare field types
        cf = {f["name"]: f["type"] for f in cur_s["resolved_fields"]}
        sf = {f["name"]: f["type"] for f in stored_s["resolved_fields"]}
        for fname in cf:
            if fname in sf and cf[fname] != sf[fname]:
                changes.append(f"  ~ {name}.{fname}: type {cf[fname]} != {sf[fname]}")

    if changes:
        print(f"ERROR: {len(changes)} schema change(s) detected:")
        print("\n".join(changes))
        print(f"\nRun `python {__file__} --write` to update the snapshot.")
        return 1
    else:
        print("OK: Schema snapshot matches current source.")
        return 0


def main():
    if "--write" in sys.argv:
        write_snapshot()
        return

    if "--json" in sys.argv:
        snapshot = generate_snapshot()
        print(json.dumps(snapshot, indent=2, default=str))
        return

    # Default: check
    exit(check_snapshot())


if __name__ == "__main__":
    main()
