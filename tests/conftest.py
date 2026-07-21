"""Session-level function-call audit.

The mathematical tests provide independent oracles.  This profiler adds a
separate engineering invariant: every source-defined module function and
method must actually execute in the complete suite, so an untested helper
cannot hide behind a passing top-level smoke test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ROOT = (Path(__file__).parents[1] / "src" / "rudolph2023").resolve()
SOURCE_PREFIX = os.path.normcase(str(SOURCE_ROOT))
CALLED: set[str] = set()
RUN_STARTED_UTC = ""


def _profile(frame, event: str, _arg):  # type: ignore[no-untyped-def]
    if event != "call":
        return
    filename = os.path.normcase(frame.f_code.co_filename)
    if not filename.startswith(SOURCE_PREFIX):
        return
    try:
        path = Path(filename).resolve()
        relative = path.relative_to(SOURCE_ROOT)
    except (OSError, ValueError):
        return
    if relative.suffix != ".py":
        return
    module = ".".join(relative.with_suffix("").parts)
    function_name = frame.f_code.co_name
    if "self" in frame.f_locals:
        function_name = f"{type(frame.f_locals['self']).__name__}.{function_name}"
    elif "cls" in frame.f_locals and isinstance(frame.f_locals["cls"], type):
        function_name = f"{frame.f_locals['cls'].__name__}.{function_name}"
    CALLED.add(f"{module}:{function_name}")


def _expected_functions() -> set[str]:
    expected: set[str] = set()
    for path in SOURCE_ROOT.glob("*.py"):
        if path.name == "__main__.py":
            continue
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                expected.add(f"{module}:{node.name}")
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        expected.add(f"{module}:{node.name}.{child.name}")
    return expected


def _tree_sha256(root: Path, pattern: str) -> str:
    records: dict[str, str] = {}
    for path in sorted(root.glob(pattern)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[path.relative_to(root).as_posix()] = digest
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    del session
    global RUN_STARTED_UTC
    CALLED.clear()
    RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
    sys.setprofile(_profile)


def pytest_runtest_setup(item) -> None:  # type: ignore[no-untyped-def]
    # The paper-scale regression repeats already-audited functions thousands
    # of times.  Profiling it adds minutes without increasing entry coverage.
    if item.get_closest_marker("slow") is not None:
        sys.setprofile(None)


def pytest_runtest_teardown(item, nextitem) -> None:  # type: ignore[no-untyped-def]
    del nextitem
    if item.get_closest_marker("slow") is not None:
        sys.setprofile(_profile)


def pytest_sessionfinish(session, exitstatus: int) -> None:  # type: ignore[no-untyped-def]
    del exitstatus
    sys.setprofile(None)
    expected = _expected_functions()
    missing = sorted(expected - CALLED)
    project_root = Path(__file__).parents[1]
    requested = [
        str(value).replace("\\", "/").rstrip("/")
        for value in session.config.getoption("file_or_dir")
    ]
    full_suite = requested in ([], ["tests"])
    record = {
        "schema_version": "rudolph2023-function-call-audit-v2",
        "started_utc": RUN_STARTED_UTC,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "pytest_argv": list(session.config.invocation_params.args),
        "full_suite_requested": full_suite,
        "source_tree_sha256": _tree_sha256(SOURCE_ROOT, "*.py"),
        "test_tree_sha256": _tree_sha256(project_root / "tests", "*.py"),
        "expected_count": len(expected),
        "called_count": len(expected & CALLED),
        "missing": missing,
        "status": "passed" if not missing and session.exitstatus == 0 else "failed",
        "scope_note": (
            "Entry-call coverage only; mathematical correctness is established by "
            "independent oracle, property, regression, and cross-implementation tests. "
            "Slow paper-scale tests are not profiled because their source functions "
            "are already exercised by focused tests."
        ),
    }
    audit_root = project_root / "artifacts" / "function_call_audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    run_id = RUN_STARTED_UTC.replace(":", "").replace("+", "_").replace(".", "_")
    run_output = audit_root / f"audit_{run_id}_{os.getpid()}.json"
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    run_output.write_text(serialized, encoding="utf-8")
    if full_suite:
        canonical = project_root / "artifacts" / "function_call_audit.json"
        canonical.write_text(serialized, encoding="utf-8")
    if full_suite and missing and session.exitstatus == 0:
        print("\nFunction-call audit missing source functions:")
        for name in missing:
            print(f"  - {name}")
        session.exitstatus = 1
