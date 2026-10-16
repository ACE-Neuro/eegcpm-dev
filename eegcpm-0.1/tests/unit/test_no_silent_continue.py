"""AST-based check for silent ``except: pass`` / ``try/except: continue`` in core paths.

Replaces the fragile string-scan in spec §3.l (S23b) and fixes audit P1-F11:
the original gate resolved the repo root one level too high (``parents[1]`` =
``tests/``), so it scanned **zero** files and could never fail. The gate now
resolves the package root (``parents[2]`` = ``eegcpm-0.1/``), asserts it
actually scanned every core path (>= 1 file per path, > 50 in total), covers
``modules/source`` and ``modules/qc`` (P2-F14/F16), and skips the ``_legacy``
tree — with a companion test that no live module imports from ``_legacy``.

Rule (deliberately blunt, no allow-list): an ``except`` handler whose body is
only ``pass``, ``continue`` or ``print(...)`` calls is a silent swallow. Every
such site must re-raise, narrow the exception and *record* what happened
(metadata / QC note / logger), or be restructured so no handler is needed. A
bare ``print`` is not a record: it disappears in batch runs, so the B6-K9
legacy handlers (``print`` + ``drop_channels``) were invisible to the earlier
pass/continue-only rule.

B7-A1 widened the scan to ``eegcpm/cli``, ``eegcpm/workflow`` and
``eegcpm/data`` (9 further offenders: a per-file ``continue`` in the
connectivity auto-detect loop, three print-only CLI handlers, the SQLite
``ALTER TABLE`` migration idiom and three ``data`` conversion handlers). All
were fixed with no allow-list: the batch loop reports every config it dropped,
the print-only handlers are narrowed and logged, the migration helper tolerates
only the duplicate-column ``OperationalError``, and the ``data`` handlers log
what they skipped.
"""

import ast
from pathlib import Path

import pytest


CORE_PATHS = [
    "eegcpm/core",
    "eegcpm/modules/preprocessing",
    "eegcpm/modules/features",
    "eegcpm/modules/connectivity",
    "eegcpm/modules/source",
    "eegcpm/modules/qc",
    "eegcpm/evaluation",
    "eegcpm/pipeline",
    # B7-A1: the CLI, the workflow state store and the dataset/ROI loaders were
    # outside the gate; they contained 9 live offenders (fixed, no allow-list).
    "eegcpm/cli",
    "eegcpm/workflow",
    "eegcpm/data",
]


def _get_repo_root() -> Path:
    """Package root (``eegcpm-0.1/``) — ``tests/unit/<file>`` -> ``parents[2]``."""
    return Path(__file__).resolve().parents[2]


def _iter_python_files(base: Path):
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if "_legacy" in path.parts:
            continue
        yield path


def _is_print_only(node: ast.AST) -> bool:
    """True for a bare ``print(...)`` expression statement."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
    )


def _has_silent_continue(tree: ast.AST) -> bool:
    """Walk the AST and detect try/except handlers whose body is
    just `pass`, `continue` or `print(...)` (an unrecorded swallow)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                body = handler.body
                if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Continue)):
                    return True
                # `print(...)` (optionally followed by pass/continue) is not a
                # record: it vanishes in batch runs (B6-K9).
                meaningful = [
                    stmt for stmt in body
                    if not isinstance(stmt, (ast.Pass, ast.Continue))
                ]
                if meaningful and all(_is_print_only(stmt) for stmt in meaningful):
                    return True
    return False


def test_no_silent_continue_in_core_paths():
    """S23b/P1-F11: AST-based check; no try/except: pass in core paths."""
    root = _get_repo_root()
    offenders = []
    scanned_total = 0
    scanned_per_path = {}
    for rel in CORE_PATHS:
        base = root / rel
        assert base.exists(), (
            f"CORE_PATHS entry {rel!r} does not exist under {root}; the gate "
            "would silently scan nothing (P1-F11)"
        )
        n_scanned = 0
        for py_file in _iter_python_files(base):
            n_scanned += 1
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            if _has_silent_continue(tree):
                offenders.append(str(py_file.relative_to(root)))
        scanned_per_path[rel] = n_scanned

    # The gate must be able to fail: assert it really looked at code (P1-F11).
    assert all(n >= 1 for n in scanned_per_path.values()), (
        f"some CORE_PATHS were scanned with 0 files: {scanned_per_path}"
    )
    scanned_total = sum(scanned_per_path.values())
    # Report the scan count (plan §B7 validation reads it): printed for `-s` /
    # failure output and embedded in the assertion messages below, so a
    # silently-empty scan is visible in CI logs (B7 rev2 fix 9).
    scan_report = (
        f"no-silent-except gate: scanned {scanned_total} files across "
        f"{len(scanned_per_path)} core paths, {len(offenders)} offenders "
        f"({scanned_per_path})"
    )
    print(scan_report)
    assert scanned_total > 50, (
        f"gate scanned only {scanned_total} files: {scanned_per_path}"
    )
    assert not offenders, (
        f"Silent try/except: pass or continue found in: {offenders} "
        f"[{scan_report}]"
    )


def test_no_module_imports_legacy():
    """The ``_legacy`` tree is excluded from the gate — nothing may import it.

    If live code imported ``_legacy``, excluding that tree would hide real
    offenders from the gate (P1-F11). ``_legacy`` files may import each other.
    """
    root = _get_repo_root()
    package_root = root / "eegcpm"
    offenders = []
    for py_file in sorted(package_root.rglob("*.py")):
        if "__pycache__" in py_file.parts or "_legacy" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if "_legacy" in module:
                    offenders.append(f"{py_file.relative_to(root)}: {module}")
    assert not offenders, (
        "live module(s) import from _legacy (excluded from the gate): "
        f"{offenders}"
    )


def test_ast_check_can_detect_violation(tmp_path):
    """Reachability test: feed the AST checker a file with a
    silent except and assert it fires."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "try:\n"
        "    do_something()\n"
        "except Exception:\n"
        "    pass\n"
    )
    tree = ast.parse(bad.read_text())
    assert _has_silent_continue(tree) is True


def test_ast_check_detects_bare_continue(tmp_path):
    """A handler whose body is only ``continue`` is silent too."""
    bad = tmp_path / "bad_continue.py"
    bad.write_text(
        "for x in items:\n"
        "    try:\n"
        "        do_something(x)\n"
        "    except ValueError:\n"
        "        continue\n"
    )
    tree = ast.parse(bad.read_text())
    assert _has_silent_continue(tree) is True


def test_ast_check_does_not_flag_named_handler_with_real_body(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        "try:\n"
        "    do_something()\n"
        "except ValueError as e:\n"
        "    logger.error('real handling', e)\n"
        "    raise\n"
    )
    tree = ast.parse(good.read_text())
    assert _has_silent_continue(tree) is False


def test_gate_scans_a_known_offender_file(tmp_path):
    """The scan loop (not just the AST helper) must be able to find an offender."""
    root = tmp_path / "pkg"
    core = root / "eegcpm" / "core"
    core.mkdir(parents=True)
    (core / "offender.py").write_text(
        "try:\n    risky()\nexcept Exception:\n    pass\n"
    )
    (core / "clean.py").write_text("x = 1\n")
    offenders = []
    scanned = 0
    for py_file in _iter_python_files(core):
        scanned += 1
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        if _has_silent_continue(tree):
            offenders.append(py_file.name)
    assert scanned == 2
    assert offenders == ["offender.py"]


@pytest.mark.parametrize("rel", CORE_PATHS)
def test_core_path_exists(rel):
    """Every CORE_PATH must resolve inside the package (P1-F11)."""
    assert (_get_repo_root() / rel).is_dir(), f"{rel} is not a directory"


def test_ast_check_detects_print_only_handler(tmp_path):
    """A handler that only prints and continues is a silent swallow (B6-K9)."""
    bad = tmp_path / "bad_print.py"
    bad.write_text(
        "try:\n"
        "    risky()\n"
        "except Exception as e:\n"
        "    print(f'warning: {e}')\n"
        "    continue\n"
    )
    tree = ast.parse(bad.read_text())
    assert _has_silent_continue(tree) is True


def test_ast_check_does_not_flag_logged_handler(tmp_path):
    """A logged/narrowed handler is a record, not a swallow."""
    good = tmp_path / "good_log.py"
    good.write_text(
        "try:\n"
        "    risky()\n"
        "except ValueError as exc:\n"
        "    logger.warning('failed: %s', exc)\n"
        "    continue\n"
    )
    tree = ast.parse(good.read_text())
    assert _has_silent_continue(tree) is False
