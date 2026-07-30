"""AST-based check for silent `except: pass` / `try/except: continue`
in core paths. Replaces the fragile string-scan in spec §3.l (S23b).
"""

import ast
from pathlib import Path

import pytest


CORE_PATHS = [
    "eegcpm/core",
    "eegcpm/modules/preprocessing",
    "eegcpm/modules/features",
    "eegcpm/modules/connectivity",
    "eegcpm/evaluation",
    "eegcpm/pipeline",
]


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _iter_python_files(base: Path):
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _has_silent_continue(tree: ast.AST) -> bool:
    """Walk the AST and detect try/except handlers whose body is
    just `pass` or `continue` (or an equivalent no-op)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                body = handler.body
                if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Continue)):
                    return True
                # Also catch: `except Exception: pass` where pass is
                # the ONLY statement in the handler.
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    return True
    return False


def test_no_silent_continue_in_core_paths():
    """S23b: AST-based check; no try/except: pass in core paths."""
    root = _get_repo_root()
    offenders = []
    for rel in CORE_PATHS:
        base = root / rel
        if not base.exists():
            continue
        for py_file in _iter_python_files(base):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            if _has_silent_continue(tree):
                offenders.append(str(py_file.relative_to(root)))
    assert not offenders, (
        f"Silent try/except: pass or continue found in: {offenders}"
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
