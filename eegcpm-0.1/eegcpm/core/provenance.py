"""Provenance and atomic-write helpers for on-disk pipeline artifacts.

Introduced by B5 (plan ``plan/261013-2240-p1p2-remediation.md`` §B5, tasks 5 and 8,
fixing P2-F10 and B4-K5):

- **Atomic writes.** Every artifact writer in :mod:`eegcpm.modules.source` goes
  through one of the ``atomic_*`` helpers: the data is written to a sibling temp
  file and ``os.replace``d onto the target (same directory → same filesystem →
  atomic rename on POSIX). An IO failure therefore never leaves a partial
  artifact behind a ``success=False`` result, and the temp file is removed on the
  way out.
- **Provenance pairing.** ``PROVENANCE_SUFFIX`` is the single naming rule that
  links an on-disk archive to the run that produced it:
  ``sub-01_roi_tc.npz`` ↔ ``sub-01_provenance.json``. Consumers use
  :func:`verify_archive_provenance` to refuse an archive whose provenance (and
  therefore config hash) is missing or does not match the file on disk, so a
  stale or partially written archive from an older run cannot be consumed
  silently.
- **Hashing.** ``sha256_file`` streams in 1 MiB chunks (large FIF/npz artifacts)
  and ``sha256_json`` hashes a resolved config with ``sort_keys=True`` so the
  same parameters always yield the same hash.

No MNE import here on purpose: the feature consumer imports this module on a
light path (``eegcpm.modules.features.source_erp_features``).
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np

#: Naming rule for provenance sidecars (see module docstring).
PROVENANCE_SUFFIX = "_provenance.json"
#: Suffix of the evoked ROI archive written by the source stage.
ARCHIVE_SUFFIX = "_roi_tc.npz"
TRIALS_ARCHIVE_SUFFIX = "_roi_tc_trials.npz"
#: Streamed hashing chunk size (plan §B5 task 5).
HASH_CHUNK_SIZE = 1024 * 1024


def package_version(distribution: str) -> str:
    """Installed version of ``distribution`` (``'not-installed'`` if absent)."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Union[str, Path], chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Streamed sha256 of a file (1 MiB chunks by default)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """sha256 of the canonical JSON encoding of ``payload``.

    Matches the plan's ``config_hash`` definition: sha256 over
    ``json.dumps(payload, sort_keys=True)`` — resolved parameters, not raw YAML
    bytes. Raises ``TypeError`` for non-JSON-native values (fail loudly instead
    of writing an unattributable hash).
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays/paths to JSON-native objects.

    ``None`` is preserved (it carries meaning in provenance: "not measured"),
    unlike the old stringifying sanitiser.
    """
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------


def atomic_save(
    path: Union[str, Path],
    writer: Callable[[Path], None],
    *,
    tmp_name: Optional[str] = None,
) -> Path:
    """Write ``path`` via ``writer(tmp)`` + atomic rename.

    Parameters
    ----------
    path : path-like
        Final artifact path.
    writer : callable
        Called with the temp path; must write the artifact (and may write
        sibling files derived from the temp stem, e.g. ``{tmp}-lh.stc``).
    tmp_name : str | None
        Temp file name (in the target directory). Defaults to
        ``.{path.name}.tmp``. Pass an explicit name when the writer validates the
        file name (``mne`` requires ``-cov.fif`` / ``-inv.fif`` endings).

    Any files whose name starts with the temp stem are removed if the writer (or
    the rename) raises, so a failed write leaves the target untouched.
    """
    path = Path(path)
    tmp = path.with_name(tmp_name) if tmp_name else path.with_name(f".{path.name}.tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        for leftover in sorted(path.parent.glob(f"{glob.escape(tmp.name)}*")):
            if leftover.is_file():
                leftover.unlink()
        raise
    return path


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """Atomically write raw bytes."""
    return atomic_save(path, lambda tmp: tmp.write_bytes(data))


def atomic_write_text(path: Union[str, Path], text: str) -> Path:
    """Atomically write text (UTF-8)."""
    return atomic_save(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def atomic_write_json(path: Union[str, Path], payload: Any) -> Path:
    """Atomically write a JSON document (numpy-aware, strict JSON).

    ``allow_nan=False``: a non-finite value is a bug in the caller's payload and
    must fail the run rather than write ``Infinity``/``NaN`` into provenance.
    """
    text = json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    return atomic_write_text(path, text)


def atomic_write_npz(
    path: Union[str, Path], arrays: Mapping[str, Any], *, compressed: bool = False
) -> Path:
    """Atomically write an ``.npz`` archive (temp name keeps the ``.npz`` suffix)."""
    path = Path(path)
    # Keep the ``.npz`` suffix on the temp name: ``np.savez`` appends it when the
    # name does not already carry it, which would break the rename target.
    tmp_name = f".{path.stem}.tmp.npz"
    save = np.savez_compressed if compressed else np.savez
    return atomic_save(path, lambda tmp: save(tmp, **arrays), tmp_name=tmp_name)


# ---------------------------------------------------------------------------
# Provenance pairing / verification
# ---------------------------------------------------------------------------


def provenance_path_for_archive(archive_path: Union[str, Path]) -> Path:
    """Map an archive to its provenance sidecar (same directory).

    Both ``sub-01_roi_tc.npz`` (evoked) and ``sub-01_roi_tc_trials.npz``
    (trials) belong to the same run, so both map to
    ``sub-01_provenance.json`` (B5-A1).
    """
    archive_path = Path(archive_path)
    name = archive_path.name
    if name.endswith(TRIALS_ARCHIVE_SUFFIX):
        name = name[: -len(TRIALS_ARCHIVE_SUFFIX)] + PROVENANCE_SUFFIX
    elif name.endswith(ARCHIVE_SUFFIX):
        name = name[: -len(ARCHIVE_SUFFIX)] + PROVENANCE_SUFFIX
    else:
        name = f"{archive_path.stem}{PROVENANCE_SUFFIX}"
    return archive_path.parent / name


def load_provenance(archive_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and minimally validate the provenance belonging to ``archive_path``.

    Raises
    ------
    ValueError
        If the provenance file is missing, unreadable, not a JSON object, or has
        no non-empty ``config_hash``. A ``*_roi_tc.npz`` without a usable
        provenance must not be consumed (B5 task 8): it cannot be attributed to a
        run or a config.
    """
    archive_path = Path(archive_path)
    provenance_path = provenance_path_for_archive(archive_path)
    if not provenance_path.is_file():
        raise ValueError(
            f"missing provenance for {archive_path.name}: expected "
            f"{provenance_path}; re-run the source stage (an archive without "
            "provenance cannot be attributed to a run/config)"
        )
    try:
        with open(provenance_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unreadable provenance {provenance_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"provenance {provenance_path} is not a JSON object")
    config_hash = payload.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise ValueError(
            f"provenance {provenance_path} has no 'config_hash'; the run that wrote "
            "this archive cannot be identified"
        )
    return payload


def verify_archive_provenance(
    archive_path: Union[str, Path], provenance: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Require the provenance of ``archive_path`` and cross-check the file hash.

    The provenance is loaded (or supplied by the caller) and **must** list the
    archive in its ``artifacts`` map with a non-empty sha256 that matches the
    file — a missing map, an unlisted archive, a partially written file or an
    edited one is rejected (audit B5-N2: the check must not be skippable by
    omitting the map).
    """
    archive_path = Path(archive_path)
    provenance = (
        load_provenance(archive_path) if provenance is None else provenance
    )
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(
            f"provenance for {archive_path.name} has no 'artifacts' hash map; "
            "the archive cannot be attributed to this run"
        )
    recorded = artifacts.get(archive_path.name)
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(
            f"archive {archive_path.name} is not listed in its provenance "
            "'artifacts' map; the file cannot be attributed to this run"
        )
    actual = sha256_file(archive_path)
    if actual != recorded:
        raise ValueError(
            f"archive {archive_path.name} does not match its provenance "
            f"(sha256 {actual} != recorded {recorded}); the file was "
            "modified or partially written"
        )
    return provenance


def stale_output_patterns(subject_id: str) -> List[str]:
    """Glob patterns of the source stage's own outputs for one subject.

    Used to delete a previous run's artifacts before writing new ones so a re-run
    can never leave a mix of old and new files in the output directory (B4-K5).

    Includes the **hidden atomic-writer temp files** (B5-A5): a SIGKILL between
    ``writer(tmp)`` and ``os.replace(tmp, path)`` leaves ``.{subject}_...tmp*``
    behind, and nothing else would ever clean them (the process that knew the
    name is gone). They are scoped to this subject's prefix, so a concurrent
    writer for another subject is never touched. Single-writer assumption: two
    processes writing the *same* subject concurrently are not supported — the
    stale sweep would delete the other writer's in-flight temp file.
    """
    return [
        f"{subject_id}_roi_tc.npz",
        f"{subject_id}_roi_tc_trials.npz",
        f"{subject_id}_provenance.json",
        f"{subject_id}_source_config.yaml",
        f"{subject_id}-cov.fif",
        f"{subject_id}-inv.fif",
        f"{subject_id}_*_stc-lh.stc",
        f"{subject_id}_*_stc-rh.stc",
        f"{subject_id}_*_source_qc.html",
        f"{subject_id}_crosstalk.npz",
        # Hidden temp leftovers of this module's atomic writers (B5-A5):
        # `.{name}.tmp`, `.{subject}_{cond}_stc.tmp*`, `.{subject}_roi_tc.tmp.npz`,
        # `.{subject}-cov.tmp-cov.fif`, `.{subject}-inv.tmp-inv.fif`, ...
        f".{subject_id}_*",
        f".{subject_id}-*",
    ]
