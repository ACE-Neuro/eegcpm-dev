"""
Config-hash machinery (per S10 in the spec).

The 5 golden configs are pinned at content-hash. Any change to a
golden config triggers the CI golden-config-hash test. The
parity-run test asserts that each stage's recorded `config_hash`
(in the STATE_SCHEMA) equals the corresponding GOLDEN_HASHES entry
on BOTH local and HPC.

P1-F7 (B6): the *preprocessing* stage is no longer pinned to the raw
YAML bytes — the YAML only selects the code-defined ``hbn_langer``
chain, so pinning bytes proved nothing about the code that actually
ran. It is pinned to the **resolved** config instead:
``resolved_config_hash(PreprocessingPipeline('hbn_langer'))`` hashes
every step's full constructor parameters plus the library versions,
and the literal value lives in ``GOLDEN_RESOLVED_HASHES`` (computed
once in the pinned venv). ``load_golden_hashes()`` keeps returning all
5 ``GOLDEN_CONFIG_NAMES`` stage keys for parity-harness compatibility:
the ``'preprocessing'`` key maps to the resolved hash, the other 4
stages keep their content-hash behaviour (P4 scope).

This module exposes:
  - GOLDEN_CONFIG_PATHS: the 5 paths the parity run names explicitly
  - GOLDEN_RESOLVED_HASHES: literal resolved-config hash per named chain
  - GOLDEN_RESOLVED_LIBRARY_VERSIONS: the library versions the literal hash
    was computed with (diagnostic for the literal-hash test)
  - GOLDEN_CONFIG_LIBRARIES: the libraries the resolved hash covers
  - hash_config(path) -> str: SHA-256 of the file contents
  - library_versions() -> dict: installed versions ('not-installed' fallback)
  - resolved_config_hash(pipeline) -> str: hash of the resolved config
  - load_golden_hashes() -> dict: the 5 pinned stage hashes
  - check_golden_config_hashes(local_dir, hpc_dir) -> None
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# The 5 golden configs (per S10; mirrored in §3.s). These are the
# ONLY configs the parity run is allowed to use.
GOLDEN_CONFIG_PATHS: Dict[str, str] = {
    "preprocessing": "examples/configs/preprocessing/hbn_langer.yaml",
    "drowsiness":    "examples/configs/features/drowsiness_metrics.yaml",
    "specparam":      "examples/configs/features/specparam_resting_ec.yaml",
    "connectivity":  "examples/configs/features/connectivity_resting_ec.yaml",
    "prediction":    "examples/configs/prediction/cpm_d_factor.yaml",
}


#: Libraries whose installed versions are part of the resolved-config hash
#: (P1-F7: the same parameters on a different MNE build are not the same
#: pipeline). ``'not-installed'`` is recorded for absent packages instead of
#: failing, so the hash can be computed in a minimal environment too.
GOLDEN_CONFIG_LIBRARIES: List[str] = [
    "eegcpm",
    "mne",
    "numpy",
    "scipy",
    "eegprep",
    "pyzaplineplus",
    "eeglabio",
    "autoreject",
    "mne-icalabel",
]


#: Literal resolved-config hashes, computed once in the pinned venv
#: (mne 1.13.2, eegprep 0.4.0, pyzaplineplus 0.1.0, eeglabio 0.1.4,
#: autoreject 0.5.1, mne-icalabel 0.10.0, numpy 2.5.4, scipy 1.18.1).
#: Re-pin after any dependency bump or chain change — the golden-hash test
#: fails loudly instead of letting the parity run drift. The re-pin command
#: (one line, run from ``eegcpm-0.1``; prints the new hash to paste below):
#:
#:   .venv/bin/python -c "from eegcpm.core.config_hash import resolved_config_hash; from eegcpm.modules.preprocessing.pipeline_builder import PreprocessingPipeline; print(resolved_config_hash(PreprocessingPipeline('hbn_langer')))"
#:
#: If the hash changed, ``GOLDEN_RESOLVED_LIBRARY_VERSIONS`` below must be
#: updated too (the literal-hash test prints the diff when they disagree).
#:
#: Computed 2026-10-16 (B6) after Checkpoint A decision (a)
#: (``RobustPCAStep`` removed from ``build_hbn_langer_chain()``; 9 steps) and
#: after the P1-F7 ``get_config`` completion, the post-reference Cz check and
#: the B4-D2 detector changes. Re-pinned 2026-10-16 (B6 rev2) after NEW-1
#: (EEG+EOG filter picks), NEW-2 (band-passed blink channel), NEW-3
#: (log-Pearson kurtosis) and NEW-4 (detector/ASR constants in
#: ``get_config``); re-pinned again after the B6 scoped re-audit fix (the ASR
#: wrapper now reports the fixed-first-60 s path it actually runs instead of
#: the data-driven constants); re-pinned 2026-10-16 (B7) after the
#: ``EOGRegressionStep`` band-limited-regressor guard (version 5.0,
#: ``allow_unfiltered_regressors`` + the guard constants in ``get_config``,
#: carried B6 item); re-pinned 2026-10-16 (**B7 rev2 — Opus final-audit
#: fixes**) after the ASR wrapper version bump 2.1 → 2.2 (fix 7: the
#: ASR-excluded interpolated channels are re-interpolated from the repaired
#: data) and the B7-N2 output blow-up guard in ``ASRStep`` 2.0 → 2.1. Previous
#: values were ``ca91b7f280d62804a63d5bc18aa4acfb18a1e80a193a0cedb64f343b3d75cbb7``,
#: ``f3a40cd6ab825513b840ffed333f1de0d425c51f584023caa8b685e28613f378``,
#: ``731756264f5c1c79783acba79dee1913b491bd5e02801c3b54105f254edef975`` and
#: ``72c394e7801d894edf7b39dc26e543e401b2d023f67b445b83d8a5460719b3dd``.
GOLDEN_RESOLVED_HASHES: Dict[str, str] = {
    "hbn_langer": "1ef8780d87f228609e8c55c233426e51778b5e2a0a8648ba42318e3433b8d421",
}


#: The installed library versions the golden resolved hash was computed with
#: (the ``library_versions()`` payload). Not hashed directly — it exists so a
#: failing literal-hash test can print the exact version diff (NEW-5, B6
#: rev2) instead of only "hash mismatch".
GOLDEN_RESOLVED_LIBRARY_VERSIONS: Dict[str, str] = {
    "eegcpm": "0.1.0",
    "mne": "1.13.2",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "eegprep": "0.4.0",
    "pyzaplineplus": "0.1.0",
    "eeglabio": "0.1.4",
    "autoreject": "0.5.1",
    "mne-icalabel": "0.10.0",
}


def hash_config(path: Path) -> str:
    """SHA-256 of the file contents (bytes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_versions() -> Dict[str, str]:
    """Installed versions of the pipeline libraries (P1-F7).

    Uses ``importlib.metadata.version``; a package that is not installed is
    reported as ``'not-installed'`` (never raises) so the resolved hash is
    computable in any environment and changes when a dependency is added.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: Dict[str, str] = {}
    for package in GOLDEN_CONFIG_LIBRARIES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolved_config_hash(pipeline: Any) -> str:
    """SHA-256 of a pipeline's **resolved** configuration (P1-F7).

    ``pipeline`` is anything exposing ``get_config()`` (e.g.
    ``PreprocessingPipeline``): the hash covers the resolved step
    parameters — not the raw YAML bytes, which can differ while resolving
    to the same pipeline — plus the installed library versions, so a
    dependency bump is visible in the recorded hash.
    """
    payload = {
        "pipeline": pipeline.get_config(),
        "libraries": library_versions(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_golden_hashes(
    root: Optional[Path] = None,
) -> Dict[str, str]:
    """The pinned hash of each golden stage (all 5 ``GOLDEN_CONFIG_NAMES``).

    The ``'preprocessing'`` stage is the **resolved-config** hash of the
    ``hbn_langer`` chain (``GOLDEN_RESOLVED_HASHES``); the other four stages
    keep the SHA-256 of their config file contents. ``root`` only affects
    the file-content stages (the resolved hash is environment-independent
    given the pinned libraries).
    """
    root = Path(root) if root is not None else _repo_root()
    hashes: Dict[str, str] = {}
    for stage, rel in GOLDEN_CONFIG_PATHS.items():
        if stage == "preprocessing":
            hashes[stage] = GOLDEN_RESOLVED_HASHES["hbn_langer"]
        else:
            hashes[stage] = hash_config(root / rel)
    return hashes


def _repo_root() -> Path:
    """Return the eegcpm-0.1 package root."""
    return Path(__file__).resolve().parent.parent.parent


def check_golden_config_hashes(
    local_state: Dict[str, Dict[str, str]],
    hpc_state: Dict[str, Dict[str, str]],
    expected_hashes: Optional[Dict[str, str]] = None,
) -> None:
    """S10: for every stage, the recorded config_hash on BOTH local
    and HPC must equal the corresponding GOLDEN_HASHES entry. This
    closes the live ENG-006 gap: the parity run is bound to the
    golden pipeline, not to a different one.
    """
    if expected_hashes is None:
        expected_hashes = load_golden_hashes()
    for stage, expected in expected_hashes.items():
        local_hash = local_state.get(stage, {}).get("config_hash")
        hpc_hash = hpc_state.get(stage, {}).get("config_hash")
        if local_hash != expected:
            raise ValueError(
                f"Local {stage} config_hash ({local_hash}) != golden "
                f"({expected}); parity run used a non-golden config."
            )
        if hpc_hash != expected:
            raise ValueError(
                f"HPC {stage} config_hash ({hpc_hash}) != golden "
                f"({expected}); parity run used a non-golden config."
            )
        if local_hash != hpc_hash:
            raise ValueError(
                f"{stage}: local ({local_hash}) != hpc ({hpc_hash})"
            )
