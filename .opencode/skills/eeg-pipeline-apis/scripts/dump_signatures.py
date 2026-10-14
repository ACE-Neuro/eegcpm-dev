#!/usr/bin/env python3
"""Dump signatures of the library APIs used by the EEGCPM preprocessing/source fixes.

Run with the repo venv (from eegcpm-0.1/):
    .venv/bin/python ../.opencode/skills/eeg-pipeline-apis/scripts/dump_signatures.py

Use the output to refresh references/mne-apis.md and references/other-libs.md.
Sourced entry point for the 2026-10-13 audit follow-ups; keep this list in sync with
the functions the remediation plan touches.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from importlib.metadata import version as pkg_version

TARGETS: list[tuple[str, str]] = [
    # --- MNE: forward / inverse / covariance ---
    ("mne", "setup_source_space"),
    ("mne", "make_forward_solution"),
    ("mne", "read_bem_solution"),
    ("mne", "compute_rank"),
    ("mne", "compute_covariance"),
    # NOTE (mne 1.13.2): these moved OUT of the top-level namespace:
    ("mne.minimum_norm", "make_inverse_operator"),
    ("mne.minimum_norm", "apply_inverse"),
    ("mne.minimum_norm", "apply_inverse_epochs"),
    ("mne", "write_source_spaces"),
    ("mne", "read_source_spaces"),
    # --- MNE: labels / ROI extraction ---
    ("mne", "Label"),
    ("mne", "BiHemiLabel"),
    ("mne", "vertex_to_mni"),
    ("mne", "extract_label_time_course"),
    # --- MNE: resolution / crosstalk QC ---
    ("mne.minimum_norm", "make_inverse_resolution_matrix"),
    ("mne.minimum_norm", "get_cross_talk"),
    ("mne.minimum_norm", "get_point_spread"),
    # --- MNE: epochs / reference / annotations / events ---
    ("mne", "BaseEpochs"),
    ("mne", "read_epochs"),
    ("mne", "set_eeg_reference"),
    ("mne", "events_from_annotations"),
    ("mne", "Annotations"),
    ("mne.io", "BaseRaw"),
    # --- eegprep ---
    ("eegprep", "clean_artifacts"),
    ("eegprep", "clean_asr"),
    ("eegprep", "clean_windows"),
    ("eegprep", "eeg_eegrej"),
    ("eegprep", "pop_loadset"),
    ("eegprep", "pop_saveset"),
    # --- pyzaplineplus ---
    ("pyzaplineplus", "zapline_plus"),
    # --- autoreject ---
    ("autoreject", "Ransac"),
    # --- eeglabio ---
    ("eeglabio.raw", "export_set"),
    ("eeglabio.epochs", "export_set"),
]

LIBS = ["mne", "mne-icalabel", "eegprep", "pyzaplineplus", "eeglabio", "autoreject"]


def main() -> int:
    print("# Installed versions")
    for lib in LIBS:
        try:
            print(f"{lib}=={pkg_version(lib)}")
        except Exception as exc:  # pragma: no cover - informational
            print(f"{lib}: NOT INSTALLED ({exc})")

    print("\n# Signatures")
    failures = 0
    for mod_name, attr in TARGETS:
        try:
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, attr)
            print(f"\n## {mod_name}.{attr}\n{inspect.signature(obj)}")
            doc = inspect.getdoc(obj)
            if doc:
                first = doc.strip().splitlines()[0]
                print(f"# doc: {first}")
        except Exception as exc:
            failures += 1
            print(f"\n## {mod_name}.{attr}\n!! UNAVAILABLE: {type(exc).__name__}: {exc}")

    print(f"\n# {failures} target(s) unavailable", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
