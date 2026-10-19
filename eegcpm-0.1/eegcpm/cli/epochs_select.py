"""CLI command: select / sort / split trials from extracted epochs (Pass 2).

``eegcpm epochs-select`` consumes epochs produced by the ``epochs`` stage,
applies a named trial selection (condition / response / accuracy / RT /
pandas query filters, an optional metadata sort, and an optional split or
task-config binning), and writes the result to a *selection* subdirectory::

    derivatives/epochs/{preprocessing}/{task}/selections/{name}/sub-{s}/ses-{y}/

The selection never overwrites the source epochs: the source stage guarantees
exactly one ``*_epo.fif`` per session directory, and every selection lives in
its own ``selections/<name>/`` subtree (name restricted to
``[a-zA-Z0-9_-]+``).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.core.task_config import SelectionRecord, TaskConfig, TrialSorter

#: A selection name becomes a directory name, so it is restricted to a safe
#: alphabet (no separators, no traversal, no whitespace).
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Tokens accepted by ``--rt`` for an open bound.
_NONE_TOKENS = {"none", "null", "nan", "na", "-"}

#: Characters replaced in derived file names (bin labels etc.).
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")

#: Files a selection owns inside its output directory. ``--force`` replaces
#: exactly these (the directory is fully derived, so a partial replacement
#: would leave stale split parts/bins from the previous selection).
_OUTPUT_PATTERNS = ("*_epo.fif", "*_ave.fif", "selection.json")


# ---------------------------------------------------------------------------
# argument parsing helpers
# ---------------------------------------------------------------------------


def validate_selection_name(name: str) -> str:
    """Return ``name`` when it is a legal selection name, else raise."""
    if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            f"Invalid selection name {name!r}: only letters, digits, '-' and "
            "'_' are allowed (it becomes a directory name)."
        )
    return name


def parse_rt_bound(token: Any) -> Optional[float]:
    """Parse one ``--rt`` bound; ``None`` means an open bound."""
    if token is None:
        return None
    if isinstance(token, (int, float)):
        return float(token)
    text = str(token).strip()
    if text.lower() in _NONE_TOKENS:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid RT bound {token!r}: use a number, or one of "
            f"{sorted(_NONE_TOKENS)} for an open bound."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"Invalid RT bound {token!r}: must be finite")
    return value


def parse_rt_bounds(rt: Optional[Sequence[Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Parse ``--rt LO HI`` into a ``(lo, hi)`` tuple."""
    if rt is None:
        return (None, None)
    if isinstance(rt, (str, bytes)) or len(rt) != 2:
        raise ValueError(
            f"--rt takes two values (LO HI), got {rt!r}. Use 'none' for an "
            "open bound."
        )
    lo, hi = parse_rt_bound(rt[0]), parse_rt_bound(rt[1])
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"--rt lower bound {lo} is greater than upper bound {hi}")
    return (lo, hi)


def _safe_label(label: str) -> str:
    """Make a label safe for use inside a file name."""
    safe = _UNSAFE_NAME.sub("_", str(label)).strip("_")
    return safe or "unnamed"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def find_source_epochs(session_dir: Path) -> Path:
    """Return the single ``*_epo.fif`` in ``session_dir``.

    Mirrors the source stage's rule: zero matches is a "not run yet" error,
    more than one is ambiguous (picking one arbitrarily would silently consume
    a stale output). ``*_ave.fif`` files do not match this glob.
    """
    matches = sorted(Path(session_dir).glob("*_epo.fif"))
    if not matches:
        raise FileNotFoundError(
            f"No '*_epo.fif' in {session_dir}. Run the epochs stage first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple '*_epo.fif' files in {session_dir}: "
            f"{[p.name for p in matches]}. The epochs stage guarantees one "
            "epochs file per session directory; remove the stale output."
        )
    return matches[0]


def resolve_task_config(
    paths: EEGCPMPaths, task: str
) -> Tuple[TaskConfig, Path]:
    """Load the project's task config for ``task`` (packaged copy as fallback).

    The project copy (``{project}/eegcpm/configs/tasks/{task}.yaml``) is
    authoritative because that is the config the epochs stage and the UI read.
    The packaged config is only a fallback so a project that has not mirrored
    its configs still resolves; the path actually used is printed and recorded
    in ``selection.json``.
    """
    candidates = [
        paths.get_configs_dir("tasks") / f"{task}.yaml",
        Path(__file__).resolve().parents[1] / "config" / "tasks" / f"{task}.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return TaskConfig.from_yaml(candidate), candidate
    raise FileNotFoundError(
        f"Task config for {task!r} not found. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _discover_subjects(paths: EEGCPMPaths, preprocessing: str, task: str) -> List[str]:
    """Subjects that have an epochs directory for the task (``sub-`` stripped)."""
    root = paths.get_epochs_root(preprocessing, task)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No epochs directory for task {task!r}: {root}. "
            "Run the epochs stage first."
        )
    return sorted(
        d.name[len("sub-"):] for d in root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


def _session_dirs(subject_dir: Path) -> List[Path]:
    sessions = sorted(d for d in subject_dir.glob("ses-*") if d.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No 'ses-*' directory in {subject_dir}")
    return sessions


def selection_output_dir(
    paths: EEGCPMPaths,
    preprocessing: str,
    task: str,
    name: str,
    subject_id: str,
    session_id: str,
) -> Path:
    """``derivatives/epochs/{preprocessing}/{task}/selections/{name}/sub-{s}/ses-{y}``."""
    return (
        paths.get_epochs_root(preprocessing, task)
        / "selections"
        / name
        / f"sub-{subject_id}"
        / f"ses-{session_id}"
    )


# ---------------------------------------------------------------------------
# selection engine
# ---------------------------------------------------------------------------


def _has_filters(args: argparse.Namespace) -> bool:
    return any(
        [
            args.conditions,
            args.response is not None,
            args.accuracy is not None,
            args.where is not None,
            parse_rt_bounds(args.rt) != (None, None),
        ]
    )


def selection_spec(args: argparse.Namespace) -> Dict[str, Any]:
    """The human-readable spec recorded in ``selection.json``."""
    return {
        "conditions": list(args.conditions) if args.conditions else None,
        "response": args.response,
        "accuracy": args.accuracy,
        "rt": list(parse_rt_bounds(args.rt)),
        "where": args.where,
        "sort_by": args.sort_by,
        "descending": bool(args.descending),
        "split_by": args.split_by,
        "split_column": args.split_column,
        "split_parts": args.split_parts,
        "bin_spec": args.bin_spec,
        "average": bool(args.average),
    }


def apply_selection(
    epochs: mne.Epochs,
    sorter: TrialSorter,
    args: argparse.Namespace,
) -> Dict[str, Tuple[mne.Epochs, List[SelectionRecord]]]:
    """Run filter -> sort -> (split | bin) and return ``{label: (epochs, records)}``.

    Filters are applied once, then sorting re-orders the filtered set, then
    the split/binning works on the result. This ordering means a positional
    split after a sort is meaningful, and a binning spec never re-applies a
    predicate the caller already made (the records are attached in order).

    Raises:
        ValueError: propagated from ``TrialSorter`` (empty selection, unknown
            column/condition/response, invalid split or bin spec).
    """
    current = epochs
    records: List[SelectionRecord] = []

    if _has_filters(args):
        current, record = sorter.select(
            current,
            conditions=args.conditions,
            response=args.response,
            accuracy=args.accuracy,
            rt=parse_rt_bounds(args.rt),
            where=args.where,
        )
        records.append(record)

    if args.sort_by:
        current, record = sorter.sort(
            current, by=args.sort_by, ascending=not args.descending
        )
        records.append(record)

    if args.split_by:
        if args.split_by == "column" and not args.split_column:
            raise ValueError("--split-by column requires --split-column COL")
        by = args.split_column if args.split_by == "column" else "position"
        parts = sorter.split(current, by=by, parts=args.split_parts)
        return {
            label: (part_epochs, records + [part_record])
            for label, (part_epochs, part_record) in parts.items()
        }

    if args.bin_spec:
        binned = sorter.select(current, group=args.bin_spec)
        return {
            label: (bin_epochs, records + [bin_record])
            for label, (bin_epochs, bin_record) in binned.items()
        }

    return {"selection": (current, records)}


def _condition_names(epochs: mne.Epochs) -> List[str]:
    """Event-id names present in ``epochs`` (sorted for determinism)."""
    if not epochs.event_id:
        return []
    codes_present = set(int(c) for c in np.unique(epochs.events[:, 2]))
    return sorted(
        name for name, code in epochs.event_id.items() if int(code) in codes_present
    )


def write_selection(
    out_dir: Path,
    subject_id: str,
    outputs: Dict[str, Tuple[mne.Epochs, List[SelectionRecord]]],
    *,
    average: bool,
    force: bool,
) -> Dict[str, List[str]]:
    """Write epochs (and optional averages) for every output label.

    A single output label is written as ``{subject_id}_epo.fif`` (the epochs
    stage's naming); multiple labels (split parts / bins) get per-label
    suffixes so the files stay distinguishable.

    Returns:
        ``{"epochs": [...], "averages": [...]}`` paths written.
    """
    multiple = len(outputs) > 1
    written: Dict[str, List[str]] = {"epochs": [], "averages": []}

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*_epo.fif"))
    if existing and not force:
        raise FileExistsError(
            f"{out_dir} already contains epochs output "
            f"({[p.name for p in existing]}); pass --force to overwrite."
        )
    if existing:
        # A selection directory is fully derived from its source epochs, so
        # --force replaces it wholesale (otherwise stale split parts / bins
        # from the previous selection would survive next to the new ones).
        stale = [
            path
            for pattern in _OUTPUT_PATTERNS
            for path in out_dir.glob(pattern)
        ]
        for path in stale:
            path.unlink()
        print(f"   ♻️  Replaced {len(stale)} existing output file(s)")

    for label, (label_epochs, _records) in outputs.items():
        suffix = f"_{_safe_label(label)}" if multiple else ""
        epochs_path = out_dir / f"{subject_id}{suffix}_epo.fif"
        label_epochs.save(epochs_path, overwrite=True, verbose=False)
        written["epochs"].append(str(epochs_path))

        if not average:
            continue

        # Overall average for this output label.
        overall_path = out_dir / f"{subject_id}{suffix}_ave.fif"
        label_epochs.average().save(overall_path, overwrite=True, verbose=False)
        written["averages"].append(str(overall_path))

        # Per-condition averages (only conditions actually present).
        for condition in _condition_names(label_epochs):
            cond_path = out_dir / (
                f"{subject_id}{suffix}_{_safe_label(condition)}_ave.fif"
            )
            label_epochs[condition].average().save(
                cond_path, overwrite=True, verbose=False
            )
            written["averages"].append(str(cond_path))

    return written


def write_record(
    out_dir: Path,
    *,
    name: str,
    subject_id: str,
    session_id: str,
    args: argparse.Namespace,
    task_config_path: Path,
    source_epochs_path: Path,
    outputs: Dict[str, Tuple[mne.Epochs, List[SelectionRecord]]],
    written: Dict[str, List[str]],
    n_source_trials: int,
) -> Path:
    """Write ``selection.json`` with the ``SelectionRecord`` provenance."""
    records = []
    for label, (label_epochs, label_records) in outputs.items():
        for record in label_records:
            entry = {"label": label, "n_output_trials": int(len(label_epochs))}
            entry.update(asdict(record))
            records.append(entry)

    payload = {
        "name": name,
        "task": args.task,
        "preprocessing": args.preprocessing,
        "subject": subject_id,
        "session": session_id,
        "source_epochs_file": str(source_epochs_path),
        "task_config": str(task_config_path),
        "n_source_trials": int(n_source_trials),
        "n_outputs": len(outputs),
        "n_output_trials": int(sum(len(e) for e, _ in outputs.values())),
        "spec": selection_spec(args),
        "records": records,
        "outputs": written,
    }
    path = out_dir / "selection.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------


def _process_session(
    args: argparse.Namespace,
    sorter: TrialSorter,
    task_config_path: Path,
    out_dir: Path,
    source_epochs_path: Path,
    subject_id: str,
    session_id: str,
) -> int:
    """Apply the selection for one session. Returns the number of outputs."""
    epochs = mne.read_epochs(source_epochs_path, preload=True, verbose=False)

    if epochs.metadata is None:
        raise ValueError(
            f"{source_epochs_path}: metadata not attached — re-run the epochs "
            "stage (it attaches the events.tsv metadata in Pass 1)."
        )

    outputs = apply_selection(epochs, sorter, args)
    n_total = sum(len(e) for e, _ in outputs.values())

    print(f"   🎯 Selection '{args.name}': {len(epochs)} → {n_total} trials"
          f" ({len(outputs)} output(s))")
    for label, (label_epochs, records) in outputs.items():
        n_before = records[0].n_before if records else len(epochs)
        print(f"      - {label}: {len(label_epochs)} trials (from {n_before})")

    if args.dry_run:
        print(f"   🔍 Dry run — would write to: {out_dir}")
        print("      (nothing written)")
        return len(outputs)

    written = write_selection(
        out_dir,
        subject_id,
        outputs,
        average=bool(args.average),
        force=bool(args.force),
    )
    record_path = write_record(
        out_dir,
        name=args.name,
        subject_id=subject_id,
        session_id=session_id,
        args=args,
        task_config_path=task_config_path,
        source_epochs_path=source_epochs_path,
        outputs=outputs,
        written=written,
        n_source_trials=len(epochs),
    )

    print(f"   📁 Output: {out_dir}")
    for path in written["epochs"]:
        print(f"      - {Path(path).name}")
    if written["averages"]:
        print(f"   📊 Averages: {len(written['averages'])}")
    print(f"   📄 Selection record: {record_path.name}")
    return len(outputs)


def epochs_select_command(args: argparse.Namespace) -> int:
    """Run ``eegcpm epochs-select``. Returns a process exit code."""
    print(f"\n{'=' * 60}")
    print("EEGCPM Epoch Selection")
    print(f"{'=' * 60}\n")

    failures: List[str] = []
    n_written = 0

    try:
        name = validate_selection_name(args.name)
        # Normalise --rt once: everything downstream (spec, predicated select)
        # sees (lo, hi) floats or None.
        args.rt = parse_rt_bounds(args.rt)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    if args.accuracy is not None and args.accuracy not in (0, 1):
        print(f"❌ --accuracy must be 0 or 1, got {args.accuracy!r}")
        return 1
    if args.split_parts < 1:
        print(f"❌ --split-parts must be >= 1, got {args.split_parts!r}")
        return 1
    if args.split_by == "column" and not args.split_column:
        print("❌ --split-by column requires --split-column COL")
        return 1
    if args.split_by is None and args.split_column:
        print("❌ --split-column requires --split-by column")
        return 1
    if args.descending and not args.sort_by:
        print("❌ --descending requires --sort-by COL")
        return 1
    if args.split_by and args.bin_spec:
        print("❌ --split-by and --bin-spec are mutually exclusive "
              "(split first, then re-run on the parts to bin)")
        return 1

    project_root = Path(args.project)
    paths = EEGCPMPaths(project_root)

    try:
        task_config, task_config_path = resolve_task_config(paths, args.task)
        sorter = TrialSorter(task_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        return 1

    if args.bin_spec:
        available = [spec.name for spec in (task_config.binning or [])]
        if args.bin_spec not in available:
            print(
                f"❌ Unknown bin spec {args.bin_spec!r}. "
                f"Available binning specs: {available}"
            )
            return 1

    subjects = list(args.subjects) if args.subjects else []
    if subjects:
        subjects = [s[len("sub-"):] if s.startswith("sub-") else s for s in subjects]
    else:
        try:
            subjects = _discover_subjects(paths, args.preprocessing, args.task)
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
            return 1

    if not subjects:
        print("❌ No subjects to process.")
        return 1

    print(f"📂 Project:        {project_root}")
    print(f"⚙️  Preprocessing:  {args.preprocessing}")
    print(f"🏷️  Task:           {args.task}")
    print(f"📄 Task config:    {task_config_path}")
    print(f"✂️  Selection:      {name}")
    print(f"👥 Subjects:       {', '.join(subjects)}")
    print(f"🧮 Spec:           {json.dumps(selection_spec(args), default=str)}")
    if args.dry_run:
        print("🔍 Dry run: no files will be written")

    for subject_id in subjects:
        subject_dir = (
            paths.get_epochs_root(args.preprocessing, args.task) / f"sub-{subject_id}"
        )
        if not subject_dir.is_dir():
            print(f"\n❌ sub-{subject_id}: no epochs directory {subject_dir}")
            failures.append(f"sub-{subject_id}: no epochs directory")
            continue

        try:
            sessions = _session_dirs(subject_dir)
        except FileNotFoundError as exc:
            print(f"\n❌ sub-{subject_id}: {exc}")
            failures.append(f"sub-{subject_id}: {exc}")
            continue

        for session_dir in sessions:
            session_id = session_dir.name[len("ses-"):]
            print(f"\n🔄 sub-{subject_id}, ses-{session_id}")

            try:
                source_epochs_path = find_source_epochs(session_dir)
                out_dir = selection_output_dir(
                    paths, args.preprocessing, args.task, name,
                    subject_id, session_id,
                )
                existing = sorted(out_dir.glob("*_epo.fif")) if out_dir.is_dir() else []
                if existing and not args.force:
                    message = (
                        f"{out_dir} already contains epochs output "
                        f"({[p.name for p in existing]}); pass --force to "
                        "overwrite."
                    )
                    if args.dry_run:
                        print(f"   ⚠️  {message}")
                    else:
                        raise FileExistsError(message)

                n_outputs = _process_session(
                    args,
                    sorter,
                    task_config_path,
                    out_dir,
                    source_epochs_path,
                    subject_id,
                    session_id,
                )
                if not args.dry_run:
                    n_written += n_outputs
            except (
                FileNotFoundError,
                FileExistsError,
                RuntimeError,
                ValueError,
                OSError,
            ) as exc:
                print(f"   ❌ {exc}")
                failures.append(f"sub-{subject_id}/ses-{session_id}: {exc}")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    if args.dry_run:
        print(f"🔍 Dry run — {len(subjects) - len(failures)} subject(s) validated, "
              "nothing written")
    else:
        print(f"✅ Outputs written: {n_written}")
    print(f"❌ Failed: {len(failures)}")
    for failure in failures:
        print(f"   - {failure}")
    print()

    return 1 if failures else 0


def add_epochs_select_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``epochs-select`` subcommand."""
    parser = subparsers.add_parser(
        "epochs-select",
        help="Select / sort / split trials from extracted epochs",
        description=(
            "Apply a named trial selection to epochs from the epochs stage and "
            "write it to derivatives/epochs/{preprocessing}/{task}/selections/"
            "{name}/. The source epochs are never modified."
        ),
    )
    parser.add_argument("--project", type=Path, required=True,
                        help="Project root directory")
    parser.add_argument("--preprocessing", type=str, required=True,
                        help="Preprocessing pipeline variant (e.g., 25susattn)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., sstvis)")
    parser.add_argument("--subjects", type=str, nargs="+",
                        help="Subject IDs (default: every subject with epochs)")
    parser.add_argument("--name", type=str, required=True,
                        help="Selection label; letters, digits, '-' and '_' only")
    parser.add_argument("--conditions", type=str, nargs="+",
                        help="Condition names from the task config to keep")
    parser.add_argument("--response", type=str,
                        help="Response category from the task config")
    parser.add_argument("--accuracy", type=int, choices=[0, 1],
                        help="Keep correct (1) or incorrect (0) trials")
    parser.add_argument("--rt", type=str, nargs=2, metavar=("LO", "HI"),
                        help="RT bounds in seconds; 'none' for an open bound")
    parser.add_argument("--where", type=str,
                        help="pandas query on the trial metadata "
                             "(use backticks for dotted column names)")
    parser.add_argument("--sort-by", type=str, metavar="COL",
                        help="Sort by a metadata column (metadata is reordered "
                             "with the epoch data)")
    parser.add_argument("--descending", action="store_true",
                        help="Sort descending (with --sort-by)")
    parser.add_argument("--split-by", choices=["position", "column"],
                        help="Split the selected trials into parts")
    parser.add_argument("--split-column", type=str, metavar="COL",
                        help="Metadata column for --split-by column")
    parser.add_argument("--split-parts", type=int, default=2, metavar="N",
                        help="Number of parts for --split-by (default: 2)")
    parser.add_argument("--bin-spec", type=str, metavar="NAME",
                        help="Bin the selected trials by a task-config binning spec")
    parser.add_argument("--average", action="store_true",
                        help="Write per-condition averages (*_ave.fif) too")
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing selection output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the resulting trial counts; "
                             "write nothing")
    return parser
