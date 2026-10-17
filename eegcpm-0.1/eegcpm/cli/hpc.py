"""``eegcpm hpc`` — SLURM batch workflow on top of :mod:`eegcpm.pipeline.slurm`.

The batch tooling itself (manifest, chunking, script generation, per-unit
execution, finalize) lives in ONE module — :mod:`eegcpm.pipeline.slurm` — which
the Streamlit pages also import; this CLI is only argument handling on top of
it.

Flow (see ``eegcpm/config/hpc/README.md``):

1. ``hpc manifest``  — BIDS tree → work-unit manifest TSV;
2. ``hpc prune``     — drop units whose requested stages are already complete;
3. ``hpc submit``    — chunk the manifest across partitions and write one
   sbatch script (SLURM job array) per partition; prints the ``sbatch``
   commands, and only runs them when ``--dry-run`` is absent;
4. ``hpc run-unit``  — the array task step (one work unit per task);
5. ``hpc status``    — read the per-unit JSON records into a per-stage table;
6. ``hpc finalize``  — the task-level features parquet (refuses while any unit
   of the task lacks its source artifact).

Every subcommand that needs paths accepts ``--config <hpc.yaml>`` (validated
through :class:`eegcpm.core.config.HpcBatchConfig`; unknown keys are hard
errors) and explicit CLI flags always override the config values. ``hpc
run-unit`` is deliberately flag-only: the generated sbatch script bakes in
absolute flags, so a compute node never has to parse the batch config.

``sbatch`` is never called with ``--dry-run``, which is what the local tests
exercise (no cluster access required).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eegcpm.core.config import HpcBatchConfig
from eegcpm.pipeline.slurm import (
    ALL_STAGES,
    PER_UNIT_STAGES,
    build_manifest,
    chunk_units,
    features_output_dir,
    finalize_task,
    generate_array_script,
    prune_units,
    read_manifest,
    run_unit,
    unit_record_path,
    write_manifest,
)


def _parse_stages(raw: Optional[str], *, allow_task_level: bool) -> List[str]:
    """Split a comma-separated ``--stages`` value and validate the names."""
    if not raw:
        raise ValueError("--stages is required (comma-separated list)")
    stages = [item.strip() for item in raw.split(",") if item.strip()]
    if not stages:
        raise ValueError(f"--stages is empty: {raw!r}")
    allowed = set(ALL_STAGES if allow_task_level else PER_UNIT_STAGES)
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(
            f"unknown stage(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
    return stages


def _parse_csv(raw: Optional[str], what: str) -> List[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        raise ValueError(f"{what} is required (comma-separated list)")
    return values


def _resolve(path: Optional[Path]) -> Optional[Path]:
    """Expand ``~`` and make a path absolute (kept as ``None`` if not given)."""
    return Path(path).expanduser().resolve() if path is not None else None


def _project_path(raw: Path, project: Path) -> Path:
    """Expand ``~`` and resolve a path against the project when relative.

    The sbatch scripts run on compute nodes with an arbitrary cwd, so every
    path baked into a script (or recorded) must be absolute; project-relative
    is the convention of the other ``hpc`` path flags.
    """
    expanded = Path(raw).expanduser()
    return expanded if expanded.is_absolute() else (project / expanded).resolve()


def _load_config(args: argparse.Namespace) -> Optional[HpcBatchConfig]:
    """Load ``--config <hpc.yaml>`` through the one validated batch schema.

    Unknown keys, bad stage names and malformed YAML are hard errors (the
    schema uses ``extra='forbid'``): a silently ignored typo in a cluster
    config would submit the wrong job.
    """
    raw = getattr(args, "config", None)
    if not raw:
        return None
    return HpcBatchConfig.from_yaml(Path(raw))


def _rebase_config_dirs(config: HpcBatchConfig, project: Path) -> None:
    """Re-derive the config's *default* dirs from an overridden ``--project``.

    ``HpcBatchConfig`` resolves ``logs_dir``/``state_db_dir``/
    ``submissions_dir`` against its own ``project``; when ``--project`` points
    somewhere else, those defaults must follow it (explicitly configured dirs —
    anything not equal to the default — are left alone).
    """
    for attr, relative in (
        ("logs_dir", Path("hpc") / "logs"),
        ("state_db_dir", Path("hpc") / "state"),
        ("submissions_dir", Path("hpc") / "submissions"),
    ):
        if getattr(config, attr) == Path(config.project) / relative:
            setattr(config, attr, project / relative)


def _resolve_project(
    args: argparse.Namespace, config: Optional[HpcBatchConfig]
) -> Optional[Path]:
    """The project root from ``--project`` (wins) or the batch config."""
    if getattr(args, "project", None) is not None:
        project = _resolve(args.project)
        if config is not None and Path(config.project) != project:
            _rebase_config_dirs(config, project)
        return project
    if config is not None:
        return Path(config.project)
    return None


def _fail(command: str, message: str) -> int:
    print(f"hpc {command}: {message}", file=sys.stderr)
    return 2


def _config_paths_from_args(
    args: argparse.Namespace,
    project: Path,
    config: Optional[HpcBatchConfig] = None,
) -> Dict[str, Path]:
    """Collect the per-stage config paths shared by submit/run-unit/finalize.

    CLI flags win over the batch config; the task-config dir falls back to the
    PROJECT's ``eegcpm/configs/tasks`` (never the package config dir).
    """
    cli_preprocessing = getattr(args, "preprocessing_config", None)
    cli_source = getattr(args, "source_config", None)
    cli_tasks_dir = getattr(args, "tasks_config_dir", None)

    config_paths: Dict[str, Path] = {}
    preprocessing = cli_preprocessing or (config.preprocessing_config if config else None)
    if preprocessing:
        config_paths["preprocessing"] = (
            _project_path(cli_preprocessing, project) if cli_preprocessing
            else Path(preprocessing)
        )
    source = cli_source or (config.source_config if config else None)
    if source:
        config_paths["source"] = (
            _project_path(cli_source, project) if cli_source else Path(source)
        )
    if cli_tasks_dir:
        config_paths["tasks_dir"] = _project_path(cli_tasks_dir, project)
    else:
        # The source/preprocess CLIs resolve task configs under the PROJECT's
        # eegcpm configs dir; never assume the package config dir.
        config_paths["tasks_dir"] = (project / "eegcpm" / "configs" / "tasks").resolve()
    return config_paths


# --------------------------------------------------------------- subcommands

def cmd_manifest(args: argparse.Namespace) -> int:
    config = _load_config(args)
    project = _resolve_project(args, config)
    if project is None:
        return _fail("manifest", "--project is required unless --config sets 'project'")
    out = _project_path(args.out, project)
    units = build_manifest(
        project / "bids",
        tasks=args.tasks,
        subjects=args.subjects,
        sessions=args.sessions,
    )
    if not units:
        print(
            f"hpc manifest: no work units found under {project / 'bids'} "
            f"(tasks={args.tasks}, subjects={args.subjects})",
            file=sys.stderr,
        )
        return 2
    write_manifest(units, out)
    subjects = sorted({unit.subject for unit in units})
    tasks = sorted({unit.task for unit in units})
    print(
        f"hpc manifest: {len(units)} work unit(s) "
        f"({len(subjects)} subjects × {len(tasks)} tasks) -> {out}"
    )
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    config = _load_config(args)
    project = _resolve_project(args, config)
    if project is None:
        return _fail("prune", "--project is required unless --config sets 'project'")
    raw_stages = args.stages or (",".join(config.stages) if config is not None else None)
    if not raw_stages:
        return _fail("prune", "--stages is required (or --config with 'stages')")
    stages = _parse_stages(raw_stages, allow_task_level=False)
    units = read_manifest(_project_path(args.manifest, project))
    pending = prune_units(
        project,
        units,
        stages,
        pipeline=args.pipeline,
        source_variant=args.source_variant,
        expected_config_hash=args.expected_config_hash,
    )
    out = _project_path(args.out, project)
    write_manifest(pending, out)
    print(
        f"hpc prune: {len(units) - len(pending)}/{len(units)} unit(s) already "
        f"complete for [{','.join(stages)}] -> {len(pending)} pending -> {out}"
    )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    config = _load_config(args)
    project = _resolve_project(args, config)
    if project is None:
        return _fail("submit", "--project is required unless --config sets 'project'")

    raw_stages = args.stages or (",".join(config.stages) if config is not None else None)
    if not raw_stages:
        return _fail("submit", "--stages is required (or --config with 'stages')")
    stages = _parse_stages(raw_stages, allow_task_level=False)

    partition_configs = {
        partition.name: partition
        for partition in (config.slurm.partitions if config is not None else [])
    }
    if args.partitions:
        partitions = _parse_csv(args.partitions, "--partitions")
    elif partition_configs:
        partitions = list(partition_configs)
    else:
        return _fail(
            "submit",
            "no partitions: pass --partitions or add `slurm.partitions` to --config",
        )

    manifest = _project_path(args.manifest, project)

    config_paths = _config_paths_from_args(args, project, config)
    if "preprocess" in stages and "preprocessing" not in config_paths:
        print(
            "hpc submit: --preprocessing-config is required when --stages "
            "includes 'preprocess'",
            file=sys.stderr,
        )
        return 2
    if "source" in stages and "source" not in config_paths:
        print(
            "hpc submit: --source-config is required when --stages includes "
            "'source'",
            file=sys.stderr,
        )
        return 2

    units = read_manifest(manifest)
    if not units:
        print(f"hpc submit: manifest has no work units: {manifest}", file=sys.stderr)
        return 2

    log_dir = args.log_dir or (config.logs_dir if config is not None else None) \
        or (project / "hpc" / "logs")
    log_dir = _project_path(log_dir, project) if args.log_dir else Path(log_dir)
    submissions_root = (
        config.submissions_dir if config is not None
        else (project / "hpc" / "submissions").resolve()
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    submission_dir = Path(submissions_root) / stamp

    state_db_dir = config.state_db_dir if config is not None else None

    venv = args.venv
    if venv is not None:
        venv = _project_path(Path(venv), project)
    elif config is not None and config.slurm.venv:
        venv = Path(config.slurm.venv).expanduser()

    modules = args.module or (list(config.slurm.modules) if config is not None else [])
    exports = dict(config.slurm.exports) if config is not None else None
    default_cpus = config.slurm.cpus_per_task if config is not None else None
    default_mem = config.slurm.mem if config is not None else None
    default_time = config.slurm.time if config is not None else None

    chunks = chunk_units(units, partitions)

    # Resolve every partition's array settings up front: a missing resource is
    # a hard error *before* anything is written.
    resolved = []
    for partition, partition_units in chunks:
        partition_config = partition_configs.get(partition)
        concurrency = (
            args.concurrency_per_partition
            or (partition_config.max_concurrent if partition_config else None)
            or 1
        )
        cpus = (
            args.cpus
            or (partition_config.cpus if partition_config else None)
            or default_cpus
        )
        mem = (
            args.mem
            or (partition_config.mem if partition_config else None)
            or default_mem
        )
        time_limit = (
            args.time
            or (partition_config.time if partition_config else None)
            or default_time
        )
        missing = [
            name for name, value in
            (("--cpus", cpus), ("--mem", mem), ("--time", time_limit))
            if not value
        ]
        if missing:
            return _fail(
                "submit",
                f"missing {', '.join(missing)} for partition {partition!r}: pass "
                "the flag(s) or set them in the config's slurm section",
            )
        resolved.append((partition, partition_units, concurrency, cpus, mem, time_limit))

    if args.dry_run:
        print(f"hpc submit: DRY RUN — {len(units)} unit(s) across "
              f"{len(partitions)} partition(s) -> {submission_dir}")

    submission_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts: List[Path] = []
    for partition, partition_units, concurrency, cpus, mem, time_limit in resolved:
        job_name = f"{args.job_name}_{partition}"
        chunk_path = submission_dir / f"{job_name}.tsv"
        write_manifest(partition_units, chunk_path)

        # Bake only the config paths of the requested stages into the command.
        stage_config_paths = dict(config_paths)
        if "preprocess" not in stages:
            stage_config_paths.pop("preprocessing", None)
        if "source" not in stages:
            stage_config_paths.pop("source", None)

        script = generate_array_script(
            job_name=job_name,
            partition=partition,
            units_chunk_path=chunk_path,
            n_units=len(partition_units),
            concurrency=concurrency,
            cpus=cpus,
            mem=mem,
            time=time_limit,
            log_dir=log_dir,
            venv=venv,
            modules=modules,
            exports=exports,
            project=project,
            stages=stages,
            pipeline=args.pipeline,
            source_variant=args.source_variant,
            config_paths=stage_config_paths,
            state_db_dir=state_db_dir,
        )
        script_path = submission_dir / f"{job_name}.sbatch"
        script_path.write_text(script, encoding="utf-8")
        scripts.append(script_path)
        print(
            f"  {partition}: {len(partition_units)} unit(s) -> {chunk_path.name}"
            f"  (array 0-{len(partition_units) - 1}%{concurrency},"
            f" cpus={cpus}, mem={mem}, time={time_limit})"
        )

    if args.dry_run:
        print("  sbatch commands (not executed):")
        for script_path in scripts:
            print(f"    sbatch {script_path}")
        return 0

    if not shutil.which("sbatch"):
        print(
            "hpc submit: sbatch not found on PATH — scripts were written to "
            f"{submission_dir}; submit them from a cluster login node or use --dry-run",
            file=sys.stderr,
        )
        return 1

    for script_path in scripts:
        completed = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            print(
                f"hpc submit: sbatch {script_path} failed "
                f"({completed.returncode}): {completed.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        print(f"  {completed.stdout.strip()}")
    return 0


def cmd_run_unit(args: argparse.Namespace) -> int:
    project = _resolve(args.project)
    stages = _parse_stages(args.stages, allow_task_level=False)
    manifest = _project_path(args.manifest, project)
    logs_dir = _project_path(args.logs, project)

    config_paths = _config_paths_from_args(args, project)
    if "preprocess" in stages and "preprocessing" not in config_paths:
        print(
            "hpc run-unit: --preprocessing-config is required when --stages "
            "includes 'preprocess'",
            file=sys.stderr,
        )
        return 2
    if "source" in stages and "source" not in config_paths:
        print(
            "hpc run-unit: --source-config is required when --stages includes "
            "'source'",
            file=sys.stderr,
        )
        return 2

    units = read_manifest(manifest)
    if not 0 <= args.index < len(units):
        print(
            f"hpc run-unit: --index {args.index} out of range for {manifest} "
            f"({len(units)} unit(s); array index must be 0..{len(units) - 1})",
            file=sys.stderr,
        )
        return 2
    unit = units[args.index]

    state_db_dir = (
        _project_path(args.state_db_dir, project) if args.state_db_dir
        else logs_dir.parent / "state"
    )

    record = run_unit(
        unit,
        project=project,
        stages=stages,
        pipeline=args.pipeline,
        source_variant=args.source_variant,
        config_paths=config_paths,
        state_db_dir=state_db_dir,
        logs_dir=logs_dir,
        dry_run=args.dry_run,
        python=args.python or sys.executable,
    )

    for stage, stage_record in record["stages"].items():
        line = f"  {stage}: {stage_record['status']}"
        if stage_record["returncode"] is not None:
            line += f" (rc={stage_record['returncode']}, {stage_record['seconds']}s)"
        print(line)
        for command in stage_record["commands"]:
            print("    " + " ".join(command))
    if not args.dry_run:
        print(f"hpc run-unit: record -> {unit_record_path(logs_dir, unit)}")
    if record["exit_code"]:
        print(
            f"hpc run-unit: {unit.key} FAILED (exit {record['exit_code']})",
            file=sys.stderr,
        )
        return record["exit_code"]
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    project = _resolve_project(args, config)
    if project is None:
        return _fail("status", "--project is required unless --config sets 'project'")
    logs_dir = (
        _project_path(args.logs, project) if args.logs
        else Path(config.logs_dir) if config is not None
        else (project / "hpc" / "logs").resolve()
    )
    units_dir = logs_dir.parent / "units"

    stages = (
        _parse_stages(args.stages, allow_task_level=False)
        if args.stages else list(PER_UNIT_STAGES)
    )

    if not units_dir.is_dir():
        print(
            f"hpc status: no per-unit records at {units_dir} "
            "(nothing has run yet; records appear after `hpc run-unit`)"
        )
        return 0

    records: List[Dict[str, Any]] = []
    unreadable: List[Path] = []
    for record_path in sorted(units_dir.glob("*.json")):
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            unreadable.append(record_path)
            print(
                f"hpc status: unreadable record {record_path.name}: {exc}",
                file=sys.stderr,
            )
            continue
        if isinstance(payload, dict):
            records.append(payload)

    if not records:
        print(
            f"hpc status: no readable per-unit records at {units_dir}"
            + (f" ({len(unreadable)} unreadable)" if unreadable else "")
        )
        return 0

    key_width = max(len("unit"), *(len(str(record.get("unit", "?"))) for record in records))
    header = "unit".ljust(key_width) + "".join(
        f"  {stage:<{max(9, len(stage))}}" for stage in stages
    )
    print(header)
    print("-" * len(header))
    counts: Dict[str, Dict[str, int]] = {
        stage: {"completed": 0, "skipped": 0, "failed": 0, "missing": 0}
        for stage in stages
    }
    for record in sorted(records, key=lambda item: str(item.get("unit", ""))):
        cells = []
        for stage in stages:
            stage_record = (record.get("stages") or {}).get(stage)
            status = stage_record.get("status") if isinstance(stage_record, dict) else None
            status = status or "missing"
            if status not in counts[stage]:
                counts[stage][status] = 0
            counts[stage][status] += 1
            cells.append(f"  {status:<{max(9, len(stage))}}")
        print(str(record.get("unit", "?")).ljust(key_width) + "".join(cells))

    print()
    for stage in stages:
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts[stage].items()))
        print(f"{stage}: {summary}")
    if unreadable:
        print(f"unreadable records: {len(unreadable)}")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    config = _load_config(args)
    project = _resolve_project(args, config)
    if project is None:
        return _fail("finalize", "--project is required unless --config sets 'project'")

    features_config = args.features_config or (
        config.features_config if config is not None else None
    )
    if not features_config:
        return _fail(
            "finalize",
            "--features-config is required (or --config with 'features_config')",
        )

    output_dir = (
        Path(args.output_dir).expanduser() if args.output_dir
        else features_output_dir(project, args.pipeline)
    )
    try:
        record = finalize_task(
            project,
            args.task,
            pipeline=args.pipeline,
            source_variant=args.source_variant,
            subjects=args.subjects,
            features_config=features_config,
            output_dir=output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"hpc finalize: {exc}", file=sys.stderr)
        return 1

    print(
        f"hpc finalize: task={record['task']} subjects={len(record['subjects'])} "
        f"units={len(record['units'])}"
        + (f" incomplete={len(record['incomplete_units'])}" if record["incomplete_units"] else "")
    )
    for command in record["commands"]:
        print("  " + " ".join(command))
    if record["dry_run"]:
        print("hpc finalize: DRY RUN — nothing executed")
        return 0

    returncode = record.get("returncode", 1)
    if returncode:
        print(
            f"hpc finalize: features failed (rc={returncode}); the task parquet "
            f"{record['output']} was not produced",
            file=sys.stderr,
        )
        return returncode
    print(f"hpc finalize: {record['output']}")
    return 0


# ------------------------------------------------------------------- parser

def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """``--config <hpc.yaml>``: the validated batch config (flags override)."""
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Batch config YAML (HpcBatchConfig schema; unknown keys are errors). "
             "Explicit flags override its values.",
    )


def add_hpc_parser(subparsers) -> argparse.ArgumentParser:
    """Register ``eegcpm hpc ...`` subcommands."""
    hpc_parser = subparsers.add_parser(
        "hpc",
        help="HPC/SLURM batch: manifest -> prune -> submit -> status -> finalize",
        description=(
            "Run the pipeline as SLURM job arrays: one work unit "
            "(subject/session/task/run) per array task, with per-unit state DBs "
            "and idempotent stage skipping. `features` is a task-level finalize "
            "step (one parquet per task)."
        ),
    )
    hpc_parser.set_defaults(_hpc_parser=hpc_parser)
    hpc_sub = hpc_parser.add_subparsers(dest="hpc_command")

    # manifest
    p = hpc_sub.add_parser("manifest", help="Build a work-unit manifest from the BIDS tree")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root directory (or use --config)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output manifest TSV (absolute, or relative to the project)")
    p.add_argument("--tasks", type=str, nargs="+", help="Restrict to these task names")
    p.add_argument("--subjects", type=str, nargs="+", help="Restrict to these subject IDs")
    p.add_argument("--sessions", type=str, nargs="+", help="Restrict to these session IDs")
    _add_config_argument(p)
    p.set_defaults(hpc_func=cmd_manifest)

    # prune
    p = hpc_sub.add_parser("prune", help="Drop units whose requested stages are complete")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root directory (or use --config)")
    p.add_argument("--manifest", type=Path, required=True, help="Input manifest TSV")
    p.add_argument("--stages", type=str, default=None,
                   help="Comma-separated per-unit stages (preprocess,epochs,source); "
                        "default from --config")
    p.add_argument("--pipeline", type=str, required=True, help="Preprocessing pipeline name")
    p.add_argument("--source-variant", type=str, default=None, help="Source variant (needed for 'source')")
    p.add_argument("--expected-config-hash", type=str, default=None,
                   help="Require this exact preprocessing resolved-config hash")
    p.add_argument("--out", type=Path, required=True, help="Output manifest TSV (pending units)")
    _add_config_argument(p)
    p.set_defaults(hpc_func=cmd_prune)

    # submit
    p = hpc_sub.add_parser("submit", help="Chunk a manifest across partitions and write sbatch scripts")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root directory (or use --config)")
    p.add_argument("--manifest", type=Path, required=True, help="Manifest TSV to submit")
    p.add_argument("--stages", type=str, default=None,
                   help="Comma-separated per-unit stages; default from --config")
    p.add_argument("--pipeline", type=str, required=True, help="Preprocessing pipeline name")
    p.add_argument("--source-variant", type=str, default=None, help="Source variant")
    p.add_argument("--partitions", type=str, default=None,
                   help="Comma-separated SLURM partition names (or from --config)")
    p.add_argument("--concurrency-per-partition", type=int, default=None,
                   help="Max simultaneously running array tasks per partition (%%k); "
                        "default: each partition's max_concurrent")
    p.add_argument("--cpus", type=int, default=None,
                   help="CPUs per array task (default: each partition's cpus)")
    p.add_argument("--mem", type=str, default=None,
                   help="Memory per array task, e.g. 16G (default: each partition's mem)")
    p.add_argument("--time", type=str, default=None,
                   help="Wall time per array task (default: each partition's time)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="SLURM log directory (default: --config logs_dir)")
    p.add_argument("--job-name", type=str, default="eegcpm", help="Job name prefix")
    p.add_argument("--venv", type=Path, default=None,
                   help="Virtualenv to activate/use (default: --config slurm.venv, "
                        "else the submitter's interpreter)")
    p.add_argument("--module", type=str, action="append", default=[],
                   help="Module to 'module load' (repeatable; default: --config slurm.modules)")
    p.add_argument("--preprocessing-config", type=Path, default=None,
                   help="Preprocessing config YAML (required for the preprocess stage)")
    p.add_argument("--source-config", type=Path, default=None,
                   help="Source reconstruction config YAML (required for the source stage)")
    p.add_argument("--tasks-config-dir", type=Path, default=None,
                   help="Task config directory (default: {project}/eegcpm/configs/tasks)")
    p.add_argument("--dry-run", action="store_true",
                   help="Write chunk scripts and print the sbatch commands without submitting")
    _add_config_argument(p)
    p.set_defaults(hpc_func=cmd_submit)

    # run-unit
    p = hpc_sub.add_parser("run-unit", help="Run one manifest row's stages (the array task step)")
    p.add_argument("--project", type=Path, required=True, help="Project root directory")
    p.add_argument("--manifest", type=Path, required=True, help="Manifest TSV (chunk)")
    p.add_argument("--index", type=int, required=True, help="0-based row index (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--stages", type=str, required=True, help="Comma-separated per-unit stages")
    p.add_argument("--pipeline", type=str, required=True, help="Preprocessing pipeline name")
    p.add_argument("--source-variant", type=str, default=None, help="Source variant")
    p.add_argument("--logs", type=Path, required=True, help="Log directory (unit records -> ../units)")
    p.add_argument("--state-db-dir", type=Path, default=None,
                   help="Per-unit state DB directory (default: <logs>/../state)")
    p.add_argument("--preprocessing-config", type=Path, default=None,
                   help="Preprocessing config YAML (required for the preprocess stage)")
    p.add_argument("--source-config", type=Path, default=None,
                   help="Source reconstruction config YAML (required for the source stage)")
    p.add_argument("--tasks-config-dir", type=Path, default=None,
                   help="Task config directory (default: {project}/eegcpm/configs/tasks)")
    p.add_argument("--python", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the stage commands without executing or writing anything")
    p.set_defaults(hpc_func=cmd_run_unit)

    # status
    p = hpc_sub.add_parser("status", help="Show per-unit stage status from the run records")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root directory (or use --config)")
    p.add_argument("--logs", type=Path, default=None,
                   help="Log directory (records read from <logs>/../units; "
                        "default: --config logs_dir)")
    p.add_argument("--stages", type=str, default=None, help="Comma-separated stage columns")
    _add_config_argument(p)
    p.set_defaults(hpc_func=cmd_status)

    # finalize
    p = hpc_sub.add_parser("finalize", help="Build the task-level source-ERP feature parquet")
    p.add_argument("--project", type=Path, default=None,
                   help="Project root directory (or use --config)")
    p.add_argument("--task", type=str, required=True, help="Task to finalize")
    p.add_argument("--pipeline", type=str, required=True, help="Preprocessing pipeline name")
    p.add_argument("--source-variant", type=str, required=True, help="Source variant")
    p.add_argument("--features-config", type=Path, default=None,
                   help="Features config YAML (or --config features_config)")
    p.add_argument("--subjects", type=str, nargs="+", default=None,
                   help="Subject IDs to include (default: every subject with BIDS input)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir (default: derivatives/features/<pipeline>/source_erp)")
    p.add_argument("--force", action="store_true",
                   help="Finalize even when units lack a complete source artifact")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the features command without executing it")
    _add_config_argument(p)
    p.set_defaults(hpc_func=cmd_finalize)

    return hpc_parser


def hpc_command(args: argparse.Namespace) -> Optional[int]:
    """Dispatch ``eegcpm hpc <subcommand>``; returns an exit code or ``None``."""
    if getattr(args, "hpc_command", None) is None:
        parser = getattr(args, "_hpc_parser", None)
        if parser is not None:
            parser.print_help()
        return 2
    try:
        return args.hpc_func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hpc {args.hpc_command}: {exc}", file=sys.stderr)
        return 2
