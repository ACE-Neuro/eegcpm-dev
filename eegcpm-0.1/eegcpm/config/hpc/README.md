# `eegcpm hpc` — batch runs on a SLURM cluster

There is **one** HPC implementation: `eegcpm/pipeline/slurm.py`. Everything
below — the CLI (`eegcpm hpc`, `eegcpm/cli/hpc.py`) and the Streamlit pages
(batch preprocessing page 2, prediction page 12) — imports its generators and
helpers; no page may inline its own `#SBATCH` header (a unit test fails the
build if one does). A future HPC **monitor** page must import the same module
too: the old mock page is archived at `eegcpm/ui/pages/_archived/hpc.py` and
must not be revived as a copy.

The commands run the pipeline as **SLURM job arrays**: one work unit
`(subject, session, task, run)` per array task, spread over one or more
partitions.

## Batch config (one schema)

`hpc.yaml` in this directory is a **template** of the `HpcBatchConfig` schema
(`eegcpm/core/config.py`). Copy it to your project (e.g.
`<project>/hpc.yaml`), replace the placeholders, and pass it with `--config`:

```bash
cd <project>
V=~/eegcpm/eegcpm-0.1/.venv/bin/eegcpm

# 1. Discover work units from the BIDS tree
$V hpc manifest --config hpc.yaml --out hpc/manifest.tsv [--tasks cptvis stroopvis]

# 2. Drop units whose requested stages are already complete (retry-safe)
$V hpc prune --config hpc.yaml --manifest hpc/manifest.tsv \
    --pipeline 25susattn --source-variant dSPM-CONN32-25sus \
    --out hpc/manifest_pending.tsv

# 3. Chunk across the config's partitions; write one job array per partition.
#    --dry-run writes/prints the scripts and never calls sbatch.
$V hpc submit --config hpc.yaml --manifest hpc/manifest_pending.tsv \
    --pipeline 25susattn --source-variant dSPM-CONN32-25sus \
    --dry-run          # omit to actually call sbatch

# 4. Inspect runs (per-unit JSON records under hpc/units/)
$V hpc status --config hpc.yaml

# 5. After every unit of a task has its source artifact, build the task-level
#    feature parquet (ONE invocation for all subjects)
$V hpc finalize --config hpc.yaml --task cptvis --pipeline 25susattn \
    --source-variant dSPM-CONN32-25sus --dry-run
```

CLI flags always override the config, so the config-free invocation still works
(`--project`, `--stages`, `--partitions normal,long`,
`--concurrency-per-partition 24 --cpus 4 --mem 16G --time 02:00:00 --log-dir
hpc/logs`, `--preprocessing-config`, `--source-config`, `--venv`, `--module`).
Unknown keys in the YAML (and unknown stage names) are **hard errors**; the
schema uses `extra='forbid'`.

What the config controls:

| Key | Meaning |
|---|---|
| `slurm.partitions[]` | partition name + `max_concurrent` (`%k`) + per-task `cpus`/`mem`/`time`; `submit` chunks the manifest round-robin across them |
| `slurm.venv` / `slurm.modules` / `slurm.exports` | environment lines in the generated scripts (`venv` = the venv to activate and to run with; empty = the submitter's interpreter) |
| `project` | project root (contains `bids/`, `derivatives/`, `eegcpm/configs/`) |
| `preprocessing_config`, `source_config`, `features_config` | the stage config YAMLs |
| `stages` | the per-unit array stages (`features` is deliberately **not** allowed) |
| `logs_dir`, `state_db_dir`, `submissions_dir` | optional; default `<project>/hpc/{logs,state,submissions}` |

`hpc submit --dry-run` writes the chunk TSVs and sbatch scripts to
`<submissions_dir>/<timestamp>/` and prints the `sbatch` commands without
running anything — this is how the workflow is tested locally. `hpc run-unit
--dry-run` prints the exact stage commands for one manifest row. (`hpc run-unit`
is deliberately flag-only: the generated script bakes in absolute paths, so a
compute node never parses the batch config.)

## Per-unit stages and state isolation

Each array task runs `eegcpm hpc run-unit`, which:

- **skips a stage whose artifact already exists** (re-submitting a job is safe;
  a completed unit does no work);
- runs `preprocess` with `--state-db hpc/state/<unit.key>.db` — **one SQLite
  file per work unit**, so concurrent array tasks never share or lock a
  database;
- runs `preprocess` with `--sessions <ses> --runs <run>` so a unit processes
  exactly its own recording, and `epochs`/`source` with `--sessions <ses>`
  (their artifacts are per subject/session/task and combine the runs of that
  session);
- writes `hpc/units/<unit.key>.json` with per-stage
  `status`/`seconds`/`returncode`/`commands`;
- exits non-zero when a stage fails, so SLURM accounting shows the failure.

Artifacts checked per stage:

| Stage | Artifact |
|---|---|
| `preprocess` | `derivatives/preprocessing/<pipeline>/sub-<s>/ses-<ses>/task-<t>/run-<r>/<s>_ses-<ses>_run-<r>_preprocessing.json` (`success: true`, matching `config_hash` if required) |
| `epochs` | `derivatives/epochs/<pipeline>/<t>/sub-<s>/ses-<ses>/<s>_epo.fif` |
| `source` | `derivatives/source/<pipeline>/<t>/variant-<variant>/sub-<s>/ses-<ses>/<s>_roi_tc.npz` |

`features` is **task-level**: it writes one parquet per task for all subjects
(`derivatives/features/<pipeline>/source_erp/source_erp_<task>.parquet`).
Running it per array task would race on the same file, so it is deliberately
not a per-unit stage; `hpc finalize` refuses while any expected unit lacks a
complete source artifact (override with `--force`).

A **failed or excluded** preprocessing run has no `*_preprocessing.json`
success sidecar, so the unit always stays pending in `hpc prune` and is
re-attempted on the next submission. That is intentional (a failed unit must
not silently disappear); units that are scientifically excluded on purpose
should be filtered out of the manifest, or `hpc finalize --force` used once the
remaining source artifacts are in place.

## Multi-run sessions

The epochs/source artifacts are per (subject, session, task). When a task has
more than one run in a session, several units of that session would produce the
same artifact: submit one unit per session/task for those stages (the artifact
check makes the other units skip), or serialize them. The 25SusAttn dataset has
one recording per (subject, task), so this does not apply there.

## Disk estimate

Budget **~0.4–0.5 GB per work unit** (preprocessed FIF + ICA + epochs + source
ROI archives + QC pages; measured 0.42 GB/unit on the first 8 complete
25SusAttn units, 2026-10-17). So 1110 units (111 subjects × 10 tasks) need
roughly **450–550 GB** on the shared filesystem — check the project quota
before submitting the full array, and prune before re-submitting to avoid
duplicated work.

## What the cluster admins must provide

- real SLURM **partition names** and their **concurrency caps** (`%k` limits)
  in `slurm.partitions[]`;
- the **module/venv** lines (`slurm.modules`, `slurm.venv`) — leave `modules`
  empty if the environment is baked into the image;
- a writable project directory (`derivatives/`, `hpc/logs/`, `hpc/state/`) on a
  filesystem visible from the compute nodes, with enough quota for the
  artifacts above.

## Other script flavours (UI, same module)

The unified module also generates the two UI scripts, so there is no second
header implementation:

- `generate_slurm_script(...)` — the batch page's per-subject array (subjects
  loop over their tasks inside the task; byte-stable);
- `generate_command_slurm_script(...)` — the prediction page's one-off job that
  runs an arbitrary command with the same `#SBATCH` header + env setup.
