# HPC Batch Preprocessing Guide

This guide covers setting up and running EEGCPM batch preprocessing on an HPC cluster using SLURM.

---

## First-Time Setup

### 1. SSH to HPC

```bash
ssh your_username@hpclogin1.eduhk.hk
```

### 2. Load Anaconda

```bash
source /usr/share/modules/init/profile.sh
module load anaconda/25.1.1
```

### 3. Create Conda Environment

```bash
conda create -p /home/$USER/.conda/envs/eegcpm python=3.12 -y
```

### 4. Activate and Install EEGCPM

```bash
conda activate /home/$USER/.conda/envs/eegcpm
cd /path/to/eegcpm-dev/eegcpm-0.1
pip install -e ".[dev]"
```

### 5. Verify Installation

```bash
eegcpm --help
```

---

## Upload Data to HPC

### BIDS Data

If your BIDS data is on a local machine, sync it to HPC:

```bash
# From your local machine
rsync -avz --progress /path/to/local/bids/ \
    your_username@hpclogin1.eduhk.hk:/share/your_group/project/bids/
```

### EEGCPM Code

```bash
rsync -avz --progress /path/to/eegcpm-dev/ \
    your_username@hpclogin1.eduhk.hk:/share/your_group/eegcpm-dev/
```

---

## Generate SLURM Script

Use the Streamlit UI on your local machine:

1. Run the UI: `streamlit run eegcpm/ui/app.py`
2. Go to **Batch Preprocessing** page
3. Configure:
   - Select preprocessing pipeline
   - Select subjects (all or specific)
   - Select tasks (all or specific)
4. Fill in **HPC Settings**:
   - **BIDS root on HPC**: Path to your BIDS directory on HPC
   - **EEGCPM root on HPC**: Path to eegcpm-0.1 on HPC
   - **Conda environment**: Name of your conda env (e.g., `eegcpm`)
   - **Email**: For job notifications
   - **Partition/Time/Memory/CPUs**: Adjust based on your allocation
5. Switch to **HPC Script** tab → click **Download**

---

## Upload Script to HPC

```bash
# From your local machine
scp batch_preprocess_standard_slurm.sh \
    your_username@hpclogin1.eduhk.hk:/share/your_group/project/
```

---

## Submit Job

```bash
# SSH to HPC
ssh your_username@hpclogin1.eduhk.hk

# Navigate to where your script is
cd /share/your_group/project

# Submit the batch job
sbatch batch_preprocess_standard_slurm.sh
# Output: Submitted batch job 12345
```

The job will run as a SLURM array — one array task per subject, processing all selected tasks sequentially.

---

## Monitor Jobs

```bash
# Check running jobs
squeue -u $USER

# Job summary (after completion)
sacct -j <JOB_ID> --brief

# View output log for a specific subject (array index 0)
cat /path/to/bids/derivatives/preprocessing/standard/slurm_<JOB_ID>_0.out

# View error log
cat /path/to/bids/derivatives/preprocessing/standard/slurm_<JOB_ID>_0.err

# Check all failed tasks
sacct -j <JOB_ID> --state=FAILED --format=JobID,State,ExitCode,Elapsed
```

---

## Manage Jobs

```bash
# Cancel entire job (all subjects)
scancel <JOB_ID>

# Cancel a specific subject (e.g., array index 5)
scancel <JOB_ID>_5

# Resubmit after cancellation or failure
# (completed subjects are automatically skipped — no --force needed)
sbatch batch_preprocess_standard_slurm.sh
```

**Key behavior:**
- `scancel` + `sbatch` = pause/resume
- Completed subject+task combinations are tracked in the state database
- Only incomplete work is reprocessed on resubmission

---

## Check Results

```bash
# Count preprocessed files
find /path/to/bids/derivatives/preprocessing/standard -name "*_preprocessed.fif" | wc -l

# List QC reports
find /path/to/bids/derivatives/preprocessing/standard -name "*_qc.html"

# Download QC reports to local machine for viewing
rsync -avz your_username@hpclogin1.eduhk.hk:/share/your_group/project/bids/derivatives/preprocessing/standard/sub-*/ses-*/task-*/*_qc.html \
    ./qc_reports/
```

---

## Troubleshooting

### Job runs out of time

Increase the time limit in HPC Settings or edit the script:

```bash
#SBATCH --time=04:00:00  # Increase from default 02:00:00
```

### Job runs out of memory

Increase memory allocation:

```bash
#SBATCH --mem=32G  # Increase from default 16G
```

### Too many bad channels detected

Edit your preprocessing config on HPC:

```yaml
# In your preprocessing config file
bad_channels:
  method: variance
  variance_threshold: 10.0  # Increase from 5.0 (less aggressive)
  drop: false               # Interpolate instead of dropping
```

### Conda environment not found

Ensure the conda env path matches your HPC Settings:

```bash
# Check your conda envs
conda env list

# The path should match what's in the script
ls /home/$USER/.conda/envs/eegcpm/bin/python
```

### Permission denied

```bash
# Make script executable
chmod +x batch_preprocess_standard_slurm.sh

# Check directory permissions
ls -la /share/your_group/project/bids/derivatives/
```

---

## Tips

- **Jobs survive SSH disconnection** — safe to close your terminal after submitting
- **Email notifications** — you'll receive email on job completion or failure
- **Parallel jobs** — the `%N` in `--array=0-X%N` limits concurrent jobs (respect cluster etiquette)
- **Logs are per-subject** — each array task writes its own `.out` and `.err` file
- **No data loss on cancel** — EEGCPM tracks state per subject+task, so resubmitting resumes from where it stopped
