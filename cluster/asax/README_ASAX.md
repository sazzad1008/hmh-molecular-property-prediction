# ASA-X Cluster Run Guide

This folder contains PBS-ready files for running the TDC ADMET HMH experiments on ASA-X.

Use the cluster like this:

1. SSH into ASA-X.
2. Clone or copy this repository.
3. Create the Python environment once.
4. Submit PBS jobs with `qsub`.
5. Monitor with `qstat`.
6. Read metrics from the output directory after jobs finish.

## 1. Login

```bash
ssh uahmsh001@asax.asc.edu
```

Approve Duo and complete the password change if prompted.

Do not run full training on the login node. Use PBS jobs.

## 2. Copy Or Clone The Project

Recommended:

```bash
git clone https://github.com/sazzad1008/hmh-molecular-property-prediction.git
cd hmh-molecular-property-prediction
```

If `git` cannot access GitHub from ASA-X, copy from your laptop:

```bash
scp -r /Users/mdsazzadhossen/projects/hmh-molecular-property-prediction \
  uahmsh001@asax.asc.edu:~/
```

## 3. Create The Environment

From the repo root on ASA-X:

```bash
bash cluster/asax/setup_env.sh
```

This creates:

```text
.venv-tdc-hmh/
```

If ASA-X uses environment modules, you may need to edit `setup_env.sh` and uncomment the correct `module load python/...` line.

## 4. Smoke Test

Submit a tiny run first:

```bash
qsub cluster/asax/pbs_smoke_test.pbs
```

Check status:

```bash
qstat -u uahmsh001
```

After it finishes, inspect logs:

```bash
ls -lh logs/
cat logs/*smoke*
```

## 5. Run One Full Dataset

Edit this line in `pbs_train_one.pbs`:

```bash
TASK="Pgp"
```

Then submit:

```bash
qsub cluster/asax/pbs_train_one.pbs
```

## 6. Run Many Datasets With A PBS Array

Submit:

```bash
qsub cluster/asax/pbs_array_admet.pbs
```

The array reads tasks from:

```text
cluster/asax/tasks_admet_full.txt
```

If your ASA-X PBS uses zero-based array IDs instead of one-based IDs, change the array line in `pbs_array_admet.pbs`.

## 7. Where Outputs Go

By default, jobs write to:

```text
$HOME/hmh_admet_runs/
$HOME/hmh_admet_cache/
$HOME/hmh_admet_data/
```

If ASA-X gives you a scratch filesystem, edit the PBS scripts and change:

```bash
BASE_DIR="$HOME"
```

to something like:

```bash
BASE_DIR="$SCRATCH"
```

or whatever ASC recommends.

## 8. Useful ASA-X Commands

```bash
hostname
pwd
ls
quota
usage
qlimits
qstat -u uahmsh001
qdel JOB_ID
```

## 9. Important Notes

- The expensive part is HMH feature computation.
- Features are cached after the first run.
- Re-running the same dataset with the same HMH parameters will load cached `.pt` files.
- CPU cores help HMH feature generation. GPU is optional for this MLP version.
- For official TDC comparison, use scaffold splits later. This script currently uses stratified random splits.

