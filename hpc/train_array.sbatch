#!/bin/bash
# =============================================================================
# Slurm array job: one train.py run per array index, read from a manifest.
#
# Usage (from the repo root on the cluster; hpc/logs must exist):
#     mkdir -p hpc/logs
#     python hpc/make_manifest.py --phase mountaincar --seeds 10 > hpc/runs_mc.txt
#     N=$(wc -l < hpc/runs_mc.txt)
#     sbatch --array=0-$((N-1))%16 hpc/train_array.sbatch hpc/runs_mc.txt
#
# %16 throttles to 16 concurrent runs -- be a good citizen on shared queues.
#
# Runs checkpoint every --ckpt-every steps and are launched with --resume, so
# if a job hits the time limit, JUST RESUBMIT THE SAME COMMAND: finished runs
# exit within seconds, interrupted ones continue from their last state.pt.
#
# GPU variant: MountainCar-sized runs are fine on CPU. For the 1M-step MuJoCo
# runs, swap the partition block for something like
#     #SBATCH --partition=gpu
#     #SBATCH --gres=gpu:1
#     #SBATCH --time=08:00:00
# and add `module load cuda/12.1` (match your cluster + installed torch).
# =============================================================================
#SBATCH --job-name=sac-explore
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=hpc/logs/%x_%A_%a.out

# NOTE: no -u here -- conda's activate scripts reference unset vars (PS1 etc.)
set -eo pipefail

MANIFEST=${1:?usage: sbatch --array=0-N%K hpc/train_array.sbatch <manifest.txt>}
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$ARGS" ]; then
    echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

# ---- environment: adjust module/env names to your cluster ------------------
module purge
module load python/3.13.5          
source ~/venvs/sac/bin/activate
echo "python: $(which python)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- outputs on scratch: state.pt files are large (up to ~250 MB each) -----
SCRATCH_BASE=${SCRATCH_BASE:-/scratch/$USER/sac-simhash}
mkdir -p "$SCRATCH_BASE/results" "$SCRATCH_BASE/checkpoints"

cd "$SLURM_SUBMIT_DIR"
echo "[task $SLURM_ARRAY_TASK_ID] python train.py $ARGS"
python train.py $ARGS --resume \
    --out-dir  "$SCRATCH_BASE/results" \
    --ckpt-dir "$SCRATCH_BASE/checkpoints"
