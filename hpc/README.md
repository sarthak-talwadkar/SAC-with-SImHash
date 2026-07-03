# Running the study on an HPC cluster (Slurm)

Written for Northeastern's Explorer cluster; everything is standard Slurm, so
only module/partition names should need adjusting elsewhere. Verify names on
your cluster with `sinfo -s` and `module avail python anaconda cuda`.

## One-time setup

```bash
ssh <you>@login.explorer.northeastern.edu
git clone <your-repo-url> && cd SAC-with-SImHash

module load anaconda3
conda create -y -n sac python=3.10
source activate sac
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU queue
# for GPU queues instead: pip install torch  (default CUDA wheels), match module load cuda
pip install -r requirements.txt

mkdir -p hpc/logs
```

MuJoCo installs headless from the `mujoco` wheel; no display or license needed
(training never renders).

## Submit a sweep

```bash
python hpc/make_manifest.py --phase mountaincar --seeds 10 > hpc/runs_mc.txt
N=$(wc -l < hpc/runs_mc.txt)
sbatch --array=0-$((N-1))%16 hpc/train_array.sbatch hpc/runs_mc.txt
```

Phases: `gate`, `mountaincar`, `sparse` (pass `--threshold`), `dense` — see
`make_manifest.py --help`. For the sparse phase, first submit only the `none`
lines and confirm the baseline FAILS at your threshold; a sparse task the
baseline solves discriminates nothing.

## Monitor / resume / collect

```bash
squeue -u $USER                          # queue state
tail -f hpc/logs/sac-explore_<jobid>_<task>.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

Timed-out or killed tasks: **resubmit the exact same sbatch command.**
Finished runs detect their completed `state.pt` and exit in seconds;
interrupted ones resume. Results land on scratch
(`/scratch/$USER/sac-simhash/results`); pull them back with:

```bash
rsync -av <you>@<transfer-node>:/scratch/<you>/sac-simhash/results/ ./results/
```

Scratch is typically purged after N days of inactivity — sync results off
promptly. Only `results/` (small CSVs) is needed for analysis; `checkpoints/`
can stay on scratch.

## Sizing notes

- MountainCar run (150k steps): well under 1 h on CPU; the 12 h limit is slack.
- MuJoCo 1M-step runs: ~2.5 h at ~110 fps on a local RTX GPU; budget 8-12 h on
  CPU or use the GPU partition (see comments in `train_array.sbatch`).
- 8 GB memory covers the largest case (1M-transition replay buffer ≈ 200 MB
  plus checkpoint serialization headroom).
