#!/usr/bin/env python3
"""Generate a run manifest for the Slurm array job (hpc/train_array.sbatch).

Each output line is one run's train.py arguments; the sbatch script picks the
line matching its $SLURM_ARRAY_TASK_ID. Typical use, from the repo root on
the cluster:

    python hpc/make_manifest.py --phase mountaincar --seeds 10 > hpc/runs_mc.txt
    N=$(wc -l < hpc/runs_mc.txt)
    sbatch --array=0-$((N-1))%16 hpc/train_array.sbatch hpc/runs_mc.txt

Phases
------
gate        : SimHash geometry ablation on MountainCar (norm / bias / affine,
              plus an affine beta=0.2 arm) -- mirrors run_gate.ps1.
mountaincar : full method matrix on MountainCar, including all hash-geometry
              ablation arms. The paper-grade rerun (use --seeds 10).
sparse      : sparse-HalfCheetah matrix. Pass --threshold; runs are tagged
              t<threshold> so different thresholds never collide on disk.
              Verify baseline SAC FAILS at your threshold before launching
              the full matrix (run the 'none' lines first).
dense       : dense MuJoCo matrix (HalfCheetah / Hopper / Walker2d).

Schedules match the local runs (run_all.ps1) so curves are comparable.
"""
from __future__ import annotations

import argparse

# (bonus, simhash-variant) arms. Ablation arms only run on MountainCar, the
# diagnostic env; the other phases compare methods with the fixed hash.
ARMS_ABLATION = [("none", None), ("simhash", "raw"), ("simhash", "norm"),
                 ("simhash", "bias"), ("simhash", "affine"),
                 ("rnd", None), ("icm", None)]
ARMS_METHODS = [("none", None), ("simhash", "raw"), ("simhash", "affine"),
                ("rnd", None), ("icm", None)]

MC_SCHED = "--steps 150000 --eval-every 20000 --eval-episodes 5 --ckpt-every 50000"
MJ_SCHED = "--steps 1000000 --eval-every 20000 --eval-episodes 5 --ckpt-every 50000"


def line(env: str, bonus: str, variant: str | None, seed: int, sched: str,
         extra: str = "") -> str:
    parts = [f"--env {env}", f"--bonus {bonus}"]
    if variant is not None:
        parts.append(f"--simhash-variant {variant}")
    parts += [f"--seed {seed}", sched]
    if extra:
        parts.append(extra)
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True,
                   choices=["gate", "mountaincar", "sparse", "dense"])
    p.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per arm")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="sparse phase: forward-distance reward threshold")
    args = p.parse_args()
    seeds = range(args.seeds)
    out: list[str] = []

    if args.phase == "gate":
        for variant, extra in [("norm", ""), ("bias", ""), ("affine", ""),
                               ("affine", "--beta 0.2 --tag b0.2")]:
            for s in seeds:
                out.append(line("MountainCarContinuous-v0", "simhash", variant,
                                s, MC_SCHED, extra))

    elif args.phase == "mountaincar":
        for bonus, variant in ARMS_ABLATION:
            for s in seeds:
                out.append(line("MountainCarContinuous-v0", bonus, variant,
                                s, MC_SCHED))

    elif args.phase == "sparse":
        t = args.threshold
        tag = f"t{t:g}"
        extra = f"--sparse --sparse-threshold {t:g} --tag {tag}"
        for bonus, variant in ARMS_METHODS:
            for s in seeds:
                out.append(line("HalfCheetah-v5", bonus, variant, s,
                                MJ_SCHED, extra))

    elif args.phase == "dense":
        for env in ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]:
            for bonus, variant in ARMS_METHODS:
                for s in seeds:
                    out.append(line(env, bonus, variant, s, MJ_SCHED))

    print("\n".join(out))


if __name__ == "__main__":
    main()
