"""Evaluate a trained checkpoint: average return over N deterministic episodes,
with an optional saved video for qualitative inspection.

Usage
-----
  python evaluate.py --env MountainCarContinuous-v0 --bonus simhash --seed 0
  python evaluate.py --env HalfCheetah-v5 --bonus rnd --seed 0 --episodes 20 --video
  python evaluate.py --ckpt checkpoints/HalfCheetah-v5/none/seed0/best.pt --env HalfCheetah-v5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from config import make_sac_config
from utils import get_device, set_seed
from algo.sac import SAC
from envs.wrappers import make_env

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def locate_ckpt(args) -> str:
    if args.ckpt:
        return args.ckpt
    env_tag = args.env + ("_sparse" if args.sparse else "")
    return os.path.join(REPO_ROOT, "checkpoints", env_tag, args.bonus,
                        f"seed{args.seed}", args.which + ".pt")


def main():
    p = argparse.ArgumentParser(description="evaluate a trained SAC checkpoint")
    p.add_argument("--env", required=True)
    p.add_argument("--bonus", default="none", choices=["none", "simhash", "rnd", "icm"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None, help="explicit path (overrides env/bonus/seed lookup)")
    p.add_argument("--which", default="best", choices=["best", "final"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--sparse-threshold", type=float, default=5.0)
    p.add_argument("--video", action="store_true", help="save mp4(s) under videos/ (needs ffmpeg/moviepy)")
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_cuda=not args.no_cuda)
    set_seed(args.seed + 999)

    ckpt_path = locate_ckpt(args)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    render_mode = "rgb_array" if args.video else None
    env = make_env(args.env, seed=args.seed + 999, sparse=args.sparse,
                   sparse_threshold=args.sparse_threshold, render_mode=render_mode)
    if args.video:
        from gymnasium.wrappers import RecordVideo
        vdir = os.path.join(REPO_ROOT, "videos", f"{args.env}_{args.bonus}_seed{args.seed}")
        env = RecordVideo(env, video_folder=vdir, episode_trigger=lambda e: True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SAC(obs_dim, act_dim, make_sac_config(args.env), device)
    agent.load(ckpt_path)
    print(f"loaded {ckpt_path}  (obs={obs_dim}, act={act_dim}, device={device})")

    returns, lengths = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, total, length = False, 0.0, 0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            length += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(length)
        print(f"  ep {ep + 1:2d}: return {total:9.2f}   length {length}")
    env.close()

    returns = np.array(returns)
    tag = args.env + ("_sparse" if args.sparse else "")
    print(f"\n{tag} | {args.bonus} | seed {args.seed} | {args.episodes} episodes")
    print(f"  return: {returns.mean():.2f} +/- {returns.std():.2f}  "
          f"(min {returns.min():.2f}, max {returns.max():.2f})")
    if args.video:
        print("  video(s) saved under videos/")


if __name__ == "__main__":
    main()
