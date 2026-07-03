"""Train SAC (optionally with an exploration bonus) on one env / one seed.

Examples
--------
  # baseline SAC
  python train.py --env HalfCheetah-v5 --bonus none --seed 0

  # SAC + SimHash on the sparse hard-exploration task
  python train.py --env MountainCarContinuous-v0 --bonus simhash --seed 0

  # quick smoke test (tiny run)
  python train.py --env Pendulum-v1 --bonus rnd --steps 3000 --start-steps 500 \
                  --eval-every 1000 --eval-episodes 3

Outputs
-------
  results/<env>/<bonus>/seed<k>/progress.csv   <- learning curve + diagnostics
  checkpoints/<env>/<bonus>/seed<k>/final.pt   <- trained networks
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch

from config import make_sac_config, BONUS_CONFIGS
from utils import set_seed, get_device, CSVLogger
from algo.sac import SAC
from algo.replay_buffer import ReplayBuffer
from algo.exploration import make_bonus
from envs.wrappers import make_env

# Anchor outputs to the repo root so results land here no matter where you
# launch the script from (the shell's working directory is irrelevant).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate(agent: SAC, env, episodes: int):
    """Average return over `episodes` deterministic (greedy) episodes."""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def parse_args():
    p = argparse.ArgumentParser(description="SAC + exploration-bonus trainer")
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--bonus", default="none", choices=["none", "simhash", "rnd", "icm"])
    p.add_argument("--seed", type=int, default=0)
    # optional overrides of config.py defaults (None -> use config)
    p.add_argument("--steps", type=int, default=None, help="total environment steps")
    p.add_argument("--start-steps", type=int, default=None, help="random-action warmup")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    # exploration-bonus hyperparameter overrides (handy for sweeps)
    p.add_argument("--beta", type=float, default=None, help="override bonus scale (beta)")
    p.add_argument("--simhash-k", type=int, default=None, help="override SimHash code length (bits)")
    p.add_argument("--simhash-variant", default=None, choices=["raw", "norm", "bias", "affine"],
                   help="SimHash hash geometry: raw=published sgn(As), norm=normalized obs, "
                        "bias=affine hyperplanes, affine=norm+bias (the proposed fix)")
    p.add_argument("--tag", default="", help="suffix for the bonus output folder (e.g. b1.0 for a sweep)")
    # sparse-reward variant for MuJoCo
    p.add_argument("--sparse", action="store_true", help="use sparse goal reward")
    p.add_argument("--sparse-threshold", type=float, default=5.0)
    # io / device
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="resume from the saved training state (state.pt) if it exists")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="env steps between full-state checkpoints (default: eval_every)")
    return p.parse_args()


def save_full_state(path, step, agent, bonus, buffer, recent_returns, best_eval, meta):
    """Atomically write the complete training state needed to resume."""
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = dict(
        step=step,
        agent=agent.state_dict(),
        buffer=buffer.state_dict(),
        bonus=(bonus.state_dict() if bonus is not None else None),
        recent_returns=list(recent_returns),
        best_eval=best_eval,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=cuda_rng),
        meta=meta,
    )
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write cannot corrupt the good file


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)

    cfg = make_sac_config(
        args.env,
        total_steps=args.steps,
        start_steps=args.start_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    # run identity (sparse runs are tagged so they never overwrite dense runs;
    # non-raw SimHash variants likewise get their own folder, e.g. simhash_affine)
    env_tag = args.env + ("_sparse" if args.sparse else "")
    variant = (args.simhash_variant
               if args.bonus == "simhash" and args.simhash_variant not in (None, "raw")
               else None)
    bonus_dir = (args.bonus + (f"_{variant}" if variant else "")
                 + (f"_{args.tag}" if args.tag else ""))
    run_dir_results = os.path.join(args.out_dir, env_tag, bonus_dir, f"seed{args.seed}")
    run_dir_ckpt = os.path.join(args.ckpt_dir, env_tag, bonus_dir, f"seed{args.seed}")
    os.makedirs(run_dir_ckpt, exist_ok=True)

    env = make_env(args.env, seed=args.seed, sparse=args.sparse, sparse_threshold=args.sparse_threshold)
    eval_env = make_env(args.env, seed=args.seed + 10_000, sparse=args.sparse, sparse_threshold=args.sparse_threshold)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SAC(obs_dim, act_dim, cfg, device)
    bonus_cls = BONUS_CONFIGS[args.bonus]
    bonus_cfg = bonus_cls() if bonus_cls is not None else None
    if bonus_cfg is not None:
        if args.beta is not None:
            bonus_cfg.beta = args.beta
        if args.simhash_k is not None and args.bonus == "simhash":
            bonus_cfg.k = args.simhash_k
        if args.simhash_variant is not None and args.bonus == "simhash":
            bonus_cfg.normalize = args.simhash_variant in ("norm", "affine")
            bonus_cfg.bias = args.simhash_variant in ("bias", "affine")
    bonus = make_bonus(args.bonus, obs_dim, act_dim, bonus_cfg, device, seed=args.seed)

    buffer = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device)
    state_path = os.path.join(run_dir_ckpt, "state.pt")
    ckpt_every = args.ckpt_every if args.ckpt_every is not None else cfg.eval_every

    # ---- resume from a saved full-state checkpoint, if requested ----
    recent_returns = deque(maxlen=20)
    best_eval = -float("inf")
    start_step = 1
    resumed = False
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        agent.load_state_dict(st["agent"])
        buffer.load_state_dict(st["buffer"])
        if bonus is not None and st.get("bonus") is not None:
            bonus.load_state_dict(st["bonus"])
        recent_returns = deque(st.get("recent_returns", []), maxlen=20)
        best_eval = st.get("best_eval", -float("inf"))
        rng = st.get("rng", {})
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([c.cpu() for c in rng["cuda"]])
        except Exception as e:
            print(f"  (warning: RNG state not fully restored: {e})")
        start_step = int(st["step"]) + 1
        resumed = True
        print(f"RESUMED [{env_tag} | {bonus_dir} | seed {args.seed}] from step {st['step']} "
              f"(buffer={len(buffer)}, best_eval={best_eval:.2f})")
    elif args.resume:
        print(f"  (--resume set but no state.pt at {state_path}; starting fresh)")

    logger = CSVLogger(os.path.join(run_dir_results, "progress.csv"), append=resumed)

    if not resumed:
        with open(os.path.join(run_dir_results, "run_config.json"), "w") as f:
            json.dump(dict(args=vars(args), obs_dim=obs_dim, act_dim=act_dim,
                           total_steps=cfg.total_steps, start_steps=cfg.start_steps,
                           bonus=args.bonus,
                           bonus_cfg=(vars(bonus_cfg) if bonus_cfg is not None else None),
                           device=str(device)), f, indent=2)
        print(f"[{env_tag} | {bonus_dir} | seed {args.seed}] obs={obs_dim} act={act_dim} "
              f"device={device} steps={cfg.total_steps}")

    if start_step > cfg.total_steps:
        print(f"  already at {start_step - 1}/{cfg.total_steps} steps; nothing to do "
              f"(raise --steps to train further).")
        logger.close()
        return

    meta = dict(env=args.env, bonus=args.bonus, tag=args.tag, seed=args.seed,
                total_steps=cfg.total_steps)
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    metrics = dict(critic_loss=0.0, actor_loss=0.0, alpha=agent.alpha,
                   entropy=0.0, q_mean=0.0, r_int_mean=0.0)
    t_last = time.time()
    step_last = start_step - 1

    for t in range(start_step, cfg.total_steps + 1):
        # ---- act: random warmup, then policy ----
        if t <= cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        # IMPORTANT: only TRUE termination cuts the bootstrap; time-limit
        # truncation should still bootstrap from the next state.
        buffer.add(obs, action, reward, next_obs, float(terminated))
        if bonus is not None:
            bonus.observe(next_obs)
        obs = next_obs

        if terminated or truncated:
            recent_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        # ---- learn ----
        if t >= cfg.update_after and t % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                metrics = agent.update(buffer.sample(cfg.batch_size), bonus=bonus)

        # ---- evaluate + log ----
        if t % cfg.eval_every == 0:
            eval_mean, eval_std = evaluate(agent, eval_env, cfg.eval_episodes)
            train_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            now = time.time()
            fps = (t - step_last) / max(now - t_last, 1e-9)
            t_last, step_last = now, t

            row = dict(step=t, time_s=round(now, 1),
                       eval_return=round(eval_mean, 3), eval_std=round(eval_std, 3),
                       train_return=round(train_ret, 3),
                       **{k: round(float(v), 5) for k, v in metrics.items()})
            if bonus is not None:
                row.update({k: round(float(v), 5) for k, v in bonus.diagnostics().items()})
            logger.write(row)

            print(f"  step {t:>8d}/{cfg.total_steps} | eval {eval_mean:8.2f} +/- {eval_std:6.2f} "
                  f"| train {train_ret:8.2f} | alpha {metrics['alpha']:.3f} "
                  f"| r_int {metrics['r_int_mean']:.4f} | {fps:5.0f} fps")

            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save(os.path.join(run_dir_ckpt, "best.pt"))

        # ---- periodic full-state checkpoint so --resume can continue ----
        if t % ckpt_every == 0:
            save_full_state(state_path, t, agent, bonus, buffer,
                            recent_returns, best_eval, meta)

    agent.save(os.path.join(run_dir_ckpt, "final.pt"))
    save_full_state(state_path, cfg.total_steps, agent, bonus, buffer,
                    recent_returns, best_eval, meta)
    logger.close()
    print(f"DONE [{env_tag} | {bonus_dir} | seed {args.seed}] best eval = {best_eval:.2f}")
    print(f"  curve : {os.path.join(run_dir_results, 'progress.csv')}")
    print(f"  model : {os.path.join(run_dir_ckpt, 'final.pt')}")


if __name__ == "__main__":
    main()
