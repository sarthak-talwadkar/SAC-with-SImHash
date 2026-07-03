"""Aggregate progress.csv across seeds into publication figures + a results table.

It walks results/<env>/<bonus>/seed<k>/progress.csv, interpolates each seed onto
a common step grid, and plots mean +/- 95% CI learning curves (one line per
bonus, one figure per env). It also writes a comparison table with final return,
best return, sample-efficiency (area under the learning curve), and the delta of
each bonus versus the no-bonus baseline.

Usage
-----
  python plot.py --all
  python plot.py --env MountainCarContinuous-v0
  python plot.py --all --smooth 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: just write PNG files, no GUI window
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_TRAPZ = getattr(np, "trapezoid", np.trapz)  # numpy 2 renamed trapz -> trapezoid

import itertools

BONUS_ORDER = ["none", "simhash", "rnd", "icm"]
BONUS_COLORS = {"none": "#666666", "simhash": "#1f77b4", "rnd": "#d62728", "icm": "#2ca02c"}
BONUS_LABEL = {"none": "SAC (no bonus)", "simhash": "SAC+SimHash",
               "rnd": "SAC+RND", "icm": "SAC+ICM"}
_FALLBACK_COLORS = itertools.cycle(["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e"])
_COLOR_CACHE = {}


def order_bonuses(present):
    """Canonical bonuses first, then any extras (e.g. sweep tags like simhash_b1.0)."""
    known = [b for b in BONUS_ORDER if b in present]
    extra = sorted(b for b in present if b not in BONUS_ORDER)
    return known + extra


def color_for(bonus):
    if bonus in BONUS_COLORS:
        return BONUS_COLORS[bonus]
    if bonus not in _COLOR_CACHE:
        _COLOR_CACHE[bonus] = next(_FALLBACK_COLORS)
    return _COLOR_CACHE[bonus]


def label_for(bonus):
    return BONUS_LABEL.get(bonus, bonus)


def read_progress(path):
    steps, rets = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rets.append(float(row["eval_return"]))
    return np.array(steps), np.array(rets)


def discover(results_dir):
    """Map env -> bonus -> [progress.csv paths] from the directory layout."""
    envs = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(results_dir, "*", "*", "seed*", "progress.csv")
    for path in glob.glob(pattern):
        parts = path.replace("\\", "/").split("/")
        env, bonus = parts[-4], parts[-3]
        envs[env][bonus].append(path)
    return envs


def aggregate(paths, n_points=200):
    """Interpolate every seed onto a shared grid; return (grid, mean, sem, n)."""
    curves = [read_progress(p) for p in paths]
    curves = [(s, r) for s, r in curves if len(s) >= 2]
    if not curves:
        return None
    lo = max(s.min() for s, _ in curves)
    hi = min(s.max() for s, _ in curves)   # no extrapolation beyond shortest run
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_points)
    stacked = np.stack([np.interp(grid, s, r) for s, r in curves])  # [n_seeds, n_points]
    mean = stacked.mean(0)
    n = stacked.shape[0]
    sem = stacked.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return grid, mean, sem, n


def smooth(y, w):
    if w <= 1:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot_env(env, bonus_paths, out_dir, smooth_w):
    plt.figure(figsize=(7, 5))
    plotted = False
    for bonus in order_bonuses(bonus_paths):
        agg = aggregate(bonus_paths[bonus])
        if agg is None:
            continue
        grid, mean, sem, n = agg
        m = smooth(mean, smooth_w)
        color = color_for(bonus)
        plt.plot(grid, m, color=color, label=f"{label_for(bonus)} (n={n})")
        if n > 1:
            ci = 1.96 * sem
            plt.fill_between(grid, m - ci, m + ci, color=color, alpha=0.2)
        plotted = True
    if not plotted:
        plt.close()
        return None
    plt.xlabel("environment steps")
    plt.ylabel("evaluation return")
    plt.title(env)
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{env}_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def final_stats(paths, last_frac=0.1):
    """Per-seed final return (avg of last 10% of evals), best return, and AULC,
    aggregated across seeds into (mean, std)."""
    finals, bests, aulcs = [], [], []
    for p in paths:
        s, r = read_progress(p)
        if len(r) == 0:
            continue
        k = max(1, int(len(r) * last_frac))
        finals.append(r[-k:].mean())
        bests.append(r.max())
        aulcs.append(_TRAPZ(r, s) / (s.max() - s.min()) if len(s) >= 2 else r.mean())
    agg = lambda x: (float(np.mean(x)), float(np.std(x)))
    return agg(finals), agg(bests), agg(aulcs), len(finals)


def make_table(envs, out_dir):
    header = ["env", "method", "seeds", "final_mean", "final_std",
              "best_mean", "aulc_mean", "delta_vs_none"]
    rows = []
    for env in sorted(envs):
        per = {}
        for bonus in order_bonuses(envs[env]):
            (fm, fs), (bm, _bs), (am, _as), n = final_stats(envs[env][bonus])
            per[bonus] = dict(n=n, fm=fm, fs=fs, bm=bm, am=am)
        base = per.get("none", {}).get("fm")
        for bonus in order_bonuses(per):
            d = per[bonus]
            delta = (d["fm"] - base) if (base is not None and bonus != "none") else float("nan")
            rows.append([env, label_for(bonus), d["n"],
                         d["fm"], d["fs"], d["bm"], d["am"], delta])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r[:3] + [f"{x:.3f}" for x in r[3:]])

    lines = ["| Env | Method | Seeds | Final (mean±std) | Best | AULC | Δ vs none |",
             "|---|---|---|---|---|---|---|"]
    for env, method, n, fm, fs, bm, am, delta in rows:
        dstr = "—" if np.isnan(delta) else f"{delta:+.2f}"
        lines.append(f"| {env} | {method} | {n} | {fm:.2f} ± {fs:.2f} | {bm:.2f} | {am:.2f} | {dstr} |")
    md = "\n".join(lines)
    # UTF-8 so the pretty ± and Δ survive on Windows (default cp1252 cannot encode them).
    with open(os.path.join(out_dir, "results_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return md


def main():
    p = argparse.ArgumentParser(description="aggregate runs into figures + table")
    p.add_argument("--env", default=None, help="plot a single env (default: all found)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figures"))
    p.add_argument("--smooth", type=int, default=1, help="rolling-mean window for curves")
    args = p.parse_args()

    envs = discover(args.results_dir)
    if not envs:
        print(f"no runs found under {args.results_dir}")
        return
    if args.env and not args.all:
        if args.env not in envs:
            print(f"env '{args.env}' not found. Available: {sorted(envs)}")
            return
        envs = {args.env: envs[args.env]}

    figs = [plot_env(env, envs[env], args.out_dir, args.smooth) for env in sorted(envs)]
    md = make_table(envs, args.out_dir)
    print("figures written:")
    for f in figs:
        if f:
            print("  ", f)
    # ASCII-safe for the Windows console; the saved .md keeps the pretty glyphs.
    console_md = md.replace("±", "+/-").replace("Δ", "delta").replace("—", "-")
    print("\n" + console_md)
    print(f"\ntable: {os.path.join(args.out_dir, 'results_table.md')}")


if __name__ == "__main__":
    main()
