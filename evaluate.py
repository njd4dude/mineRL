"""
Compare models fairly: play each one on the SAME set of Minecraft worlds (fixed seeds), plus a
random-action baseline, and print logs collected per world.

  python evaluate.py --models random runs/model_A runs/model_B/treechop_ckpt_30000_steps.zip
  python evaluate.py --models random latest --seeds 1 2 3 4 5 6 7 8 --steps 2000
  python evaluate.py --models random latest --workers 4      # faster if nothing else is running

Each model can be "random", "latest", a run folder (its newest model) or a .zip checkpoint.
Results are also saved to runs/evals/<date>_<time>.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

import numpy as np

import train_treechop as T

_env = None      # one Minecraft per worker process, reused for every job
_models = {}     # loaded models per worker, by path


def _init_worker(mc_mem):
    global _env
    T.set_minecraft_memory(mc_mem)
    _env = T.TreechopEnv()


def _run_job(job):
    """Play one policy on one seeded world. Returns (policy label, seed, logs, seconds)."""
    label, path, seed, steps = job
    import torch
    from stable_baselines3 import PPO
    start = time.time()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)  # the model samples actions; fix that randomness too
    if path is not None and path not in _models:
        _models[path] = PPO.load(path, device="cpu")  # tiny network, CPU is fine
    obs, _ = _env.reset(seed=seed)
    info = {"logs": 0}
    for _ in range(steps):
        if path is None:
            action = rng.randint(len(T.ACTIONS))
        else:
            action = int(_models[path].predict(obs, deterministic=False)[0])
        obs, _, done, _, info = _env.step(action)
        if done:
            break
    return label, seed, info["logs"], time.time() - start


def label_for(arg, path):
    if path is None:
        return "random"
    run = os.path.basename(os.path.dirname(path))
    return f"{run}/{os.path.basename(path).replace('.zip', '')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["random", "latest"],
                        help='"random", "latest", run folders or .zip checkpoints')
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="world seeds to test on")
    parser.add_argument("--steps", type=int, default=1000,
                        help="agent steps per world (a full episode is 2000)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Minecraft instances in parallel (about 3.5 GB RAM each)")
    parser.add_argument("--mc-mem", default="2G")
    args = parser.parse_args()

    policies = []
    for arg in args.models:
        path = None if arg == "random" else T.resolve_model(arg)
        if arg != "random" and (not path or not os.path.exists(path)):
            print(f"Skipping {arg}: no model found")
            continue
        policies.append((label_for(arg, path), path))

    # Each Minecraft instance plus its Python worker uses about 4 GB; don't run WSL out of memory
    with open("/proc/meminfo") as f:
        avail_gb = int(next(l for l in f if l.startswith("MemAvailable")).split()[1]) / 1024 ** 2
    if args.workers * 4 > avail_gb - 1:
        print(f"Not enough free memory for {args.workers} Minecraft instance(s): {avail_gb:.1f} GB available, "
              f"need about {args.workers * 4 + 1} GB. Use fewer --workers or stop other runs.")
        return

    jobs = [(label, path, seed, args.steps) for label, path in policies for seed in args.seeds]
    print(f"{len(policies)} policies x {len(args.seeds)} worlds = {len(jobs)} episodes of "
          f"{args.steps} steps, {args.workers} Minecraft instance(s)")

    results = {label: {} for label, _ in policies}
    ctx = mp.get_context("forkserver")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.mc_mem,)) as pool:
        for label, seed, logs, secs in pool.imap_unordered(_run_job, jobs):
            results[label][seed] = logs
            print(f"  {label:55s} seed {seed:3d}: {logs:3d} logs  ({secs:.0f}s)", flush=True)

    # Table: one row per policy, one column per seed
    print("\nLogs collected per world (same worlds for every policy):")
    header = f"{'policy':55s} " + " ".join(f"s{s:<4d}" for s in args.seeds) + "   mean   std"
    print(header)
    print("-" * len(header))
    for label, _ in policies:
        vals = [results[label][s] for s in args.seeds]
        print(f"{label:55s} " + " ".join(f"{v:<5d}" for v in vals) + f"  {np.mean(vals):5.1f}  {np.std(vals):4.1f}")

    os.makedirs(os.path.join(T.RUNS_DIR, "evals"), exist_ok=True)
    out = os.path.join(T.RUNS_DIR, "evals", datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "steps": args.steps, "results": results}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
