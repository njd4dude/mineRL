"""
Compare models fairly: play each one on the SAME set of Minecraft worlds (fixed seeds), plus a
random-action baseline, and print logs collected per world.

  python evaluate.py --models random runs/model_A runs/model_B/treechop_ckpt_30000_steps.zip
  python evaluate.py --models random latest --seeds 1 2 3 4 5 6 7 8 --steps 2000
  python evaluate.py --models random latest --workers 4      # faster if nothing else is running
  python evaluate.py --models latest --record                # also save every game as an .mp4

Each model can be "random", "latest", a run folder (its newest model) or a .zip checkpoint.
Results are also saved to runs/evals/<date>_<time>.json. With --record, every game is also saved to
runs/evals/<date>_<time>_videos/<policy>_seed<N>.mp4, showing the game, the step, logs so far, the
chosen action, and (top right) the 64x64 image the agent actually sees.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

import cv2
import numpy as np

import train_treechop as T

# Videos (--record): a 1000-step game is 4000 game ticks, so keep them small: 640x360 (320x180
# scaled up with sharp pixels), every 2nd game tick at 10 fps, which still plays at real game
# speed (Minecraft runs 20 ticks per second). About 80 MB per game.
VIDEO_SCALE = 2
VIDEO_TICK_STRIDE = 2
VIDEO_FPS = 20 // VIDEO_TICK_STRIDE


def open_video(path):
    """An OpenCV video writer for an .mp4 file. Uses MPEG-4 (mp4v): the pip OpenCV has no H.264
    encoder. Plays in VLC and the Windows Media Player / Movies & TV apps."""
    size = (T.RENDER_RES[0] * VIDEO_SCALE, T.RENDER_RES[1] * VIDEO_SCALE)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV can't write .mp4 files here ({path})")
    return writer

_env = None      # one Minecraft per worker process, reused for every job
_models = {}     # loaded models per worker, by path


def _init_worker(mc_mem):
    global _env
    T.set_minecraft_memory(mc_mem)
    _env = T.TreechopEnv()


def _annotate(pov, step, steps, logs, action):
    """Full game frame enlarged, with a text bar and the agent's own 64x64 view in the corner."""
    big = cv2.resize(pov, None, fx=VIDEO_SCALE, fy=VIDEO_SCALE, interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
    agent_view = cv2.resize(T.process_pov(pov), (128, 128), interpolation=cv2.INTER_NEAREST)
    big[6:134, -134:-6] = cv2.cvtColor(agent_view, cv2.COLOR_RGB2BGR)
    cv2.rectangle(big, (0, 0), (390, 46), (0, 0, 0), -1)
    act = T.describe_action(action) if action is not None else "-"
    for i, text in enumerate([f"step {step}/{steps}   logs {logs}", f"action: {act}"]):
        cv2.putText(big, text, (8, 19 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return big


def _run_job(job):
    """Play one policy on one seeded world. Returns (policy label, seed, logs, seconds, step it died
    at or None)."""
    label, path, seed, steps, video_path = job
    import torch
    from stable_baselines3 import PPO
    start = time.time()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)  # the model samples actions; fix that randomness too
    if path is not None and path not in _models:
        _models[path] = PPO.load(path, device="cpu")  # tiny network, CPU is fine
    writer = None
    state = {"step": 0, "action": None, "tick": 0}   # what to draw on each recorded frame
    info = {"logs": 0}
    if video_path:
        writer = open_video(video_path)

        def on_frame(pov):
            state["last_pov"] = pov
            state["last_written"] = state["tick"] % VIDEO_TICK_STRIDE == 0
            if state["last_written"]:
                writer.write(_annotate(pov, state["step"], steps, _env.prev_logs, state["action"]))  # updated every tick
            state["tick"] += 1
        _env.on_frame = on_frame
    try:
        obs, _ = _env.reset(seed=seed)
        for step in range(1, steps + 1):
            if path is None:
                action = [rng.randint(n) for n in T.ACTION_DIMS]
            else:
                action = _models[path].predict(obs, deterministic=False)[0]
            state["step"], state["action"] = step, action
            obs, _, done, truncated, info = _env.step(action)
            if done or truncated:
                break
    finally:
        _env.on_frame = None
        if writer:
            # Always end on the game's very last frame (it may fall between recorded ticks), held
            # for 2 seconds, saying how the game ended
            if "last_pov" in state:
                ending = ("DIED (lost the axe)" if info.get("died") else
                          "Minecraft failed" if info.get("env_error") else
                          f"won: {T.LOGS_TO_WIN} logs" if info["logs"] >= T.LOGS_TO_WIN else "out of steps")
                frame = _annotate(state["last_pov"], state["step"], steps, _env.prev_logs, state["action"])
                cv2.putText(frame, f"END: {ending}", (8, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
                for _ in range(2 * VIDEO_FPS):
                    writer.write(frame)
            writer.release()
    if info.get("env_error"):
        print(f"  WARNING: Minecraft failed during {label} seed {seed} at step {step}; "
              f"its score only counts the steps before that", flush=True)
    return label, seed, info["logs"], time.time() - start, step if info.get("died") else None


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
    parser.add_argument("--record", action="store_true",
                        help="save every game as an .mp4 in runs/evals/<date>_<time>_videos/")
    args = parser.parse_args()

    policies = []
    for arg in args.models:
        path = None if arg == "random" else T.resolve_model(arg)
        if arg != "random" and (not path or not os.path.exists(path)):
            print(f"Skipping {arg}: no model found")
            continue
        from stable_baselines3 import PPO
        problem = path and T.model_mismatch(PPO.load(path, device="cpu"))
        if problem:
            print(f"Skipping {arg}: {problem}")
            continue
        policies.append((label_for(arg, path), path))
    if not policies:
        print("No models to evaluate.")
        return

    # Each Minecraft instance plus its Python worker uses about 4 GB; don't run WSL out of memory
    with open("/proc/meminfo") as f:
        avail_gb = int(next(l for l in f if l.startswith("MemAvailable")).split()[1]) / 1024 ** 2
    if args.workers * 4 > avail_gb - 1:
        print(f"Not enough free memory for {args.workers} Minecraft instance(s): {avail_gb:.1f} GB available, "
              f"need about {args.workers * 4 + 1} GB. Use fewer --workers or stop other runs.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(T.RUNS_DIR, "evals", stamp + "_videos") if args.record else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    def video_for(label, seed):
        return os.path.join(video_dir, f"{label.replace('/', '_')}_seed{seed}.mp4") if video_dir else None

    jobs = [(label, path, seed, args.steps, video_for(label, seed))
            for label, path in policies for seed in args.seeds]
    print(f"{len(policies)} policies x {len(args.seeds)} worlds = {len(jobs)} episodes of "
          f"{args.steps} steps, {args.workers} Minecraft instance(s)")

    results = {label: {} for label, _ in policies}
    deaths = {label: {} for label, _ in policies}
    ctx = mp.get_context("forkserver")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.mc_mem,)) as pool:
        for label, seed, logs, secs, died_at in pool.imap_unordered(_run_job, jobs):
            results[label][seed] = logs
            if died_at:
                deaths[label][seed] = died_at
            death = f"  DIED at step {died_at}" if died_at else ""
            print(f"  {label:55s} seed {seed:3d}: {logs:3d} logs  ({secs:.0f}s){death}", flush=True)

    # Table: one row per policy, one column per seed
    print("\nLogs collected per world (same worlds for every policy):")
    header = f"{'policy':55s} " + " ".join(f"s{s:<4d}" for s in args.seeds) + "   mean   std  deaths"
    print(header)
    print("-" * len(header))
    for label, _ in policies:
        vals = [results[label][s] for s in args.seeds]
        print(f"{label:55s} " + " ".join(f"{v:<5d}" for v in vals) +
              f"  {np.mean(vals):5.1f}  {np.std(vals):4.1f}  {len(deaths[label])}/{len(vals)}")

    os.makedirs(os.path.join(T.RUNS_DIR, "evals"), exist_ok=True)
    out = os.path.join(T.RUNS_DIR, "evals", stamp + ".json")
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "steps": args.steps, "results": results,
                   "died_at_step": deaths}, f, indent=2)
    print(f"\nSaved to {out}")
    if video_dir:
        print(f"Videos in {video_dir}/")


if __name__ == "__main__":
    main()
