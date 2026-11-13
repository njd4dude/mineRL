"""
Compare models fairly: play each one on the SAME set of Minecraft worlds (fixed seeds), plus a
random-action baseline, and print logs collected per world.

  python evaluate.py --models random runs/bc/model_A runs/ppo/model_B/treechop_ckpt_30000_steps.zip
  python evaluate.py --models random latest --seeds 1 2 3 4 5 6 7 8 --steps 2000
  python evaluate.py --models random latest --workers 4      # faster if nothing else is running
  python evaluate.py --models latest --record                # also save every game as an .mp4
  python evaluate.py --models latest --record --video agent  # video of only what the agent sees
  python evaluate.py --models random --seeds 7 --workers 1 --record   # one game, e.g. to debug the env

Each model can be "random", "latest", a run folder (its newest model) or a .zip checkpoint.
Results are saved in each model's own run folder: runs/<kind>/<run>/evals/<date>_<time>/results.json
(that run's models only, plus which other models were scored on the same worlds), and for random
actions in runs/random/evals/<date>_<time>/. With --record, every game is also saved there as
<checkpoint>_seed<N>.mp4 (e.g. final_seed3.mp4), with the step, logs so far, the chosen action and
how the game ended. --video picks the layout:
  both  (default) the full game frame and, beside it, the 64x64 image the agent actually sees
  game  only the full game frame, as a player would see it
  agent only the agent's 64x64 view, enlarged 8x (what the network gets: no HUD, squashed)
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

# Videos (--record): a 1000-step game is 4000 game ticks, so keep them small: every 2nd game tick
# at 10 fps, which still plays at real game speed (Minecraft runs 20 ticks per second).
VIDEO_TICK_STRIDE = 2
VIDEO_FPS = 20 // VIDEO_TICK_STRIDE
GAME_SCALE = 2       # the 320x180 game frame is shown at 640x360
AGENT_BIG = 512      # --video agent: the 64x64 view enlarged 8x
BAR_H = 60           # --video agent: text bar under the view
VIDEO_LAYOUTS = ["both", "game", "agent"]


def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # MineRL gives RGB, OpenCV writes BGR


def _enlarge(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)   # sharp pixels, no blur


def _text(frame, lines, x, y, scale=0.5, color=(255, 255, 255), thick=1):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + int(40 * scale) * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thick, cv2.LINE_AA)


def video_size(layout):
    gw, gh = T.RENDER_RES[0] * GAME_SCALE, T.RENDER_RES[1] * GAME_SCALE
    return {"game": (gw, gh), "both": (gw + gh, gh), "agent": (AGENT_BIG, AGENT_BIG + BAR_H)}[layout]


def render_frame(layout, pov, step, steps, logs, action, ending=None):
    """One video frame. pov is the full 320x180 game frame; the agent's view is made from it with
    the same process_pov the env uses, so it is exactly the image the network got."""
    act = T.describe_action(action) if action is not None else "-"
    lines = [f"step {step}/{steps}   logs {logs}", f"action: {act}"]
    agent = _bgr(T.process_pov(pov))
    gw, gh = T.RENDER_RES[0] * GAME_SCALE, T.RENDER_RES[1] * GAME_SCALE
    if layout == "agent":
        frame = np.zeros((AGENT_BIG + BAR_H, AGENT_BIG, 3), np.uint8)
        frame[:AGENT_BIG] = _enlarge(agent, AGENT_BIG, AGENT_BIG)
        _text(frame, lines, 8, AGENT_BIG + 22)
    else:
        game = _enlarge(_bgr(pov), gw, gh)
        if layout == "game":
            frame = game
        else:   # both: game on the left, the agent's view (same height) on the right
            frame = np.hstack([game, _enlarge(agent, gh, gh)])
            cv2.rectangle(frame, (gw, gh - 24), (gw + gh, gh), (0, 0, 0), -1)
            _text(frame, ["what the agent sees (64x64)"], gw + 8, gh - 7)
        cv2.rectangle(frame, (0, 0), (390, 46), (0, 0, 0), -1)
        _text(frame, lines, 8, 19)
    if ending:
        y = AGENT_BIG - 16 if layout == "agent" else gh - 40
        _text(frame, [f"END: {ending}"], 8, y, scale=0.9, color=(0, 0, 255), thick=2)
    return frame


def open_video(path, layout):
    """An OpenCV video writer for an .mp4 file. Uses MPEG-4 (mp4v): the pip OpenCV has no H.264
    encoder. Plays in VLC and the Windows Media Player / Movies & TV apps."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, video_size(layout))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV can't write .mp4 files here ({path})")
    return writer


_env = None      # one Minecraft per worker process, reused for every job
_models = {}     # loaded models per worker, by path


def _init_worker(mc_mem):
    global _env
    T.set_minecraft_memory(mc_mem)
    _env = T.TreechopEnv()


def _run_job(job):
    """Play one policy on one seeded world. Returns (policy label, seed, logs, seconds, step it died
    at or None)."""
    label, path, seed, steps, video_path, layout = job
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
        writer = open_video(video_path, layout)

        def on_frame(pov):
            state["last_pov"] = pov
            state["last_written"] = state["tick"] % VIDEO_TICK_STRIDE == 0
            if state["last_written"]:
                # _env.prev_logs (not info) is updated every tick, so the count is never a step behind
                writer.write(render_frame(layout, pov, state["step"], steps, _env.prev_logs, state["action"]))
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
                frame = render_frame(layout, state["last_pov"], state["step"], steps, _env.prev_logs,
                                     state["action"], ending=ending)
                for _ in range(2 * VIDEO_FPS):
                    writer.write(frame)
            writer.release()
    if info.get("env_error"):
        print(f"  WARNING: Minecraft failed during {label} seed {seed} at step {step}; "
              f"its score only counts the steps before that", flush=True)
    won_at = step if info["logs"] >= T.LOGS_TO_WIN else None   # how fast it cleared the world
    return label, seed, info["logs"], time.time() - start, step if info.get("died") else None, won_at


RANDOM_DIR = os.path.join(T.RUNS_DIR, "random")   # random actions have no run folder; their evals go here


def eval_dir_for(path, stamp):
    """Where one eval's results and videos go for this model: <its run folder>/evals/<stamp>/."""
    run_dir = os.path.dirname(path) if path else RANDOM_DIR
    return os.path.join(run_dir, "evals", stamp)


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
                        help="save every game as an .mp4 in each run's evals/<date>_<time>/")
    parser.add_argument("--video", choices=VIDEO_LAYOUTS, default="both",
                        help="with --record: both = game + agent's view side by side, game = only the game, "
                             "agent = only what the agent sees")
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
    out_dirs = {label: eval_dir_for(path, stamp) for label, path in policies}
    for d in set(out_dirs.values()):
        os.makedirs(d, exist_ok=True)

    def video_for(label, seed):
        # inside the run's own folder, so the run name isn't repeated: final_seed3.mp4, random_seed3.mp4
        return os.path.join(out_dirs[label], f"{label.split('/')[-1]}_seed{seed}.mp4") if args.record else None

    jobs = [(label, path, seed, args.steps, video_for(label, seed), args.video)
            for label, path in policies for seed in args.seeds]
    print(f"{len(policies)} policies x {len(args.seeds)} worlds = {len(jobs)} episodes of "
          f"{args.steps} steps, {args.workers} Minecraft instance(s)")

    results = {label: {} for label, _ in policies}
    deaths = {label: {} for label, _ in policies}
    wins = {label: {} for label, _ in policies}   # seed -> step the game reached LOGS_TO_WIN
    ctx = mp.get_context("forkserver")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.mc_mem,)) as pool:
        for label, seed, logs, secs, died_at, won_at in pool.imap_unordered(_run_job, jobs):
            results[label][seed] = logs
            if died_at:
                deaths[label][seed] = died_at
            if won_at:
                wins[label][seed] = won_at
            death = f"  DIED at step {died_at}" if died_at else ""
            won = f"  (64 at step {won_at})" if won_at else ""
            print(f"  {label:55s} seed {seed:3d}: {logs:3d} logs  ({secs:.0f}s){death}{won}", flush=True)

    # Table: one row per policy, one column per seed
    print("\nLogs collected per world (same worlds for every policy):")
    header = f"{'policy':55s} " + " ".join(f"s{s:<4d}" for s in args.seeds) + "   mean   std  deaths  won  steps to win (median)"
    print(header)
    print("-" * len(header))
    for label, _ in policies:
        vals = [results[label][s] for s in args.seeds]
        print(f"{label:55s} " + " ".join(f"{v:<5d}" for v in vals) +
              f"  {np.mean(vals):5.1f}  {np.std(vals):4.1f}  {len(deaths[label])}/{len(vals)}"
              f"  {len(wins[label]):2d}/{len(vals)}  " +
              (f"{np.median(list(wins[label].values())):.0f}" if wins[label] else "-"))

    # One results.json per run folder, holding that run's models (a run can have several checkpoints)
    print()
    for d in sorted(set(out_dirs.values())):
        mine = [label for label, _ in policies if out_dirs[label] == d]
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump({"seeds": args.seeds, "steps": args.steps,
                       "results": {l: results[l] for l in mine},
                       "died_at_step": {l: deaths[l] for l in mine},
                       "won_at_step": {l: wins[l] for l in mine},
                       "evaluated_with": [l for l, _ in policies if l not in mine]}, f, indent=2)
        print(f"Saved to {d}/" + (" (results.json + videos)" if args.record else " (results.json)"))


if __name__ == "__main__":
    main()
