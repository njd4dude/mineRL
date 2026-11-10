"""
Build a behaviour-cloning dataset from OpenAI's VPT contractor recordings (made with MineRL 1.0 /
Minecraft 1.16, the same version we train on).

  python vpt_data.py                       # download + convert 300 treechop segments
  python vpt_data.py --segments 50         # quicker test
  python vpt_data.py --preview             # also save a contact sheet of frames + labels per segment

What it does:
  - The VPT 6.x index has ~4,000 one-minute "treechop-*" segments: contractors asked to chop trees.
    Each has a 640x360 20 fps .mp4 and a .jsonl with one line per game tick (keys, mouse, inventory).
  - Skips segments where the player broke fewer than --min-logs logs (some are just walking or
    sitting in menus). Log breaks come from the recording's Minecraft stats (mine_block:*_log), so
    picking up logs dropped earlier doesn't count as chopping.
  - Turns every 4 ticks (our frame_skip) into one training example: the frame at the start of the
    4 ticks, processed exactly like the env does (process_pov), plus what the human did in those
    ticks (camera turn in degrees, W / space / attack held). bc_train.py maps that to the env's
    [move, yaw, pitch] actions, so changing the action set never needs a new download.
  - Drops ticks with a menu open or the player idle, and anything more than --keep seconds before
    the next log break. Each example stores how many ticks remain until that break
    (until_break), so bc_train.py --window can pick a tighter window without re-downloading.
  - Saves one data/vpt_treechop/<segment>.npz per segment (obs uint8 [N,64,64,3], until_break,
    dyaw, dpitch, forward, jump, attack [N]) and deletes the video afterwards. Re-running skips
    segments already done.
  - python vpt_data.py --summary     shows how the data maps onto the env's actions
"""
import argparse
import json
import os
import random
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from train_treechop import (LOG_ITEMS, MOVE_NAMES, OBS_H, OBS_W, PITCH_STEPS, RENDER_RES, YAW_STEPS,
                            describe_action, human_to_action, nearest_step, process_pov)

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json"
OUT_DIR = os.path.join("data", "vpt_treechop")
FRAME_SKIP = 4                 # must match TreechopEnv's frame_skip
TICKS_PER_SEC = 20
CAMERA_SCALER = 360.0 / 2400.0 # VPT: mouse dx/dy -> degrees (checked: equals the next tick's yaw change)
PREVIEW_WINDOW_S = 5           # previews show examples within this many seconds of a log break
MINE_STATS = {f"minecraft.mine_block:minecraft.{item}" for item in LOG_ITEMS}
NO_BREAK = 32000               # until_break value for examples with no later log break
# What each example stores about the human's input (bc_train.py turns these into env actions with
# human_to_action, so the action set can change without downloading anything again)
INPUT_FIELDS = ["dyaw", "dpitch", "forward", "jump", "attack"]


def count_mined(tick):
    """Logs the player has broken so far (Minecraft's own statistic), same log types the env rewards."""
    return sum(v for k, v in tick.get("stats", {}).items() if k in MINE_STATS)


MAX_BREAKS_PER_TICK = 2       # bigger jumps are the stats loading, not chopping (see logs_mined)


def logs_mined(ticks):
    """(logs broken in the recording, running count per tick starting at 0). Some recordings'
    stats only arrive a few ticks in, jumping from 0 to the player's lifetime total (98,038 in
    one case); those jumps are ignored, or they'd pass --min-logs and count as a "break"."""
    raw = [count_mined(t) for t in ticks]
    counts = [0]
    for a, b in zip(raw, raw[1:]):
        step = b - a
        counts.append(counts[-1] + (step if 0 < step <= MAX_BREAKS_PER_TICK else 0))
    return counts[-1], counts


def chunk_inputs(ticks):
    """What the human did over FRAME_SKIP ticks: camera turn in degrees (yaw: + = right,
    pitch: + = down), whether W was held (at least half the ticks), space pressed, attack held."""
    keys = [set(t["keyboard"]["keys"]) for t in ticks]
    dyaw = sum(t["mouse"]["dx"] for t in ticks) * CAMERA_SCALER
    dpitch = sum(t["mouse"]["dy"] for t in ticks) * CAMERA_SCALER
    attack = any(0 in t["mouse"]["buttons"] for t in ticks)
    return {
        "dyaw": dyaw,
        "dpitch": dpitch,
        "forward": sum("key.keyboard.w" in k for k in keys) * 2 >= len(ticks),
        "jump": any("key.keyboard.space" in k for k in keys),
        "attack": attack,
        "idle": not any(keys) and not attack and dyaw == 0 and dpitch == 0,  # away from keyboard
    }


def parse_ticks(jsonl_text):
    return [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]


def select_chunks(ticks, n_frames, keep_s):
    """The FRAME_SKIP-tick chunks worth learning from: no menu open, player not idle, and at most
    keep_s seconds before the next log break. Returns [(start tick, ticks until break, inputs)]."""
    return [(start, ub, inputs) for start, ub, inputs, reason in classify_chunks(ticks, n_frames, keep_s)
            if reason is None]


def classify_chunks(ticks, n_frames, keep_s):
    """Every FRAME_SKIP-tick chunk with why it was dropped: [(start tick, ticks until break,
    inputs, reason)], reason None for kept chunks (vpt_review.py shows these)."""
    _, mined = logs_mined(ticks)
    until_break = [NO_BREAK] * len(ticks)
    next_break = None
    for i in range(len(ticks) - 2, -1, -1):
        if mined[i + 1] > mined[i]:
            next_break = i + 1
        if next_break is not None:
            until_break[i] = next_break - i
    chunks = []
    for start in range(0, min(len(ticks), n_frames) - FRAME_SKIP + 1, FRAME_SKIP):
        chunk = ticks[start:start + FRAME_SKIP]
        inputs = chunk_inputs(chunk)
        if any(t["isGuiOpen"] for t in chunk):
            reason = "menu open"
        elif until_break[start] == NO_BREAK:
            reason = "after the last log break"
        elif until_break[start] > keep_s * TICKS_PER_SEC:
            reason = f"over {keep_s:g}s before a break"
        elif inputs["idle"]:
            reason = "idle"
        else:
            reason = None
        chunks.append((start, until_break[start], inputs, reason))
    return chunks


def chunk_arrays(chosen):
    """Per-example arrays (saved in the .npz) from select_chunks' output."""
    return {
        "tick": np.array([c[0] for c in chosen], dtype=np.int32),
        "until_break": np.array([c[1] for c in chosen], dtype=np.int16),
        "dyaw": np.array([c[2]["dyaw"] for c in chosen], dtype=np.float32),
        "dpitch": np.array([c[2]["dpitch"] for c in chosen], dtype=np.float32),
        "forward": np.array([c[2]["forward"] for c in chosen], dtype=bool),
        "jump": np.array([c[2]["jump"] for c in chosen], dtype=bool),
        "attack": np.array([c[2]["attack"] for c in chosen], dtype=bool),
    }


def labels(d):
    """Env actions [N, 3] (move, yaw, pitch) for the examples in a loaded .npz."""
    return np.array([human_to_action(y, p, f, j) for y, p, f, j in
                     zip(d["dyaw"], d["dpitch"], d["forward"], d["jump"])], dtype=np.int64).reshape(-1, 3)


def convert_segment(ticks, video_path, keep_s):
    """Return (obs, per-example arrays) for one segment."""
    # Read every frame up front: a 1-minute segment at 640x360 is small. Frame i is what the
    # player saw when action i (jsonl line i) was taken.
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    chosen = select_chunks(ticks, len(frames), keep_s)
    obs = []
    for start, _, _ in chosen:
        rgb = cv2.cvtColor(frames[start], cv2.COLOR_BGR2RGB)
        pov = cv2.resize(rgb, RENDER_RES, interpolation=cv2.INTER_AREA)  # 640x360 -> env's 320x180
        obs.append(process_pov(pov))
    return np.array(obs, dtype=np.uint8).reshape(-1, OBS_H, OBS_W, 3), chunk_arrays(chosen)


def save_preview(path, obs, actions, n=48):
    """actions: [N, 3] env actions."""
    """Grid of evenly spaced examples with their action labels, to eyeball the conversion."""
    if len(obs) == 0:
        return
    idx = np.linspace(0, len(obs) - 1, min(n, len(obs))).astype(int)
    tiles = []
    for i in idx:
        tile = cv2.resize(obs[i], (128, 128), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        move, rest = describe_action(actions[i]).split(", ", 1)
        cv2.putText(tile, move, (2, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(tile, rest, (2, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)]
    cv2.imwrite(path, np.vstack(rows))


def fetch(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout).read()


def process_one(base, relpath, args):
    """Download, filter and convert one segment. Returns (status, name, logs gained, examples).
    Any problem with one segment (corrupt file, network error) just skips that segment."""
    try:
        return _process_one(base, relpath, args)
    except Exception as e:
        name = os.path.splitext(os.path.basename(relpath))[0]
        return f"error ({type(e).__name__}: {str(e)[:80]})", name, 0, 0


def _process_one(base, relpath, args):
    name = os.path.splitext(os.path.basename(relpath))[0]
    out = os.path.join(OUT_DIR, name + ".npz")
    skip = os.path.join(OUT_DIR, "skipped", name)
    if os.path.exists(out) or os.path.exists(skip):
        return "done", name, 0, 0
    url = base + relpath
    try:
        ticks = parse_ticks(fetch(url.replace(".mp4", ".jsonl")).decode())
    except urllib.error.HTTPError as e:
        return f"missing ({e.code})", name, 0, 0
    # Cheap check before downloading the ~30 MB video
    gained, _ = logs_mined(ticks)
    if gained < args.min_logs:
        open(skip, "w").close()   # remember, so re-runs don't download it again
        return "few logs", name, gained, 0
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=OUT_DIR, delete=True) as tmp:
        try:
            tmp.write(fetch(url, timeout=300))
            tmp.flush()
        except urllib.error.HTTPError as e:
            return f"missing ({e.code})", name, 0, 0
        obs, arrays = convert_segment(ticks, tmp.name, args.keep)
    np.savez_compressed(out + ".tmp.npz", obs=obs, **arrays, logs_mined=gained, source=relpath)
    os.replace(out + ".tmp.npz", out)  # so a half-written file is never picked up
    if args.preview:
        save_preview_for(out)
    return "ok", name, gained, len(obs)


def save_preview_for(npz_path):
    d = np.load(npz_path)
    near = d["until_break"] <= PREVIEW_WINDOW_S * TICKS_PER_SEC   # what bc_train.py uses by default
    name = os.path.splitext(os.path.basename(npz_path))[0]
    save_preview(os.path.join(OUT_DIR, "previews", name + ".jpg"), d["obs"][near], labels(d)[near])


def summarize():
    """Print how the stored examples map onto the env's actions, and how well the camera steps
    fit what the humans actually did."""
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".npz"))
    acts, dyaw, dpitch, stored = [], [], [], 0
    for f in files:
        d = np.load(os.path.join(OUT_DIR, f))
        if "dyaw" not in d:
            print(f"(skipping {f}: old format, run --relabel)")
            continue
        near = d["until_break"] <= PREVIEW_WINDOW_S * TICKS_PER_SEC
        stored += len(near)
        acts.append(labels(d)[near])
        dyaw.append(d["dyaw"][near])
        dpitch.append(d["dpitch"][near])
    if not acts:
        return
    acts, dyaw, dpitch = np.concatenate(acts), np.concatenate(dyaw), np.concatenate(dpitch)
    n = len(acts)
    print(f"\nDataset: {len(files)} segments, {stored} examples stored; {n} within "
          f"{PREVIEW_WINDOW_S}s of a log break ({n * FRAME_SKIP / TICKS_PER_SEC / 60:.0f} min of play)")
    for title, col, names in [("move", 0, MOVE_NAMES), ("yaw", 1, [f"{v:+d}" for v in YAW_STEPS]),
                              ("pitch", 2, [f"{v:+d}" for v in PITCH_STEPS])]:
        counts = np.bincount(acts[:, col], minlength=len(names))
        print(f"  {title:6s}" + "  ".join(f"{nm} {100 * c / n:.1f}%" for nm, c in zip(names, counts)))
    for title, human, steps in [("yaw", dyaw, YAW_STEPS), ("pitch", dpitch, PITCH_STEPS)]:
        nearest = np.asarray(steps)[[nearest_step(v, steps) for v in human]]
        err = np.abs(human - nearest)
        moving = np.abs(human) >= 1
        print(f"  {title} fit: mean error {err.mean():.2f} deg ({err[moving].mean():.2f} when the camera moved), "
              f"{100 * (np.abs(human) > max(steps)).mean():.1f}% of moves bigger than the largest step")


def relabel(path, base):
    """Add the raw human inputs to an .npz made by the first version of this script (which only
    stored 7-action labels), using the small .jsonl alone - no video download. The kept examples
    are re-selected with the same rules, and must line up with the stored frames."""
    d = dict(np.load(path))
    ticks = parse_ticks(fetch(base + str(d["source"]).replace(".mp4", ".jsonl")).decode())
    chosen = select_chunks(ticks, 10 ** 9, keep_s=20)   # the old run kept 20 s before a break
    n = len(d["obs"])
    arrays = chunk_arrays(chosen[:n])  # the old run stopped where the video ended (at most a few ticks early)
    if len(chosen) < n or not np.array_equal(arrays["until_break"], d["until_break"]):
        raise ValueError("re-selected examples don't line up with the stored frames")
    d.pop("actions", None)
    d.update(arrays)
    np.savez_compressed(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)


def relabel_all(workers):
    index = json.loads(fetch(INDEX_URL))
    todo = [os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
            if f.endswith(".npz") and "dyaw" not in np.load(os.path.join(OUT_DIR, f))]
    print(f"Relabelling {len(todo)} old-format segments (downloads only their .jsonl action logs)")
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(relabel, p, index["basedir"]): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:
                failed += 1
                print(f"  {os.path.basename(futures[fut])}: {type(e).__name__}: {e} (deleting it)")
                os.remove(futures[fut])
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"Done: {len(todo) - failed} relabelled, {failed} removed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=300, help="treechop segments to convert (that pass the filter)")
    parser.add_argument("--min-logs", type=int, default=5, help="skip segments where fewer logs were broken")
    parser.add_argument("--keep", type=float, default=20,
                        help="store examples up to this many seconds before a log break (bc_train.py --window picks from these)")
    parser.add_argument("--workers", type=int, default=6, help="parallel downloads")
    parser.add_argument("--seed", type=int, default=0, help="which random segments to pick")
    parser.add_argument("--preview", action="store_true", help="save a labelled contact sheet per segment")
    parser.add_argument("--relabel", action="store_true",
                        help="upgrade .npz files from the first version (7-action labels) instead of downloading")
    parser.add_argument("--summary", action="store_true", help="just print the dataset summary")
    args = parser.parse_args()

    if args.relabel or args.summary:
        if args.relabel:
            relabel_all(args.workers)
        summarize()
        return

    os.makedirs(os.path.join(OUT_DIR, "skipped"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "previews"), exist_ok=True)
    index = json.loads(fetch(INDEX_URL))
    relpaths = [p for p in index["relpaths"] if os.path.basename(p).startswith("treechop-")]
    random.Random(args.seed).shuffle(relpaths)
    print(f"{len(relpaths)} treechop segments in the VPT index; converting until {args.segments} pass the filter")

    ok = sum(f.endswith(".npz") for f in os.listdir(OUT_DIR))
    it = iter(relpaths)
    with ThreadPoolExecutor(args.workers) as pool:
        running = set()
        try:
            while ok < args.segments:
                while len(running) < args.workers:
                    p = next(it, None)
                    if p is None:
                        break
                    running.add(pool.submit(process_one, index["basedir"], p, args))
                if not running:
                    break
                fut = next(as_completed(running))
                running.remove(fut)
                status, name, gained, n = fut.result()
                if status == "done":
                    continue
                if status == "ok":
                    ok += 1
                    print(f"[{ok}/{args.segments}] {name}: {gained} logs, {n} examples")
                else:
                    print(f"  skip {name}: {status}" + (f" ({gained} logs)" if status == "few logs" else ""))
            # downloads already in flight when the target was reached still finish; report them
            for fut in as_completed(running):
                status, name, gained, n = fut.result()
                if status == "ok":
                    ok += 1
                    print(f"[{ok}] {name}: {gained} logs, {n} examples")
        except KeyboardInterrupt:
            print("\nStopped by Ctrl+C (finished segments are kept)")
            for f in running:
                f.cancel()
    summarize()


if __name__ == "__main__":
    main()
