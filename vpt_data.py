"""
Build a behaviour-cloning dataset from OpenAI's VPT contractor recordings (made with MineRL 1.0 /
Minecraft 1.16, the same version we train on).

  python vpt_data.py                       # download + convert 300 treechop segments
  python vpt_data.py --segments 50         # quicker test
  python vpt_data.py --preview             # also save a contact sheet of frames + labels per segment
  python vpt_data.py --version 8 --all     # every 8.x (house building) recording that passes the filter

What it does:
  - The VPT 6.x index has ~4,000 one-minute "treechop-*" segments: contractors asked to chop trees.
    Each has a 640x360 20 fps .mp4 and a .jsonl with one line per game tick (keys, mouse, inventory).
    --version 7/8/9/10 reads the other datasets instead (early game, house building, diamond
    pickaxe: ~5-minute recordings of other tasks, where players also chop trees). Each version is
    saved to its own folder, data/vpt_<N>xx (for 6.x only the treechop segments are used).
  - Skips segments where the player broke fewer than --min-logs logs (some are just walking or
    sitting in menus). Log breaks come from the recording's Minecraft stats (mine_block:*_log), so
    picking up logs dropped earlier doesn't count as chopping.
  - Only learns from log breaks that look like our agent's chopping (see classify_chunks): an axe
    in the inventory, the log broken within --max-hold seconds of attacking (not by hand), and
    no logs placed earlier in the recording (later "trees" may be the player's own house).
  - Turns every 4 ticks (our frame_skip) into one training example: the frame at the start of the
    4 ticks, processed exactly like the env does (process_pov), plus what the human did in those
    ticks (camera turn in degrees, W / space / attack held). bc_train.py maps that to the env's
    [move, yaw, pitch] actions, so changing the action set never needs a new download.
  - Drops ticks with a menu open or the player idle, and anything more than --keep seconds before
    the next log break. Each example stores how many ticks remain until that break
    (until_break), so bc_train.py --window can pick a tighter window without re-downloading.
  - Saves one data/vpt_6xx/<segment>.npz per segment (obs uint8 [N,64,64,3], until_break,
    dyaw, dpitch, forward, jump, attack [N]) and deletes the video afterwards. Re-running skips
    segments already done.
  - python vpt_data.py --summary     shows how the data maps onto the env's actions
"""
import argparse
import json
import os
import random
import shutil
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from train_treechop import (LOG_ITEMS, MOVE_NAMES, OBS_H, OBS_W, PITCH_STEPS, RENDER_RES, YAW_STEPS,
                            describe_action, human_to_action, nearest_step, process_pov)

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json"
# One index per VPT dataset version. Only 6.x has "treechop-*" segments; the others are other
# tasks, but players chop trees in them too (the log filter works the same on any recording).
INDEX_URLS = {6: INDEX_URL} | {
    v: f"https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_{v}xx_{d}.json"
    for v, d in ((7, "Apr_6"), (8, "Jun_29"), (9, "Jun_29"), (10, "Jun_29"))}
OUT_DIR = os.path.join("data", "vpt_6xx")   # 6.x (treechop segments only); see version_dir()
FRAME_SKIP = 4                 # must match TreechopEnv's frame_skip
TICKS_PER_SEC = 20
CAMERA_SCALER = 360.0 / 2400.0 # VPT: mouse dx/dy -> degrees (checked: equals the next tick's yaw change)
PREVIEW_WINDOW_S = 5           # previews show examples within this many seconds of a log break
MINE_STATS = {f"minecraft.mine_block:minecraft.{item}" for item in LOG_ITEMS}
PLACE_STATS = {f"minecraft.use_item:minecraft.{item}" for item in LOG_ITEMS}
NO_BREAK = 32000               # until_break value for examples with no later log break
MAX_HOLD_S = 2.0               # default --max-hold: iron axe ~0.5 s per log, wooden ~1.5 s, by hand ~3 s


def version_dir(version):
    return os.path.join("data", f"vpt_{version}xx")


def has_axe(tick):
    if "axe" in tick:          # slimmed tick (slim())
        return tick["axe"]
    return any(i.get("type", "").endswith("_axe") for i in tick.get("inventory", []))


def logs_placed(tick):
    if "placed" in tick:
        return tick["placed"]
    return sum(v for k, v in tick.get("stats", {}).items() if k in PLACE_STATS)


def slim(tick):
    """Just what the filter needs from one tick, so the 5-minute recordings' 20-60 MB logs don't
    fill the memory when several are converted at once."""
    stats = tick.get("stats", {})
    return {
        "isGuiOpen": tick["isGuiOpen"],
        "mouse": {k: tick["mouse"][k] for k in ("dx", "dy", "buttons")},
        "keyboard": {"keys": tick["keyboard"]["keys"]},
        "stats": {k: v for k, v in stats.items() if k in MINE_STATS},
        "placed": sum(v for k, v in stats.items() if k in PLACE_STATS),
        "axe": has_axe(tick),
    }
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
    """Slimmed ticks from a .jsonl. Some recordings end with a cut-off line (the recorder
    stopped mid-write); that last line is dropped. A broken line anywhere else is an error,
    since every later tick would no longer line up with its video frame."""
    lines = [line for line in jsonl_text.splitlines() if line.strip()]
    ticks = []
    for i, line in enumerate(lines):
        try:
            ticks.append(slim(json.loads(line)))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    return ticks


def select_chunks(ticks, n_frames, keep_s, max_hold_s=MAX_HOLD_S, filter_breaks=True):
    """The FRAME_SKIP-tick chunks worth learning from: no menu open, player not idle, and at most
    keep_s seconds before the next log break. Returns [(start tick, ticks until break, inputs)]."""
    return [(start, ub, inputs) for start, ub, inputs, reason in classify_chunks(ticks, n_frames, keep_s, max_hold_s, filter_breaks)
            if reason is None]


def break_problems(ticks, mined, max_hold_s=MAX_HOLD_S):
    """{break tick: why it's not like our agent's chopping} for the log breaks to learn nothing
    from: no axe in the inventory, attack held longer than max_hold_s (by hand or a slow tool:
    the timing would be wrong for our iron axe), or after the player placed logs (it may be
    their own building). Breaks not in the dict are fine."""
    placed = [logs_placed(t) for t in ticks]
    first_place = next((i for i in range(1, len(ticks)) if placed[i] > placed[i - 1]), None)
    attacking = [0 in t["mouse"]["buttons"] for t in ticks]
    problems, prev = {}, 0
    for b in range(1, len(ticks)):
        if mined[b] <= mined[b - 1]:
            continue
        i = b - 1      # attack held before this break, since it started or the previous break
        while i > prev and attacking[i]:
            i -= 1
        hold = b - 1 - i
        prev = b
        if first_place is not None and b > first_place:
            problems[b] = "break after placing logs"
        elif not has_axe(ticks[b]):
            problems[b] = "break without an axe"
        elif hold > max_hold_s * TICKS_PER_SEC:
            problems[b] = "slow break (by hand?)"
    return problems


def good_breaks(ticks, max_hold_s=MAX_HOLD_S):
    """How many log breaks pass break_problems (the --min-logs check)."""
    _, mined = logs_mined(ticks)
    total = sum(mined[i + 1] > mined[i] for i in range(len(mined) - 1))
    return total - len(break_problems(ticks, mined, max_hold_s))


def classify_chunks(ticks, n_frames, keep_s, max_hold_s=MAX_HOLD_S, filter_breaks=True):
    """Every FRAME_SKIP-tick chunk with why it was dropped: [(start tick, ticks until break,
    inputs, reason)], reason None for kept chunks (vpt_review.py shows these)."""
    _, mined = logs_mined(ticks)
    problems = break_problems(ticks, mined, max_hold_s) if filter_breaks else {}
    until_break = [NO_BREAK] * len(ticks)
    break_reason = [None] * len(ticks)
    next_break = None
    for i in range(len(ticks) - 2, -1, -1):
        if mined[i + 1] > mined[i]:
            next_break = i + 1
        if next_break is not None:
            until_break[i] = next_break - i
            break_reason[i] = problems.get(next_break)
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
        elif break_reason[start]:
            reason = break_reason[start]
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


def convert_segment(ticks, video_path, keep_s, max_hold_s=MAX_HOLD_S):
    """Return (obs, per-example arrays) for one segment. Frame i is what the player saw when
    action i (jsonl line i) was taken."""
    # Decide which frames are needed from the action log alone, then read the video once and keep
    # only those (a 5-minute recording is ~6,000 frames, too many to hold at 640x360).
    chosen = select_chunks(ticks, len(ticks), keep_s, max_hold_s)
    wanted = {start for start, _, _ in chosen}
    cap = cv2.VideoCapture(video_path)
    processed = {}
    i = 0
    while len(processed) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pov = cv2.resize(rgb, RENDER_RES, interpolation=cv2.INTER_AREA)  # 640x360 -> env's 320x180
            processed[i] = process_pov(pov)
        i += 1
    cap.release()
    chosen = [c for c in chosen if c[0] in processed]   # the video can end a few frames early
    obs = [processed[start] for start, _, _ in chosen]
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
    out_dir = version_dir(args.version)
    out = os.path.join(out_dir, name + ".npz")
    skip = os.path.join(out_dir, "skipped", name)
    if os.path.exists(out) or os.path.exists(skip):
        return "done", name, 0, 0

    def remember_skip(reason):   # so re-runs don't download it again
        with open(skip, "w") as f:
            f.write(reason)

    url = base + relpath
    try:
        # errors="replace": a few logs have a stray non-UTF-8 byte (in typed text), which we never use
        ticks = parse_ticks(fetch(url.replace(".mp4", ".jsonl"), timeout=300).decode(errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            remember_skip("missing (404)")
        return f"missing ({e.code})", name, 0, 0
    # Cheap check before downloading the 30-170 MB video: enough breaks that pass break_problems
    gained = good_breaks(ticks, args.max_hold)
    if gained < args.min_logs:
        remember_skip(f"few logs ({gained})")
        return "few logs", name, gained, 0
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=out_dir, delete=True) as tmp:
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                shutil.copyfileobj(r, tmp, 1 << 20)
            tmp.flush()
        except urllib.error.HTTPError as e:
            return f"missing ({e.code})", name, 0, 0
        obs, arrays = convert_segment(ticks, tmp.name, args.keep, args.max_hold)
    np.savez_compressed(out + ".tmp.npz", obs=obs, **arrays, logs_mined=gained, source=relpath,
                        version=args.version, max_hold=args.max_hold)
    os.replace(out + ".tmp.npz", out)  # so a half-written file is never picked up
    if args.preview:
        save_preview_for(out)
    return "ok", name, gained, len(obs)


def save_preview_for(npz_path):
    d = np.load(npz_path)
    near = d["until_break"] <= PREVIEW_WINDOW_S * TICKS_PER_SEC   # what bc_train.py uses by default
    name = os.path.splitext(os.path.basename(npz_path))[0]
    save_preview(os.path.join(os.path.dirname(npz_path), "previews", name + ".jpg"), d["obs"][near], labels(d)[near])


def summarize(out_dir=OUT_DIR):
    """Print how the stored examples map onto the env's actions, and how well the camera steps
    fit what the humans actually did."""
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    acts, dyaw, dpitch, stored = [], [], [], 0
    for f in files:
        d = np.load(os.path.join(out_dir, f))
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
    # the old run kept 20 s before a break, and didn't have break_problems' checks
    chosen = select_chunks(ticks, 10 ** 9, keep_s=20, filter_breaks=False)
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
    parser.add_argument("--version", type=int, default=6, choices=sorted(INDEX_URLS),
                        help="VPT dataset version, saved to data/vpt_<N>xx (6 = its treechop segments only, 7-10 = other tasks)")
    parser.add_argument("--segments", type=int, default=300, help="segments to convert (that pass the filter)")
    parser.add_argument("--all", action="store_true", help="go through every recording in the index (ignores --segments)")
    parser.add_argument("--min-logs", type=int, default=5, help="skip segments where fewer good log breaks happened")
    parser.add_argument("--max-hold", type=float, default=MAX_HOLD_S,
                        help="ignore log breaks that took longer than this many seconds of attacking (by hand / slow tool)")
    parser.add_argument("--keep", type=float, default=20,
                        help="store examples up to this many seconds before a log break (bc_train.py --window picks from these)")
    parser.add_argument("--workers", type=int, default=6, help="parallel downloads")
    parser.add_argument("--seed", type=int, default=0, help="which random segments to pick")
    parser.add_argument("--preview", action="store_true", help="save a labelled contact sheet per segment")
    parser.add_argument("--relabel", action="store_true",
                        help="upgrade .npz files from the first version (7-action labels) instead of downloading")
    parser.add_argument("--summary", action="store_true", help="just print the dataset summary")
    args = parser.parse_args()

    out_dir = version_dir(args.version)
    if args.relabel or args.summary:
        if args.relabel:
            relabel_all(args.workers)
        summarize(out_dir)
        return

    os.makedirs(os.path.join(out_dir, "skipped"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    for f in os.listdir(out_dir):   # leftovers from a run that was killed mid-download
        if f.endswith(".tmp.npz") or (f.startswith("tmp") and f.endswith(".mp4")):
            os.remove(os.path.join(out_dir, f))
    index = json.loads(fetch(INDEX_URLS[args.version]))
    relpaths = index["relpaths"]
    if args.version == 6:
        relpaths = [p for p in relpaths if os.path.basename(p).startswith("treechop-")]
    random.Random(args.seed).shuffle(relpaths)
    if args.all:
        args.segments = len(relpaths)
    print(f"{len(relpaths)} recordings in the VPT {args.version}.x index; converting until {args.segments} "
          f"pass the filter -> {out_dir}/")

    ok = sum(f.endswith(".npz") for f in os.listdir(out_dir))
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
    summarize(out_dir)


if __name__ == "__main__":
    main()
