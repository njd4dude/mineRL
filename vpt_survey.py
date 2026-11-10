"""
How much usable tree-chopping is in each VPT dataset version? Reads only the .jsonl action logs
(no videos), runs vpt_data.py's filter on them, and measures what the filter doesn't check yet.

  python vpt_survey.py                 # 100 random recordings per group
  python vpt_survey.py --per-group 30  # quicker

Groups: 6.x treechop (what we train on now, as the baseline), 6.x other (free play), 7.x, 8.x,
9.x, 10.x. For each recording it records:
  - logs broken (Minecraft's mine_block stats, same as vpt_data.py) and the frames vpt_data.py
    would keep: all of them (--keep 20) and those bc_train.py uses (--window 5)
  - logs placed (use_item stats): building with logs, so a later "break" may be the player's own
    wall rather than a tree
  - for each log break: how long attack was held on it (a fist takes ~3 s, an iron axe like our
    agent's ~0.5 s) and whether an axe was in the inventory
  - whether the fields the filter needs are present (the recorder's format changed over versions)
Results: printed per group, and one line per recording in runs/vpt_survey/<date>_<time>.jsonl.
"""
import argparse
import json
import os
import random
import urllib.error
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

import vpt_data as V
from train_treechop import LOG_ITEMS, RUNS_DIR
from vpt_review import INDEX_URLS

PLACE_STATS = {f"minecraft.use_item:minecraft.{item}" for item in LOG_ITEMS}
NEEDED = ("isGuiOpen", "mouse", "keyboard", "stats", "inventory")
IRON_AXE_TICKS = 10      # ~0.5 s to break a log with an iron axe (our agent's tool)
FIST_TICKS = 60          # ~3 s by hand


def slim(tick):
    """Just what the filter and this survey need, so 100 MB logs don't fill the memory."""
    stats = tick.get("stats", {})
    return {
        "isGuiOpen": tick["isGuiOpen"],
        "mouse": {k: tick["mouse"][k] for k in ("dx", "dy", "buttons")},
        "keyboard": {"keys": tick["keyboard"]["keys"]},
        "stats": {k: v for k, v in stats.items() if k in V.MINE_STATS},
        "placed": sum(v for k, v in stats.items() if k in PLACE_STATS),
        "axe": any(i.get("type", "").endswith("_axe") for i in tick.get("inventory", [])),
    }


def survey_one(base, relpath, group):
    out = {"group": group, "name": os.path.basename(relpath)[:-4]}
    try:
        text = V.fetch(base + relpath.replace(".mp4", ".jsonl"), timeout=300).decode()
    except urllib.error.HTTPError as e:
        return out | {"status": f"missing ({e.code})"}
    except Exception as e:
        return out | {"status": f"error: {type(e).__name__}"}
    ticks = []
    for line in text.splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        missing = [k for k in NEEDED if k not in t]
        if missing:
            return out | {"status": "format: no " + ",".join(missing)}
        ticks.append(slim(t))
    del text
    if len(ticks) < V.FRAME_SKIP:
        return out | {"status": "empty"}

    gained, mined = V.logs_mined(ticks)
    placed = [t["placed"] for t in ticks]
    breaks = [i + 1 for i in range(len(mined) - 1) if mined[i + 1] > mined[i]]
    first_place = next((i for i in range(1, len(placed)) if placed[i] > placed[i - 1]), None)
    # attack held before each break, counted from when this attack started or the previous break
    attacking = [0 in t["mouse"]["buttons"] for t in ticks]
    hold, prev = [], 0
    for b in breaks:
        i = b - 1
        while i > prev and attacking[i]:
            i -= 1
        hold.append(b - 1 - i)
        prev = b
    chunks = V.classify_chunks(ticks, len(ticks), 20)
    kept = [c for c in chunks if c[3] is None]
    in_window = [c for c in kept if c[1] <= 5 * V.TICKS_PER_SEC]
    return out | {
        "status": "ok",
        "minutes": round(len(ticks) / V.TICKS_PER_SEC / 60, 2),
        "logs": gained,
        "placed": placed[-1] - placed[0],
        "breaks_after_placing": sum(first_place is not None and b > first_place for b in breaks),
        "hold_ticks": hold,
        "axe_at_break": sum(ticks[b]["axe"] for b in breaks),
        "kept": len(kept),
        "window5": len(in_window),
        "window5_axe": sum(ticks[c[0]]["axe"] for c in in_window),
    }


def report(rows, min_logs):
    groups = list(dict.fromkeys(r["group"] for r in rows))
    print(f"\n{'group':16s}{'tried':>6s}{'ok':>5s}{'404':>5s}{'fmt':>5s}  {'>=1 log':>8s}{f'>={min_logs} logs':>9s}"
          f"{'logs/rec':>9s}{'win5/rec':>9s}{'win5/min':>9s}")
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        ok = [r for r in rs if r["status"] == "ok"]
        miss = sum(r["status"].startswith("missing") for r in rs)
        fmt = sum(r["status"].startswith("format") for r in rs)
        if not ok:
            print(f"{g:16s}{len(rs):6d}{0:5d}{miss:5d}{fmt:5d}")
            continue
        passed = [r for r in ok if r["logs"] >= min_logs]
        w5 = sum(r["window5"] for r in passed)   # what vpt_data.py would actually keep
        mins = sum(r["minutes"] for r in ok)
        print(f"{g:16s}{len(rs):6d}{len(ok):5d}{miss:5d}{fmt:5d}  "
              f"{100 * np.mean([r['logs'] > 0 for r in ok]):7.0f}%{100 * len(passed) / len(ok):8.0f}%"
              f"{np.mean([r['logs'] for r in ok]):9.1f}{w5 / len(ok):9.0f}{w5 / mins:9.1f}")
    print(f"\n(win5 = frames within 5 s of a break, from recordings with >={min_logs} logs: what bc_train.py "
          f"would train on. /rec = per recording tried successfully, /min = per minute of recording)")

    print(f"\n{'group':16s}{'breaks':>7s}{'axe held?':>10s}{'hold med':>9s}{'<=1s':>6s}{'>=2.5s':>7s}"
          f"{'placing':>9s}{'breaks after':>13s}{'win5 w/ axe':>12s}")
    for g in groups:
        ok = [r for r in rows if r["group"] == g and r["status"] == "ok" and r["logs"] >= min_logs]
        hold = np.array([h for r in ok for h in r["hold_ticks"]])
        if not len(hold):
            continue
        n_breaks = len(hold)
        w5 = sum(r["window5"] for r in ok)
        print(f"{g:16s}{n_breaks:7d}{100 * sum(r['axe_at_break'] for r in ok) / n_breaks:9.0f}%"
              f"{np.median(hold) / 20:8.2f}s{100 * np.mean(hold <= 20):5.0f}%{100 * np.mean(hold >= 50):6.0f}%"
              f"{100 * np.mean([r['placed'] > 0 for r in ok]):8.0f}%"
              f"{100 * sum(r['breaks_after_placing'] for r in ok) / n_breaks:12.0f}%"
              f"{100 * sum(r['window5_axe'] for r in ok) / max(w5, 1):11.0f}%")
    print("\n(over recordings with enough logs. axe held? = an axe in the inventory at the break (not "
          "necessarily in hand); hold = attack held before each break: iron axe ~0.5s, fist ~3s; "
          "placing = recordings that placed logs; breaks after = breaks after the first log placement, "
          "which may be the player's own blocks)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-logs", type=int, default=5, help="same as vpt_data.py")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    jobs = []
    for version, url in INDEX_URLS.items():
        index = json.loads(V.fetch(url))
        paths = index["relpaths"]
        if version == 6:
            tc = [p for p in paths if os.path.basename(p).startswith("treechop-")]
            other = [p for p in paths if not os.path.basename(p).startswith("treechop-")]
            groups = [("6.x treechop", tc), ("6.x other", other)]
        else:
            groups = [(f"{version}.x", paths)]
        for name, ps in groups:
            jobs += [(index["basedir"], p, name) for p in rng.sample(ps, min(args.per_group, len(ps)))]

    os.makedirs(os.path.join(RUNS_DIR, "vpt_survey"), exist_ok=True)
    out_path = os.path.join(RUNS_DIR, "vpt_survey", datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
    print(f"Surveying {len(jobs)} action logs ({args.per_group} per group) -> {out_path}")
    rows = []
    with ProcessPoolExecutor(args.workers) as pool, open(out_path, "w") as f:
        futures = [pool.submit(survey_one, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    order = {g: i for i, g in enumerate(dict.fromkeys(j[2] for j in jobs))}
    rows.sort(key=lambda r: order[r["group"]])
    report(rows, args.min_logs)


if __name__ == "__main__":
    main()
