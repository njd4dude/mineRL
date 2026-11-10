"""
Watch VPT recordings the way vpt_data.py saw them, to check its filtering by eye.

  python vpt_review.py                         # 1 kept + 1 skipped segment, picked at random
  python vpt_review.py --kept 3 --skipped 3    # more of each
  python vpt_review.py --new 2                 # segments vpt_data.py never looked at
  python vpt_review.py treechop-05e006bbcdec-20210904-183454   # specific segments

Re-downloads each segment (vpt_data.py deletes the videos after converting them) and writes
runs/vpt_review/<segment>.mp4 at real speed (20 fps), with:
  - the original 640x360 video, a green or red border for kept or dropped frames
  - a side panel: the segment's verdict (kept, or why it was skipped), why this frame was dropped,
    time to the next log break, logs broken so far, the keys/mouse the human used and the env
    action that becomes the label, and the 64x64 image the network gets
  - a timeline along the bottom: bright green = trained on (inside bc_train.py's --window),
    dark green = saved in the .npz but outside --window,
    red/orange/grey = dropped (menu / too early / idle), white ticks = log breaks
Nothing in data/ is changed. For a kept segment it also checks that re-running the filter picks
exactly the frames stored in its .npz, and that the "network sees" images are pixel-for-pixel
the ones stored there (which bc_train.py trains on).
"""
import argparse
import json
import os
import random
import tempfile

import cv2
import numpy as np

import vpt_data as V
from train_treechop import OBS_H, OBS_W, RENDER_RES, RUNS_DIR, describe_action, human_to_action, process_pov

OUT_DIR = os.path.join(RUNS_DIR, "vpt_review")
VID_W, VID_H = 640, 360
PANEL_W = 470   # text column + the network's view beside it
BAR_H = 34
CANVAS = (VID_W + PANEL_W, VID_H + BAR_H)   # (width, height)

GREEN, BRIGHT, RED, ORANGE, GREY = (60, 170, 60), (90, 255, 90), (60, 60, 220), (40, 150, 240), (120, 120, 120)
REASON_COLOURS = {"menu open": RED, "idle": GREY}   # anything else (too early / after last break): ORANGE


def chunk_colour(reason, until_break, window_s):
    if reason is None:
        return BRIGHT if until_break <= window_s * V.TICKS_PER_SEC else GREEN
    return REASON_COLOURS.get(reason, ORANGE)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def text(img, s, y, colour=(230, 230, 230), scale=0.45, x=None, thick=1):
    cv2.putText(img, s, (VID_W + 10 if x is None else x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick,
                cv2.LINE_AA)


def timeline(chunks, n_ticks, breaks, window_s):
    bar = np.zeros((BAR_H, CANVAS[0], 3), np.uint8)
    scale = CANVAS[0] / max(n_ticks, 1)
    for start, ub, _, reason in chunks:
        x0, x1 = int(start * scale), int((start + V.FRAME_SKIP) * scale)
        bar[6:BAR_H - 6, x0:max(x1, x0 + 1)] = chunk_colour(reason, ub, window_s)
    for t in breaks:
        x = int(t * scale)
        bar[0:BAR_H, max(x - 1, 0):x + 1] = (255, 255, 255)
    return bar


def render(name, ticks, frames, args, verdict, verdict_colour, out_path):
    chunks = V.classify_chunks(ticks, len(frames), args.keep)
    by_start = {c[0]: c for c in chunks}
    _, mined = V.logs_mined(ticks)
    breaks = [i + 1 for i in range(len(mined) - 1) if mined[i + 1] > mined[i]]
    n = min(len(ticks), len(frames))
    bar = timeline(chunks, n, breaks, args.window)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), V.TICKS_PER_SEC, CANVAS)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV can't write .mp4 files here ({out_path})")
    model_view = np.zeros((OBS_H * 3, OBS_W * 3, 3), np.uint8)
    shown = {}   # chunk start tick -> the exact 64x64 RGB image drawn in the "network sees" box
    flash = 0
    for t in range(n):
        start = t - t % V.FRAME_SKIP
        chunk = by_start.get(start)
        canvas = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
        canvas[:VID_H, :VID_W] = cv2.resize(frames[t], (VID_W, VID_H))
        if t == start:   # the frame the network would see for this chunk
            pov = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB), RENDER_RES, interpolation=cv2.INTER_AREA)
            shown[t] = process_pov(pov)
            small = cv2.cvtColor(shown[t], cv2.COLOR_RGB2BGR)
            model_view = cv2.resize(small, (OBS_W * 3, OBS_H * 3), interpolation=cv2.INTER_NEAREST)
        if t in breaks:
            flash = 10

        # panel
        text(canvas, name.replace("treechop-", "")[:34], 18, (180, 180, 180), 0.4)
        text(canvas, verdict, 38, verdict_colour, 0.45)
        text(canvas, f"{t / V.TICKS_PER_SEC:5.1f}s / {n / V.TICKS_PER_SEC:.1f}s   logs broken: {mined[t] - mined[0]}", 58)
        if chunk is None:
            status, colour, why = "NOT USED", GREY, "past the end of the video"
            ub = None
        else:
            _, ub, inputs, reason = chunk
            colour = chunk_colour(reason, ub, args.window)
            if reason is not None:
                status, why = "DROPPED", reason
            elif ub <= args.window * V.TICKS_PER_SEC:
                status, why = "TRAINED ON", f"within {args.window:g}s of a break (--window)"
            else:
                status, why = "SAVED, NOT TRAINED ON", f"over {args.window:g}s before a break (--window)"
        scale = 0.8   # shrink long statuses to fit the text column (left of the network's view)
        while cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > 250:
            scale -= 0.05
        text(canvas, status, 90, colour, scale, thick=2)
        text(canvas, why, 112, colour)
        if ub is not None:
            nb = "no more breaks" if ub == V.NO_BREAK else f"next break in {ub / V.TICKS_PER_SEC:.1f}s"
            text(canvas, nb, 132)
        if chunk is not None:
            keys = [k for k, on in (("W", inputs["forward"]), ("jump", inputs["jump"]), ("attack", inputs["attack"])) if on]
            text(canvas, "held: " + (" ".join(keys) or "-"), 158)
            text(canvas, f"mouse: yaw {inputs['dyaw']:+.1f}  pitch {inputs['dpitch']:+.1f}", 176)
            act = human_to_action(inputs["dyaw"], inputs["dpitch"], inputs["forward"], inputs["jump"])
            move, rest = describe_action(act).split(", ", 1)
            text(canvas, f"label: {move}", 194, (200, 200, 120))
            text(canvas, f"       {rest}", 210, (200, 200, 120))
        if ticks[t].get("isGuiOpen"):
            text(canvas, "menu open", 228, RED)
        mx, my = CANVAS[0] - OBS_W * 3 - 10, 90
        text(canvas, "network sees:", my - 8, (180, 180, 180), 0.4, x=mx)
        canvas[my:my + OBS_H * 3, mx:mx + OBS_W * 3] = model_view

        # border on the video, log-break flash, timeline with cursor
        cv2.rectangle(canvas, (1, 1), (VID_W - 2, VID_H - 2), colour, 3)
        if flash:
            text(canvas, "LOG BROKEN", 30, (255, 255, 255), 0.9, x=VID_W // 2 - 90, thick=2)
            flash -= 1
        canvas[VID_H:] = bar
        x = int(t * CANVAS[0] / n)
        canvas[VID_H:, max(x - 1, 0):x + 2] = (0, 255, 255)
        writer.write(canvas)
    writer.release()
    return chunks, shown


def review(relpath, base, args, stored=None):
    name = os.path.splitext(os.path.basename(relpath))[0]
    ticks = V.parse_ticks(V.fetch(base + relpath.replace(".mp4", ".jsonl")).decode())
    gained, _ = V.logs_mined(ticks)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(V.fetch(base + relpath, timeout=300))
        tmp.flush()
        frames = read_frames(tmp.name)
    if gained < args.min_logs:
        verdict, vcol = f"SKIPPED: {gained} logs (needs {args.min_logs})", RED
    else:
        verdict, vcol = f"KEPT: {gained} logs", GREEN
    out = os.path.join(OUT_DIR, name + ".mp4")
    chunks, shown = render(name, ticks, frames, args, verdict, vcol, out)

    kept = [c for c in chunks if c[3] is None]
    reasons = {}
    for c in chunks:
        reasons[c[3] or "kept"] = reasons.get(c[3] or "kept", 0) + 1
    print(f"{name}: {verdict.lower()}; {len(ticks)} ticks, {len(frames)} video frames")
    print("   chunks: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    if stored is not None:
        same = np.array_equal(stored["tick"], [c[0] for c in kept])
        print(f"   matches the stored .npz: {'yes' if same else 'NO'} ({len(stored['tick'])} stored, {len(kept)} re-selected)")
        # the "network sees" box against the images bc_train.py trains on (d["obs"]), pixel by pixel
        diffs = [np.abs(shown[t].astype(int) - o.astype(int)) for t, o in zip(stored["tick"], stored["obs"]) if t in shown]
        if diffs:
            exact = sum(int(d.max() == 0) for d in diffs)
            worst = max(int(d.max()) for d in diffs)
            print(f"   network view vs stored obs: {exact}/{len(diffs)} images identical"
                  + ("" if exact == len(diffs) else f", largest pixel difference {worst}/255"))
    print(f"   -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", nargs="*", help="segment names (e.g. treechop-05e006bbcdec-20210904-183454)")
    parser.add_argument("--kept", type=int, default=None, help="random segments from data/vpt_treechop/")
    parser.add_argument("--skipped", type=int, default=None, help="random segments that failed the log filter")
    parser.add_argument("--new", type=int, default=0, help="random segments vpt_data.py never processed")
    parser.add_argument("--min-logs", type=int, default=5, help="same as vpt_data.py")
    parser.add_argument("--keep", type=float, default=20, help="same as vpt_data.py")
    parser.add_argument("--window", type=float, default=5, help="same as bc_train.py (for colouring only)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not args.segments and args.kept is None and args.skipped is None and not args.new:
        args.kept, args.skipped = 1, 1
    rng = random.Random(args.seed)

    index = json.loads(V.fetch(V.INDEX_URL))
    by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in index["relpaths"]}
    kept = sorted(f[:-4] for f in os.listdir(V.OUT_DIR) if f.endswith(".npz"))
    skip_dir = os.path.join(V.OUT_DIR, "skipped")
    skipped = sorted(os.listdir(skip_dir)) if os.path.isdir(skip_dir) else []
    done = set(kept) | set(skipped)
    unseen = [n for n in by_name if n.startswith("treechop-") and n not in done]

    names = list(args.segments)
    names += rng.sample(kept, min(args.kept or 0, len(kept)))
    names += rng.sample(skipped, min(args.skipped or 0, len(skipped)))
    names += rng.sample(unseen, min(args.new, len(unseen)))
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in names:
        if name not in by_name:
            print(f"{name}: not in the VPT index")
            continue
        npz = os.path.join(V.OUT_DIR, name + ".npz")
        stored = np.load(npz) if os.path.exists(npz) else None
        try:
            review(by_name[name], index["basedir"], args, stored)
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
