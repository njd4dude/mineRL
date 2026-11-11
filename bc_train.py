"""
Behaviour cloning: teach the PPO network to copy the human actions in data/vpt_6xx/
(built by vpt_data.py). No Minecraft needed, just the GPU.

  python bc_train.py                     # new run: runs/bc/model_<date>_<time>_bc/
  python bc_train.py --epochs 30 --name bc_long
  python bc_train.py --window 3          # only copy the last 3 s before each log break
  python bc_train.py --data data/vpt_6xx data/vpt_8xx --name bc_6xx_8xx   # several VPT versions
  tensorboard --logdir runs              # graphs per epoch (runs/bc/<run>/tb/BC), next to the PPO runs
  python bc_train.py --tb-backfill       # add graphs for BC runs made before this was logged
  python evaluate.py --models random latest --record   # then score it, and watch the saved games

The result is a normal PPO model (same network and [move, yaw, pitch] actions as
train_treechop.py), so it can be watched, evaluated, and later fine-tuned with PPO.

Training details:
  - Each human input (camera turn in degrees, W / space held) is mapped to the closest env action
    with train_treechop.human_to_action, then the network learns to predict it from the frame
    (cross-entropy on each of the three parts: move, yaw, pitch).
  - Only examples from the --window seconds before a log break are used (walking to the tree,
    aiming, chopping), not wandering around.
  - 10% of segments are held out to check the network isn't just memorising, picked by a hash of
    the segment's name, so a segment is held out in every run that uses it (validation loss is
    comparable between runs on the same data, and is also printed per data folder).
  - The frames used are copied once into an uncompressed cache in data/bc_cache/ (one per data
    set + window; about 12 KB per frame), read from disk in batches, so the data doesn't have to
    fit in RAM. Delete data/bc_cache/ any time; it's rebuilt when needed.
  - Augmentation: random brightness/contrast (VPT players often used darker settings than our env),
    small random shifts, and left-right mirroring (with the yaw turn mirrored too).
"""
import argparse
import glob
import hashlib
import json
import os
import queue
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import train_treechop as T
from vpt_data import labels

DATA_DIR = os.path.join("data", "vpt_6xx")
OLD_DIR_NAMES = {"vpt_treechop": "vpt_6xx", "vpt_6xx_treechop": "vpt_6xx"}   # earlier names of data/vpt_6xx
CACHE_DIR = os.path.join("data", "bc_cache")
FRAME_BYTES = T.OBS_H * T.OBS_W * 3
PARTS = [("move", T.MOVE_NAMES),
         ("yaw", [f"{v:+d}" for v in T.YAW_STEPS]),
         ("pitch", [f"{v:+d}" for v in T.PITCH_STEPS])]
# yaw index after a left-right flip: YAW_STEPS is symmetric, so reverse it
assert T.YAW_STEPS == [-v for v in reversed(T.YAW_STEPS)]
YAW_MIRROR = torch.arange(len(T.YAW_STEPS)).flip(0)


class SpacesOnlyEnv(gymnasium.Env):
    """Has the env's observation/action spaces but no Minecraft, just so PPO can be built."""
    observation_space = gymnasium.spaces.Box(0, 255, (T.OBS_H, T.OBS_W, 3), dtype=np.uint8)
    action_space = gymnasium.spaces.MultiDiscrete(T.ACTION_DIMS)

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def is_val(path, val_frac):
    """Held-out segment? Decided by the file name alone, so it's the same in every run."""
    return zlib.crc32(os.path.basename(path).encode()) % 1000 < val_frac * 1000


def load_data(dirs, val_frac, window_s):
    """Frames (a uint8 [N, H, W, 3] memmap of the on-disk cache), actions [N, 3], held-out mask [N]
    and data-folder index [N] for every example within window_s seconds of a log break."""
    files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, "*.npz")))
             if not f.endswith(".tmp.npz")]
    if not files:
        raise SystemExit(f"No data in {dirs} - run vpt_data.py first")
    key = hashlib.sha1(json.dumps([window_s] + [(f, os.path.getsize(f), int(os.path.getmtime(f)))
                                                for f in files]).encode()).hexdigest()[:16]
    cache = os.path.join(CACHE_DIR, key)
    if not os.path.exists(os.path.join(cache, "meta.npz")):
        build_cache(cache, files, window_s)
    meta = np.load(os.path.join(cache, "meta.npz"))
    obs = np.memmap(os.path.join(cache, "obs.u8"), dtype=np.uint8, mode="r",
                    shape=(len(meta["actions"]), T.OBS_H, T.OBS_W, 3))
    file_dir = np.array([dirs.index(os.path.dirname(f)) for f in files])
    file_val = np.array([is_val(f, val_frac) for f in files])
    fi = meta["file_index"]
    return obs, meta["actions"], file_val[fi], file_dir[fi], files, file_val


def build_cache(cache, files, window_s):
    print(f"Building the frame cache {cache}/ from {len(files)} segments (once per data set)...")
    os.makedirs(cache, exist_ok=True)

    def keep_mask(f):
        d = np.load(f)
        if "dyaw" not in d:
            raise SystemExit(f"{f} is from an older vpt_data.py; run: python vpt_data.py --relabel")
        return d["until_break"] <= window_s * 20   # 20 game ticks per second

    with ThreadPoolExecutor(8) as pool:
        masks = list(pool.map(keep_mask, files))
        counts = [int(m.sum()) for m in masks]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        obs = np.memmap(os.path.join(cache, "obs.u8.tmp"), dtype=np.uint8, mode="w+",
                        shape=(int(offsets[-1]), T.OBS_H, T.OBS_W, 3))

        def fill(i):
            d = np.load(files[i])
            obs[offsets[i]:offsets[i + 1]] = d["obs"][masks[i]]
            return labels(d)[masks[i]]

        actions = []
        for i, a in enumerate(pool.map(fill, range(len(files)))):
            actions.append(a)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} segments")
    obs.flush()
    del obs
    os.replace(os.path.join(cache, "obs.u8.tmp"), os.path.join(cache, "obs.u8"))
    np.savez(os.path.join(cache, "meta.npz"), actions=np.concatenate(actions).astype(np.int64),
             file_index=np.repeat(np.arange(len(files)), counts), files=np.array(files))
    print(f"  {offsets[-1]} frames, {offsets[-1] * FRAME_BYTES / 1e9:.1f} GB")


def batches(obs, actions, idx, batch, shuffle):
    """Yields (obs uint8 [B,H,W,3], actions [B,3]) numpy batches over the examples idx, read from
    the memmap in a background thread so the GPU doesn't wait on the disk."""
    if shuffle:
        idx = np.random.permutation(idx)
    q = queue.Queue(maxsize=8)
    stop = threading.Event()

    def worker():
        n = len(idx) // batch if shuffle else -(-len(idx) // batch)   # training drops the last partial batch
        for s in range(n):
            if stop.is_set():
                return
            b = np.sort(idx[s * batch:(s + 1) * batch])   # sorted: fewer disk seeks
            q.put((np.asarray(obs[b]), actions[b]))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()


def augment(obs, actions):
    """obs: float [B,3,H,W] in 0..255, actions: long [B,3], both on the GPU."""
    b = obs.shape[0]
    dev = obs.device
    # brightness (multiply) and contrast (stretch around the image mean)
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.8, 1.8)
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    mean = obs.mean(dim=(1, 2, 3), keepdim=True)
    obs = ((obs - mean) * contrast + mean) * bright
    # random shift of up to 4 pixels (pad with edge pixels, crop back)
    pad = 4
    padded = F.pad(obs, (pad, pad, pad, pad), mode="replicate")
    dx, dy = np.random.randint(0, 2 * pad + 1, 2)
    obs = padded[:, :, dy:dy + T.OBS_H, dx:dx + T.OBS_W]
    # mirror half the batch: flip the image and turn the other way
    flip = torch.rand(b, device=dev) < 0.5
    obs = torch.where(flip[:, None, None, None], obs.flip(3), obs)
    actions = actions.clone()
    actions[:, 1] = torch.where(flip, YAW_MIRROR.to(dev)[actions[:, 1]], actions[:, 1])
    return obs.clamp(0, 255), actions


def to_tensor(obs_np, device):
    return torch.as_tensor(obs_np, device=device).permute(0, 3, 1, 2).float()  # NHWC -> NCHW


def evaluate(policy, obs, actions, idx, device, batch=1024):
    """Mean loss and per-part accuracy on the held-out examples idx, plus each part's "always
    guess the most common answer" accuracy to compare against."""
    policy.set_training_mode(False)
    loss_sum = 0.0
    hits = np.zeros(len(PARTS))
    with torch.no_grad():
        for o_np, a_np in batches(obs, actions, idx, batch, shuffle=False):
            o = to_tensor(o_np, device)
            a = torch.as_tensor(a_np, device=device)
            dist = policy.get_distribution(o)
            loss_sum += -dist.log_prob(a).sum().item()
            for k, cat in enumerate(dist.distribution):   # one Categorical per part
                hits[k] += (cat.probs.argmax(1) == a[:, k]).sum().item()
    va = actions[idx]
    baseline = [np.bincount(va[:, k]).max() / len(va) for k in range(len(PARTS))]
    return loss_sum / len(va), hits / len(va), np.array(baseline)


def log_epoch(writer, entry):
    """One epoch of bc_history.json as TensorBoard scalars (x axis: epoch)."""
    e = entry["epoch"]
    writer.add_scalar("bc/train_loss", entry["train_loss"], e)
    writer.add_scalar("bc/val_loss", entry["val_loss"], e)
    for key, v in entry.items():     # val_acc_move etc. (older runs had other action sets)
        if not key.startswith("val_acc"):
            continue
        if isinstance(v, dict):      # the first BC run: accuracy per action
            for k, x in v.items():
                writer.add_scalar(f"bc/{key}/{k}", x, e)
        else:
            writer.add_scalar(f"bc/{key}", v, e)
    for d, v in entry.get("val_loss_per_dir", {}).items():
        name = OLD_DIR_NAMES.get(os.path.basename(d), os.path.basename(d))
        writer.add_scalar(f"bc/val_loss_{name}", v, e)
    writer.flush()


def tb_backfill():
    """Write TensorBoard graphs from bc_history.json for BC runs that don't have them yet."""
    for hist in sorted(os.path.join(d, "bc_history.json") for d in T.all_runs()):
        if not os.path.exists(hist):
            continue
        tb = os.path.join(os.path.dirname(hist), "tb", "BC")
        if os.path.isdir(tb) and os.listdir(tb):
            continue
        with open(hist) as f, SummaryWriter(tb) as writer:
            for entry in json.load(f):
                log_epoch(writer, entry)
        print(f"wrote {tb}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent", type=float, default=0.01, help="entropy bonus: keeps the policy from becoming over-confident")
    parser.add_argument("--window", type=float, default=5,
                        help="only copy what players did in the last this-many seconds before breaking a log")
    parser.add_argument("--val-frac", type=float, default=0.1, help="fraction of segments held out")
    parser.add_argument("--data", nargs="+", default=[DATA_DIR],
                        help="vpt_data.py output folders to train on (e.g. data/vpt_6xx data/vpt_8xx)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="bc", help='label for the run folder, e.g. "bc_6xx-1762_8xx-1957" (data used)')
    parser.add_argument("--tb-backfill", action="store_true",
                        help="just write TensorBoard graphs for existing BC runs (from their bc_history.json)")
    args = parser.parse_args()
    if args.tb_backfill:
        tb_backfill()
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    args.data = [os.path.normpath(d) for d in args.data]
    obs, actions, val, src, files, file_val = load_data(args.data, args.val_frac, args.window)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    n_tr, n_va = int((~file_val).sum()), int(file_val.sum())
    print(f"Train: {len(tr_idx)} examples from {n_tr} segments; validation: {len(va_idx)} from {n_va}")
    for k, d in enumerate(args.data):
        print(f"  {d}: {int(((src == k) & ~val).sum())} train / {int(((src == k) & val).sum())} validation examples")

    run_dir = T.new_run_dir(args.name, "bc")
    model = T.new_model(SpacesOnlyEnv(), tb_dir=os.path.join(run_dir, "tb"))
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    steps_per_epoch = len(tr_idx) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"type": "behaviour cloning", "segments": n_tr + n_va,
                   "segments_per_dir": {d: sum(os.path.dirname(f) == d for f in files) for d in args.data},
                   "examples": int(len(actions)), **vars(args),
                   "env": T.ENV_NAME, "render_res": T.RENDER_RES, "hud_top_row": T.HUD_TOP_ROW,
                   "obs_size": [T.OBS_W, T.OBS_H], "frame_skip": 4, "moves": T.MOVES,
                   "yaw_steps": T.YAW_STEPS, "pitch_steps": T.PITCH_STEPS,
                   "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Run folder: {run_dir}")

    best = float("inf")
    history = []
    writer = SummaryWriter(os.path.join(run_dir, "tb", "BC"))
    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        total = 0.0
        for o_np, a_np in batches(obs, actions, tr_idx, args.batch, shuffle=True):
            o = to_tensor(o_np, device)
            act = torch.as_tensor(a_np, device=device)
            o, act = augment(o, act)
            dist = policy.get_distribution(o)
            loss = -dist.log_prob(act).mean() - args.ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            sched.step()
            total += loss.item()
        val_loss, acc, baseline = evaluate(policy, obs, actions, va_idx, device)
        per_dir = {}
        if len(args.data) > 1:
            for k, d in enumerate(args.data):
                idx = va_idx[src[va_idx] == k]
                if len(idx):
                    per_dir[d] = evaluate(policy, obs, actions, idx, device)[0]
        history.append({"epoch": epoch, "train_loss": total / steps_per_epoch, "val_loss": val_loss,
                        **{f"val_acc_{name}": float(a) for (name, _), a in zip(PARTS, acc)},
                        **({"val_loss_per_dir": per_dir} if per_dir else {})})
        log_epoch(writer, history[-1])
        mark = ""
        if val_loss < best:
            best = val_loss
            model.save(os.path.join(run_dir, T.FINAL_NAME))
            mark = "  (saved)"
        print(f"epoch {epoch:2d}  train loss {total / steps_per_epoch:.3f}  val loss {val_loss:.3f}  " +
              "  ".join(f"{name} {100 * a:.1f}%" for (name, _), a in zip(PARTS, acc)) + mark +
              "".join(f"\n          val loss {os.path.basename(d)} {v:.3f}" for d, v in per_dir.items()))
    print("Always guessing the most common answer would get: " +
          "  ".join(f"{name} {100 * b:.1f}%" for (name, _), b in zip(PARTS, baseline)))
    writer.close()
    with open(os.path.join(run_dir, "bc_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best model (lowest validation loss) saved to {os.path.join(run_dir, T.FINAL_NAME)}")


if __name__ == "__main__":
    main()
