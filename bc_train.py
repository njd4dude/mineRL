"""
Behaviour cloning: teach the PPO network to copy the human actions in data/vpt_treechop/
(built by vpt_data.py). No Minecraft needed, just the GPU.

  python bc_train.py                     # new run: runs/model_<date>_<time>_bc/
  python bc_train.py --epochs 30 --name bc_long
  python bc_train.py --window 3          # only copy the last 3 s before each log break
  python train_treechop.py --watch       # then watch it (newest run)
  python evaluate.py --models random latest

The result is a normal PPO model (same network and [move, yaw, pitch] actions as
train_treechop.py), so it can be watched, evaluated, and later fine-tuned with PPO.

Training details:
  - Each human input (camera turn in degrees, W / space held) is mapped to the closest env action
    with train_treechop.human_to_action, then the network learns to predict it from the frame
    (cross-entropy on each of the three parts: move, yaw, pitch).
  - Only examples from the --window seconds before a log break are used (walking to the tree,
    aiming, chopping), not wandering around.
  - 10% of segments are held out to check the network isn't just memorising.
  - Augmentation: random brightness/contrast (VPT players often used darker settings than our env),
    small random shifts, and left-right mirroring (with the yaw turn mirrored too).
"""
import argparse
import glob
import json
import os
from datetime import datetime

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F

import train_treechop as T
from vpt_data import labels

DATA_DIR = os.path.join("data", "vpt_treechop")
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


def load_data(val_frac, seed, window_s):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        raise SystemExit(f"No data in {DATA_DIR}/ - run vpt_data.py first")
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))

    def stack(fs):
        obs, actions = [], []
        for f in fs:
            d = np.load(f)
            if "dyaw" not in d:
                raise SystemExit(f"{f} is from an older vpt_data.py; run: python vpt_data.py --relabel")
            keep = d["until_break"] <= window_s * 20   # 20 game ticks per second
            obs.append(d["obs"][keep])
            actions.append(labels(d)[keep])
        return np.concatenate(obs), np.concatenate(actions)

    return stack(files[n_val:]), stack(files[:n_val]), len(files) - n_val, n_val


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


def evaluate(policy, obs, actions, device, batch=1024):
    """Mean loss and per-part accuracy on held-out data, plus each part's "always guess the most
    common answer" accuracy to compare against."""
    policy.set_training_mode(False)
    loss_sum = 0.0
    hits = np.zeros(len(PARTS))
    with torch.no_grad():
        for i in range(0, len(actions), batch):
            o = to_tensor(obs[i:i + batch], device)
            a = torch.as_tensor(actions[i:i + batch], device=device)
            dist = policy.get_distribution(o)
            loss_sum += -dist.log_prob(a).sum().item()
            for k, cat in enumerate(dist.distribution):   # one Categorical per part
                hits[k] += (cat.probs.argmax(1) == a[:, k]).sum().item()
    baseline = [np.bincount(actions[:, k]).max() / len(actions) for k in range(len(PARTS))]
    return loss_sum / len(actions), hits / len(actions), np.array(baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent", type=float, default=0.01, help="entropy bonus: keeps the policy from becoming over-confident")
    parser.add_argument("--window", type=float, default=5,
                        help="only copy what players did in the last this-many seconds before breaking a log")
    parser.add_argument("--val-frac", type=float, default=0.1, help="fraction of segments held out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="bc", help="label for the run folder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    (tr_obs, tr_act), (va_obs, va_act), n_tr, n_va = load_data(args.val_frac, args.seed, args.window)
    print(f"Train: {len(tr_act)} examples from {n_tr} segments; validation: {len(va_act)} from {n_va}")

    run_dir = T.new_run_dir(args.name)
    model = T.new_model(SpacesOnlyEnv(), tb_dir=os.path.join(run_dir, "tb"))
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    steps_per_epoch = len(tr_act) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
    tr_act_t = torch.as_tensor(tr_act)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"type": "behaviour cloning", "data": DATA_DIR, "segments": n_tr + n_va,
                   "examples": int(len(tr_act) + len(va_act)), **vars(args),
                   "env": T.ENV_NAME, "render_res": T.RENDER_RES, "hud_top_row": T.HUD_TOP_ROW,
                   "obs_size": [T.OBS_W, T.OBS_H], "frame_skip": 4, "moves": T.MOVES,
                   "yaw_steps": T.YAW_STEPS, "pitch_steps": T.PITCH_STEPS,
                   "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Run folder: {run_dir}")

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        perm = torch.randperm(len(tr_act))
        total = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            obs = to_tensor(tr_obs[idx.numpy()], device)
            act = tr_act_t[idx].to(device)
            obs, act = augment(obs, act)
            dist = policy.get_distribution(obs)
            loss = -dist.log_prob(act).mean() - args.ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            sched.step()
            total += loss.item()
        val_loss, acc, baseline = evaluate(policy, va_obs, va_act, device)
        history.append({"epoch": epoch, "train_loss": total / steps_per_epoch, "val_loss": val_loss,
                        **{f"val_acc_{name}": float(a) for (name, _), a in zip(PARTS, acc)}})
        mark = ""
        if val_loss < best:
            best = val_loss
            model.save(os.path.join(run_dir, T.FINAL_NAME))
            mark = "  (saved)"
        print(f"epoch {epoch:2d}  train loss {total / steps_per_epoch:.3f}  val loss {val_loss:.3f}  " +
              "  ".join(f"{name} {100 * a:.1f}%" for (name, _), a in zip(PARTS, acc)) + mark)
    print("Always guessing the most common answer would get: " +
          "  ".join(f"{name} {100 * b:.1f}%" for (name, _), b in zip(PARTS, baseline)))
    with open(os.path.join(run_dir, "bc_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best model (lowest validation loss) saved to {os.path.join(run_dir, T.FINAL_NAME)}")


if __name__ == "__main__":
    main()
