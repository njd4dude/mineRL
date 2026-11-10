"""
Train an agent to chop trees in MineRLTreechop-v0 using PPO (reinforcement learning).

  python train_treechop.py                 # train (resumes from the latest save if there is one)
  python train_treechop.py --envs 3        # change how many Minecraft instances run in parallel (default 4)
  python train_treechop.py --cpu-render    # draw Minecraft on the CPU instead of the GPU
  python train_treechop.py --watch         # watch the trained agent play
  tensorboard --logdir tb_logs             # view training graphs (in another terminal)

How it learns:
  - Each step the agent sees a small 128x56 image of the world and picks one of a few simple actions.
  - It gets +1 reward for each log collected.
  - PPO adjusts the neural network so actions that led to logs become more likely.
"""
import argparse
import functools
import glob
import logging
import os
import sys

# In WSL, OpenGL defaults to "llvmpipe" (drawing on the CPU). These settings make Minecraft draw
# on the NVIDIA GPU through Windows DirectX 12 instead. Set before any Minecraft starts, so every
# instance inherits them. Pass --cpu-render to turn this off.
if "--cpu-render" not in sys.argv:
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import cv2
import gym  # MineRL uses the old gym library
import gymnasium  # Stable-Baselines3 uses the newer gymnasium library
import minerl  # noqa: F401  (registers the MineRL envs)
import numpy as np
from minerl.herobraine.env_specs.treechop_specs import Treechop
from minerl.herobraine.hero import handlers as H
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

logging.basicConfig(level=logging.WARNING)

MODEL_DIR = "models"
FINAL_MODEL = os.path.join(MODEL_DIR, "treechop_ppo.zip")

# A small set of actions makes learning much easier than MineRL's full action space.
# The agent always holds "attack", so it only has to learn where to go and look.
ACTIONS = [
    {"forward": 1},                   # 0: walk forward
    {"forward": 1, "jump": 1},        # 1: walk forward + jump
    {},                               # 2: stand still (just swing)
    {"camera": [0, -10]},             # 3: turn left
    {"camera": [0, 10]},              # 4: turn right
    {"camera": [-10, 0]},             # 5: look up
    {"camera": [10, 0]},              # 6: look down
]


# Minecraft 1.16 has a separate item for each wood type
LOG_ITEMS = ["acacia_log", "birch_log", "dark_oak_log", "jungle_log", "oak_log", "spruce_log"]
LOGS_TO_WIN = 64
ENV_NAME = "MineRLTreechopInventory320-v0"

# Minecraft's HUD (hotbar, hearts, hunger, XP) has a fixed size in pixels: the hotbar alone is
# 182 px wide, so at 64x64 it filled ~60% of the image. Instead, render at 320x180, cut off the
# HUD rows at the bottom, and shrink what's left to a small image for the network.
RENDER_RES = (320, 180)      # (width, height) Minecraft draws at
HUD_TOP_ROW = 138            # at 320x180 the HUD starts around row 140; keep rows above this
OBS_W, OBS_H = 128, 56       # what the network sees; keeps the 320x138 aspect ratio (~2.3:1)


def process_pov(pov):
    """Crop off the HUD and shrink the frame to what the network sees."""
    world = pov[:HUD_TOP_ROW]
    small = cv2.resize(world, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small)


class TreechopWithInventory(Treechop):
    """MineRL 1.0's Treechop never gives any reward: its reward looks for the old item name "log"
    (Minecraft 1.16 uses "oak_log" etc.), and MineRL 1.0's Minecraft mod has no reward code at all.
    This version also reports the log counts in the inventory, so we can compute the reward ourselves
    (the same way MineRL's own ObtainDiamondShovel does)."""

    def create_observables(self):
        return super().create_observables() + [H.FlatInventoryObservation(LOG_ITEMS)]


def register_env():
    try:
        gym.spec(ENV_NAME)
    except gym.error.Error:
        TreechopWithInventory(name=ENV_NAME, resolution=RENDER_RES).register()


class TreechopEnv(gymnasium.Env):
    """Wraps MineRL Treechop so Stable-Baselines3 can use it:
    image-only observations, a few discrete actions, and each action held for several game steps.
    Reward: +1 for each log added to the inventory (computed here, since MineRL's is always 0)."""

    def __init__(self, frame_skip=4, render=False):
        super().__init__()
        register_env()  # runs in each worker process too
        self.env = gym.make(ENV_NAME)
        self.prev_logs = 0
        self.frame_skip = frame_skip
        self.render_enabled = render
        self.observation_space = gymnasium.spaces.Box(0, 255, (OBS_H, OBS_W, 3), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(len(ACTIONS))

    def _make_action(self, index):
        action = self.env.action_space.no_op()
        action["attack"] = 1
        for key, value in ACTIONS[index].items():
            action[key] = np.array(value) if key == "camera" else value
        return action

    @staticmethod
    def _count_logs(obs):
        return sum(int(obs["inventory"][item]) for item in LOG_ITEMS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        self.prev_logs = self._count_logs(obs)
        return process_pov(obs["pov"]), {}

    def step(self, index):
        action = self._make_action(index)
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, _, done, info = self.env.step(action)  # MineRL's own reward is always 0 here
            logs = self._count_logs(obs)
            # +1 per log beyond the most held so far this episode, so logs are never counted twice
            # (e.g. if the inventory reading drops, or the agent dies and picks logs back up)
            total_reward += max(logs - self.prev_logs, 0)
            self.prev_logs = max(self.prev_logs, logs)
            if logs >= LOGS_TO_WIN:
                done = True
            if self.render_enabled:
                self.env.render()
            if done:
                break
        return process_pov(obs["pov"]), total_reward, done, False, {"logs": self.prev_logs}

    def close(self):
        self.env.close()


def latest_model():
    """Return the newest saved model (final or checkpoint), or None."""
    files = glob.glob(os.path.join(MODEL_DIR, "*.zip"))
    return max(files, key=os.path.getmtime) if files else None


def set_minecraft_memory(max_mem):
    """MineRL starts every Minecraft with a 4G Java memory limit and gives no option to change it,
    so patch the launcher to pass our own limit (e.g. "2G")."""
    from minerl.env import malmo
    original_init = malmo.MinecraftInstance.__init__
    if getattr(original_init, "_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        if kwargs.get("max_mem") is None:
            kwargs["max_mem"] = max_mem
        original_init(self, *args, **kwargs)

    patched_init._patched = True
    malmo.MinecraftInstance.__init__ = patched_init


def make_env(mc_mem=None):
    # Runs inside each worker process, so the memory patch applies to that process's Minecraft
    if mc_mem:
        set_minecraft_memory(mc_mem)
    # Monitor records episode rewards/lengths so they show up in the logs and TensorBoard
    return Monitor(TreechopEnv())


def train(timesteps, n_steps, n_envs, mc_mem=None):
    env_fn = functools.partial(make_env, mc_mem)
    if n_envs == 1:
        env = DummyVecEnv([env_fn])
    else:
        # Each Minecraft runs in its own process so they all step at the same time
        env = SubprocVecEnv([env_fn for _ in range(n_envs)])
    print(f"Running {n_envs} Minecraft instance(s) with {mc_mem or '4G (default)'} memory each, "
          f"{n_steps} steps each per update ({n_steps * n_envs} steps per update)")

    path = latest_model()
    if path:
        print(f"Resuming from {path}")
        model = PPO.load(path, env=env, device="cuda")
    else:
        print("Starting a new model")
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=n_steps,
            batch_size=64,
            learning_rate=2.5e-4,
            ent_coef=0.01,  # encourages trying different actions
            verbose=1,
            tensorboard_log="tb_logs",
            device="cuda",
        )

    # save_freq counts per environment, so divide to save about every 10,000 total steps
    checkpoint = CheckpointCallback(save_freq=max(10_000 // n_envs, 1), save_path=MODEL_DIR,
                                    name_prefix="treechop_ckpt")
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    finally:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(FINAL_MODEL)
        print(f"Saved model to {FINAL_MODEL}")
        try:
            env.close()
        except Exception:
            pass  # after Ctrl+C the Minecraft processes may already be gone


def watch(episodes):
    path = latest_model()
    if not path:
        print("No trained model found in models/. Train first.")
        return
    print(f"Loading {path}")
    model = PPO.load(path, device="cuda")
    env = TreechopEnv(render=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(int(action))
            total += reward
            if reward > 0:
                print(f"  got {reward:.0f} log(s), total = {total:.0f}")
        print(f"Episode {ep + 1}: {total:.0f} logs")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="watch the trained agent instead of training")
    parser.add_argument("--timesteps", type=int, default=500_000, help="agent steps to train for")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="steps collected per Minecraft instance per PPO update")
    # Defaults picked from benchmarks on this machine (Ryzen 7 7700X, 27 GB RAM in WSL):
    # 4 instances x 2G was fastest (~27-33 steps/s, ~18 GB RAM peak); 5 was no faster.
    parser.add_argument("--envs", type=int, default=4, help="Minecraft instances to run in parallel")
    parser.add_argument("--cpu-render", action="store_true",
                        help="draw Minecraft on the CPU (llvmpipe) instead of the GPU")
    parser.add_argument("--mc-mem", default="2G",
                        help="Java memory limit per Minecraft (MineRL's own default is 4G); try 3G if it crashes")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to watch")
    args = parser.parse_args()

    if args.watch:
        watch(args.episodes)
    else:
        train(args.timesteps, args.n_steps, args.envs, args.mc_mem)
