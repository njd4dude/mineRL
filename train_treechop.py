"""
Train an agent to chop trees in MineRLTreechop-v0 using PPO (reinforcement learning).

  python train_treechop.py                 # train (resumes from the latest save if there is one)
  python train_treechop.py --envs 4        # train with 4 Minecraft instances in parallel (default 3)
  python train_treechop.py --watch         # watch the trained agent play
  tensorboard --logdir tb_logs             # view training graphs (in another terminal)

How it learns:
  - Each step the agent sees the 64x64 image and picks one of a few simple actions.
  - It gets +1 reward for each log collected.
  - PPO adjusts the neural network so actions that led to logs become more likely.
"""
import argparse
import glob
import logging
import os

import gym  # MineRL uses the old gym library
import gymnasium  # Stable-Baselines3 uses the newer gymnasium library
import minerl  # noqa: F401  (registers the MineRL envs)
import numpy as np
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


class TreechopEnv(gymnasium.Env):
    """Wraps MineRL Treechop so Stable-Baselines3 can use it:
    image-only observations, a few discrete actions, and each action held for several game steps."""

    def __init__(self, frame_skip=4, render=False):
        super().__init__()
        self.env = gym.make("MineRLTreechop-v0")
        self.frame_skip = frame_skip
        self.render_enabled = render
        self.observation_space = gymnasium.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(len(ACTIONS))

    def _make_action(self, index):
        action = self.env.action_space.no_op()
        action["attack"] = 1
        for key, value in ACTIONS[index].items():
            action[key] = np.array(value) if key == "camera" else value
        return action

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs["pov"], {}

    def step(self, index):
        action = self._make_action(index)
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if self.render_enabled:
                self.env.render()
            if done:
                break
        return obs["pov"], total_reward, done, False, {}

    def close(self):
        self.env.close()


def latest_model():
    """Return the newest saved model (final or checkpoint), or None."""
    files = glob.glob(os.path.join(MODEL_DIR, "*.zip"))
    return max(files, key=os.path.getmtime) if files else None


def make_env():
    # Monitor records episode rewards/lengths so they show up in the logs and TensorBoard
    return Monitor(TreechopEnv())


def train(timesteps, n_steps, n_envs):
    if n_envs == 1:
        env = DummyVecEnv([make_env])
    else:
        # Each Minecraft runs in its own process so they all step at the same time
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    print(f"Running {n_envs} Minecraft instance(s), {n_steps} steps each per update "
          f"({n_steps * n_envs} steps per update)")

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
    parser.add_argument("--envs", type=int, default=3, help="Minecraft instances to run in parallel")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to watch")
    args = parser.parse_args()

    if args.watch:
        watch(args.episodes)
    else:
        train(args.timesteps, args.n_steps, args.envs)
