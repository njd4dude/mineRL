"""
Train an agent to chop trees in MineRLTreechop-v0 using PPO (reinforcement learning).

  python train_treechop.py                 # start a NEW run in runs/model_<date>_<time>/
  python train_treechop.py --name test1    # new run with a label: runs/model_<date>_<time>_test1/
  python train_treechop.py --resume        # continue the latest run
  python train_treechop.py --init runs/model_20261109_224055_bc_multi --name finetune
                                           # fine-tune a behaviour-cloned model (bc_train.py) with PPO
  python train_treechop.py --resume runs/model_20261109_163045   # continue a specific run
  python train_treechop.py --envs 3        # change how many Minecraft instances run in parallel (default 4)
  python train_treechop.py --cpu-render    # draw Minecraft on the CPU instead of the GPU
  tensorboard --logdir runs                # view and compare all runs' training graphs

To see how a model plays:
  python debug_view.py --model latest      # live window of what the agent sees
  python evaluate.py --models latest --record   # score it on fixed worlds and save each game as .mp4

How it learns:
  - Each step the agent sees a small 64x64 image of the world (HUD cropped off) and picks one of a
    few simple actions.
  - It gets +1 reward for each log collected.
  - PPO adjusts the neural network so actions that led to logs become more likely.
"""
import argparse
import functools
import glob
import json
import logging
import os
import sys
from datetime import datetime

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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

logging.basicConfig(level=logging.WARNING)

# Every training run gets its own folder: runs/model_<date>_<time>[_<name>]/ holding its
# checkpoints, final model, TensorBoard logs (tb/) and settings (config.json).
RUNS_DIR = "runs"
FINAL_NAME = "final.zip"

# Each agent decision is three choices made together (a MultiDiscrete action [move, yaw, pitch]),
# so it can walk, turn and look up/down in the same step, with small turns for aiming at a log and
# big ones for turning toward a tree, like the human players in the VPT data do.
# The agent always holds "attack", so it only has to learn where to go and look.
# Camera values are the TOTAL turn per agent decision, in degrees. Each decision lasts frame_skip
# game ticks, so the turn is spread evenly over those ticks (see _make_action).
MOVES = [
    {},                               # 0: stand still (just swing)
    {"forward": 1},                   # 1: walk forward
    {"forward": 1, "jump": 1},        # 2: walk forward + jump
]
MOVE_NAMES = ["stand", "forward", "forward+jump"]
YAW_STEPS = [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]   # degrees; negative = turn left
PITCH_STEPS = [-20, -10, -5, -2, 0, 2, 5, 10, 20]          # degrees; negative = look up
ACTION_DIMS = [len(MOVES), len(YAW_STEPS), len(PITCH_STEPS)]


def describe_action(action):
    """Human-readable text for one [move, yaw, pitch] action."""
    move, yaw, pitch = (int(a) for a in action)
    return f"{MOVE_NAMES[move]}, yaw {YAW_STEPS[yaw]:+d}, pitch {PITCH_STEPS[pitch]:+d}"


def nearest_step(value, steps):
    """Index of the step closest to value (degrees)."""
    return int(np.argmin(np.abs(np.asarray(steps) - value)))


def human_to_action(dyaw, dpitch, forward, jump):
    """The [move, yaw, pitch] action closest to what a human did in one decision's worth of ticks
    (used to label the VPT recordings). Strafing, walking backwards and sprinting have no
    equivalent and are ignored."""
    move = (2 if jump else 1) if forward else 0
    return [move, nearest_step(dyaw, YAW_STEPS), nearest_step(dpitch, PITCH_STEPS)]


# Minecraft 1.16 has a separate item for each wood type
LOG_ITEMS = ["acacia_log", "birch_log", "dark_oak_log", "jungle_log", "oak_log", "spruce_log"]
LOGS_TO_WIN = 64
SPAWN_BIOME = "forest"       # Minecraft 1.16 biome category to spawn in
AXE_ITEM = "iron_axe"         # Treechop starts the agent with one
ENV_NAME = "MineRLTreechopAxe320-v0"   # renamed whenever the env's observations change

# Minecraft's HUD (hotbar, hearts, hunger, XP) has a fixed size in pixels: the hotbar alone is
# 182 px wide, so at 64x64 it filled ~60% of the image. Instead, render at 320x180, cut off the
# HUD rows at the bottom, and shrink what's left to a small image for the network.
RENDER_RES = (320, 180)      # (width, height) Minecraft draws at
HUD_TOP_ROW = 138            # at 320x180 the HUD starts around row 140; keep rows above this
OBS_W, OBS_H = 64, 64        # what the network sees; squashes the 320x138 view sideways, which is
                             # fine as long as training and playing use the same processing


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
        # The axe count tells when the agent died (dying drops the inventory), see TreechopEnv.step.
        # Not H.ObservationFromLifeStats (health): with it, Minecraft instances often hung when
        # 4 ran at once (1-2 of 4 per test, versus none without it).
        return super().create_observables() + [H.FlatInventoryObservation(LOG_ITEMS + [AXE_ITEM])]

    def create_agent_start(self):
        # Treechop's own world settings (generatorOptions with "fixedBiome": 4 = forest) are in the
        # Minecraft 1.11 format and MineRL 1.0 ignores them, so worlds were random (plains, snow...).
        # Spawning in a forest biome is what MineRL 1.0 itself uses (BASALT's PreferredSpawnBiome).
        return super().create_agent_start() + [H.PreferredSpawnBiome(SPAWN_BIOME)]


def register_env():
    try:
        gym.spec(ENV_NAME)
    except gym.error.Error:
        TreechopWithInventory(name=ENV_NAME, resolution=RENDER_RES).register()


class TreechopEnv(gymnasium.Env):
    """Wraps MineRL Treechop so Stable-Baselines3 can use it:
    image-only observations, [move, yaw, pitch] actions, and each action held for several game steps.
    Reward: +1 for each log added to the inventory (computed here, since MineRL's is always 0).
    The episode ends when the agent loses its axe, which almost always means it died (e.g.
    drowned) and dropped everything: otherwise it respawns without the axe, and gets no reward
    until it holds more logs than before it died. (An axe can also wear out after 250 uses;
    without it the agent is just as stuck, so ending the episode is right then too.)"""

    def __init__(self, frame_skip=4):
        super().__init__()
        register_env()  # runs in each worker process too
        self.env = gym.make(ENV_NAME)
        self.prev_logs = 0
        self.last_pov = np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)
        self.needs_restart = False
        self.died = False
        self.frame_skip = frame_skip
        self.on_frame = None  # optional function called with every full-size game frame (evaluate.py --record)
        self.observation_space = gymnasium.spaces.Box(0, 255, (OBS_H, OBS_W, 3), dtype=np.uint8)
        self.action_space = gymnasium.spaces.MultiDiscrete(ACTION_DIMS)

    def _make_action(self, choice):
        """The MineRL action sent on each of the frame_skip ticks of one decision. Buttons are held
        every tick; the camera turn is divided by frame_skip, because MineRL applies the camera
        value on every tick (before that fix, "turn 10" actually turned 40 degrees)."""
        move, yaw, pitch = (int(c) for c in choice)
        action = self.env.action_space.no_op()
        action["attack"] = 1
        for key, value in MOVES[move].items():
            action[key] = value
        action["camera"] = np.array([PITCH_STEPS[pitch], YAW_STEPS[yaw]], dtype=np.float32) / self.frame_skip
        return action

    @staticmethod
    def _count_logs(obs):
        return sum(int(obs["inventory"][item]) for item in LOG_ITEMS)

    def _restart_minecraft(self, reason):
        """Close this env's Minecraft and start a fresh one. MineRL instances sometimes stop
        responding (e.g. a timeout while generating a new world); without this, one stuck
        instance crashed the whole training run."""
        print(f"Minecraft instance problem ({reason}); restarting it", flush=True)
        instances = getattr(self.env.unwrapped, "instances", [])
        try:
            self.env.close()
        except Exception:
            pass
        # MineRL's close() skips instances it no longer thinks are running (e.g. after a timeout),
        # which left their Minecraft processes behind; kill them explicitly
        for instance in instances:
            try:
                instance.kill()
            except Exception:
                pass
        self.env = gym.make(ENV_NAME)
        self.needs_restart = False

    def reset(self, seed=None, options=None, attempts=3):
        super().reset(seed=seed)
        for attempt in range(1, attempts + 1):
            try:
                if self.needs_restart:
                    self._restart_minecraft("failed during the last episode")
                if seed is not None:
                    self.env.seed(seed)  # same seed = same world (MineRL only applies it to this reset)
                obs = self.env.reset()
                break
            except Exception as e:
                if attempt == attempts:
                    raise
                self.needs_restart = True
                print(f"reset failed ({type(e).__name__}: {e}), attempt {attempt}/{attempts}", flush=True)
        self.prev_logs = self._count_logs(obs)
        self.died = False
        if self.on_frame:
            self.on_frame(obs["pov"])
        self.last_pov = process_pov(obs["pov"])
        return self.last_pov, {}

    def step(self, choice):
        action = self._make_action(choice)
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            try:
                obs, _, done, info = self.env.step(action)  # MineRL's own reward is always 0 here
                # When Minecraft stops responding, MineRL doesn't raise: it returns a RANDOM
                # observation (random inventory, so thousands of fake "logs") with info["error"]
                if "error" in info:
                    raise RuntimeError(info["error"])
            except Exception as e:
                # End this episode early ("truncated") without using that step's observation; the
                # next reset() restarts this Minecraft instance
                print(f"step failed ({type(e).__name__}: {e}); ending the episode", flush=True)
                self.needs_restart = True
                return self.last_pov, total_reward, False, True, {"logs": self.prev_logs, "died": False,
                                                                   "env_error": True}
            logs = self._count_logs(obs)
            # +1 per log beyond the most held so far this episode, so logs are never counted twice
            # (e.g. if the inventory reading drops for a moment)
            total_reward += max(logs - self.prev_logs, 0)
            self.prev_logs = max(self.prev_logs, logs)
            if logs >= LOGS_TO_WIN:
                done = True
            if int(obs["inventory"][AXE_ITEM]) == 0:
                self.died = True   # lost the axe: died and dropped everything (or it wore out)
                done = True
            if self.on_frame:
                self.on_frame(obs["pov"])
            if done:
                break
        self.last_pov = process_pov(obs["pov"])
        return self.last_pov, total_reward, done, False, {"logs": self.prev_logs, "died": self.died}

    def close(self):
        self.env.close()


def new_run_dir(name=None):
    """Create runs/model_<date>_<time>[_<name>]/ for a new training run."""
    folder = "model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{name}" if name else "")
    path = os.path.join(RUNS_DIR, folder)
    os.makedirs(path)
    return path


def latest_run():
    """The most recently used run folder, or None. Only training runs (runs/model_*) count, not
    e.g. runs/evals/, which changes whenever an evaluation saves its results."""
    runs = [d for d in glob.glob(os.path.join(RUNS_DIR, "model_*")) if os.path.isdir(d)]
    return max(runs, key=os.path.getmtime) if runs else None


def latest_model(run_dir=None):
    """Newest saved model (final or checkpoint) in a run folder (default: the latest run), or None."""
    run_dir = run_dir or latest_run()
    if not run_dir:
        return None
    files = glob.glob(os.path.join(run_dir, "*.zip"))
    return max(files, key=os.path.getmtime) if files else None


def resolve_model(arg=None):
    """Turn a --model value into a .zip path: None/"latest" = newest model in the latest run,
    a run folder = newest model in that folder, a .zip = that file."""
    if arg in (None, "latest"):
        return latest_model()
    if os.path.isdir(arg):
        return latest_model(arg)
    return arg


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


def model_mismatch(model):
    """Why a loaded model can't drive the current env (older image size or action set), or None."""
    if model.observation_space.shape not in ((OBS_H, OBS_W, 3), (3, OBS_H, OBS_W)):
        return (f"This model was trained on images of shape {model.observation_space.shape}, but the "
                f"environment now gives {(OBS_H, OBS_W, 3)}. It can't be used with this version.")
    space = model.action_space
    if not isinstance(space, gymnasium.spaces.MultiDiscrete) or list(space.nvec) != ACTION_DIMS:
        return (f"This model was trained with a different action set ({space}); the environment now "
                f"uses [move, yaw, pitch] = MultiDiscrete({ACTION_DIMS}). It can't be used with this version.")
    return None


# PPO settings for a model trained from scratch, and for fine-tuning a behaviour-cloned one (--init).
# Fine-tuning changes the network more gently, so PPO doesn't wreck what was copied from humans
# before its value estimates are any good: a lower learning rate, a smaller push toward random
# actions (the entropy bonus), and stopping an update early if the policy changed too much.
SCRATCH = {"learning_rate": 2.5e-4, "ent_coef": 0.01, "target_kl": None}
FINETUNE = {"learning_rate": 1e-4, "ent_coef": 0.001, "target_kl": 0.02}


def new_model(env, n_steps=1024, tb_dir=None, learning_rate=2.5e-4, ent_coef=0.01, target_kl=None):
    """A fresh PPO model. Also used by bc_train.py, so a behaviour-cloned model has exactly the
    same network and settings and can be watched, evaluated or resumed like any other run."""
    return PPO(
        "CnnPolicy",
        env,
        n_steps=n_steps,
        batch_size=64,
        learning_rate=learning_rate,
        ent_coef=ent_coef,  # encourages trying different actions
        target_kl=target_kl,
        verbose=1,
        tensorboard_log=tb_dir,
        device="cuda",
    )


class ValueWarmup(BaseCallback):
    """For the first `updates` PPO updates, train only the value head and keep the policy frozen.
    A behaviour-cloned model's value head was never trained, so its estimates of "how good is this
    action" start out as noise; letting PPO change the policy based on them would undo the cloning.
    Counted with the model's own update counter, so resuming a run doesn't warm up again."""

    def __init__(self, updates):
        super().__init__()
        self.updates = updates
        self.frozen = None

    def _set_frozen(self, frozen):
        if frozen == self.frozen:
            return
        for name, param in self.model.policy.named_parameters():
            if not name.startswith("value_net."):
                param.requires_grad_(not frozen)
        self.frozen = frozen
        print("Value warm-up: policy frozen, training only the value head" if frozen
              else "Value warm-up done: training the whole network")

    def _on_rollout_end(self):
        # runs right before each PPO update; _n_updates counts epochs, n_epochs per update
        done = self.model._n_updates // self.model.n_epochs
        self._set_frozen(done < self.updates)

    def _on_training_end(self):
        self._set_frozen(False)

    def _on_step(self):
        return True


def train(timesteps, n_steps, n_envs, mc_mem=None, resume=None, name=None, init=None, warmup=5):
    # Pick the run folder: continue an existing run, or start a new one
    if resume:
        run_dir = latest_run() if resume == "latest" else resume
        path = latest_model(run_dir) if run_dir else None
        if not path:
            print(f"Nothing to resume: no saved model found in {run_dir or RUNS_DIR + '/'}")
            return
    else:
        init_path = resolve_model(init) if init else None
        if init and (not init_path or not os.path.exists(init_path)):
            print(f"No model found for --init {init}")
            return
        settings = FINETUNE if init else SCRATCH
        run_dir, path = new_run_dir(name), None
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"env": ENV_NAME, "render_res": RENDER_RES, "hud_top_row": HUD_TOP_ROW,
                       "obs_size": [OBS_W, OBS_H], "frame_skip": 4, "moves": MOVES,
                       "yaw_steps": YAW_STEPS, "pitch_steps": PITCH_STEPS,
                       "camera_note": "camera values are total degrees per decision",
                       "envs": n_envs, "n_steps": n_steps, "mc_mem": mc_mem,
                       "init": init_path, "value_warmup_updates": warmup if init else 0, **settings,
                       "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Run folder: {run_dir}")

    env_fn = functools.partial(make_env, mc_mem)
    if n_envs == 1:
        env = DummyVecEnv([env_fn])
    else:
        # Each Minecraft runs in its own process so they all step at the same time
        env = SubprocVecEnv([env_fn for _ in range(n_envs)])
    print(f"Running {n_envs} Minecraft instance(s) with {mc_mem or '4G (default)'} memory each, "
          f"{n_steps} steps each per update ({n_steps * n_envs} steps per update)")

    tb_dir = os.path.join(run_dir, "tb")
    if path:
        print(f"Resuming from {path}")
        try:
            model = PPO.load(path, env=env, device="cuda")
        except ValueError as e:  # e.g. the run used a different image size
            print(f"Can't resume this run with the current settings: {e}")
            env.close()
            return
        model.tensorboard_log = tb_dir  # keep logging into this run's folder
    else:
        model = new_model(env, n_steps, tb_dir, **settings)
        if init_path:
            start = PPO.load(init_path, device="cuda")
            problem = model_mismatch(start)
            if problem:
                print(problem)
                env.close()
                return
            model.policy.load_state_dict(start.policy.state_dict())
            print(f"Starting from the weights of {init_path} (fine-tuning: {settings})")
        else:
            print("Starting a new model")

    # save_freq counts per environment, so divide to save about every 10,000 total steps
    checkpoint = CheckpointCallback(save_freq=max(10_000 // n_envs, 1), save_path=run_dir,
                                    name_prefix="treechop_ckpt")
    callbacks = [checkpoint]
    if not path and init and warmup:
        callbacks.append(ValueWarmup(warmup))
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=False,
                    tb_log_name="PPO")
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    finally:
        final = os.path.join(run_dir, FINAL_NAME)
        model.save(final)
        print(f"Saved model to {final}")
        try:
            env.close()
        except Exception:
            pass  # after Ctrl+C the Minecraft processes may already be gone


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="continue a run instead of starting a new one: a run folder, or no value for the latest run")
    parser.add_argument("--init", default=None,
                        help="start a new run from another model's weights (e.g. a bc_train.py run) and fine-tune it")
    parser.add_argument("--warmup", type=int, default=5,
                        help="with --init: PPO updates at the start that train only the value head")
    parser.add_argument("--name", default=None,
                        help="optional label added to a new run's folder name, e.g. --name smallturns")
    args = parser.parse_args()

    if args.init and args.resume:
        parser.error("use --init to start a fine-tuning run, or --resume to continue one, not both")
    train(args.timesteps, args.n_steps, args.envs, args.mc_mem, args.resume, args.name,
          args.init, args.warmup)
