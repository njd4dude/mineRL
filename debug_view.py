"""
See exactly what the agent sees: the same images, frame skip and reward as training.

  python debug_view.py                      # random actions
  python debug_view.py --model latest       # actions from the newest saved model in models/
  python debug_view.py --model models/treechop_ckpt_50000_steps.zip
  python debug_view.py --save-every 10      # also save every 10th frame to debug_frames/
  python debug_view.py --no-window          # no window (e.g. just saving frames)

Press q in the window to quit.
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines3 import PPO

import train_treechop as T  # same env wrapper, actions and settings as training

ACTION_NAMES = ["forward", "forward+jump", "stand still", "turn left",
                "turn right", "look up", "look down"]


def annotate(pov, scale, step, action, reward, logs, total):
    """Enlarge the observation (nearest-neighbor, so pixels stay sharp) and add a text bar."""
    big = cv2.resize(pov, (pov.shape[1] * scale, pov.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)  # MineRL gives RGB, OpenCV draws BGR
    bar = np.zeros((70, big.shape[1], 3), dtype=np.uint8)
    lines = [f"step {step}   action: {ACTION_NAMES[action] if action is not None else '-'} (+attack)",
             f"reward {reward:.0f}   logs this episode {logs}   total reward {total:.0f}"]
    for i, text in enumerate(lines):
        color = (0, 255, 0) if reward > 0 and i == 1 else (255, 255, 255)
        cv2.putText(bar, text, (8, 25 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return np.vstack([big, bar])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help='"latest", a .zip path, or omit for random actions')
    parser.add_argument("--steps", type=int, default=2000, help="agent steps to run (one episode is 2000)")
    parser.add_argument("--scale", type=int, default=6, help="how much to enlarge the agent's image")
    parser.add_argument("--delay", type=int, default=50, help="ms to wait between frames")
    parser.add_argument("--save-every", type=int, default=0, help="save every Nth frame to debug_frames/ (0 = off)")
    parser.add_argument("--no-window", action="store_true", help="don't open a window")
    args = parser.parse_args()

    model = None
    if args.model:
        path = T.latest_model() if args.model == "latest" else args.model
        if not path:
            print("No saved model found in models/")
            return
        print(f"Using model {path}")
        model = PPO.load(path, device="cuda")
    else:
        print("Using random actions")

    if args.save_every:
        os.makedirs("debug_frames", exist_ok=True)

    env = T.TreechopEnv()
    obs, _ = env.reset()
    print(f"Observation: shape {obs.shape}, dtype {obs.dtype}, values {obs.min()}-{obs.max()}")
    print("(The network gets this same image, rescaled to 0-1 and channels-first.)")

    total, action, reward, info = 0.0, None, 0.0, {"logs": 0}
    try:
        for step in range(args.steps):
            frame = annotate(obs, args.scale, step, action, reward, info["logs"], total)
            if args.save_every and step % args.save_every == 0:
                cv2.imwrite(f"debug_frames/step_{step:05d}.png", frame)
            if not args.no_window:
                cv2.imshow("What the agent sees", frame)
                if cv2.waitKey(args.delay) & 0xFF == ord("q"):
                    break

            if model is not None:
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total += reward
            if reward > 0:
                print(f"step {step}: +{reward:.0f} log(s), logs this episode = {info['logs']}")
            if done:
                print(f"Episode over at step {step}, total reward {total:.0f}")
                break
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
